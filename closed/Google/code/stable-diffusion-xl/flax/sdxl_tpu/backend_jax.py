# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from jax.sharding import PartitionSpec as P
from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache(os.path.expanduser("~/jax_cache"))
from maxdiffusion import pyconfig
from maxdiffusion import (
            FlaxStableDiffusionXLPipeline
          )

import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import PositionalSharding
import numpy as np
import functools
from infer_jax_utils import vae_decode, tokenize, get_embeddings, get_add_time_ids
from maxdiffusion.max_utils import (
  create_device_mesh,
  get_states,
  device_put_replicated,
  get_flash_block_sizes,
)
import base64
from maxdiffusion.image_processor import VaeImageProcessor
from io import BytesIO
import jax.numpy as jnp
from jax.sharding import PositionalSharding
import functools
from flax.linen import partitioning as nn_partitioning


class StableDiffusion:

    def __init__(self, dataset_path, config='configs/config.yml', latents_path="coco2014/latents/latents.npy"):
        pyconfig.initialize([None, config])
        self.config = pyconfig.config
        self.rng = jax.random.PRNGKey(self.config.seed)

        # Setup Mesh
        self.devices_array = create_device_mesh(self.config)
        self.mesh = Mesh(self.devices_array, self.config.mesh_axes)

        self.batch_size = self.config.per_device_batch_size * jax.device_count()
        flash_block_sizes = get_flash_block_sizes(self.config)
        self.pipeline, self.params = FlaxStableDiffusionXLPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            revision=self.config.revision,
            dtype=jnp.bfloat16,
            split_head_dim=self.config.split_head_dim,
            attention_kernel=self.config.attention,
            flash_block_sizes=flash_block_sizes,
            mesh=self.mesh,
        )
        self.scheduler_state = self.params.pop("scheduler")
        self.params = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16), self.params)
        self.params["scheduler"] = self.scheduler_state

        self.data_sharding = jax.sharding.NamedSharding(
            self.mesh, P(*self.config.data_sharding))
        self.sharding = PositionalSharding(self.devices_array).replicate()
        partial_device_put_replicated = functools.partial(device_put_replicated, sharding=self.sharding)
        self.params["text_encoder"] = jax.tree_util.tree_map(partial_device_put_replicated, self.params["text_encoder"])
        self.params["text_encoder_2"] = jax.tree_util.tree_map(partial_device_put_replicated, self.params["text_encoder_2"])
        self.unet_state, self.unet_state_mesh_shardings, self.vae_state, self.vae_state_mesh_shardings = get_states(
            self.mesh, None, self.rng, self.config, self.pipeline, self.params["unet"], self.params["vae"], training=False)
        del self.params["vae"]
        del self.params["unet"]

        print(dataset_path)
        print(latents_path)
        latents = [np.load(latents_path) for _ in range(self.batch_size)]
        self.latents = np.concatenate(latents, axis=0)

        negative_prompt_ids = [self.config.negative_prompt] * self.batch_size
        self.negative_prompt_ids = tokenize(
            negative_prompt_ids, self.pipeline)

        self.negative_prompt_embeds, self.negative_pooled_embeds = get_embeddings(
            self.negative_prompt_ids, self.pipeline, self.params)

        self.p_generate_image = jax.jit(
            functools.partial(self.generate_image),
            in_shardings=(self.unet_state_mesh_shardings,
                          self.vae_state_mesh_shardings, None),
            out_shardings=None
        )
        prompt_ids = self.prepare_inputs_for_generate(
            ["warmup"]*self.batch_size)

        self.p_generate_image(self.unet_state, self.vae_state, prompt_ids)

        # Init Complete
        print("Initialized")

    def get_unet_inputs(self, prompt_ids, latents):

        guidance_scale = self.config.guidance_scale
        num_inference_steps = self.config.num_inference_steps
        height = self.config.resolution
        width = self.config.resolution
        prompt_embeds, pooled_embeds = get_embeddings(
            prompt_ids, self.pipeline, self.params)
        batch_size = prompt_embeds.shape[0]
        
        add_time_ids = get_add_time_ids(
            (height, width), (0, 0), (height, width), prompt_embeds.shape[0], dtype=prompt_embeds.dtype
        )

        prompt_embeds = jnp.concatenate(
            [self.negative_prompt_embeds, prompt_embeds], axis=0)
        add_text_embeds = jnp.concatenate(
            [self.negative_pooled_embeds, pooled_embeds], axis=0)
        add_time_ids = jnp.concatenate([add_time_ids, add_time_ids], axis=0)
        # Ensure model output will be `float32` before going into the scheduler
        guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)
        scheduler_state = self.pipeline.scheduler.set_timesteps(
            self.params["scheduler"],
            num_inference_steps=num_inference_steps,
            shape=latents.shape
        )

        latents = latents * scheduler_state.init_noise_sigma

        added_cond_kwargs = {
            "text_embeds": add_text_embeds, "time_ids": add_time_ids}
        latents = jax.device_put(latents, self.data_sharding)
        prompt_embeds = jax.device_put(prompt_embeds, self.data_sharding)
        added_cond_kwargs['text_embeds'] = jax.device_put(
            added_cond_kwargs['text_embeds'], self.data_sharding)
        added_cond_kwargs['time_ids'] = jax.device_put(
            added_cond_kwargs['time_ids'], self.data_sharding)

        return latents, prompt_embeds, added_cond_kwargs, guidance_scale, scheduler_state

    def generate_image(self, unet_state, vae_state, prompt_ids):
        (latents,
         prompt_embeds,
         added_cond_kwargs,
         guidance_scale,
         scheduler_state) = self.get_unet_inputs(prompt_ids, self.latents)

        def loop_body(step, args, model, pipeline, added_cond_kwargs, prompt_embeds, guidance_scale):
            latents, scheduler_state, state = args
            latents_input = jnp.concatenate([latents] * 2)

            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
            timestep = jnp.broadcast_to(t, latents_input.shape[0])

            latents_input = pipeline.scheduler.scale_model_input(
                scheduler_state, latents_input, t)
            noise_pred = model.apply(
                {"params": state.params},
                jnp.array(latents_input),
                jnp.array(timestep, dtype=jnp.int32),
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs
            ).sample

            noise_pred_uncond, noise_prediction_text = jnp.split(
                noise_pred, 2, axis=0)
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_prediction_text - noise_pred_uncond)

            latents, scheduler_state = pipeline.scheduler.step(
                scheduler_state, noise_pred, t, latents).to_tuple()

            return latents, scheduler_state, state
        
        loop_body_p = functools.partial(loop_body, model=self.pipeline.unet,
                                        pipeline=self.pipeline,
                                        added_cond_kwargs=added_cond_kwargs,
                                        prompt_embeds=prompt_embeds,
                                        guidance_scale=guidance_scale)

        vae_decode_p = functools.partial(
            vae_decode, pipeline=self.pipeline)
        with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            latents, _, _ = jax.lax.fori_loop(0, self.config.num_inference_steps,
                                              loop_body_p, (latents, scheduler_state, unet_state))
        image = vae_decode_p(latents, vae_state)
        return image

    def prepare_inputs_for_generate(self, inputs):
        input_given_len = len(inputs)
        if input_given_len < self.batch_size:
            input_padding = [inputs[-1]]*(self.batch_size - input_given_len)
            inputs += input_padding

        prompt_ids = tokenize(inputs, self.pipeline)
        return prompt_ids

    def predict(self, prompts):
        prompt_len = len(prompts)
        try:

            prompt_ids = self.prepare_inputs_for_generate(prompts)
            images = self.p_generate_image(
                self.unet_state, self.vae_state, prompt_ids)

            images.block_until_ready()
            images = (np.array(images) * 255).round().astype("uint8")
            return images[:prompt_len]
            # return VaeImageProcessor.numpy_to_pil(np.array(images))[:prompt_len]
        except Exception as e:
            pass

        return []
