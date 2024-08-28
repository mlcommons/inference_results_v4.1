# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
import json
import matplotlib.pyplot as plt
import sys
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    # Adding arguments
    parser.add_argument('--token_recv_stats_file', type=str, help='Path to the frontend stats file', default='/work/token_recv_stats.json')
    parser.add_argument('--sent_sample_stats', type=str, help='Path to the frontend stats file', default='/work/sent_sample_stats.json')
    parser.add_argument('--dispatcher_stats_file', type=str, help='Path to the dispatcher stats file', default='/work/seq_stats.json')
    parser.add_argument('--k', type=int, help='Num samples for running average of output seq len', default=5)
    parser.add_argument('-t1', '--title1', type=str, help='Token plot title', default="rate of tokens recd")
    parser.add_argument('-t2', '--title2', type=str, help='Output plot title', default="running average of last {k} out seqs")

    # Parsing arguments
    args = parser.parse_args()

    return args


def plot_ttft_stats(args, idx):
    with open(args.sent_sample_stats, 'r') as f:
        disp_stats = json.load(f)
    with open(args.token_recv_stats_file, 'r') as f:
        tok_recv_stats = json.load(f)
        num_gpus = len(tok_recv_stats)
        idcs = tok_recv_stats[0][0]
        cum_tok_recv_stats = [idcs]
        for gpu_tokens in tok_recv_stats:
            gpu_tokens.pop(0)
            cum_tok_recv_stats.extend(gpu_tokens)
        tok_recv_stats = cum_tok_recv_stats

    ttft = {}  # ttft[sample_id] = ttft
    isl = {}
    sample_id_order = []

    # get sample dispatch times
    idcs = disp_stats.pop(0)
    id_idx = idcs.index('sample_id')
    ts_idx = idcs.index('timestamp')
    isl_idx = idcs.index('isl')

    for dispatched_sample in disp_stats:
        sample_id, ts = int(dispatched_sample[id_idx]), dispatched_sample[ts_idx]
        ttft[sample_id] = ts
        isl[sample_id] = dispatched_sample[isl_idx]
        sample_id_order.append(sample_id)

    # get first token times
    idcs = tok_recv_stats.pop(0)
    ts_idx = idcs.index('timestamp')
    ttyp_idx = idcs.index('tok_type')
    dsptch_idx = idcs.index('num_dispatched')
    id_idx = idcs.index('sample_id')
    gpu_util_idx = idcs.index('gpu_util')
    out_len_idx = idcs.index('out_len')

    ttft_2 = {}
    for token in tok_recv_stats:
        tok_type = token[ttyp_idx]
        if tok_type != '1':
            continue

        sample_id, recv_time = int(token[id_idx]), token[ts_idx]
        ttft_2[sample_id] = recv_time - ttft[sample_id]

    ordered_ttft = []
    ordered_ttft_last_k = [0] * args.k
    ordered_isl = []
    isl_last_k = [0] * args.k
    num_invalid_ttfts = 0
    max_ttft = []
    max_ttft_so_far = 0
    k_idx = 0
    for sample_id in sample_id_order:
        if sample_id not in ttft_2:
            continue
        ordered_ttft_last_k[k_idx] = ttft_2[sample_id]
        isl_last_k[k_idx] = isl[sample_id]
        max_ttft_so_far = max(max_ttft_so_far, ttft_2[sample_id])
        max_ttft.append(max_ttft_so_far)
        if ttft_2[sample_id] > 2:
            num_invalid_ttfts += 1
        k_idx += 1
        k_idx %= args.k
        ttft_avg = sum(ordered_ttft_last_k) / args.k
        isl_avg = sum(isl_last_k) / args.k
        ordered_ttft.append(ttft_avg)
        ordered_isl.append(isl_avg)

    print("Plotting TTFTs")
    xlabels = list(range(len(ttft_2)))
    plt.figure(idx)
    plt.plot(xlabels, ordered_ttft, label=f"TTFT ({num_invalid_ttfts} samples have TTFT > 2s)", color="red")
    plt.plot(xlabels, max_ttft, label=f"Max TTFT so far", color="blue")
    plt.xlabel('sample in order received')
    plt.ylabel('TTFT in sec')
    plt.title(f"Avg TTFT of last {args.k} samples")
    plt.legend()
    plt.savefig('plot_ttft_raw.png', dpi=300)
    plt.close(idx)
    idx += 1

    print("Plotting ISLs")
    plt.figure(idx)
    plt.plot(xlabels, ordered_isl, label=f"Running avg of last {args.k} input sequence lengths", color="green")
    plt.xlabel('sample in order received')
    plt.ylabel('Input seq length')
    plt.legend()
    plt.savefig('plot_isl.png', dpi=300)
    plt.close(idx)
    idx += 1
    return idx


def plot_dispatcher_stats(args, idx):
    with open(args.dispatcher_stats_file, 'r') as f:
        stats = json.load(f)

    idcs = stats.pop(0)

    id_idx = idcs.index('sample_id')
    is_final_idx = idcs.index('is_final')
    token_idx = idcs.index('token')
    ts_idx = idcs.index('ts')

    times = []
    sample_stats = {}
    k = 20
    last_k_samples_idx = 0
    first_k_samples = []
    last_k_samples = [-1] * k
    last_k_set = set()
    start_time = stats[0][ts_idx]
    for stat in stats:
        ts = stat[ts_idx] - start_time
        times.append(ts)
        sample_id = stat[id_idx]
        if sample_id in sample_stats:
            sample_stats[sample_id].append(ts - sample_stats[sample_id][0])
        else:
            sample_stats[sample_id] = [ts]

        if stat[is_final_idx]:
            if sample_id not in last_k_set:
                remove = last_k_samples[last_k_samples_idx]
                if remove >= 0:
                    last_k_set.remove(remove)
                last_k_samples[last_k_samples_idx] = sample_id
                last_k_samples_idx += 1
                last_k_samples_idx %= k
                last_k_set.add(sample_id)
            if len(first_k_samples) < k:
                first_k_samples.append(sample_id)

    print("Plotting samples token time..")
    plt.figure(idx)
    plt.ylabel('Times')
    plt.xlabel('sample IDs')
    x_labels = []
    labels = []
    idx_x = 1
    for sample_id in last_k_samples:
        sample_stats[sample_id][0] = 0
        y = sample_stats[sample_id]
        x = [idx_x] * len(y)
        x_labels.append(idx_x)
        plt.scatter(x, y, s=2)
        labels.append("{}".format(len(y)))
        idx_x += 1
    plt.xticks(ticks=x_labels, labels=labels, rotation=45, ha='right')
    plt.title("last {k} completed sequences".format(k=k))
    plt.savefig('plot_last_sample_tokens.png', dpi=300)

    plt.figure(idx + 1)
    plt.ylabel('Times')
    plt.xlabel('sample IDs')
    x_labels = []
    labels = []
    idx_x = 1
    for sample_id in first_k_samples:
        sample_stats[sample_id][0] = 0
        y = sample_stats[sample_id]
        x = [idx_x] * len(y)
        x_labels.append(idx_x)
        plt.scatter(x, y, s=2)
        labels.append("{}".format(len(y)))
        idx_x += 1
    plt.xticks(ticks=x_labels, labels=labels, rotation=45, ha='right')
    plt.title("first {k} completed sequences".format(k=k))
    plt.savefig('plot_first_sample_tokens.png', dpi=300)


def plot_token_rate_stats(args, idx):
    with open(args.token_recv_stats_file, 'r') as f:
        stats = json.load(f)

    num_gpus = len(stats)
    for gpu_idx in range(num_gpus):
        gpu_stat = stats[gpu_idx]
        idcs = gpu_stat.pop(0)

        ts_idx = idcs.index('timestamp')
        ttyp_idx = idcs.index('tok_type')
        dsptch_idx = idcs.index('num_dispatched')
        id_idx = idcs.index('sample_id')
        gpu_util_idx = idcs.index('gpu_util')
        out_len_idx = idcs.index('out_len')

        times = []
        first_tok_count = []
        interm_tok_count = []
        final_tok_count = []
        total_tok_count = []
        num_dispatched = []
        gpu_util = []
        out_lens = []
        out_lens_last_k = [0] * args.k
        out_lens_last_k_index = 0

        first, interm, final = 0, 0, 0
        num_samples_dispatched = 0

        max_out_len, min_out_len = 0, 0

        start_time = gpu_stat[0][ts_idx]
        num = 0
        for stat in gpu_stat:
            times.append(stat[ts_idx] - start_time)
            if stat[ttyp_idx] == "1":
                first += 1
            elif stat[ttyp_idx] == "I":
                interm += 1
            elif stat[ttyp_idx] == "C":
                final += 1
            else:
                raise AssertionError

            first_tok_count.append(first)
            interm_tok_count.append(interm)
            final_tok_count.append(final)
            total_tok_count.append(first + interm + final)

            num_dispatched.append(stat[dsptch_idx])
            num_samples_dispatched = stat[dsptch_idx]
            util = stat[gpu_util_idx] / 100
            if util >= 0:
                gpu_util.append(stat[gpu_util_idx] / 100)
            num += 1
            if num % 100000 == 0:
                print("Done {} toks".format(num))

            if stat[out_len_idx] >= 0:
                out_lens_last_k[out_lens_last_k_index] = stat[out_len_idx]
                out_lens_last_k_index += 1
                out_lens_last_k_index %= args.k
            len_avg = sum(out_lens_last_k) / args.k
            out_lens.append(len_avg)

        first_tok_count = [f / num_samples_dispatched for f in first_tok_count]
        interm_tok_count = [f / interm for f in interm_tok_count]
        final_tok_count = [f / num_samples_dispatched for f in final_tok_count]
        num_dispatched = [f / num_samples_dispatched for f in num_dispatched]

        print("Plotting token stats..")
        plt.figure(idx)
        plt.plot(times, first_tok_count, label=f"first tokens ({first})", color="red")
        plt.plot(times, interm_tok_count, label=f"interm tokens ({interm})", color="green")
        plt.plot(times, final_tok_count, label=f"final tokens ({final})", color="blue")
        plt.plot(times, num_dispatched, label=f"number samples dispatched ({num_samples_dispatched})", color="yellow")
        if len(gpu_util) > 0:
            plt.plot(times, gpu_util, label="GPU utilization", color="black", linestyle=":", linewidth=1)
        plt.xlabel('Times')
        plt.ylabel('Ratio of tokens done')
        plt.title(args.title1)
        plt.legend()

        # Save the plot as a PNG file
        print("Saving token plot..")
        plt.savefig(f'plot_tokens_{gpu_idx}.png', dpi=300)
        plt.close(idx)
        idx += 1

        print("Plotting output lengths..")
        plt.figure(idx)
        plt.plot(times, out_lens, label=f"output sequence lengths", color="black")
        plt.xlabel('Times')
        plt.ylabel('Token length of response')
        plt.title(args.title2.format(k=args.k))
        plt.savefig(f'plot_lens_{gpu_idx}.png', dpi=300)
        plt.close(idx)
        idx += 1

        print("Plotting cumulative token counts..")
        plt.figure(idx)
        plt.plot(times, total_tok_count, label=f"number_tokens", color="red")
        plt.xlabel('Times')
        plt.ylabel('Num tokens')
        plt.title("all tokens received")
        plt.savefig(f'plot_all_tokens_{gpu_idx}.png', dpi=300)
        plt.close(idx)
        idx += 1
    return idx


fig_idx = 0
args = parse_args()
fig_idx = plot_ttft_stats(args, fig_idx)
fig_idx = plot_token_rate_stats(args, fig_idx)

# if os.path.isfile(args.dispatcher_stats_file):
#     plot_dispatcher_stats(args, 4)

print("Done")
# Close the plot
plt.close('all')
