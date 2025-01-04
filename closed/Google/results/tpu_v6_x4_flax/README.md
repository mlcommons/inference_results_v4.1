
See the HTML preview [here](https://htmlpreview.github.io/?https://github.com/GATEOverflow/inference_results_v4.1/blob/main/closed/Google/results/tpu_v6_x4_flax/summary.html)



<div class="resultpage">
 <div class="titlebarcontainer">
  <div class="logo">
   <a href="/" style="border: none"><img src="" alt="" /></a>
  </div>
  <div class="titlebar">
   <h1 class="title">MLPerf Inference v4.1</h1>
   <p style="font-size: smaller">Copyright 2019-2025 MLCommons</p>
  </div>
 </div>
 <table class="titlebarcontainer">
  <tr>
   <td class="headerbar" rowspan="2">
    <p>Google     </p>
    <p>tpu-v6-4    </p>
   </td>
  </tr>
 </table>
 <table class="datebar">
  <tbody>
   <tr>
    <th id="license_num"><a href="">MLPerf Inference Category:</a></th>
    <td id="license_num_val">datacenter</td>
    <th id="test_date"><a href="">MLPerf Inference Division:</a></th>
    <td id="test_date_val">closed</td>
   </tr>
   <tr>
    <th id="tester"><a href="">Submitted by:</a></th>
    <td id="tester_val">Google</td>
    <th id="sw_avail"><a href="">Availability:</a></th>
    <td id="sw_avail_val">Preview as of February 2025, should be avaiable within 180 days</td>
   </tr>
  </tbody>
 </table>
  
<table>
            <tr><td><h3>Accelerator Details</h3><table><tr><td>accelerator_host_interconnect</td><td>TODO</td></tr><tr><td>accelerator_interconnect</td><td>TODO</td></tr><tr><td>accelerator_memory_capacity</td><td>32 GB</td></tr><tr><td>accelerator_memory_configuration</td><td>HBM3</td></tr><tr><td>accelerator_model_name</td><td>TPU v6</td></tr><tr><td>accelerators_per_node</td><td>4</td></tr><tr><td>accelerator_frequency</td><td></td></tr><tr><td>accelerator_interconnect_topology</td><td></td></tr><tr><td>accelerator_on-chip_memories</td><td></td></tr></table></td> <td><h3>Processor and Memory Details</h3><table><tr><td>host_memory_capacity</td><td>720 GB</td></tr><tr><td>host_memory_configuration</td><td>TODO</td></tr><tr><td>host_processor_core_count</td><td>180</td></tr><tr><td>host_processor_model_name</td><td>AMD EPYC 9B14</td></tr><tr><td>host_processors_per_node</td><td>1</td></tr><tr><td>host_processor_caches</td><td></td></tr><tr><td>host_processor_frequency</td><td></td></tr><tr><td>host_processor_interconnect</td><td></td></tr></table></td> </tr>
            <tr><td ><h3>Other Hardware Details</h3><table><tr><td>cooling</td><td>TODO</td></tr><tr><td>hw_notes</td><td></td></tr></table></td> <td><h3>Network and Interconnect Details</h3><table><tr><td>host_network_card_count</td><td>TODO</td></tr><tr><td>host_networking_topology</td><td>TODO</td></tr><tr><td>host_networking</td><td>TODO</td></tr></table></td> </tr>
            <tr><td colspan="2"><h3>Software Details</h3><table><tr><td>operating_system</td><td>Linux version 6.2.0-1019-gcp (buildd@lcy02-amd64-032) (x86_64-linux-gnu-gcc-12 (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0, GNU ld (GNU Binutils for Ubuntu) 2.38) #21~22.04.1-Ubuntu SMP Thu Nov 16 18:18:34 UTC 2023</td></tr><tr><td>framework</td><td>flax</td></tr><tr><td>other_software_stack</td><td>{'JAX TPU runtime': 'flax==0.8.5, jax[tpu]==0.4.30'}</td></tr><tr><td>sw_notes</td><td></td></tr></table></td> </tr>
            </table>

<h3>Results Table</h3>
<table>
    <tr>
        <th rowspan="2">Model</th>
        <th rowspan="2">Accuracy Target</th>
        <th colspan="2">Server</th>
        <th colspan="2">Offline</th>
    </tr>
    <tr>
    <td>Metric</td>
    <td>Performance</td>
    <td>Metric</td>
    <td>Performance</td>
    </tr><tr><td>stable-diffusion-xl</td><td>CLIP_SCORE: 31.6863, FID_SCORE: 23.0109</td><td>Queries/s</td> <td>4.48577</td><td>Samples/s</td> <td>5.43896</td></table>

