
See the HTML preview [here](https://htmlpreview.github.io/?https://github.com/mlcommons/inference_results_v4.1/blob/main/open/NeuralMagic/results/vLLM_8xL40S-reference-cpu-pytorch-v2.3.1-default_config/summary.html)



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
    <p>NeuralMagic     </p>
    <p>Crusoe Cloud L40S (8x L40S PCIe, vLLM)    </p>
   </td>
  </tr>
 </table>
 <table class="datebar">
  <tbody>
   <tr>
    <th id="license_num"><a href="">MLPerf Inference Category:</a></th>
    <td id="license_num_val">datacenter</td>
    <th id="test_date"><a href="">MLPerf Inference Division:</a></th>
    <td id="test_date_val">open</td>
   </tr>
   <tr>
    <th id="tester"><a href="">Submitted by:</a></th>
    <td id="tester_val">NeuralMagic</td>
    <th id="sw_avail"><a href="">Availability:</a></th>
    <td id="sw_avail_val">Available  as of August 2024</td>
   </tr>
  </tbody>
 </table>
  
<table>
            <tr><td><h3>Accelerator Details</h3><table><tr><td>accelerator_frequency</td><td></td></tr><tr><td>accelerator_host_interconnect</td><td>PCIe Gen4 x16</td></tr><tr><td>accelerator_interconnect</td><td>N/A</td></tr><tr><td>accelerator_interconnect_topology</td><td></td></tr><tr><td>accelerator_memory_capacity</td><td>48 GB</td></tr><tr><td>accelerator_memory_configuration</td><td>GDDR6</td></tr><tr><td>accelerator_model_name</td><td>NVIDIA L40S</td></tr><tr><td>accelerator_on-chip_memories</td><td></td></tr><tr><td>accelerators_per_node</td><td>8</td></tr></table></td> <td><h3>Processor and Memory Details</h3><table><tr><td>host_memory_capacity</td><td>1.5T</td></tr><tr><td>host_memory_configuration</td><td>undefined</td></tr><tr><td>host_processor_caches</td><td>L1d cache: 1.3 MiB (40 instances), L1i cache: 1.3 MiB (40 instances), L2 cache: 40 MiB (40 instances), L3 cache: 160 MiB (5 instances)</td></tr><tr><td>host_processor_core_count</td><td>4</td></tr><tr><td>host_processor_frequency</td><td>undefined</td></tr><tr><td>host_processor_interconnect</td><td></td></tr><tr><td>host_processor_model_name</td><td>AMD EPYC 9254 24-Core Processor</td></tr><tr><td>host_processors_per_node</td><td>2</td></tr></table></td> </tr>
            <tr><td ><h3>Other Hardware Details</h3><table><tr><td>cooling</td><td>air</td></tr><tr><td>hw_notes</td><td></td></tr></table></td> <td><h3>Network and Interconnect Details</h3><table><tr><td>host_network_card_count</td><td>1</td></tr><tr><td>host_networking</td><td>Gig Ethernet</td></tr><tr><td>host_networking_topology</td><td>N/A</td></tr></table></td> </tr>
            <tr><td colspan="2"><h3>Software Details</h3><table><tr><td>framework</td><td>vLLM 0.5.2</td></tr><tr><td>operating_system</td><td>Ubuntu 22.04 (linux-5.15.0-94-generic-glibc2.35)</td></tr><tr><td>other_software_stack</td><td>Python: 3.10.12, LLVM-15.0.6</td></tr><tr><td>sw_notes</td><td>Automated by MLCommons CM v2.3.3. </td></tr></table></td> </tr>
            </table>

<h3>Results Table</h3>
<table>
    <tr>
        <th rowspan="2">Model</th>
        <th rowspan="2">Accuracy Target</th>
        <th colspan="3">Server</th>
        <th colspan="3">Offline</th>
    </tr>
    <tr><td> Accuracy </td>
    <td>Metric</td>
    <td>Performance</td><td> Accuracy </td>
    <td>Metric</td>
    <td>Performance</td>
    </tr><tr><td>llama2-70b-99</td><td>ROUGE1: 43.9869, ROUGE2: 21.8148, ROUGEL: 28.33, TOKENS_PER_SAMPLE: 265.005</td><td></td><td></td><td></td><td>ROUGE1: 44.1411  ROUGE2: 21.8139  ROUGEL: 28.4395  TOKENS_PER_SAMPLE: 292.9</td><td>Tokens/s</td> <td>923.333</td><tr><td>llama2-70b-99.9</td><td>ROUGE1: 44.3868, ROUGE2: 22.0132, ROUGEL: 28.5876, TOKENS_PER_SAMPLE: 265.005</td><td></td><td></td><td></td><td>ROUGE1: 44.1411  ROUGE2: 21.8139  ROUGEL: 28.4395  TOKENS_PER_SAMPLE: 292.9</td><td>Tokens/s</td> <td>923.333</td></table>

