
See the HTML preview [here](https://htmlpreview.github.io/?https://github.com/GATEOverflow/inference_results_v4.1/blob/main/closed/RedHat/results/L40S-RedHat-OpenShift/summary.html)



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
    <p>RedHat     </p>
    <p>L40S-RedHat-OpenShift    </p>
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
    <td id="tester_val">RedHat</td>
    <th id="sw_avail"><a href="">Availability:</a></th>
    <td id="sw_avail_val">Available  as of August 2024</td>
   </tr>
  </tbody>
 </table>
  
<table>
            <tr><td><h3>Accelerator Details</h3><table><tr><td>accelerators_per_node</td><td>4</td></tr><tr><td>accelerator_model_name</td><td>NVIDIA L40S</td></tr><tr><td>accelerator_host_interconnect</td><td>PCIe Gen5</td></tr><tr><td>accelerator_frequency</td><td></td></tr><tr><td>accelerator_on-chip_memories</td><td></td></tr><tr><td>accelerator_memory_configuration</td><td>HBM2</td></tr><tr><td>accelerator_memory_capacity</td><td>48 GB</td></tr><tr><td>accelerator_interconnect</td><td>PCIe</td></tr><tr><td>accelerator_interconnect_topology</td><td></td></tr></table></td> <td><h3>Processor and Memory Details</h3><table><tr><td>host_processors_per_node</td><td>2</td></tr><tr><td>host_processor_model_name</td><td>Intel(R) Xeon(R) Platinum 8480CL</td></tr><tr><td>host_processor_core_count</td><td>112</td></tr><tr><td>host_processor_vcpu_count</td><td>-</td></tr><tr><td>host_processor_frequency</td><td></td></tr><tr><td>host_processor_caches</td><td></td></tr><tr><td>host_processor_interconnect</td><td>PCIe</td></tr><tr><td>host_memory_capacity</td><td>2 TB</td></tr><tr><td>host_memory_configuration</td><td>32x 64GB Micron DDR5</td></tr></table></td> </tr>
            <tr><td ><h3>Other Hardware Details</h3><table><tr><td>cooling</td><td>NA</td></tr><tr><td>hw_notes</td><td>NVIDIA L40S-48GB</td></tr></table></td> <td><h3>Network and Interconnect Details</h3><table><tr><td>host_networking</td><td>Management: 1x Ethernet 10GB/Sec</td></tr><tr><td>host_networking_topology</td><td>-</td></tr><tr><td>host_network_card_count</td><td>-</td></tr></table></td> </tr>
            <tr><td colspan="2"><h3>Software Details</h3><table><tr><td>framework</td><td>CUDA 12.2</td></tr><tr><td>other_software_stack</td><td>{'cuda_driver_version': '535.129.03', 'cuda_version': '12.2', 'vllm': '0.5.1'}</td></tr><tr><td>operating_system</td><td>Red Hat Enterprise Linux CoreOS release 4.14</td></tr><tr><td>sw_notes</td><td>Red Hat OpenShift Container Platform 4.14 + OpenShift AI</td></tr></table></td> </tr>
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
    </tr><tr><td>llama2-70b-99.9</td><td>ROUGE1: 44.3868, ROUGE2: 22.0132, ROUGEL: 28.5876, TOKENS_PER_SAMPLE: 265.005</td><td>Tokens/s</td> <td>1469.19</td><td>Tokens/s</td> <td>1717.77</td></table>

