
See the HTML preview [here](https://htmlpreview.github.io/?https://github.com/mlcommons/inference_results_v4.1/blob/main/closed/NVIDIA/results/B200-SXM-180GBx1_TRT/summary.html)



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
    <p>NVIDIA     </p>
    <p>NVIDIA B200 (1x B200-SXM-180GB, TensorRT)    </p>
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
    <td id="tester_val">NVIDIA</td>
    <th id="sw_avail"><a href="">Availability:</a></th>
    <td id="sw_avail_val">Preview  as of August 2024, should be avaiable within 180 days</td>
   </tr>
  </tbody>
 </table>
  
<table>
            <tr><td><h3>Accelerator Details</h3><table><tr><td>accelerator_frequency</td><td></td></tr><tr><td>accelerator_host_interconnect</td><td>PCIe Gen5 x16</td></tr><tr><td>accelerator_interconnect</td><td>N/A</td></tr><tr><td>accelerator_interconnect_topology</td><td></td></tr><tr><td>accelerator_memory_capacity</td><td>180 GB</td></tr><tr><td>accelerator_memory_configuration</td><td>HBM3e</td></tr><tr><td>accelerator_model_name</td><td>NVIDIA B200-SXM-180GB</td></tr><tr><td>accelerator_on-chip_memories</td><td></td></tr><tr><td>accelerators_per_node</td><td>1</td></tr></table></td> <td><h3>Processor and Memory Details</h3><table><tr><td>host_memory_capacity</td><td>408GB</td></tr><tr><td>host_memory_configuration</td><td>6x 64GB MTC40F2046S1RC48BA1</td></tr><tr><td>host_processor_caches</td><td></td></tr><tr><td>host_processor_core_count</td><td>12</td></tr><tr><td>host_processor_frequency</td><td></td></tr><tr><td>host_processor_interconnect</td><td></td></tr><tr><td>host_processor_model_name</td><td>Intel(R) Xeon(R) Silver 4410Y</td></tr><tr><td>host_processors_per_node</td><td>1</td></tr></table></td> </tr>
            <tr><td ><h3>Other Hardware Details</h3><table><tr><td>cooling</td><td>Air-cooled</td></tr><tr><td>disk_controllers</td><td>NVMe</td></tr><tr><td>disk_drives</td><td>SSD</td></tr><tr><td>hw_notes</td><td>B200 TGP 1000W</td></tr><tr><td>other_hardware</td><td></td></tr><tr><td>power_management</td><td></td></tr><tr><td>power_supply_details</td><td></td></tr><tr><td>power_supply_quantity_and_rating_watts</td><td></td></tr></table></td> <td><h3>Network and Interconnect Details</h3><table><tr><td>host_network_card_count</td><td>2x 10Gbe</td></tr><tr><td>host_networking</td><td>Gig Ethernet</td></tr><tr><td>host_networking_topology</td><td>Ethernet on switching network</td></tr><tr><td>network_speed_mbit</td><td></td></tr><tr><td>nics_enabled_connected</td><td></td></tr><tr><td>nics_enabled_firmware</td><td></td></tr><tr><td>nics_enabled_os</td><td></td></tr><tr><td>number_of_type_nics_installed</td><td></td></tr></table></td> </tr>
            <tr><td colspan="2"><h3>Software Details</h3><table><tr><td>boot_firmware_version</td><td></td></tr><tr><td>framework</td><td>TensorRT 10.1.0, CUDA 12.7</td></tr><tr><td>management_firmware_version</td><td></td></tr><tr><td>nics_enabled_firmware</td><td></td></tr><tr><td>operating_system</td><td>Ubuntu 22.04.4</td></tr><tr><td>other_software_stack</td><td>TensorRT 10.1.0, CUDA 12.7, cuDNN 8.9.7, Driver 565</td></tr><tr><td>sw_notes</td><td>Private git hash for code and TRTLLM were used to generate the preview results. The git hash are eba031f9e2e6cf3d1cdce0549511d27adf01a3f4 and 4e5b175cc80320789ba6846ced80a87f25e70fb2</td></tr></table></td> </tr>
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
    </tr><tr><td>llama2-70b-99</td><td>ROUGE1: 43.9869, ROUGE2: 21.8148, ROUGEL: 28.33, TOKENS_PER_SAMPLE: 265.005</td><td>Tokens/s</td> <td>10755.6</td><td>Tokens/s</td> <td>11264.4</td><tr><td>llama2-70b-99.9</td><td>ROUGE1: 44.3868, ROUGE2: 22.0132, ROUGEL: 28.5876, TOKENS_PER_SAMPLE: 265.005</td><td>Tokens/s</td> <td>10755.6</td><td>Tokens/s</td> <td>11264.4</td></table>

