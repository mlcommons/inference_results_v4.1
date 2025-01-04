
See the HTML preview [here](https://htmlpreview.github.io/?https://github.com/GATEOverflow/inference_results_v4.1/blob/main/open/NVIDIA/results/Orin_TRT_DepthPruned/summary.html)



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
    <p>NVIDIA Jetson AGX Orin Developer Kit 64G (TensorRT)    </p>
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
    <td id="tester_val">NVIDIA</td>
    <th id="sw_avail"><a href="">Availability:</a></th>
    <td id="sw_avail_val">Available  as of August 2024</td>
   </tr>
  </tbody>
 </table>
  
<table>
            <tr><td><h3>Accelerator Details</h3><table><tr><td>accelerator_frequency</td><td></td></tr><tr><td>accelerator_host_interconnect</td><td>N/A</td></tr><tr><td>accelerator_interconnect</td><td>N/A</td></tr><tr><td>accelerator_interconnect_topology</td><td></td></tr><tr><td>accelerator_memory_capacity</td><td>Shared with host</td></tr><tr><td>accelerator_memory_configuration</td><td>LPDDR5</td></tr><tr><td>accelerator_model_name</td><td>NVIDIA Jetson AGX Orin 64G</td></tr><tr><td>accelerator_on-chip_memories</td><td></td></tr><tr><td>accelerators_per_node</td><td>1</td></tr></table></td> <td><h3>Processor and Memory Details</h3><table><tr><td>host_memory_capacity</td><td>64 GB</td></tr><tr><td>host_memory_configuration</td><td>64GB 256-bit LPDDR5</td></tr><tr><td>host_processor_caches</td><td></td></tr><tr><td>host_processor_core_count</td><td>12</td></tr><tr><td>host_processor_frequency</td><td></td></tr><tr><td>host_processor_interconnect</td><td></td></tr><tr><td>host_processor_model_name</td><td>12-core ARM Cortex-A78AE CPU</td></tr><tr><td>host_processors_per_node</td><td>1</td></tr></table></td> </tr>
            <tr><td ><h3>Other Hardware Details</h3><table><tr><td>cooling</td><td>Air-cooled</td></tr><tr><td>disk_controllers</td><td>eMMC 5.1</td></tr><tr><td>disk_drives</td><td>eMMC 5.1</td></tr><tr><td>hw_notes</td><td>GPU and both DLAs are used in resnet50 and Retinanet, in Offline scenario</td></tr><tr><td>other_hardware</td><td></td></tr><tr><td>power_management</td><td></td></tr><tr><td>power_supply_details</td><td>Dell USB-C 130.0W Adapter (HA130PM170)</td></tr><tr><td>power_supply_quantity_and_rating_watts</td><td>130W</td></tr></table></td> <td><h3>Network and Interconnect Details</h3><table><tr><td>host_network_card_count</td><td>1 Integrated</td></tr><tr><td>host_networking</td><td>Gig Ethernet</td></tr><tr><td>host_networking_topology</td><td>USB forwarded</td></tr><tr><td>network_speed_mbit</td><td></td></tr><tr><td>nics_enabled_connected</td><td></td></tr><tr><td>nics_enabled_firmware</td><td></td></tr><tr><td>nics_enabled_os</td><td></td></tr><tr><td>number_of_type_nics_installed</td><td></td></tr></table></td> </tr>
            <tr><td colspan="2"><h3>Software Details</h3><table><tr><td>boot_firmware_version</td><td></td></tr><tr><td>framework</td><td>Jetpack 6.0, TensorRT 10.1, CUDA 12.2</td></tr><tr><td>management_firmware_version</td><td></td></tr><tr><td>nics_enabled_firmware</td><td></td></tr><tr><td>operating_system</td><td>Jetson r36.3.1 L4T</td></tr><tr><td>other_software_stack</td><td>Jetpack 6.0, TensorRT 10.1, CUDA 12.2, cuDNN 8.9.4</td></tr><tr><td>sw_notes</td><td></td></tr></table></td> </tr>
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
    </tr><tr><td>llama2-70b-99</td><td>ROUGE1: 43.9869, ROUGE2: 21.8148, ROUGEL: 28.33, TOKENS_PER_SAMPLE: 265.005</td><td></td><td></td><td></td><td>ROUGE1: 43.2263  ROUGE2: 24.3279  ROUGEL: 31.3816  TOKENS_PER_SAMPLE: 166.3</td><td>Tokens/s</td> <td>184.893</td></table>

