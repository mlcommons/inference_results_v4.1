
See the HTML preview [here](https://htmlpreview.github.io/?https://github.com/mlcommons/inference_results_v4.1/blob/main/closed/Lenovo/results/SR650_V3_3xL40S_TRT/summary.html)



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
    <p>Lenovo     </p>
    <p>ThinkSystem SR650 V3 (3x NVIDIA L40S, TensorRT)    </p>
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
    <td id="tester_val">Lenovo</td>
    <th id="sw_avail"><a href="">Availability:</a></th>
    <td id="sw_avail_val">Available  as of August 2024</td>
   </tr>
  </tbody>
 </table>
  
<table>
            <tr><td><h3>Accelerator Details</h3><table><tr><td>accelerator_frequency</td><td></td></tr><tr><td>accelerator_host_interconnect</td><td>N/A</td></tr><tr><td>accelerator_interconnect</td><td>N/a</td></tr><tr><td>accelerator_interconnect_topology</td><td></td></tr><tr><td>accelerator_memory_capacity</td><td>48 GB</td></tr><tr><td>accelerator_memory_configuration</td><td>GDDR6</td></tr><tr><td>accelerator_model_name</td><td>NVIDIA L40S</td></tr><tr><td>accelerator_on-chip_memories</td><td></td></tr><tr><td>accelerators_per_node</td><td>3</td></tr></table></td> <td><h3>Processor and Memory Details</h3><table><tr><td>host_memory_capacity</td><td>512 GB</td></tr><tr><td>host_memory_configuration</td><td>16x 32GB M321R4GA3BB6-CQKDS</td></tr><tr><td>host_processor_caches</td><td></td></tr><tr><td>host_processor_core_count</td><td>32</td></tr><tr><td>host_processor_frequency</td><td></td></tr><tr><td>host_processor_interconnect</td><td></td></tr><tr><td>host_processor_model_name</td><td>Intel(R) Xeon(R) Gold 6438N</td></tr><tr><td>host_processors_per_node</td><td>2</td></tr></table></td> </tr>
            <tr><td ><h3>Other Hardware Details</h3><table><tr><td>cooling</td><td>Air-cooled</td></tr><tr><td>disk_controllers</td><td></td></tr><tr><td>disk_drives</td><td></td></tr><tr><td>hw_notes</td><td></td></tr><tr><td>other_hardware</td><td></td></tr><tr><td>power_management</td><td></td></tr><tr><td>power_supply_details</td><td></td></tr><tr><td>power_supply_quantity_and_rating_watts</td><td></td></tr></table></td> <td><h3>Network and Interconnect Details</h3><table><tr><td>host_network_card_count</td><td>1</td></tr><tr><td>host_networking</td><td>Gig Ethernet</td></tr><tr><td>host_networking_topology</td><td>N/A</td></tr><tr><td>network_speed_mbit</td><td></td></tr><tr><td>nics_enabled_connected</td><td></td></tr><tr><td>nics_enabled_firmware</td><td></td></tr><tr><td>nics_enabled_os</td><td></td></tr><tr><td>number_of_type_nics_installed</td><td></td></tr></table></td> </tr>
            <tr><td colspan="2"><h3>Software Details</h3><table><tr><td>boot_firmware_version</td><td></td></tr><tr><td>framework</td><td>TensorRT 10.2.0, CUDA 12.5</td></tr><tr><td>management_firmware_version</td><td></td></tr><tr><td>nics_enabled_firmware</td><td></td></tr><tr><td>operating_system</td><td>Ubuntu 22.04.4</td></tr><tr><td>other_software_stack</td><td>TensorRT 10.2.0, CUDA 12.5, cuDNN 8.9.7, Driver 555.42</td></tr><tr><td>sw_notes</td><td></td></tr></table></td> </tr>
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
    </tr><tr><td>resnet</td><td>acc: 75.6954</td><td>Queries/s</td> <td>132008.0</td><td>Samples/s</td> <td>132436.0</td><tr><td>retinanet</td><td>mAP: 37.1745</td><td>Queries/s</td> <td>2201.83</td><td>Samples/s</td> <td>2290.08</td><tr><td>bert-99</td><td>F1: 89.9653</td><td>Queries/s</td> <td>9302.0</td><td>Samples/s</td> <td>9480.82</td><tr><td>3d-unet-99</td><td>DICE: 0.8531</td><td colspan="2"> N/A </td><td>Samples/s</td> <td>11.6149</td><tr><td>gptj-99</td><td>ROUGE1: 42.5566, ROUGE2: 19.9223, ROUGEL: 29.6882, GEN_LEN: 3615190.2</td><td>Tokens/s</td> <td>2475.22</td><td>Tokens/s</td> <td>2593.65</td></table>

