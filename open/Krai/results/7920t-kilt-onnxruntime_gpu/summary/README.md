

<div class="resultpage">
 <div class="titlebarcontainer">
  <div class="logo">
   <a href="/" style="border: none"><img src="" alt="" /></a>
  </div>
  <div class="titlebar">
   <h1 class="title">MLPerf Inference v4.1</h1>
   <p style="font-size: smaller">Copyright 2019-2024 MLCommons</p>
  </div>
 </div>
 <table class="titlebarcontainer">
  <tr>
   <td class="headerbar" rowspan="2">
    <p>Krai     </p>
    <p>Dell Precision 7920 Tower (1x NVIDIA RTX A5000 GPU)    </p>
   </td>
  </tr>
 </table>
 <table class="datebar">
  <tbody>
   <tr>
    <th id="license_num"><a href="">MLPerf Inference Category:</a></th>
    <td id="license_num_val">Datacenter</td>
    <th id="test_date"><a href="">MLPerf Inference Division:</a></th>
    <td id="test_date_val">Closed</td>
   </tr>
   <tr>
    <th id="tester"><a href="">Submitted by:</a></th>
    <td id="tester_val">Krai</td>
    <th id="sw_avail"><a href="">Availability:</a></th>
    <td id="sw_avail_val">Available as of Aug 2024</td>
   </tr>
  </tbody>
 </table>
  
<table>
            <tr><td><h3>Accelerator Details</h3><table><tr><td>accelerator_frequency</td><td>1170 MHz (base); 1695 MHz (turbo)</td></tr><tr><td>accelerator_host_interconnect</td><td>PCIe Gen 4</td></tr><tr><td>accelerator_interconnect</td><td>NVIDIA NVLink</td></tr><tr><td>accelerator_interconnect_topology</td><td></td></tr><tr><td>accelerator_memory_capacity</td><td>24 GB</td></tr><tr><td>accelerator_memory_configuration</td><td>1x 24 GB</td></tr><tr><td>accelerator_model_name</td><td>NVIDIA RTX A5000 GPU</td></tr><tr><td>accelerator_on-chip_memories</td><td></td></tr><tr><td>accelerators_per_node</td><td>1</td></tr></table></td> <td><h3>Processor and Memory Details</h3><table><tr><td>host_memory_capacity</td><td>96 GB</td></tr><tr><td>host_memory_configuration</td><td>6x 16 GB</td></tr><tr><td>host_processor_caches</td><td>L1d cache: 768 KiB (24 instances); L1i cache: 768 KiB (24 instances); L2 cache: 24 MiB (24 instances); L3 cache: 35.8 MiB (1 instance)</td></tr><tr><td>host_processor_core_count</td><td>24</td></tr><tr><td>host_processor_frequency</td><td>1000 MHz (min); 2400 MHz (base); 4000 MHz (boost)</td></tr><tr><td>host_processor_interconnect</td><td></td></tr><tr><td>host_processor_model_name</td><td>Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz</td></tr><tr><td>host_processors_per_node</td><td>1</td></tr><tr><td>host_processor_url</td><td>https://www.intel.com/content/www/us/en/products/sku/199343/intel-xeon-gold-6240r-processor-35-75m-cache-2-40-ghz/specifications.html</td></tr></table></td> </tr>
            <tr><td ><h3>Other Hardware Details</h3><table><tr><td>cooling</td><td>air</td></tr><tr><td>hw_notes</td><td></td></tr></table></td> <td><h3>Network and Interconnect Details</h3><table><tr><td>host_networking</td><td>Ethernet</td></tr><tr><td>host_networking_topology</td><td>Integrated</td></tr><tr><td>host_network_card_count</td><td>1</td></tr></table></td> </tr>
            <tr><td colspan="2"><h3>Software Details</h3><table><tr><td>framework</td><td>KRAI Inference Library Technology (KILT) with ONNX Runtime support</td></tr><tr><td>operating_system</td><td>Ubuntu 22.04.4 LTS (Linux kernel 5.15.0-76-generic #83-Ubuntu SMP Thu Jun 15 19:16:32 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux)</td></tr><tr><td>other_software_stack</td><td>CUDA v12.5; ONNX Runtime v1.18.1; Python v3.9.19; GCC v11.4.0</td></tr><tr><td>sw_notes</td><td>Powered by the KRAI X and KILT technologies</td></tr></table></td> </tr>
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
    </tr>
    <tr><td>bert-99</td><td></td><td></td><td></td><td>Samples/s</td> <td>65.5708</td><tr><td>resnet</td><td></td><td></td><td></td><td>Samples/s</td> <td>1090.09</td></table>
