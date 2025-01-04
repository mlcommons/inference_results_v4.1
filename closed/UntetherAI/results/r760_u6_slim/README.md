
See the HTML preview [here](https://htmlpreview.github.io/?https://github.com/GATEOverflow/inference_results_v4.1/blob/main/closed/UntetherAI/results/r760_u6_slim/summary.html)



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
    <p>UntetherAI     </p>
    <p>Dell PowerEdge R760xa (6x speedAI240 Slim)    </p>
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
    <td id="tester_val">UntetherAI</td>
    <th id="sw_avail"><a href="">Availability:</a></th>
    <td id="sw_avail_val">Available as of February 2025</td>
   </tr>
  </tbody>
 </table>
  
<table>
            <tr><td><h3>Accelerator Details</h3><table><tr><td>accelerator_frequency</td><td></td></tr><tr><td>accelerator_host_interconnect</td><td>PCIe Gen 5 16x (32 GT/s)</td></tr><tr><td>accelerator_interconnect</td><td>N/A</td></tr><tr><td>accelerator_interconnect_topology</td><td>N/A</td></tr><tr><td>accelerator_memory_capacity</td><td>disabled</td></tr><tr><td>accelerator_memory_configuration</td><td>LPDDR5 64x</td></tr><tr><td>accelerator_model_name</td><td>UntetherAI speedAI240 Slim</td></tr><tr><td>accelerator_on-chip_memories</td><td>238 MB SRAM</td></tr><tr><td>accelerators_per_node</td><td>6</td></tr></table></td> <td><h3>Processor and Memory Details</h3><table><tr><td>host_memory_capacity</td><td>256 GB</td></tr><tr><td>host_memory_configuration</td><td>16x 16 GB DDR5 (Samsung M321R2GA3BB6-CQKDS 4800 MT/s)</td></tr><tr><td>host_processor_caches</td><td>L1d cache: 3 MiB (64 instances); L1i cache: 2 MiB (64 instances); L2 cache: 128 MiB (64 instances); L3 cache: 120 MiB (2 instances)</td></tr><tr><td>host_processor_core_count</td><td>32</td></tr><tr><td>host_processor_frequency</td><td>800 MHz (min); 2100 MHz (base); 4100 MHz (boost)</td></tr><tr><td>host_processor_interconnect</td><td></td></tr><tr><td>host_processor_model_name</td><td>Intel(R) Xeon(R) Gold 6448Y</td></tr><tr><td>host_processor_url</td><td>https://www.intel.com/content/www/us/en/products/sku/232384/intel-xeon-gold-6448y-processor-60m-cache-2-10-ghz/specifications.html</td></tr><tr><td>host_processors_per_node</td><td>2</td></tr></table></td> </tr>
            <tr><td ><h3>Other Hardware Details</h3><table><tr><td>cooling</td><td>air</td></tr><tr><td>hw_notes</td><td></td></tr></table></td> <td><h3>Network and Interconnect Details</h3><table><tr><td>host_network_card_count</td><td>2</td></tr><tr><td>host_networking</td><td>embedded; integrated</td></tr><tr><td>host_networking_topology</td><td>Broadcom NetXtreme 1GbE (BCM5720); Broadcom Adv. Dual 25GbE</td></tr></table></td> </tr>
            <tr><td colspan="2"><h3>Software Details</h3><table><tr><td>framework</td><td>UntetherAI imAIgine SDK v24.07.19</td></tr><tr><td>operating_system</td><td>Ubuntu 22.04.4 LTS (Linux kernel 6.5.0-44-generic #44~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Jun 18 14:36:16 UTC 2 x86_64 x86_64 x86_64 GNU/Linux)</td></tr><tr><td>other_software_stack</td><td>{'KILT': 'mlperf_4.1', 'Docker': '27.1.0, build 6312585', 'Python': '3.10.12'}</td></tr><tr><td>sw_notes</td><td>Powered by the KRAI X and KILT technologies</td></tr></table></td> </tr>
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
    </tr><tr><td>resnet</td><td>acc: 75.6954</td><td>Queries/s</td> <td>309752.0</td><td>Samples/s</td> <td>334462.0</td></table>

