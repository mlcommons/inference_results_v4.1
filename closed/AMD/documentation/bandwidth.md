# Bandwidth Requirement

 Formulas for minimum network bandwidth are as [follows](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#b1-ingress-bandwidth): 
 
 Llama2-70b: $throughput * 1024* dtype size$
 
 SDXL: $throughput * 1024 * dtype size$
 
 GPT-J: $throughput * 2048 * dtype size$
 
 DLRM v2: throughput * 270 * (13*dtype size1+26*dtype) size2

 Modern servers use PCIe gen5, which provides throughput of 4 gigabyte/second per lane and 64 GB/s for x16 connection. Assuming that
 a NIC is connected via a PCIe x16 to the DRAM and data types used in our submission, the maximum supported throughput is well above what was found in our submission



 | Benchmark |   Precision |       QPS    |
 | --------- |-------------|--------------|
 | Llama2-70b| 8 bit| 64 GB/(1024 * 1 Byte) = 62500000 |
 | SDXL| 8 bit| 64 GB/(1024 * 1 Byte) = 62500000 |
 | GPT-J| 8 bit| 64 GB/(2048 * 1 Byte) = 31250000 |
 | DLRM v2| 8 bit| 64 GB/(39*1 Byte)/270 = 6077872 |

