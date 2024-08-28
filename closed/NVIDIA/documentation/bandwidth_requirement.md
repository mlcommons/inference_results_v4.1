# Bandwidth measurements of GPU to network for MLPerf Inference 4.0

## Introduction
As of MLPerf Inf v4.0, submissions are required to prove that the systems used provide a certain level of ingress (network to GPU) and egress (GPU to network) bandwidth bypassing the CPU. This is required for all systems, and is additional to the requirements to validate benchmark runs using `start_from_device` and `end_on_device`. Put simply, the throughput at which the accelerator accepts queries and generates responses should not exceed the maximum data bandwidth the system is capable of supporting.  

This round, NVIDIA is making submissions using the following systems:
- DGX H100
- H200-SXM-141GBx8
- H200-SXM-141GB-CTSx8
- GH200 144GB

## Procedure
This document assumes that the following architecture is used in all systems. Each GPU is connected to the PCIe bus. The PCIe bus is also connected to each NIC (network interface cards). Hence, the ingress data flow is: 
1. network to NIC
2. NIC to PCIe bus
3. PCIe bus to GPU

The egress data flow is the reverse:
1. GPU to PCIe
2. PCIe to NIC
3. NIC to network

Hence, we must measure (1) GPU to PCIe bandwidth and (2) PCIe to NIC bandiwdth. The minimum of (1) and (2) is the theoretical max bandwidth that our system will support.  

### Getting GPU to PCIe bandwidth
We use the `nvidia-smi` utility to fetch the GPU PCIe information. Below is the information from L40S:
```
$ nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv

index, pcie.link.gen.current, pcie.link.gen.max, pcie.link.width.current, pcie.link.width.max
0, 1, 4, 16, 16
1, 1, 4, 16, 16
2, 1, 4, 16, 16
3, 1, 4, 16, 16
4, 1, 4, 16, 16
5, 1, 4, 16, 16
6, 1, 4, 16, 16
7, 1, 4, 16, 16
```


While in practice `current` and `max` gen/width may differ, we use the `max` field here since we are reporting theoretical maximum bandwidth of the system.  
From [wiki](https://en.wikipedia.org/wiki/PCI_Express#Comparison_table), we can extract the throughput, given width and PCIe generation.  

In our case, per GPU throughput is 31.508 GB/s. When all 8 accelerators are used, the GPU-PCIe bandwidth is 252.064 GB/s. 

### Getting PCIe to NIC bandwidth
First, we get the number of NICs that are available for use, using `networkctl` and `lshw`. Below is the information from L40S.

```
$ networkctl
IDX LINK            TYPE     OPERATIONAL SETUP     
  1 lo              loopback carrier     unmanaged
  2 enxb2fb207cc2ec ether    off         unmanaged
  3 enp196s0f0      ether    routable    configured
  4 enp196s0f1      ether    off         unmanaged
  5 docker0         bridge   no-carrier  unmanaged

$ sudo lshw -class network -businfo
Bus info          Device           Class          Description
=============================================================
pci@0000:c4:00.0  enp196s0f0       network        Ethernet Controller 10G X550T
pci@0000:c4:00.1  enp196s0f1       network        Ethernet Controller 10G X550T
usb@3:1.3         enxb2fb207cc2ec  network        Ethernet interface

```

`lshw` tells us that one of the three ethernet NICs is connected to a usb port - this can not use DMA, hence we do not consider it.  
The other two NICs are PCIe ports, as reflected by the `pci` prefix on their Bus info.  

To get PCIe - NIC bandwidth of `enp196s0f0`, we use `lspci` in verbose mode and dump the output to a file.
```
$ sudo lspci -vv > lspci_dump.log
```

- `sudo` is required to get bandwidth capabilities.
- Pipe the output to a file so that NIC specific information can be searched.


In the output, search for the NIC using the respective PCIe bus ID. (`c4:00.0`, for example).

```
c4:00.0 Ethernet controller: Intel Corporation Ethernet Controller 10G X550T (rev 01)
    Subsystem: Intel Corporation Ethernet Converged Network Adapter X550-T2
    ....

    Capabilities: [a0] Express (v2) Endpoint, MSI 00
        DevCap: MaxPayload 512 bytes, PhantFunc 0, Latency L0s <512ns, L1 <64us
        ....
        LnkCap: Port #0, Speed 8GT/s, Width x4, ASPM L0s L1, Exit Latency L0s <2us, L1 <16us
        LnkSta: Speed 8GT/s (ok), Width x4 (ok)
        ....
        LnkCap2: Supported Link Speeds: 2.5-8GT/s, Crosslink- Retimer- 2Retimers- DRS-
        LnkCtl2: Target Link Speed: 2.5GT/s, EnterCompliance- SpeedDis-
```

This suggests that the NIC supports 8GT/s with a width of 4, but currently uses 2.5GT/s. We use 8GT/s for the purpose of reporting theoretical maximum.  
Similarly, for the second NIC with ID `c4:00:1`, it's easy to verify that the current bandwidth is 8GT/s x4.  

The transfer rate per lane of 8GT/s corresponds to PCIe v3.0, and for 4 lanes, the throughput per NIC is 3.938 GB/s. Hence, total NIC-PCIe bandwidth is 7.877 GB/s.


### Theoretical maximum bandwidth of system
Even though each GPU supports ~31 GB/s, we are constrained by the NIC bandwidth in L40S system. The maximum permissible bandwidth (ingress and egress) is 7.877 GB/s.

## Bandwidth requirements per scenario
For each benchmark model, the offline scenario uses the most bandwidth. We list the bandwidth used per model `used_bw` (in byte/sec) for a run in Offline scenario.  
The `used_bw` is a function of throughput of input samples `tput` (sample/second) and size of each input sample `bytes/sample`.  

### Ingress bandwidth requirements

| Benchmark | Formula                                                                                                               | Bandwidth used                  | Values                                                                                                               |
|-----------|-----------------------------------------------------------------------------------------------------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------|
| ResNet50  | ```used_bw = tput x (C x H x W) x dtype_size = tput x (3 x 224 x 224) = tput x 150528```                              | ```used_bw = tput x 150528```   | ```H = W = 224; C =3; dtype_size = 1byte```                                                                          |
| BERT      | ```used_bw = tput x (num_inputs x max_input_len x dtype_size)```                                                      | ```used_bw = tput x 4608```     | ```num_inputs = 3; max_input_len = 384; dtype_size = 4bytes```                                                         |
| DLRM      | ```used_bw = tput x num_pairs_per_sample x ((num_numerical_inputs x dtype_0) + (num_categorical_inputs x dtype_1))``` | ```used_bw = tput x 35100```    | ```num_pairs_per_sample = 270; num_numerical_inputs = 13; num_categorical_inputs = 26; dtype_0 = 2B; dtype_1 = 4B``` |
| 3D U-Net  | ```used_bw = tput x avg(C x D x H x W)```                                                                             | ```used_bw = tput x 32944795``` | ```avg = 32944795```                                                                                                 |
| GPT-J     | ```used_bw = tput x max_input_len x dtype_size```                                                                     | ```used_bw = tput x 7676```     | ```max_input_len = 1919; dtype_size = 4B```                                                                            |
| Llama2    | ```used_bw = tput x  max_input_len x dtype_size```                                                                    | ```used_bw = tput x 4096```     | ```max_input_len = 1024; dtype_size = 4B```                                                                            |
| MoE       | ```used_bw = tput x  max_input_len x dtype_size```                                                                    | ```used_bw = tput x 8192```     | ```max_input_len = 2048; dtype_size = 4B```                                                                            |
| RetinaNet | ```used_bw = tput x C x H x W x dtype_size```                                                                         | ```used_bw = tput x 1920000```  | ```H = W = 800; C = 3; dtype_size = 1B```                                                                            |
| SDXL      | ```used_bw = num_inputs x max_prompt_len x dtype_size```                                                              | ```used_bw = tput x 308```      | ```num_inputs = 1; max_prompt_len = 77```                                                                            |

### Egress bandwidth requirements

According to the [rules set out by MLCommons](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#b2-egress-bandwidth), we only need to measure egress bandwidth for 3D U-Net and SDXL. 
| Benchmark | Formula                                                                                                               | Bandwidth used                  | Values                                                                                                               |
|-----------|-----------------------------------------------------------------------------------------------------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------|
| 3D U-Net  | ```used_bw = tput x avg(C x D x H x W)```                                                                             | ```used_bw = tput x 32944795``` | ```avg = 32944795```                                                                                                 |
| SDXL      | ```used_bw = num_inputs x image_height x image_width x image_channel x dtype_size```                                  | ```used_bw = tput x 3145728```  | ```num_inputs = 1; image_height = image_width = 1024; image_channel = 3```                                           |

## Calculating maximum permissible QPS per system-model submission pair
Using the formulae in the previous section, we calculate for each system-benchmark pair the maximum permissible QPS by setting `system_bw = used_bw` and calculating for `tput`.
For workloads with constraint on both ingress and egress bandwidth, we take the max of the two. (for example, we take egress for SDXL)

PLEASE NOTE - The numbers are calculated below for NVIDIA's systems and are provided for reference only. Each systems configuration (and hence, bandwidth) may be different. It is imperative that each participant does such calculations individually for their own systems.

| System                       | Bandwidth of system | ResNet50    | BERT        | DLRM        | 3D U-Net | GPT-J       | Llama2      | RetinaNet   | SDXL         | MoE         |
|------------------------------|---------------------|-------------|-------------|-------------|----------|-------------|-------------|-------------|--------------|-------------|
| DGX H100                     | 500GB/s             | 2.60 x 10^7 | 8.68 x 10^9 | 1.13 x 10^8 | 121,408  | 4.49 x 10^8 | 2.40 x 10^8 | 2.08 x 10^6 | 170,666      | 4.80 x 10^8 |
| H200-SXM-141GBx8             | 500GB/s             | 2.60 x 10^7 | 8.68 x 10^9 | 1.13 x 10^8 | 121,408  | 4.49 x 10^8 | 2.40 x 10^8 | 2.08 x 10^6 | 170,666      | 4.80 x 10^8 |
| H200-SXM-141GB-CTSx8         | 500GB/s             | 2.60 x 10^7 | 8.68 x 10^9 | 1.13 x 10^8 | 121,408  | 4.49 x 10^8 | 2.40 x 10^8 | 2.08 x 10^6 | 170,666      | 4.80 x 10^8 |
| GH200-144GB                  | 252.06GB/s          | 1.32 x 10^7 | 4.37 x 10^9 | 5.69 x 10^7 | 61,204   | 2.26 x 10^8 | 1.21 x 10^8 | 1.04 x 10^6 | 86,036       | 2.41 x 10^8 |