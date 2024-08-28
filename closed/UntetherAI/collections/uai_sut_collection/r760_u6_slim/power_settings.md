# Boot/BIOS Firmware Settings

### Processor Settings

#### Logical Processor: Enabled
#### SST-Performance Profile: Operating Point 1 | P1: 2.1 GHz, TDP: 225W, Core Count: 32 

# Firmware Settings

Key | Value
-|-
BIOS Version	              | 2.1.5
Lifecycle Controller Firmware | 7.10.30.05
System Revision               | I

# Fan Settings

## Host Fan

### Thermal Profile Optimization: Minimum Power (Performance per Watt Optimized)

### Fan Speed: 38% of Max (8640 RPM)

### IDRAC 9 custom PCIE airflow settings:

#### Slot 2: 350 LMF
#### Slot 7: 200 LMF
#### Slots 31/33/36/38: 55 LMF

## Accelerator Fans (95%)
<pre>
<b>&dollar;</b> speedai smi --set-fanspeed 95
</pre>

# Accelerator Mode

## Offline and Server (ECO)
<pre>
<b>&dollar;</b> speedai smi --set-mode <b>ECO</b>
</pre>

# CPU Frequency Policy
The CPU frequency policy is controlled through CPU governors.

### Offline and Server (ondemand)

<pre>
<b>&dollar;</b> sudo cpupower frequency-set --governor <b>ondemand</b>
</pre>
