# Firmware Settings

Key | Value
-|-
Firmware Version      |	01.01.08
Firmware Build Time   |	10/06/2023
Redfish Version	      | 1.11.0
BIOS Firmware Version |	1.6
BIOS Build Time	      | 11/03/2023A
CPLD Version          | F5.0E.13

# Fan Settings

## Host Fan

System > Component Info > Fan > Advanced Settings > Fan Mode > Standard Speed (1960 RPM)

## Accelerator Fans (95%)
<pre>
<b>&dollar;</b> speedai smi --set-fanspeed 95
</pre>


# Accelerator Mode

## Offline (ECO)
<pre>
<b>&dollar;</b> speedai smi --set-mode <b>ECO</b>
</pre>

## SingleStream (SPORT)
<pre>
<b>&dollar;</b> speedai smi --set-mode <b>SPORT</b>
</pre>

## MultiStream (SPORT)
<pre>
<b>&dollar;</b> speedai smi --set-mode <b>SPORT</b>
</pre>


# CPU Frequency Policy
The CPU frequency policy is controlled through CPU governors.

## Offline (powersave)
<pre>
<b>&dollar;</b> sudo cpupower frequency-set --governor <b>powersave</b>
</pre>

## SingleStream (ondemand)
<pre>
<b>&dollar;</b> sudo cpupower frequency-set --governor <b>ondemand</b>
</pre>

## MultiStream (schedutil)
<pre>
<b>&dollar;</b> sudo cpupower frequency-set --governor <b>schedutil</b>
</pre>
