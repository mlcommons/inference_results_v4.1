import argparse
import csv
import re
import sys

fieldnames = [
    "timestamp",
    "prompt_tokens_per_second",
    "generation_tokens_per_second",
    "running_requests",
    "swapped_requests",
    "pending_requests",
    "gpu_kvcache_usage",
    "cpu_kvcache_usage",
    ]

pattern = re.compile(r"INFO (?P<month>[0-1][0-9])-(?P<day>[0-3][0-9]) (?P<time>[0-9:]+).* "
    r"Avg prompt throughput: (?P<prompt_throughput>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<gen_throughput>[0-9.]+) tokens/s, "
    r"Running: (?P<running>[0-9]+) reqs, "
    r"Swapped: (?P<swapped>[0-9]+) reqs, "
    r"Pending: (?P<pending>[0-9]+) reqs, "
    r"GPU KV cache usage: (?P<gpu_kvcache_percent>[0-9.]+)%, "
    r"CPU KV cache usage: (?P<cpu_kvcache_percent>[0-9.]+)%."
)

parser = argparse.ArgumentParser(description="Parse statistics from the vLLM stats log entries")

parser.add_argument("infile", type=argparse.FileType("r"))
parser.add_argument("outfile", nargs="?", type=argparse.FileType("w"), default=sys.stdout)

args = parser.parse_args()

writer = csv.DictWriter(args.outfile, fieldnames=fieldnames)
writer.writeheader()

log_entries = []
for line in args.infile:
    for match in pattern.finditer(line):
        writer.writerow({
            "timestamp": f"2024-{match.group('month')}-{match.group('day')} {match.group('time')}",
            "prompt_tokens_per_second": match.group('prompt_throughput'),
            "generation_tokens_per_second": match.group('gen_throughput'),
            "running_requests": match.group('running'),
            "swapped_requests": match.group('swapped'),
            "pending_requests": match.group('pending'),
            "gpu_kvcache_usage": float(match.group('gpu_kvcache_percent'))/100.0,
            "cpu_kvcache_usage": float(match.group('cpu_kvcache_percent'))/100.0,
        })
