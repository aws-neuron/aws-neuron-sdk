latency_list = []
with open('latencies.txt', 'r') as f:
    for line in f:
        latency_list.append(float(line.rstrip()))

latency_list = sorted(latency_list)
l = len(latency_list)

print(f'p50 latency is {latency_list[int(.5 * l)]} seconds')
print(f'p90 latency is {latency_list[int(.9 * l)]} seconds')
print(f'p95 latency is {latency_list[int(.95 * l)]} seconds')
print(f'p99 latency is {latency_list[int(.99 * l)]} seconds')
