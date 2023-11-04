#!/usr/bin/python3

import argparse

parser = argparse.ArgumentParser(description='Report')
parser.add_argument('outdir')
parser.add_argument('secs', type=float)
args = parser.parse_args()

dir = args.outdir
merge = dir + '/merge.csv'
n_txs = 0
n_millisecs = 0
w_ms = 0
r_ms = 0
n_writes = 0
with open(merge, 'r') as f:
    for l in f:
        l = l.strip()
        [_,type,name,time] = l.split(',')
        n_millisecs += float(time)
        n_txs += 1
        if type=='write':
            n_writes+=1
            w_ms += float(time)
        elif type=='read':
            r_ms += float(time)

avg_latency = n_millisecs / n_txs
tps = n_txs / args.secs
print('avg_latency =', avg_latency)

avg_w_latency = w_ms / n_writes
print('avg_w_latency =', avg_w_latency)

avg_r_latency = r_ms / (n_txs - n_writes)
print('avg_r_latency =', avg_r_latency)

print('normalized w latency', avg_w_latency / avg_r_latency)

print('tps =', tps)
print('write ratio =', n_writes / n_txs)
