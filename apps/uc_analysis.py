#!/usr/bin/env python3

# Compute the unsuccessful core.

import pandas as pd

df = pd.read_csv('final.csv', header=None)
all_paths = list(set(list(df[0])))  # dedup
print('Number of paths =', len(all_paths))

df = df[df[3] == True]  # leave only those restricted

def count_restrictions_wrt(path):
    self_check = len(df[df[0]==path][df[1]==path])
    first = len(df[df[0]==path])
    second = len(df[df[1]==path])
    return first+second-self_check

annotated_all_paths = [(count_restrictions_wrt(path), count_restrictions_wrt(path) / len(all_paths), path) for path in all_paths]
annotated_all_paths.sort(reverse=True)
result = pd.DataFrame(annotated_all_paths)
result.to_csv('uc.csv')

