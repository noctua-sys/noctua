import re
import os

open_re = re.compile(r'^[ \t]*# *BEGIN\(.*?\)[ \t]*$')
close_re = re.compile(r'^[ \t]*# *END\(.*?\)[ \t]*$')

def count_file(path) -> int:
    cnt = 0
    outside = True
    with open(path, 'r') as f:
        for line in f:
            if outside:
                if open_re.match(line):
                    outside = False
            else:
                if close_re.match(line):
                    outside = True
                else:
                    cnt += 1
    return cnt


def count_tree(path) -> int:
    total = 0
    for root, dirs, files in os.walk(path):
        for filename in files:
            if os.path.splitext(filename)[1] == '.py':
                path = os.path.join(root, filename)
                cnt = count_file(path)
                if cnt > 0:
                    total += cnt
                    print(cnt, path)
    return total


