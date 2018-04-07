# coding=utf8
import numpy as np

data = open('names', 'r', encoding='utf8')

filtered = []
for line in data:
    line = line.strip()
    if len(line) > 0 \
        and not line.startswith('0')\
        and not line.startswith('1') \
            and not line.startswith('2')\
            and not line.startswith('3'):
        filtered.append(line)

for i in filtered:
    print(i)

