#!/usr/bin/env python3

class RangePair:
    def __init__(self, start, end):
        self.start = start
        self.end = end

# Example list of instances
range_pairs = [
        RangePair(2.0, 5.0),
        RangePair(float('-inf'), 3.0),
        RangePair(1.0, 4.0)
        ]

def key_func(x):
    return [x.start, x.end]

# Key function to sort by (start, end)
# sorted_pairs = sorted(range_pairs, key=lambda x: ((x.start, x.end)))
sorted_pairs = sorted(range_pairs, key=key_func)

# Print sorted pairs for verification
for pair in sorted_pairs:
    print(f"Range: ({pair.start}, {pair.end})")
