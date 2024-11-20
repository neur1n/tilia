#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np

import util


def normalize_array(arr):
    max_positive = max(filter(lambda x: x > 0, arr), default=0)
    min_negative = min(filter(lambda x: x < 0, arr), default=0)

    normalized = [
        (x / max_positive if x > 0 else x / abs(min_negative) if x < 0 else 0)
        for x in arr
    ]
    return normalized

if __name__ == "__main__":
    data = [3, -2, 0, 6, -1, 4]
    normalized_data = normalize_array(data)
    print(normalized_data)

    normalized_data = util.normalize(np.array(data), -1.0, 1.0)
    print(normalized_data)
