#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial.distance


def jaccard_index(mask1, mask2):
    # mask1 = np.asarray(mask1).astype(bool)
    # mask2 = np.asarray(mask2).astype(bool)

    # intersection = np.logical_and(mask1, mask2).sum()
    # union = np.logical_or(mask1, mask2).sum()

    # return intersection / union if union != 0 else 0
    a = set(mask1)
    b = set(mask2)
    # return len(a & b)/ len(a | b)
    print(a.intersection(b))
    print(a.union(b))
    return len(a.intersection(b)) / len(a.union(b))


if __name__ == "__main__":
    a = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            ]
    a = np.array(a).flatten()
    b = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            ]
    b = np.array(b).flatten()
    # sim = scipy.spatial.distance.jaccard(a, b)

    sim = jaccard_index(a, b)
    print(sim)

    a = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            ]
    a = np.array(a).flatten()
    b = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            ]
    b = np.array(b).flatten()

    sim = jaccard_index(a, b)
    print(sim)


    a = [12, 34, 46, 78, 90]
    a = np.array(a).flatten()
    b = [34, 12, 46, 90, 70]
    b = np.array(b).flatten()

    sim = jaccard_index(a, b)
    print(sim)

    a = [0.0, 1.0, 0.0]
    b = [0.2, 0.6, 0.2]
    print(1 - scipy.spatial.distance.cosine(a, b))
