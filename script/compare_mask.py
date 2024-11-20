#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import datetime
import itertools
import glob
import numpy as np
import PIL.Image
import scipy.spatial.distance

import config


def pairwise_jaccard(boundaries):
    indices = []
    for (mask1, mask2) in itertools.combinations(boundaries, 2):
        indices.append(1.0 - scipy.spatial.distance.jaccard(mask1.flatten(), mask2.flatten()))
    return indices


dataset = [
        # "n02085936_Maltese_dog.JPEG.jpg",
        # "alligator.JPEG",
        # "beeflower.JPEG",
        "dog.JPEG",
        ]


patch = [5]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--regressor", default=None, required=False, help="Regressor, linear or tree")
    ap.add_argument("-t", "--timestamp", default=None, required=False, help="Timestamp")
    args = ap.parse_args()

    # NOTE: `None` is for linear.
    if args.regressor is not None:
        config.REGRESSOR = args.regressor

    timestamp: str
    if args.timestamp:
        timestamp = args.timestamp
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d")

    for name in dataset:
        output_dir = f"{config.ROOT_DIR}/output/{timestamp}/{name}/{config.REGRESSOR}"
        os.makedirs(output_dir, exist_ok=True)

        for p in patch:
            file = glob.glob(f"{output_dir}/mask*patch{p}*.png")
            mask = []
            for f in file:
                mask.append(np.asarray(PIL.Image.open(f)))

            jaccard = pairwise_jaccard(mask)
            mean = np.mean(jaccard)
            print(f"(#Patch: {p}) {jaccard} - {mean}")
