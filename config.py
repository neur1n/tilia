#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os


DELIMITER = ","

EXPLAINER = {"None": "Ridge", "dtr": "DecisionTreeRegressor"}

FIGSIZE = {
        "diabetes": (6, 9),  # TODO
        "glass": (6, 9),  # TODO
        "iris": (6, 9),
        "liver-disorders": (6, 9),  # TODO
        "skin-segmentation": (6, 9),  # TODO
        }

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

SEED = [3, 11, 23, 37, 42]
# SEED = [43, 43, 43, 43, 43]


def figsize(n_feature: int, n_label: int) -> tuple:
    """ Calculate the figure size based on the number of features and labels.

    The reference figure size is determined by the number of features and
    labels in the reference dataset `iris`.
    """
    xtick_offset = 3
    ref_w = 12
    ref_h = 6
    ref_n_feature = 4
    ref_n_label = 3 + xtick_offset

    scale_w = n_feature / ref_n_feature
    scale_h = (n_label + xtick_offset) / ref_n_label
    return (int(ref_w * scale_w), int(ref_h * scale_h))


if __name__ == "__main__":
    print(f"{ROOT_DIR}")
