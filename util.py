#!/usr/bin/env python3

import collections
import numpy as np
import pandas as pd
import re
import scipy.stats


def bin_importance(value, positive_quantile, negative_quantile):
    if value > 0:
        if value <= positive_quantile[0]:
            return 1
        else:
            return 2
    elif value < 0:
        if value <= negative_quantile[0]:
            return -1
        else:
            return -2
    else:
        return 0


def feature_key(feature: str):
    match = re.match(r"(?:(\d*\.?\d*)\s*<\s*)?([a-zA-Z\s\(\)]+)\s*(<=?|>=?|==)\s*(\d*\.?\d*)", feature)
    assert match, f"Invalid feature: {feature}"
    start, name, operator, end = match.groups()

    key = [name, 0.0, 0.0]

    if start:
        key[1] = float(start)
    elif operator in (">", ">="):
        key[1] = float(end)
    else:
        key[1] = float("-inf")

    if operator in ("<", "<=", "=="):
        key[2] = float(end)
    else:
        key[2] = float("inf")

    return key


def get_column_entropy(column: pd.Series):
    quantile = []

    for i, item in enumerate(column):
        parsed: dict = parse_condition(item)
        for _, v in parsed.items():
            quantile.append(v["quantile"])

    counts = collections.Counter(quantile)
    return scipy.stats.entropy(list(counts.values()), base=2)

def normalize(matrix: np.ndarray, lower: float = 0.0, upper: float = 1.0, scale: float = 1.0) -> np.ndarray:
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    if min_val == max_val:
        return np.full(matrix.shape, lower)

    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    scaled_matrix = normalized_matrix * (upper - lower) + lower
    return scaled_matrix


def parse_condition(condition: str):
    parsed = {}

    expr, quan = condition.split(":")

    # Match patterns like "1.50 < petal length (cm) <= 4.25" and "sepal width (cm) > 3.40"
    match = re.match(r"(?:(\d*\.?\d*)\s*<\s*)?([a-zA-Z\s\(\)]+)\s*(<=?|>=?|==)\s*(\d*\.?\d*)", expr)

    if match:
        lower, feature, operator, upper = match.groups()

        if feature not in parsed:
            parsed[feature] = {}

        if lower:
            parsed[feature]["lower"] = float(lower)
        elif operator in (">", ">="):
            parsed[feature]["lower"] = float(upper)
        else:
            parsed[feature]["lower"] = None

        if operator in ("<", "<="):
            parsed[feature]["upper"] = float(upper)
        elif operator == "==":
            parsed[feature]["equals"] = float(upper)
        else:
            parsed[feature]["upper"] = None

        parsed[feature]["quantile"] = quan

    return parsed
