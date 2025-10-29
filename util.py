#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd
import re


def bin_importance(value, positive_quantile, negative_quantile):
    if value > 0:
        if value <= positive_quantile:
            return 1
        else:
            return 2
    elif value < 0:
        if value >= negative_quantile:
            return -1
        else:
            return -2
    else:
        return 0


def feature_key(feature: str):
    # NOTE: <num> <op> <feature> <op> <num>
    match = re.match(r"(?:(-?\d+(?:\.\d+)?)\s*)?([<>]=?|=)?\s*([\w\s()]+)\s*([<>]=?|=)?\s*(?:(-?\d+(?:\.\d+)?))?", feature)
    assert match, f"Invalid feature: {feature}"

    lower = None
    upper = None
    name = match.group(3).strip()

    if match.group(4) in [">", ">="]:
        lower = match.group(5)
        upper = float("inf")
    elif match.group(4) in ["<", "<="]:
        upper = match.group(5)

    if match.group(2) in [">", ">="]:
        upper = match.group(1)
    elif match.group(2) in ["<", "<="]:
        lower = match.group(1)

    lower = float(lower) if lower is not None else float("-inf")
    upper = float(upper) if upper is not None else float("inf")

    return [name, lower, upper]


def normalize(data: np.ndarray, ifnan="lower") -> np.ndarray:
    min_val = np.min(data)
    max_val = np.max(data)

    if min_val == max_val:
        return np.full(data.shape, 0.0 if "lower" else 1.0)

    normalized = (data - min_val) / (max_val - min_val)

    return normalized


def normalize_rows_to_range(matrix, lower=-1, upper=1):
    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    normalized = lower + (upper - lower) * (matrix - row_min) / (row_max - row_min + 1e-8)
    return normalized


def normalize_with_mean_reference(data: np.ndarray) -> np.ndarray:
    output = np.zeros_like(data)

    for i, row in enumerate(data):
        if np.all(row == 0):
            output[i] = copy.deepcopy(row)
            continue

        pos = [x for x in row if x > 0]
        neg = [x for x in row if x < 0]

        normalized = []

        if pos and not neg:
            mean_pos = sum(pos) / len(pos)
            for x in row:
                if x == 0:
                    normalized.append(x)
                else:
                    normalized.append((x - mean_pos) / (max(pos) - mean_pos))
        elif neg and not pos:
            mean_neg = sum(neg) / len(neg)
            for x in row:
                if x == 0:
                    normalized.append(x)
                else:
                    normalized.append((x - mean_neg) / (mean_neg - min(neg)))
        else:
            max_pos = max(pos, default=0)
            min_neg = min(neg, default=0)
            normalized = [(x / max_pos if x > 0 else x / abs(min_neg) if x < 0 else 0) for x in row]

        output[i] = copy.deepcopy(normalized)

    return output


def normalize_positive_negative(data: np.ndarray) -> np.ndarray:
    normalized = data.copy()

    pos_mask = data > 0
    neg_mask = data < 0

    if pos_mask.any():
        pos_min = data[pos_mask].min()
        pos_max = data[pos_mask].max()
        if pos_min == pos_max:
            normalized[pos_mask] = 1
        else:
            normalized[pos_mask] = (data[pos_mask] - pos_min) / (pos_max - pos_min)

    if neg_mask.any():
        neg_min = data[neg_mask].min()
        neg_max = data[neg_mask].max()
        if neg_min == neg_max:
            normalized[neg_mask] = -1
        else:
            normalized[neg_mask] = (data[neg_mask] - neg_max) / (neg_min - neg_max) * -1

    return normalized

    # shape = data.shape
    # data = data.flatten()

    # pos = [x for x in data if x > 0]
    # neg = [x for x in data if x < 0]

    # normalized = []

    # if pos and not neg:
    #     mean_pos = sum(pos) / len(pos)
    #     normalized = [(x - mean_pos) / (max(pos) - mean_pos) for x in data]
    # elif neg and not pos:
    #     mean_neg = sum(neg) / len(neg)
    #     normalized = [(x - mean_neg) / (mean_neg - min(neg)) for x in data]
    # else:
    #     max_pos = max(pos, default=0)
    #     min_neg = min(neg, default=0)
    #     normalized = [(x / max_pos if x > 0 else x / abs(min_neg) if x < 0 else 0) for x in data]

    # return np.array(normalized).reshape(shape)


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
