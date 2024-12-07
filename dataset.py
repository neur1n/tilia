#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass


@dataclass
class Dataset:
    name: str
    task: str
    categorical_feature: list = None  # type: ignore
    categorical_label: list = None  # type: ignore
    label: list = None  # type: ignore
    feature: list = None  # type: ignore
    openml_id: int = None  # type: ignore
