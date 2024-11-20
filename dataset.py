#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass


@dataclass
class Dataset:
    name: str
    task: str
    label: list = None  # type: ignore
    feature: list = None  # type: ignore
    openml_id: int = None  # type: ignore
