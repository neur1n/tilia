#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class Dataset:
    name: str
    task: str
    label: list = None
    feature: list = None
