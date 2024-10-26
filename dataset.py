#!/usr/bin/env python3

from dataclasses import dataclass
import typing


@dataclass
class Dataset:
    name: str
    task: str
    label: typing.Union[list, None] = None
    feature: typing.Union[list, None] = None
