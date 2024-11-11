#!/usr/bin/env python3

import os


DELIMITER = ","


REGRESSOR = None
# REGRESSOR = "tree"


REPEAT = 20


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


SEED = 42


if __name__ == "__main__":
    print(f"{ROOT_DIR}/{REGRESSOR}")
