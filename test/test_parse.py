#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import util


if __name__ == "__main__":
    conditions = [
            "1.50 < petal length (cm) <= 4.25:p1",
            "sepal width (cm) > 3.40:n1",
            ]
    for c in conditions:
        print(util.parse_condition(c))

    # Output:
    # {'petal length (cm) ': {'lower': 1.5, 'upper': 4.25, 'quantile': 'p1'}}
    # {'sepal width (cm) ': {'lower': 3.4, 'upper': None, 'quantile': 'n1'}}
