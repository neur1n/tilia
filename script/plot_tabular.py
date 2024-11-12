#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import collections
import datetime
import glob

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.utils

import config
import dataset
import util


dataset = [
        # dataset.Dataset("breast-cancer", "classification"),
        # dataset.Dataset("digits", "classification"),
        dataset.Dataset("iris", "classification"),
        ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="sum", required=False, help="Data used for plotting")
    ap.add_argument("-r", "--regressor", default=None, required=False, help="Regressor, linear or tree")
    ap.add_argument("-t", "--timestamp", default=None, required=False, help="Timestamp")
    args = ap.parse_args()

    # NOTE: `None` is for linear.
    if args.regressor == "tree":
        config.REGRESSOR = args.regressor

    timpstamp: str
    if args.timestamp:
        timestamp = args.timestamp
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d")

    bunch: sklearn.utils.Bunch

    for ds in dataset:
        if ds.name == "breast-cancer":
            bunch = sklearn.datasets.load_breast_cancer()  # type: ignore
            ds.label = bunch.target_names
            ds.feature = bunch.feature_names
        elif ds.name == "digits":
            bunch = sklearn.datasets.load_digits()  # type: ignore
            ds.label = bunch.target_names
            ds.feature = bunch.feature_names
        elif ds.name == "iris":
            bunch = sklearn.datasets.load_iris()  # type: ignore
            ds.label = bunch.target_names
            ds.feature = bunch.feature_names
        else:
            pass
            # X, y = sklearn.datasets.load_svmlight_file(f"../dataset/{ds.name}")

        input_dir = f"{config.ROOT_DIR}/output/{timestamp}/{ds.name}/{config.REGRESSOR}"

        importance = {}
        stability = {}

        for l in range(len(ds.label)):
            local_importance = collections.defaultdict(int)
            local_stability = collections.defaultdict(float)

            input_file: list = glob.glob(f"{input_dir}/lbl{l}_exp*.csv")
            for file in input_file:
                input: pd.DataFrame = pd.read_csv(file, sep=config.DELIMITER, index_col=0)
                imp = input.loc["mode"]
                if args.data == "sum":
                    imp = input.loc["sum"]
                elif args.data == "binned_sum":
                    imp = input.loc["binned_sum"]
                var = input.loc["var"]

                for f, _ in input.items():
                    local_importance[f] += float(imp[f])  # type: ignore
                    local_stability[f] += float(var[f])  # type: ignore

            importance[l] = local_importance
            stability[l] = local_stability

        all_feature = list({feat for imp in importance.values() for feat in imp.keys()})
        all_feature = sorted(all_feature, key=util.feature_key)

        matrix = np.zeros((len(importance), len(all_feature)))
        for lbl, imp in importance.items():
            for i, feat in enumerate(all_feature):
                matrix[lbl, i] = imp.get(feat, 0)

        """ Plot with matshow """
        # FIXME: Have problem with the colorbar when using [-1, 1].
        vmin: float
        vmax: float
        if args.data == "mode":
            vmin = -2.0
            vmax = 2.0
        elif args.data == "sum":
            vmin = -1.0
            vmax = 1.0
        elif args.data == "binned_sum":
            vmin = config.REPEAT * (-2.0)
            vmax = config.REPEAT * 2.0
        else:
            raise ValueError(f"Unknown --data: {args.data}")

        cax: matplotlib.image.AxesImage
        fig, axes = plt.subplots(1, len(ds.label), figsize=(18, 6))

        fig.suptitle(f"Dataset: {ds.name} Interpretor: {config.REGRESSOR if config.REGRESSOR else 'linear'} importance: {args.data}")

        for i, (lbl, ax) in enumerate(zip(importance.keys(), axes)):
            vector: np.ndarray
            if args.data == "sum":
                vector = util.normalize(matrix[i:i+1], lower=vmin, upper=vmax).T
            else:
                vector = matrix[i:i+1].T
            cax = ax.matshow(vector, cmap="coolwarm", vmin=vmin, vmax=vmax)

            for j, feat in enumerate(all_feature):
                ax.text(
                        1.1, j,
                        f"{stability[lbl].get(feat, 0.0):.2f}",
                        va="center", ha="left", color="black", fontsize=10,
                        transform=ax.get_yaxis_transform()
                        )

            ax.set_xticks([0])
            ax.set_xticklabels([ds.label[lbl]])
            ax.set_yticks(range(len(all_feature)))
            ax.set_yticklabels(all_feature)

        fig.colorbar(cax, ax=axes, orientation="vertical")  # type: ignore
        plt.show()
