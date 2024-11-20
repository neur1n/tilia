#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        # dataset.Dataset("iris", "classification", openml_id=61),
        dataset.Dataset("diabetes", "classification", openml_id=37),
        # dataset.Dataset("liver-disorders", "classification", openml_id=8),
        ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="sum", required=False, help="Data used for plotting")
    ap.add_argument("-r", "--regressor", default=None, required=False, help="Regressor, linear or tree")
    ap.add_argument("-t", "--timestamp", default=None, required=False, help="Timestamp")
    args = ap.parse_args()

    if args.timestamp is None:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d")

    bunch: sklearn.utils.Bunch

    for ds in dataset:
        input_dir = f"{config.ROOT_DIR}/output/{args.timestamp}/{ds.name}/{args.regressor}"
        output_dir = f"{input_dir}/fig"
        os.makedirs(output_dir, exist_ok=True)

        bunch = sklearn.datasets.fetch_openml(data_id=ds.openml_id)  # type: ignore
        X, y = bunch.data.to_numpy(), bunch.target.to_numpy()
        ds.label = np.unique(y).tolist()
        ds.feature = bunch.feature_names

        importance = {}
        stability = {}

        for l in ds.label:
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
        for i, (_, imp) in enumerate(importance.items()):
            for j, feat in enumerate(all_feature):
                matrix[i, j] = imp.get(feat, 0)

        if args.data == "sum":
            matrix = util.normalize_with_mean_reference(matrix)

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
            vmin = len(config.SEED) * (-2.0)
            vmax = len(config.SEED) * 2.0
        else:
            raise ValueError(f"Unknown --data: {args.data}")

        fig, ax = plt.subplots(figsize=config.figsize(len(ds.feature), len(ds.label)))
        cax = ax.matshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                cell_value = stability[list(importance.keys())[i]].get(all_feature[j], 0.0)
                ax.text(j, i, f"{cell_value:.2f}", va="center", ha="center", color="black", fontsize=10)

        # Configure axes
        ax.set_xticks(range(len(all_feature)))
        ax.set_xticklabels(all_feature, rotation=45, ha="left")
        ax.set_yticks(range(len(importance.keys())))
        ax.set_yticklabels(ds.label)

        fig.colorbar(cax, ax=ax, orientation="horizontal")
        fig.suptitle(f"Feature Importance and Stability of Dataset {ds.name} (Explainer: {config.EXPLAINER[str(args.regressor)]})")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{datetime.datetime.now().strftime('%H%M%S')}.png")





        # cax: matplotlib.image.AxesImage
        # fig, axes = plt.subplots(1, len(ds.label), figsize=(30, 12))

        # fig.suptitle(f"Dataset: {ds.name} Interpretor: {args.regressor if args.regressor else 'linear'} importance: {args.data}")

        # for i, (lbl, ax) in enumerate(zip(importance.keys(), axes)):
        #     vector: np.ndarray
        #     if args.data == "sum":
        #         # vector = util.normalize(matrix[i:i+1], lower=vmin, upper=vmax).T
        #         vector = util.normalize_with_mean_reference(matrix[i:i+1]).T
        #     else:
        #         vector = matrix[i:i+1].T
        #     cax = ax.matshow(vector, cmap="coolwarm", vmin=vmin, vmax=vmax)

        #     for j, feat in enumerate(all_feature):
        #         ax.text(
        #                 1.1, j,
        #                 f"{stability[lbl].get(feat, 0.0):.2f}",
        #                 va="center", ha="left", color="black", fontsize=10,
        #                 transform=ax.get_yaxis_transform()
        #                 )

        #     ax.set_xticks([0])
        #     ax.set_xticklabels([ds.label[lbl]])
        #     ax.set_yticks(range(len(all_feature)))
        #     ax.set_yticklabels(all_feature)

        # fig.colorbar(cax, ax=axes, orientation="vertical")  # type: ignore
        # plt.savefig(f"{output_dir}/{datetime.datetime.now().strftime('%H%M%S')}.png")
