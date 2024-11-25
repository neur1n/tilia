#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import collections
import datetime
import glob
import re

import matplotlib.image
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.axes_divider
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.utils

import config
import dataset
import util


dataset = [
        # dataset.Dataset("liver-disorders", "classification", openml_id=8),
        # dataset.Dataset("diabetes", "classification", openml_id=37),
        # dataset.Dataset("glass", "classification", openml_id=41),
        dataset.Dataset("iris", "classification", openml_id=61),
        # dataset.Dataset("skin-segmentation", "classification", openml_id=1502),
        ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="default", required=False, help="Data used for plotting")
    ap.add_argument("-r", "--regressor", default=None, required=False, help="Regressor, linear or tree")
    ap.add_argument("-s", "--sample", default=-1, type=int, required=False, help="Number of samples to visualize.")
    ap.add_argument("-t", "--timestamp", default=None, required=False, help="Timestamp")
    args = ap.parse_args()

    if args.timestamp is None:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d")

    np.set_printoptions(precision=4, linewidth=200)

    bunch: sklearn.utils.Bunch

    for ds in dataset:
        input_dir = f"{config.ROOT_DIR}/output/{args.timestamp}/{ds.name}/{args.regressor}"
        output_dir = f"{input_dir}/fig/{args.data}"
        os.makedirs(output_dir, exist_ok=True)

        bunch = sklearn.datasets.fetch_openml(data_id=ds.openml_id)  # type: ignore
        X, y = bunch.data.to_numpy(), bunch.target.to_numpy()
        ds.label = np.unique(y).tolist()
        ds.feature = bunch.feature_names

        all_feature = set()
        all_sample = collections.defaultdict(set)

        for label in ds.label:
            sample_file: list = glob.glob(f"{input_dir}/lbl{label}*.csv")
            for file in sample_file:
                input: pd.DataFrame = pd.read_csv(file, sep=config.DELIMITER, index_col=0)
                all_feature.update(input.columns)

                match = re.search(r"samp(\d+)", file)
                assert match, f"Cannot find sample ID in {file}"
                sample_list = int(match.group(1))
                all_sample[label].add(sample_list)

        all_feature = sorted(list(all_feature), key=util.feature_key)

        temp = [util.feature_key(feat) for feat in all_feature]
        x_label = []
        for name, lower, upper in temp:
            left = "(" if lower == float("-inf") else "["
            right = ")" if upper == float("inf") else "]"
            x_label.append(f"{name} $\\in$ {left}{lower}, {upper}{right}")

        if args.sample > 0:
            all_sample = {
                    label: sorted(list(sample)[:min(args.sample, len(sample))])
                    for label, sample in all_sample.items()
                    }
        else:
            all_sample = {
                    label: sorted(list(sample))
                    for label, sample in all_sample.items()
                    }

        """
        For each label, choose S samples to visualize the importance and
        stability for 3 different seeds. This gives 3 figures for each label.
        """
        vmin: float
        vmax: float
        if args.data == "default":
            vmin = -1.0
            vmax = 1.0
        elif args.data == "binned":
            vmin = -2.0
            vmax = 2.0
        elif args.data == "mode":
            vmin = -2.0
            vmax = 2.0
        else:
            raise ValueError(f"Unknown --data: {args.data}")

        for label, sample_list in all_sample.items():
            for sample in sample_list:
                importance = {
                        seed: np.zeros((3, len(all_feature)), dtype=float)
                        for seed in config.SEED
                        }
                binned_importance = {
                        seed: np.zeros((3, len(all_feature)), dtype=int)
                        for seed in config.SEED
                        }

                # NOTE: Must not miss the `_` after `samp{sample}`.
                sample_file = glob.glob(f"{input_dir}/lbl{label}_samp{sample}_*.csv")

                # Files of the same sample but different explained labels.
                for r, file in enumerate(sample_file):
                    input = pd.read_csv(file, sep=config.DELIMITER, index_col=0)

                    for seed in config.SEED:
                        row = input.loc[f"{seed}"]
                        for c, feat in enumerate(all_feature):
                            if feat not in input.columns:
                                continue
                            _, imp, bimp = row.loc[feat].split(":")
                            importance[seed][r, c] += float(imp)
                            binned_importance[seed][r, c] += int(bimp)

                # NOTE: Two extra axes for overall importances and colorbar.
                fig, axes = plt.subplots(
                        nrows=len(importance.keys()) + 1, ncols=1,
                        figsize=config.FIGSIZE[ds.name],
                        gridspec_kw={"height_ratios": [1] * len(importance.keys()) + [0.1]})

                # fig.suptitle(f"{ds.name} Feature Importance Explained by {config.EXPLAINER[str(args.regressor)]} (Sample {sample}: {label})")
                fig.suptitle(f"{ds.name} Sample {sample}: {label}")

                img: matplotlib.image.AxesImage
                for i, (seed, ax) in enumerate(zip(config.SEED, axes[:-1])):
                    matrix: np.ndarray
                    if args.data == "default":
                        matrix = util.normalize_with_mean_reference(importance[seed])
                    elif args.data == "binned":
                        matrix = binned_importance[seed]
                    else:
                        raise NotImplementedError(f"Not implemented `--data`: {args.data}")

                    img = ax.matshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)

                    ax.set_title(f"Seed: {seed}")

                    ax.set_xticks(range(len(all_feature)))
                    if i == 0:
                        # ax.set_xticklabels(x_label, rotation=45, ha="left")
                        ax.set_xticklabels(x_label, rotation=90)
                    else:
                        ax.set_xticklabels([])

                    ax.set_yticks(range(len(ds.label)))
                    ax.set_yticklabels(ds.label)

                fig.colorbar(img, cax=axes[-1], orientation="horizontal")  # type: ignore
                if args.data == "binned":
                    axes[-1].set_xticks(range(-2, 3, 1))
                    axes[-1].set_xticklabels(["-2",  "-1", "0", "1", "2"])

                plt.tight_layout()
                plt.savefig(f"{output_dir}/lbl{label}_samp{sample}.png")

        #     # Collect all features
        #     all_feature = set()


        #     all_feature = sorted(list(all_feature), key=util.feature_key)

        #     importance = {seed: np.zeros((3, len(all_feature))) for seed in config.SEED}

        #     for explbl in ds.label:
        #         input_file = glob.glob(f"{input_dir}/lbl{label}*exp{explbl}.csv")



        #     # Select samples
        #     if args.sample > 0:
        #         input_file = input_file[:min(args.sample, len(input_file))]

        #     for file in input_file:
        #         match = re.search(r"exp(\w+)", file)
        #         assert match, f"Cannot find explained label in {file}"
        #         explbl = match.group(1)

        #         # match = re.search(r"samp(\d+)", file)
        #         # assert match, f"Cannot find sample ID in {file}"
        #         # sample = int(match.group(1))

        #         importance = {}

        #         input: pd.DataFrame = pd.read_csv(file, sep=config.DELIMITER, index_col=0)

        #         # for seed in input.index[:len(config.SEED)]:
        #         breakpoint()



        #         imp = input.loc["mode"]
        #         if args.data == "sum":
        #             imp = input.loc["sum"]
        #         elif args.data == "binned_sum":
        #             imp = input.loc["binned_sum"]
        #         sta = input.loc["var"]

        #         for feature in input.columns:
        #             importance[feature] += float(imp[feature])  # type: ignore
        #             stability[feature] += float(sta[feature])  # type: ignore

        #         breakpoint()




        # all_feature = set()
        # all_sample = collections.defaultdict(list)
        # importance = {}
        # stability = {}

        # for label in ds.label:
        #     importance[label] = {}
        #     stability[label] = {}

        #     input_file: list = glob.glob(f"{input_dir}/lbl{label}*exp{label}.csv")
        #     for file in input_file:
        #         match = re.search(r"samp(\d+)", file)
        #         assert match, f"Cannot find sample ID in {file}"
        #         sample = int(match.group(1))
        #         all_sample[label].append(sample)

        #         if sample not in importance:
        #             importance[label][sample] = collections.defaultdict(int)
        #             stability[label][sample] = collections.defaultdict(float)

        #         input: pd.DataFrame = pd.read_csv(file, sep=config.DELIMITER, index_col=0)

        #         all_feature.update(input.columns)

        #         imp = input.loc["mode"]
        #         if args.data == "sum":
        #             imp = input.loc["sum"]
        #         elif args.data == "binned_sum":
        #             imp = input.loc["binned_sum"]
        #         sta = input.loc["var"]

        #         for feature in input.columns:
        #             importance[label][sample][feature] += float(imp[feature])  # type: ignore
        #             stability[label][sample][feature] += float(sta[feature])  # type: ignore

        # all_feature = sorted(list(all_feature), key=util.feature_key)
        # # all_sample = sorted(all_sample)
        # breakpoint()

        # """ Plot with matshow """
        # # FIXME: Have problem with the colorbar when using [-1, 1].
        # vmin: float
        # vmax: float
        # if args.data == "mode":
        #     vmin = -2.0
        #     vmax = 2.0
        # elif args.data == "sum":
        #     vmin = -1.0
        #     vmax = 1.0
        # elif args.data == "binned_sum":
        #     vmin = len(config.SEED) * (-2.0)
        #     vmax = len(config.SEED) * 2.0
        # else:
        #     raise ValueError(f"Unknown --data: {args.data}")

        # selected_sample = all_sample[:min(3, len(all_sample))]

        # # NOTE: Two extra axes, one for accumulated importances and one for colorbar.
        # fig, axes = plt.subplots(
        #         nrows=len(selected_sample) + 2, ncols=1,
        #         figsize=(10, 14),
        #         gridspec_kw={"height_ratios": [1] * len(selected_sample) + [0.1]})
        #         # figsize=config.figsize(len(ds.feature), len(ds.label)))

        # fig.suptitle(f"Feature Importance and Stability of Dataset {ds.name} (Explainer: {config.EXPLAINER[str(args.regressor)]})")

        # img: matplotlib.image.AxesImage
        # for i, (sample, ax) in enumerate(zip(selected_sample, axes[:-1])):
        #     matrix = np.zeros((len(ds.label), len(all_feature)))
        #     for r, lbl in enumerate(ds.label):
        #         for c, feat in enumerate(all_feature):
        #             matrix[r, c] = importance[lbl][sample].get(feat, 0)

        #             cell_value = stability[lbl][sample].get(feat, 0.0)
        #             ax.text(c, r, f"{cell_value:.2f}", va="center", ha="center", color="black", fontsize=10)

        #     if args.data == "sum":
        #         matrix = util.normalize_with_mean_reference(matrix)

        #     img = ax.matshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)

        #     ax.set_title(f"Seed: {sample}")

        #     ax.set_xticks(range(len(all_feature)))
        #     if i == 0:
        #         ax.set_xticklabels(all_feature, rotation=45, ha="left")
        #     else:
        #         ax.set_xticklabels([])

        #     ax.set_yticks(range(len(ds.label)))
        #     ax.set_yticklabels(ds.label)

        #     # if i == len(all_seed) - 1:
        #     #     divider = mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable(ax)
        #     #     cax = divider.append_axes("bottom", size="10%", pad=0.25)
        #     #     fig.colorbar(img, cax=cax, orientation="horizontal")

        # fig.colorbar(img, cax=axes[-1], orientation="horizontal")  # type: ignore

        # plt.tight_layout()
        # plt.savefig(f"{output_dir}/{datetime.datetime.now().strftime('%H%M%S')}.png")





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
