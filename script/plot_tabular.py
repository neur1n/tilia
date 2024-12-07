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
        dataset.Dataset("iris", "classification", openml_id=61),
        dataset.Dataset("glass", "classification", openml_id=41),
        dataset.Dataset("ionosphere", "classification", openml_id=59),
        # dataset.Dataset("fri_c4_1000_100", "classification", openml_id=718),
        # dataset.Dataset("tecator", "classification", openml_id=851),
        # dataset.Dataset("clean1", "classification", openml_id=40665),
        ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="default", required=False, help="Data used for plotting.")
    ap.add_argument("-f", "--format", default="png", required=False, help="Format of the output figure, png or pdf.")
    ap.add_argument("-o", "--orientation", default="h", required=False, help="Orientation of the plot, h or v.")
    ap.add_argument("-r", "--regressor", default=None, required=False, help="Regressor, linear or tree.")
    ap.add_argument("-s", "--sample", default=-1, type=int, required=False, help="Number of samples to visualize.")
    ap.add_argument("-t", "--timestamp", default=None, required=False, help="Timestamp")
    args = ap.parse_args()

    if args.timestamp is None:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d")

    np.set_printoptions(precision=4, linewidth=200)

    bunch: sklearn.utils.Bunch

    for ds in dataset:
        print(f"Plotting {ds.name}...")

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
        feature_tick = []
        for name, lower, upper in temp:
            left = "(" if lower == float("-inf") else "["
            right = ")" if upper == float("inf") else "]"
            feature_tick.append(f"{name} $\\in$ {left}{lower}, {upper}{right}")

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
                key_type = "index" if len(np.unique(config.SEED)) == 1 else "seed"

                importance = {
                        seed if key_type == "seed" else i: np.zeros((len(ds.label), len(all_feature)), dtype=float)
                        for i, seed in enumerate(config.SEED)
                        }
                binned_importance = {
                        seed if key_type == "seed" else i: np.zeros((len(ds.label), len(all_feature)), dtype=int)
                        for i, seed in enumerate(config.SEED)
                        }

                # NOTE: Must not miss the `_` after `samp{sample}`.
                sample_file = glob.glob(f"{input_dir}/lbl{label}_samp{sample}_*.csv")


                # Files of the same sample but different explained labels.
                for r, file in enumerate(sample_file):
                    input = pd.read_csv(file, sep=config.DELIMITER, index_col=0)

                    for i, seed in enumerate(config.SEED):
                        row = input.loc[f"{seed}"] if key_type == "seed" else input.iloc[i]
                        for c, feat in enumerate(all_feature):
                            if feat not in input.columns:
                                continue
                            _, imp, bimp = row.loc[feat].split(":")
                            importance[seed if key_type == "seed" else i][r, c] += float(imp)
                            binned_importance[seed if key_type == "seed" else i][r, c] += int(bimp)

                img: matplotlib.image.AxesImage
                matrix: np.ndarray
                if args.orientation == "h":
                    # NOTE: Two extra axes for overall importances and colorbar.
                    fig, axes = plt.subplots(
                            nrows=len(importance.keys()) + 1, ncols=1,
                            figsize=config.FIGSIZE[args.orientation][ds.name],
                            gridspec_kw={"height_ratios": [1] * len(importance.keys()) + [0.1]})

                    # fig.suptitle(f"{ds.name} Feature Importance Explained by {config.EXPLAINER[str(args.regressor)]} (Sample {sample}: {label})")
                    fig.suptitle(f"{ds.name} Sample {sample}: {label}")

                    for i, (seed, ax) in enumerate(zip(config.SEED, axes[:-1])):
                        if args.data == "default":
                            matrix = util.normalize_with_mean_reference(importance[seed if key_type == "seed" else i])
                        elif args.data == "binned":
                            matrix = binned_importance[seed if key_type == "seed" else i]
                        else:
                            raise NotImplementedError(f"Not implemented `--data`: {args.data}")

                        img = ax.matshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)

                        for r in range(matrix.shape[0]):
                            for c in range(matrix.shape[1]):
                                cell_value = importance[seed if key_type == "seed" else i][r, c]
                                if cell_value != 0.0:
                                    ax.text(c, r, f"{cell_value:.2f}", va="center", ha="center", color="white", fontsize=8)

                        ax.set_title(f"Seed: {seed}")

                        ax.set_xticks(range(len(all_feature)))
                        if i == 0:
                            # ax.set_xticklabels(x_label, rotation=45, ha="left")
                            # ax.set_xticklabels(feature_tick, rotation=90)
                            ax.set_xticklabels([f"$f_{{{i}}}$" for i in range(len(all_feature))])
                        else:
                            ax.set_xticklabels([])

                        ax.set_yticks(range(len(ds.label)))
                        ax.set_yticklabels(ds.label)

                    fig.colorbar(img, cax=axes[-1], orientation="horizontal")  # type: ignore
                    if args.data == "binned":
                        axes[-1].set_xticks(range(-2, 3, 1))
                        axes[-1].set_xticklabels(["-2",  "-1", "0", "1", "2"])

                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/lbl{label}_samp{sample}_{args.orientation}.{args.format}")
                    plt.cla()
                else:
                    # NOTE: Two extra axes for overall importances and colorbar.
                    fig, axes = plt.subplots(
                            nrows=1, ncols=len(importance.keys()) + 1,
                            figsize=config.FIGSIZE[args.orientation][ds.name],
                            gridspec_kw={"width_ratios": [1] * len(importance.keys()) + [0.1]})

                    fig.suptitle(f"{ds.name} Sample {sample}: {label}")

                    for i, (seed, ax) in enumerate(zip(config.SEED, axes[:-1])):
                        if args.data == "default":
                            # matrix = util.normalize_with_mean_reference(importance[seed])
                            matrix = util.normalize_rows_to_range(importance[seed])
                        elif args.data == "binned":
                            matrix = binned_importance[seed]
                        else:
                            raise NotImplementedError(f"Not implemented `--data`: {args.data}")

                        img = ax.matshow(matrix.T, cmap="coolwarm", vmin=vmin, vmax=vmax)

                        ax.set_title(f"Seed: {seed}")

                        ax.set_yticks(range(len(all_feature)))
                        if i == 0:
                            ax.set_yticklabels(feature_tick)
                        else:
                            ax.set_yticklabels([])

                        ax.set_xticks(range(len(ds.label)))
                        ax.set_xticklabels(ds.label, rotation=90)

                    fig.colorbar(img, cax=axes[-1], orientation="vertical")  # type: ignore
                    if args.data == "binned":
                        axes[-1].set_yticks(range(-2, 3, 1))
                        axes[-1].set_yticklabels(["-2",  "-1", "0", "1", "2"])

                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/lbl{label}_samp{sample}_{args.orientation}.{args.format}")
                    plt.cla()
