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

import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import sklearn.datasets
import sklearn.metrics
import sklearn.utils

import config
import dataset
import util


def pairwise_jaccard(matrices):
    n = len(matrices)
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # if i <= j:  # Compute only for upper triangle (including diagonal)
            flat_i = matrices[i].flatten(order="F")  # Flatten column-wise
            flat_j = matrices[j].flatten(order="F")  # Flatten column-wise
            jaccard[i, j] = sklearn.metrics.jaccard_score(flat_i, flat_j, average="macro")
            jaccard[j, i] = jaccard[i, j]  # Symmetric matrix
    return 1 - jaccard


def mean_std_without_outliers_iqr(data, axis=0):
    q1 = np.percentile(data, 25, axis=axis, keepdims=True)
    q3 = np.percentile(data, 75, axis=axis, keepdims=True)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = (data >= lower_bound) & (data <= upper_bound)
    filtered_data = np.where(mask, data, np.nan)  # Replace outliers with NaN
    mean = np.nanmean(filtered_data, axis=axis)
    std = np.nanstd(filtered_data, axis=axis)
    return mean, std

def weighted_avg_and_std(values, weights, axis=0):
    average = np.average(values, weights=weights, axis=axis)
    variance = np.average((values - average) ** 2, weights=weights, axis=axis)
    return average, np.sqrt(variance)


dataset = [
        dataset.Dataset("iris", "classification", openml_id=61),
        dataset.Dataset("phoneme", "classification", openml_id=1489),
        dataset.Dataset("diabetes", "classification", openml_id=37),
        dataset.Dataset("glass", "classification", openml_id=41),
        dataset.Dataset("ionosphere", "classification", openml_id=59),
        dataset.Dataset("fri_c4_1000_100", "classification", openml_id=718),
        dataset.Dataset("tecator", "classification", openml_id=851),
        dataset.Dataset("clean1", "classification", openml_id=40665),
        ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="binned", required=False, help="Data used for plotting.")
    ap.add_argument("-f", "--format", default="png", required=False, help="Format of the output figure, png or pdf.")
    ap.add_argument("-p", "--plot", default=1, type=int, required=False, help="Plot, or not.")
    ap.add_argument("-r", "--regressor", default=None, required=False, help="Regressor, linear or tree.")
    ap.add_argument("-s", "--sample", default=-1, type=int, required=False, help="Number of samples to visualize.")
    ap.add_argument("-t", "--timestamp", default=None, required=False, help="Timestamp")
    args = ap.parse_args()

    if args.regressor == "linear":
        args.regressor = None

    if args.timestamp is None:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d")

    np.set_printoptions(precision=4, linewidth=200)

    bunch: sklearn.utils.Bunch

    for ds in dataset:
        print(f"\nPlotting {ds.name}...")

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

        if args.sample > 0:
            total_count = sum(len(sample) for sample in all_sample.values())
            args.sample = min(args.sample, total_count)

            label_count = {
                    label: round(len(sample) * args.sample / total_count)
                    for label, sample in all_sample.items()
                    }

            remain_count = args.sample - sum(label_count.values())
            if remain_count > 0:
                # NOTE: Not using `ds.label` here since some labels may not have any samples.
                for l in all_sample.keys():
                    extra = len(all_sample[l]) - label_count[l]
                    additional = min(extra, remain_count)
                    label_count[l] += additional
                    remain_count -= additional

            all_sample = {
                    label: sorted(list(sample)[:label_count[label]])
                    for label, sample in all_sample.items()
                    }
        else:
            all_sample = {
                    label: sorted(list(sample))
                    for label, sample in all_sample.items()
                    }

        stability = collections.defaultdict(list)

        for label, sample_list in all_sample.items():
            for sample in sample_list:
                importance = {}
                binned_importance = {}
                for i in range(len(config.SEED)):
                    importance[f"seed{i}"] = np.zeros((len(ds.label), len(all_feature)), dtype=float)
                    binned_importance[f"seed{i}"] = np.zeros((len(ds.label), len(all_feature)), dtype=int)

                # NOTE: Must not miss the `_` after `samp{sample}`.
                sample_file = glob.glob(f"{input_dir}/lbl{label}_samp{sample}_*.csv")

                # Files of the same sample but different explained labels.
                for r, file in enumerate(sample_file):
                    input = pd.read_csv(file, sep=config.DELIMITER, index_col=0)

                    for i, seed in enumerate(config.SEED):
                        row = input.iloc[i]
                        for c, feat in enumerate(all_feature):
                            if feat not in input.columns:
                                continue

                            imp = 0.0
                            bimp = 0
                            if pd.isna(row.loc[feat]):
                                imp = np.mean(importance[f"seed{i}"][:, c])
                                bimp = np.mean(binned_importance[f"seed{i}"][:, c])
                            else:
                                _, imp, bimp = row.loc[feat].split(":")
                            importance[f"seed{i}"][r, c] += float(imp)
                            binned_importance[f"seed{i}"][r, c] += int(bimp)

                matrix = []
                for i in range(len(config.SEED)):
                    if args.data == "raw":
                        matrix.append(util.normalize_rows_to_range(importance[f"seed{i}"]).T)
                    elif args.data == "binned":
                        matrix.append(binned_importance[f"seed{i}"].T)
                    else:
                        raise NotImplementedError(f"Not implemented `--data`: {args.data}")

                stability[label].append(pairwise_jaccard(matrix))

        if not args.plot:
            all_mean = []
            all_std = []
            all_cnt = []

            for label, data in stability.items():
                mean, std = mean_std_without_outliers_iqr(data, axis=0)
                print(f"{label}: {np.mean(mean):.2f} +- {np.mean(std):.2f}")
                # print(f"{label}: {np.nanmean(data):.2f} +- {np.nanstd(data):.2f}")
                all_mean.append(mean)
                all_std.append(std)
                all_cnt.append(len(data))

            weight = np.array(all_cnt) / np.sum(all_cnt)
            w_mean, w_std = weighted_avg_and_std(np.array(all_mean), np.array(weight), axis=0)
            print(f"Weighted average: {np.mean(w_mean):.2f} +- {np.mean(w_std):.2f}\n")
            continue

        """============================================================
        Plotting {{{
        """
        # fig = plt.figure(figsize=config.FIGSIZE_v2[ds.name], layout="constrained")
        # fig = plt.figure(layout="tight")
        fig = plt.figure(layout="constrained")
        fig_w = 0
        fig_h = 0

        w_ratio =[5] * len(ds.label) + [0.25]
        h_ratio = [5]
        gs = fig.add_gridspec(
                nrows=len(h_ratio), ncols=len(w_ratio),
                width_ratios=w_ratio, height_ratios=h_ratio,
                wspace=0.0, hspace=0.0,
                )

        axes = {}
        for i in range(len(ds.label)):
            axes[f"{ds.label[i]}"] = fig.add_subplot(gs[0, i])
        axes["colorbar"] = fig.add_subplot(gs[0, -1])

        """ colormap """
        palette = ["#faebd7", "#cfcefd"]
        cmap_2 = matplotlib.colors.LinearSegmentedColormap.from_list("var", palette)
        norm_2 = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

        ttl = fig.suptitle(f"Stability of Dataset {ds.name} (Jaccard $\\downarrow$)")

        """ stability """
        for i, label in enumerate(ds.label):
            if stability[label]:
                data = np.nanmean(stability[label], axis=0)
            else:
                data = np.zeros((len(config.SEED), len(config.SEED)), dtype=float)
                axes[label].text(2, 2, "No data", ha="center", va="center", fontsize=20)
            hmap = seaborn.heatmap(
                    data,
                    ax=axes[label], cmap=cmap_2,
                    cbar=False, square=True)
            fig_w += hmap.get_window_extent().width

            axes[label].invert_yaxis()
            axes[label].margins(x=0, y=0)
            axes[label].set_aspect(1)
            axes[label].set_title(label)
            axes[label].set_xticks(np.arange(len(config.SEED)) + 0.5)
            axes[label].set_xticklabels(range(len(config.SEED)))
            if i != 0:
                axes[label].set_yticks([])

        fig_h = hmap.get_window_extent().height

        # # axes["stability"].yaxis.tick_right()
        # axes["stability"].invert_yaxis()
        # axes["stability"].margins(x=0, y=0)
        # axes["stability"].set_aspect(1)
        # # ax.set_title(f"Jaccard $\\downarrow$")
        # # axes["stability"].set_xticks(range(len(ds.label)))
        # # axes["stability"].set_xticklabels([f"$L_{{{i}}}$" for i in range(len(ds.label))])
        # # axes["stability"].set_yticks([])
        # # axes["stability"].set_yticklabels([])
        # axes["stability"].set_yticklabels(axes["stability"].get_yticklabels(), rotation=0)

        """ colorbar """
        cbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm_2, cmap=cmap_2),
                cax=axes["colorbar"],
                orientation="vertical",
                )
        fig_w += cbar.ax.get_window_extent().width
        # if args.data == "binned":
        #     cbar.set_ticks([-2, -1, 0, 1, 2])
        #     cbar.set_ticklabels(["-2",  "-1", "0", "1", "2"])
        # axes["colorbar_1"].tick_params(length=0)
        axes["colorbar"].set_aspect(len(config.SEED) * 4)  # / 0.25

        h = (fig_h / 50) / 3 * len(ds.label)
        w = (fig_w / 60) / 3 * len(ds.label)
        fig.set_size_inches(w, h)
        # fig.set_size_inches(fig_w / 60, fig_h / 50)

        plt.savefig(f"{output_dir}/stability.{args.format}", bbox_inches="tight")
        plt.close()
        # plt.show()
        # exit()
        """ Plotting }}} """
