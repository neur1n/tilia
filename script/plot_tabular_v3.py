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


dataset = [
        dataset.Dataset("iris", "classification", openml_id=61),
        # dataset.Dataset("phoneme", "classification", openml_id=1489),
        # dataset.Dataset("diabetes", "classification", openml_id=37),
        # dataset.Dataset("glass", "classification", openml_id=41),
        # dataset.Dataset("ionosphere", "classification", openml_id=59),
        # dataset.Dataset("fri_c4_1000_100", "classification", openml_id=718),
        # dataset.Dataset("tecator", "classification", openml_id=851),
        # dataset.Dataset("clean1", "classification", openml_id=40665),
        ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="binned", required=False, help="Data used for plotting.")
    ap.add_argument("-f", "--format", default="png", required=False, help="Format of the output figure, png or pdf.")
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

        """
        For each label, choose S samples to visualize the importance and
        stability for 3 different seeds. This gives 3 figures for each label.
        """
        vmin: float
        vmax: float
        if args.data == "raw":
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
                            _, imp, bimp = row.loc[feat].split(":")
                            importance[f"seed{i}"][r, c] += float(imp)
                            binned_importance[f"seed{i}"][r, c] += int(bimp)

                """============================================================
                Plotting {{{
                """
                # fig = plt.figure(figsize=config.FIGSIZE_v2[ds.name], layout="constrained")
                fig = plt.figure(layout="constrained")
                fig_w = 0
                fig_h = 0

                # NOTE: Layout: seed0, seed1, seed2, seed3, seed4, colorbar, stability, colorbar
                w_ratio =[len(ds.label)] * len(config.SEED) + [0.25] + [len(all_feature), 0.25]
                h_ratio = [len(all_feature)]
                gs = fig.add_gridspec(
                        nrows=1, ncols=len(config.SEED)+3,
                        width_ratios=w_ratio, height_ratios=h_ratio,
                        wspace=0.0, hspace=0.0,
                        )

                w, h = matplotlib.figure.figaspect((sum(h_ratio)) / sum(w_ratio))
                fig.set_size_inches(w, h)

                axes = {}
                axid = 0
                for i in range(len(config.SEED)):
                    axes[f"seed{i}"] = fig.add_subplot(gs[:, i])
                    axid += 1
                axes["colorbar_1"] = fig.add_subplot(gs[:, axid])
                axid += 1
                axes["stability"] = fig.add_subplot(gs[:, axid])
                axid += 1
                axes["colorbar_2"] = fig.add_subplot(gs[:, axid])

                """ colormap """
                # palette = ["#84a0c6", "#c2d0e3", "#faebd7", "#f1bcbc", "#e27878"]  # iceberg
                palette = ["#58539f", "#bbbbd6", "#f1faee", "#eebabb", "#d86967"]
                bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
                cmap_1 = matplotlib.colors.ListedColormap(palette)
                norm_1 = matplotlib.colors.BoundaryNorm(bounds, cmap_1.N, clip=True)

                # palette = ["#faebd7", "#cfcefd"]
                palette = ["#f1faee", "#9797f8"]
                # palette = ["#f1faee", "#00007f"]
                cmap_2 = matplotlib.colors.LinearSegmentedColormap.from_list("var", palette)
                norm_2 = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

                # ttl = fig.suptitle(f"{ds.name} Sample {sample}: {label}")
                ttl = fig.suptitle(f"Label: {label}")
                bbox = ttl.get_window_extent()
                fig_h += bbox.y1 - bbox.y0

                matrix = []
                """ matshow """
                for i in range(len(config.SEED)):
                    if args.data == "raw":
                        matrix.append(util.normalize_rows_to_range(importance[f"seed{i}"]).T)
                    elif args.data == "binned":
                        matrix.append(binned_importance[f"seed{i}"].T)
                    else:
                        raise NotImplementedError(f"Not implemented `--data`: {args.data}")

                    img = axes[f"seed{i}"].matshow(matrix[-1], cmap=cmap_1, vmin=vmin, vmax=vmax, origin="lower")
                    bbox = img.get_window_extent()
                    # print(f"matshow: {bbox.x1 - bbox.x0}, {bbox.y1 - bbox.y0}")
                    fig_w += bbox.x1 - bbox.x0

                    """ title """
                    ttl = axes[f"seed{i}"].set_title(f"Seed: {config.SEED[i]}")
                    """ x-axis """
                    axes[f"seed{i}"].xaxis.set_ticks_position('bottom')
                    axes[f"seed{i}"].set_xticks(range(len(ds.label)))
                    axes[f"seed{i}"].set_xticklabels([f"$C_{{{i}}}$" for i in range(len(ds.label))])
                    """ y-axis """
                    if i == 0:
                        axes[f"seed{i}"].set_yticks(range(len(all_feature)))
                        axes[f"seed{i}"].set_yticklabels([f"$F_{{{i}}}$" for i in range(len(all_feature))])
                    else:
                        axes[f"seed{i}"].set_yticks([])
                        axes[f"seed{i}"].set_yticklabels([])

                bbox = img.get_window_extent()
                fig_h += bbox.y1 - bbox.y0
                bbox = ttl.get_window_extent()
                fig_h += bbox.y1 - bbox.y0

                    # for f in range(len(all_feature)):
                    #     axes[f"seed{i}"].axhline(y=f+0.5, color="black", linewidth=0.5)
                    # for l in range(len(ds.label)):
                    #     axes[f"seed{i}"].axvline(x=l+0.5, color="black", linewidth=0.5)

                """ colorbar_1 """
                cbar = fig.colorbar(
                        plt.cm.ScalarMappable(norm=norm_1, cmap=cmap_1),
                        cax=axes["colorbar_1"],
                        orientation="vertical",
                        )
                bbox = cbar.ax.get_window_extent()
                fig_w += bbox.x1 - bbox.x0
                if args.data == "binned":
                    cbar.set_ticks([-2, -1, 0, 1, 2])
                    cbar.set_ticklabels(["-2",  "-1", "0", "1", "2"])
                axes["colorbar_1"].tick_params(length=0)
                axes["colorbar_1"].set_aspect(len(all_feature) * 4)  # / 0.25

                """ stability """
                # for i in np.arange(0, len(config.SEED) + 2):
                #     axes["stability"].axhline(y=i, color="black", linewidth=0.5)
                #     axes["stability"].axvline(x=i, color="black", linewidth=0.5)

                stability = pairwise_jaccard(matrix)
                hmap = seaborn.heatmap(
                        stability, ax=axes["stability"], cmap=cmap_2,
                        cbar=False, square=True)
                bbox = hmap.get_window_extent()
                fig_w += bbox.x1 - bbox.x0

                # axes["stability"].yaxis.tick_right()
                axes["stability"].invert_yaxis()
                axes["stability"].margins(x=0, y=0)
                axes["stability"].set_aspect(1)
                axes["stability"].set_title(f"Stability $\\downarrow$")
                # axes["stability"].set_xticks(range(len(ds.label)))
                # axes["stability"].set_xticklabels([f"$L_{{{i}}}$" for i in range(len(ds.label))])
                # axes["stability"].set_yticks([])
                # axes["stability"].set_yticklabels([])
                axes["stability"].set_yticklabels(axes["stability"].get_yticklabels(), rotation=0)

                """ colorbar_2 """
                cbar = fig.colorbar(
                        plt.cm.ScalarMappable(norm=norm_2, cmap=cmap_2),
                        cax=axes["colorbar_2"],
                        orientation="vertical",
                        )
                bbox = cbar.ax.get_window_extent()
                fig_w += bbox.x1 - bbox.x0
                # if args.data == "binned":
                #     cbar.set_ticks([-2, -1, 0, 1, 2])
                #     cbar.set_ticklabels(["-2",  "-1", "0", "1", "2"])
                # axes["colorbar_1"].tick_params(length=0)
                axes["colorbar_2"].set_aspect(len(all_feature) * 4)  # / 0.25

                # print(fig_w, fig_h)
                fig.set_size_inches(fig_w / 100, fig_h / 100)

                # plt.savefig(f"{output_dir}/lbl{label}_samp{sample}.{args.format}", bbox_inches="tight")
                # plt.close()
                plt.show()
                exit()
                """ Plotting }}} """
