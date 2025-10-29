#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import datetime
import glob

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.utils

import config
import dataset


dataset = [
        # dataset.Dataset("phoneme", "classification", openml_id=1489),
        # dataset.Dataset("iris", "classification", openml_id=61),
        # dataset.Dataset("glass", "classification", openml_id=41),
        dataset.Dataset("diabetes", "classification", openml_id=37),
        # dataset.Dataset("ionosphere", "classification", openml_id=59),
        # dataset.Dataset("fri_c4_1000_100", "classification", openml_id=718),
        # dataset.Dataset("tecator", "classification", openml_id=851),
        # dataset.Dataset("clean1", "classification", openml_id=40665),
        ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="default", required=False, help="Data used for plotting.")
    ap.add_argument("-f", "--format", default="png", required=False, help="Format of the output figure, png or pdf.")
    ap.add_argument("-m", "--mode", default="discretized", required=False, help="discretized or original")
    ap.add_argument("-o", "--orientation", default="h", required=False, help="Orientation of the plot, h or v.")
    ap.add_argument("-p", "--plot", default=1, type=int, required=False, help="Plot or not.")
    ap.add_argument("-r", "--regressor", default=None, required=False, help="Regressor, linear or tree.")
    ap.add_argument("-s", "--sample", default=-1, type=int, required=False, help="Number of samples to visualize.")
    ap.add_argument("-t", "--timestamp", default=None, required=False, help="Timestamp")
    args = ap.parse_args()

    if args.regressor == "linear":
        args.regressor = None

    if args.timestamp is None:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d")

    np.set_printoptions(suppress=True, linewidth=200)

    bunch: sklearn.utils.Bunch

    for ds in dataset:
        print(f"Aggregating {ds.name}...")

        bunch = sklearn.datasets.fetch_openml(data_id=ds.openml_id)  # type: ignore
        X, y = bunch.data.to_numpy(), bunch.target.to_numpy()
        ds.label = np.unique(y).tolist()
        ds.feature = bunch.feature_names

        input_dir_pattern = f"{config.ROOT_DIR}/output/{args.timestamp}*/{ds.name}/{args.regressor}"
        input_dir = glob.glob(input_dir_pattern)

        n_feature = 0

        for label_gt in ds.label:
            all_input = []
            for dir in input_dir:
                file = f"{dir}/stability_{label_gt}.csv"
                input = np.loadtxt(file, delimiter=config.DELIMITER)
                n_feature = input.shape[1]
                all_input.append(input)
            all_input = np.array(all_input)

            if args.mode == "original":
                n_feature = len(ds.feature)

                i, j, k = all_input.shape
                all_input = all_input.reshape(i, j, k // 4, 4)
                all_input = np.nanmean(all_input, axis=-1)

            grouped_input = {}
            # NOTE: Quadtiled features
            for f in range(n_feature):
                grouped_input[f"f{f}"] = [all_input[:, l, f].flatten() for l in range(len(ds.label))]

            positions = []
            box_data = []
            group_width = 0.8
            num_labels = len(ds.label)
            offset = group_width / num_labels

            for f, (feature, stddev) in enumerate(grouped_input.items()):
                for label_idx, label_data in enumerate(stddev):
                    box_data.append(label_data)
                    positions.append(f + label_idx * offset)

            avg = np.nanmean(box_data, axis=1)
            std = np.nanstd(box_data, axis=1)
            print(f"\n>>>>>>>>>>>>>>>>>>>> avg of {label_gt}")
            for l in range(len(ds.label)):
                for f in range(n_feature):
                    print(f"{avg[l * n_feature + f]:.5f}+-{std[l * n_feature + f]:.5f}", end=" ")
                print()

            if args.plot:
                fig, ax = plt.subplots(figsize=(20, 6))

                ax.boxplot(box_data, positions=positions, widths=offset * 0.8)

                # ax.set_ylim(0, 0.5)
                ax.set_title(f"{ds.name} (Label {label_gt})")

                plt.xticks(
                    [i + group_width / 2 - offset / 2 for i in range(n_feature)],
                    [f"Feature {i+1}" for i in range(n_feature)],
                )

                plt.show()
