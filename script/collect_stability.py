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

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.datasets
import sklearn.utils

import config
import dataset
import util


dataset = [
        dataset.Dataset("credit", "classification", openml_id=46543),
        # dataset.Dataset("iris", "classification", openml_id=61),
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
    ap.add_argument("-d", "--data", default="binned", required=False, help="Data used for stability, raw or binned.")
    ap.add_argument("-f", "--format", default="png", required=False, help="Format of the output figure, png or pdf.")
    ap.add_argument("-o", "--orientation", default="h", required=False, help="Orientation of the plot, h or v.")
    ap.add_argument("-r", "--regressor", default=None, required=False, help="Regressor, linear or tree.")
    ap.add_argument("-s", "--sample", default=-1, type=int, required=False, help="Number of samples to visualize.")
    ap.add_argument("-t", "--timestamp", default=None, required=False, help="Timestamp")
    args = ap.parse_args()

    if args.regressor == "linear":
        args.regressor = None

    if args.timestamp is None:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d")

    np.set_printoptions(suppress=False)

    bunch: sklearn.utils.Bunch

    for ds in dataset:
        print(f"Aggregating {ds.name}...")

        input_dir = f"{config.ROOT_DIR}/output/{args.timestamp}/{ds.name}/{args.regressor}"

        bunch = sklearn.datasets.fetch_openml(data_id=ds.openml_id)  # type: ignore
        X, y = bunch.data.to_numpy(), bunch.target.to_numpy()
        ds.label = np.unique(y).tolist()
        ds.feature = bunch.feature_names

        all_feature = set()
        all_sample = collections.defaultdict(set)

        for label_gt in ds.label:
            sample_file: list = glob.glob(f"{input_dir}/lbl{label_gt}*.csv")
            for file in sample_file:
                input: pd.DataFrame = pd.read_csv(file, sep=config.DELIMITER, index_col=0)
                all_feature.update(input.columns)

                match = re.search(r"samp(\d+)", file)
                assert match, f"Cannot find sample ID in {file}"
                sample_id = int(match.group(1))
                all_sample[label_gt].add(sample_id)

        all_feature = sorted(list(all_feature), key=util.feature_key)

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

        for label_gt, sample_list in all_sample.items():
            importance = {
                    label: np.zeros((len(config.SEED) * len(all_sample[label_gt]), len(all_feature)), dtype=float)
                    for label in ds.label
                    }

            for chunk, sample in enumerate(sample_list):
                key_type = "index" if len(np.unique(config.SEED)) == 1 else "seed"

                for label_exp in ds.label:
                    file = f"{input_dir}/lbl{label_gt}_samp{sample}_exp{label_exp}.csv"
                    input = pd.read_csv(file, sep=config.DELIMITER, index_col=0)

                    for r, seed in enumerate(config.SEED):
                        row = input.loc[f"{seed}"] if key_type == "seed" else input.iloc[r]
                        for c, feat in enumerate(all_feature):
                            if feat not in input.columns:
                                continue

                            # NOTE: 0 is the feature name, 1 is the importance value, 2 is the binned importance value
                            importance[label_exp][chunk * len(config.SEED) + r, c] = row.loc[feat].split(":")[2]

            if os.path.isfile(f"{input_dir}/stability_{label_gt}.csv"):
                os.remove(f"{input_dir}/stability_{label_gt}.csv")
            with open(f"{input_dir}/stability_{label_gt}.csv", "a") as f:
                for label, imp in importance.items():
                    stability: np.ndarray
                    if args.data == "raw":
                        imp = util.normalize_positive_negative(np.where(np.isclose(imp, 0), np.nan, imp))
                        stability = np.nanstd(imp, axis=0)
                    else:
                        stability = np.zeros(len(all_feature), dtype=float)
                        for i in range(len(all_feature)):
                            column = imp[:, i]
                            unique, count = np.unique(column, return_counts=True)
                            prob_dict = dict(zip(unique, count))
                            prob = np.array([prob_dict.get(level, 0) / len(column) for level in [-2, -1, 0, 1, 2]])
                            stability[i] = scipy.stats.entropy(prob, base=2)

                    np.savetxt(f, stability.reshape(1, -1), delimiter=config.DELIMITER, fmt="%.4f")
