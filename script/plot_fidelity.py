#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.tree
import sklearn.utils

import config
import dataset


dataset = [
        dataset.Dataset("iris", "classification", openml_id=61),
        dataset.Dataset("glass", "classification", openml_id=41),
        dataset.Dataset("ionosphere", "classification", openml_id=59),
        dataset.Dataset("fri_c4_1000_100", "classification", openml_id=718),
        dataset.Dataset("tecator", "classification", openml_id=851),
        dataset.Dataset("clean1", "classification", openml_id=40665),
        ]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--topk", default=5, type=int, required=False, help="Top-k features.")
    ap.add_argument("-p", "--plot", default=1, type=int, required=False, help="Plot, or not.")
    ap.add_argument("-r", "--regressor", default=None, type=str, required=False, help="Regressor, linear, dt, gbdt, or rf.")
    ap.add_argument("-s", "--sample", default=-1, type=int, required=False, help="Number of samples to explain.")
    ap.add_argument("-t", "--timestamp", default=None, type=str, required=False, help="Timestamp.")
    args = ap.parse_args()

    if args.timestamp is None:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d")

    bunch: sklearn.utils.Bunch
    X: sp.sparse.csr_matrix
    y: np.ndarray
    X_train: sp.sparse.csr_matrix
    y_train: np.ndarray
    X_test: sp.sparse.csr_matrix
    y_test: np.ndarray

    if args.plot:
        fig, axes = plt.subplots(nrows=1, ncols=len(dataset), figsize=(12, 3))

    for d, ds in enumerate(dataset):
        input_dir = f"{config.ROOT_DIR}/output/{args.timestamp}/{ds.name}/{args.regressor}"

        bunch = sklearn.datasets.fetch_openml(data_id=ds.openml_id)  # type: ignore
        X, y = bunch.data.to_numpy(), bunch.target.to_numpy()

        ds.label = np.unique(y).tolist()
        ds.feature = bunch.feature_names

        ds.label = np.unique(y).tolist()

        fidelity = np.load(f"{input_dir}/fidelity.npy")
        print(f"{ds.name}: {np.round(fidelity.mean(axis=0), 2)} +- {np.round(fidelity.std(axis=0), 2)}\n")

        if args.plot:
            axes[d].boxplot(fidelity, showmeans=True)

            axes[d].set_title(ds.name)

            # axes[d].set_xticks(list(range(len(ds.label))))
            # axes[d].set_xticklabels(ds.label, rotation=45, ha="right")
            axes[d].set_xticks(range(1, len(ds.label) + 1))
            axes[d].set_xticklabels([f"$c_{{{i}}}$" for i in range(len(ds.label))])

            axes[d].set_ylim(0, 1)
            axes[d].yaxis.grid(True)

    if args.plot:
        plt.tight_layout()
        plt.show()
