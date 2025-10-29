#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import collections
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
        dataset.Dataset("phoneme", "classification", openml_id=1489),
        dataset.Dataset("iris", "classification", openml_id=61),
        dataset.Dataset("glass", "classification", openml_id=41),
        dataset.Dataset("diabetes", "classification", openml_id=37),
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

    if args.regressor == "linear":
        args.regressor = None

    if args.timestamp is None:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d")

    bunch: sklearn.utils.Bunch
    X: sp.sparse.csr_matrix
    y: np.ndarray
    X_train: sp.sparse.csr_matrix
    y_train: np.ndarray
    X_test: sp.sparse.csr_matrix
    y_test: np.ndarray

    fidelity = collections.defaultdict(list)

    for d, ds in enumerate(dataset):
        input_dir = f"{config.ROOT_DIR}/output/{args.timestamp}/{ds.name}/{args.regressor}"

        bunch = sklearn.datasets.fetch_openml(data_id=ds.openml_id)  # type: ignore
        X, y = bunch.data.to_numpy(), bunch.target.to_numpy()

        ds.label = np.unique(y).tolist()
        ds.feature = bunch.feature_names

        ds.label = np.unique(y).tolist()

        fidelity[ds.name].append(len(ds.label))
        fidelity[ds.name].append(np.load(f"{input_dir}/fidelity.npy"))
        # print(f"{ds.name}: {np.round(fidelity[ds.name].mean(axis=0), 2)} +- {np.round(fidelity[ds.name].std(axis=0), 2)}\n")

    """====================================================================
    Plotting {{{
    """
    fig = plt.figure(figsize=(6, 8), layout="tight")
    # ttl = fig.suptitle("Fidelity $\\uparrow$")
    # print(ttl.get_window_extent())

    # NOTE: Layout:
    #   iris (3);
    #   glass (6);
    #   phoneme (2), diabetes (2), ionosphere (2);
    #   fri_c4 (2), tecator (2), clean1 (2)
    # NOTE: Extra half column for y-axis labels.
    w_ratio =[1, 1, 1, 1, 1, 1]
    h_ratio = [1, 1, 1, 1]

    gs = fig.add_gridspec(
            nrows=len(h_ratio), ncols=len(w_ratio),
            width_ratios=w_ratio, height_ratios=h_ratio,
            wspace=0.1, hspace=0.1,
            )

    axes = {}
    axes["title"] = fig.add_subplot(gs[0, 3:6])

    axes["iris"] = fig.add_subplot(gs[0, 0:3])

    axes["glass"] = fig.add_subplot(gs[1, :])

    axes["phoneme"] = fig.add_subplot(gs[2, 0:2])
    axes["diabetes"] = fig.add_subplot(gs[2, 2:4])
    axes["ionosphere"] = fig.add_subplot(gs[2, 4:6])

    axes["fri_c4_1000_100"] = fig.add_subplot(gs[3, 0:2])
    axes["tecator"] = fig.add_subplot(gs[3, 2:4])
    axes["clean1"] = fig.add_subplot(gs[3, 4:6])

    for name, ax in axes.items():
        if name == "title":
            ax.margins(x=0, y=0)
            ax.set_aspect(1)
            ax.set_xticks([0, 1, 2])
            ax.set_ylim(0.0, 1.0)
            ax.axis("off")
            t = ax.text(
                2, 1,
                f"Fidelity $\\uparrow$",
                fontsize=24,
                ha="right",
                va="top",
                )
            # print(t.get_window_extent())

        else:
            ax.margins(x=0, y=0)
            ax.set_aspect(1)
            ax.set_title(name if name != "fri_c4_1000_100" else "fri_c4")

            ax.set_xticks(range(fidelity[name][0]))
            ax.set_xticklabels([f"$C_{{{i}}}$" for i in range(fidelity[name][0])])

            ax.set_ylim(0.0, 1.0)
            if name in ["iris", "glass", "phoneme", "fri_c4_1000_100"]:
                ax.set_yticks([0.0, 0.5, 1.0])
            else:
                ax.set_yticks([])

            img = ax.bar(
                range(fidelity[name][0]),
                fidelity[name][1].mean(axis=0),
                yerr=fidelity[name][1].std(axis=0),
                capsize=8,
                capstyle="butt",
                color="#cfcefd",
                hatch="/",
                )

    plt.savefig(f"./{config.ROOT_DIR}/output/{args.timestamp}/fidelity_{args.regressor}.png", bbox_inches='tight')
    plt.close()
    # plt.show()
