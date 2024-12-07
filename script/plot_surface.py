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
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.ensemble
import sklearn.inspection
import sklearn.model_selection
import sklearn.utils
import tqdm

import config
import dataset
import lime
import lime.lime_tabular


dataset = [
        dataset.Dataset("iris", "classification", openml_id=61),
        ]


# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02


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

        bunch = sklearn.datasets.fetch_openml(data_id=ds.openml_id)  # type: ignore
        X, y = bunch.data.to_numpy(), bunch.target.to_numpy()
        ds.label = np.unique(y).tolist()
        ds.feature = bunch.feature_names

        X_train, X_test, y_train, y_test = \
                sklearn.model_selection.train_test_split(
                        X, y, train_size=0.8, random_state=42)  # type: ignore

        opaque_model = sklearn.ensemble.RandomForestClassifier()
        opaque_model.fit(X_train, y_train)

        for seed in tqdm.tqdm(config.SEED, desc="Seed", leave=False):
            explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X_train,
                    mode=ds.task,
                    training_labels=y_train,
                    feature_names=ds.feature,
                    discretize_continuous=False,
                    discretizer="quartile",
                    class_names=ds.label,
                    random_state=seed)

            exp_inst, surrogate_model, X_perturbed, y_perturbed = explainer.explain_instance(
                    data_row=X_test[4],
                    predict_fn=opaque_model.predict_proba,
                    labels=[l for l in range(len(ds.label))],
                    num_features=len(ds.feature),
                    model_regressor=args.regressor,
                    return_surrogate=True)


            for l in range(len(ds.label)):
                for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
                    surrogate_model[l].fit(X_perturbed[l][:, pair], y_perturbed[l])

                    ax = plt.subplot(2, 3, pairidx + 1)
                    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
                    sklearn.inspection.DecisionBoundaryDisplay.from_estimator(
                            surrogate_model[l],
                            X_perturbed[l][:, pair],
                            cmap=plt.cm.RdYlBu,
                            response_method="predict",
                            ax=ax,
                            xlabel=ds.feature[pair[0]],
                            ylabel=ds.feature[pair[1]],
                            )

                    # Plot the training points
                    for i, color in zip(range(n_classes), plot_colors):
                        idx = np.where(y == i)
                        plt.scatter(
                            X_perturbed[l][idx, 0],
                            X_perturbed[l][idx, 1],
                            c=color,
                            label=ds.label[i],
                            edgecolor="black",
                            s=15,
                        )

                plt.suptitle("Decision surface of decision trees trained on pairs of features")
                plt.legend(loc="lower right", borderpad=0, handletextpad=0)
                _ = plt.axis("tight")
                plt.show()
