#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import collections
import datetime
import numpy as np
import pandas as pd
import scipy as sp
import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.tree
import sklearn.utils
import tqdm

import config
import dataset
import lime
import lime.lime_tabular


dataset = [
        # dataset.Dataset("diabetes", "classification", openml_id=37),
        # dataset.Dataset("glass", "classification", openml_id=41),
        dataset.Dataset("iris", "classification", openml_id=61),
        # dataset.Dataset("skin-segmentation", "classification", openml_id=1502),
        ]


def jaccard(x, y):
    set_x = set(x)
    set_y = set(y)
    return 1 - len(set_x.intersection(set_y)) / len(set_x.union(set_y))

def custom_distance(X, Y=None):
    """
    Wrapper function for pairwise distances with the custom Jaccard metric.
    """
    # Ensure Y is set to X if no comparison array is provided
    if Y is None:
        Y = X

    # Compute the pairwise distances
    distances = np.zeros((len(X), len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            distances[i, j] = jaccard(x, y)
    return distances


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--topk", default=5, type=int, required=False, help="Top-k features.")
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

    for ds in dataset:
        output_dir = f"{config.ROOT_DIR}/output/{args.timestamp}/{ds.name}/{args.regressor}"
        os.makedirs(output_dir, exist_ok=True)

        bunch = sklearn.datasets.fetch_openml(data_id=ds.openml_id)  # type: ignore
        X, y = bunch.data.to_numpy(), bunch.target.to_numpy()

        # NOTE: The labels of `liver-disorders` are numeric.
        if np.issubdtype(X.dtype, np.number):
            y = y.astype(str)

        # NOTE: There is only one instance with the label "15.0" in `liver-disorders`.
        if ds.name == "liver-disorders":
            outlier_idx = np.where(y == "15.0")
            X = np.delete(X, outlier_idx, axis=0)
            y = np.delete(y, outlier_idx, axis=0)

        ds.label = np.unique(y).tolist()
        ds.feature = bunch.feature_names

        n_feature = X.shape[1]
        max_depth = "adaptive" if args.regressor is not None else 0

        X_train, X_test, y_train, y_test = \
                sklearn.model_selection.train_test_split(
                        X, y, train_size=0.8, random_state=42)  # type: ignore

        opaque_model = sklearn.ensemble.RandomForestClassifier()
        opaque_model.fit(X_train, y_train)
        # print(f"Name | Classes | Features | Instances | Score")
        print(f"{ds.name} & {len(ds.label)} & {X.shape[1]} & {X.shape[0]} & {opaque_model.score(X_test, y_test):.2f}\n")

        if args.sample <= 0:
            args.sample = len(X_test)
        else:
            args.sample = min(args.sample, len(X_test))

        for idx in tqdm.tqdm(range(args.sample), desc="Sample"):
            print(f"\n >>>>>>>>>>>>>>>>>>>> idx: {idx} - {y_test[idx]}")
        # for idx in tqdm.tqdm(range(2)):
            df = [pd.DataFrame() for _ in range(len(ds.label))]

            feature = collections.defaultdict(list)
            for seed in tqdm.tqdm(config.SEED, desc="Seed", leave=False):
                explainer = lime.lime_tabular.LimeTabularExplainer(
                        training_data=X_train,
                        mode=ds.task,
                        training_labels=y_train,
                        feature_names=ds.feature,
                        discretize_continuous=True,
                        discretizer="quartile",
                        class_names=ds.label,
                        random_state=seed)

                exp_inst = explainer.explain_instance(
                        data_row=X_test[idx],
                        predict_fn=opaque_model.predict_proba,
                        labels=[l for l in range(len(ds.label))],
                        num_features=n_feature,
                        model_regressor=args.regressor,
                        max_depth=max_depth)
                # exp_inst.save_to_file(f"{output_dir}/lbl{y_test[idx]}_samp{idx}_seed{seed}.html")

                print(f"\nseed: {seed}")
                for l, label in enumerate(ds.label):
                    explanation: list[tuple] = exp_inst.as_list(label=l)
                    print(f"{l}: {sorted([f for f, v in explanation])}")
                    # feature[label].append([f for f, v in explanation])

            # for l, f in feature.items():
            #     distance = sklearn.metrics.pairwise_distances(np.array(f), metric=jaccard)
            #     if np.any(distance):
            #         print(distance)
