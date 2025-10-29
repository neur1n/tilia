#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import datetime
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.tree
import sklearn.utils
import tqdm

import config
import dataset
import MAPLE.MAPLE
import MAPLE.Misc
import util


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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--topk", default=5, type=int, required=False, help="Top-k features.")
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

    print(f"Name | OpenML ID | Classes | Features | Instances | Score")
    for d, ds in enumerate(dataset):
        output_dir = f"{config.ROOT_DIR}/output/{args.timestamp}/{ds.name}/{args.regressor}"
        os.makedirs(output_dir, exist_ok=True)

        bunch = sklearn.datasets.fetch_openml(data_id=ds.openml_id)  # type: ignore
        X, y = bunch.data.to_numpy(), bunch.target.to_numpy()

        original_y = y
        original_label = np.unique(original_y).tolist()

        label_encoder = sklearn.preprocessing.LabelEncoder()
        y = label_encoder.fit_transform(y)

        ds.label = np.unique(y).tolist()
        ds.feature = bunch.feature_names

        n_feature = X.shape[1]
        max_depth = "adaptive" if args.regressor is not None else 0

        X_train, X_test, y_train, y_test = \
                sklearn.model_selection.train_test_split(
                        X, y, train_size=0.8, random_state=42)  # type: ignore

        opaque_model = sklearn.ensemble.RandomForestClassifier()
        opaque_model.fit(X_train, y_train)
        print(f"{ds.name} & {len(ds.label)} & {X.shape[1]} & {X.shape[0]} & {opaque_model.score(X_test, y_test):.2f}\n")

        if args.sample <= 0:
            args.sample = len(X_test)
        else:
            args.sample = min(args.sample, len(X_test))

        for idx in tqdm.tqdm(range(args.sample), desc="Sample"):
            df = [pd.DataFrame() for _ in range(len(ds.label))]

            for seed in tqdm.tqdm(config.SEED, desc="Seed", leave=False):
                explainer = MAPLE.MAPLE.MAPLE(
                        X_train,
                        opaque_model.predict(X_train),
                        X_train,
                        opaque_model.predict(X_train),
                        random_state=seed)

                exp_inst = explainer.explain(X_test[idx])

                if True:
                    for l in range(len(ds.label)):
                        importance: list = exp_inst["coefs"][1:]
                        pos_i = [imp for imp in importance if imp > 0]
                        neg_i = [imp for imp in importance if imp < 0]

                        pos_q = None
                        neg_q = None

                        if len(pos_i) > 0:
                            pos_q = np.quantile(pos_i, 0.5)

                        if len(neg_i) > 0:
                            neg_q = np.quantile(neg_i, 0.5)

                        # NOTE: <feature>:<importance>:<binned importance>
                        # NOTE: Keeping `feature` for FP-Growth compatibility.
                        formatted = {f: f"{f}:{i}:{util.bin_importance(i, pos_q, neg_q)}" for f, i in zip(ds.feature, importance)}
                        new_row = pd.DataFrame([formatted], index=[seed])
                        df[l] = pd.concat([df[l], new_row], ignore_index=False)

            for l in range(len(ds.label)):
                mode = {}
                sum = {}
                binned_sum = {}
                ent = {}  # entropy
                aad = {}  # average absolute deviation
                std = {}  # standard deviation

                for feature, column in df[l].items():
                    column.dropna(inplace=True)

                    importance = []
                    binned_importance = []
                    for item in column:
                        imp, bimp = item.split(":")[1:]
                        importance.append(float(imp))
                        binned_importance.append(int(bimp))

                    sum[feature] = np.sum(importance)
                    binned_sum[feature] = np.sum(binned_importance)

                    """ entropy """
                    _, cnt = np.unique(binned_importance, return_counts=True)
                    ent[feature] = scipy.stats.entropy(cnt, base=2)
                    """ aad using mode """
                    mode[feature] = sp.stats.mode(binned_importance).mode
                    aad[feature] = np.mean(np.abs(np.array(binned_importance) - mode[feature]))
                    """ std """
                    std[feature] = np.std(importance)

                df_mode = pd.DataFrame(mode, index=["mode"])
                df_sum = pd.DataFrame(sum, index=["sum"])
                df_binned_sum = pd.DataFrame(binned_sum, index=["binned_sum"])
                df_ent = pd.DataFrame(ent, index=["ent"])
                df_aad = pd.DataFrame(aad, index=["aad"])
                df_var = pd.DataFrame(std, index=["var"])
                df[l] = pd.concat([df[l], df_mode, df_sum, df_binned_sum, df_ent, df_aad, df_var])

                file = f"{output_dir}/lbl{original_label[y_test[idx]]}_samp{idx}_exp{original_label[l]}.csv"
                df[l].to_csv(file, sep=config.DELIMITER, index=True)
