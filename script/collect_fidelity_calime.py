#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import datetime
import networkx as nx
import numpy as np
import pandas as pd
import pickle
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
import CALIME.calime_explainer
import CALIME.causal_model


dataset = [
        dataset.Dataset("iris", "classification", openml_id=61),
        dataset.Dataset("glass", "classification", openml_id=41),
        dataset.Dataset("ionosphere", "classification", openml_id=59),
        dataset.Dataset("fri_c4_1000_100", "classification", openml_id=718),
        dataset.Dataset("tecator", "classification", openml_id=851),
        dataset.Dataset("clean1", "classification", openml_id=40665),
        dataset.Dataset("cnae-9", "classification", openml_id=1468),
        ]


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

    print(f"Name | OpenML ID | Classes | Features | Instances | Score")
    for d, ds in enumerate(dataset):
        output_dir = f"{config.ROOT_DIR}/output/{args.timestamp}/{ds.name}/{args.regressor}"
        os.makedirs(output_dir, exist_ok=True)

        bunch = sklearn.datasets.fetch_openml(data_id=ds.openml_id)  # type: ignore
        X, y = bunch.data.to_numpy(), bunch.target.to_numpy()

        ds.label = np.unique(y).tolist()
        ds.feature = bunch.feature_names

        n_feature = X.shape[1]
        max_depth = "adaptive" if args.regressor is not None else 0

        X_train, X_test, y_train, y_test = \
                sklearn.model_selection.train_test_split(
                        X, y, train_size=0.8, random_state=42)  # type: ignore

        best_opaque: sklearn.ensemble.RandomForestClassifier
        last_score = 0.0
        score = np.zeros((10), dtype=float)
        for i in range(10):
            score[i]
            opaque_model = sklearn.ensemble.RandomForestClassifier()
            opaque_model.fit(X_train, y_train)
            score[i] = opaque_model.score(X_test, y_test)
            if score[i] > last_score:
                best_opaque = opaque_model
                last_score = score[i]

        print(f"{ds.name} & {ds.openml_id} & {len(ds.label)} & {X.shape[1]} & {X.shape[0]} & {score.mean():.2f}\n")

        graph_path = os.path.join("./", "ground_truth.gpickle")
        generation_path = os.path.join("./", "generative_model.pkl")
        if os.path.exists(graph_path) and os.path.exists(generation_path):
            with open(graph_path, "rb") as f:
                graph = pickle.load(f)
            nx.draw_networkx(graph)
            with open(generation_path, "rb") as file:
                generative_model = pickle.load(file)
        else:
            generative_model, graph = CALIME.causal_model.get_causal_model(pd.DataFrame(X_test), np.array(range(len(ds.feature))).astype(str), "./")

        if args.sample <= 0:
            args.sample = len(X_test)
        else:
            args.sample = min(args.sample, len(X_test))

        fidelity = np.zeros((args.sample * len(config.SEED), len(ds.label)), dtype=float)
        for i in tqdm.tqdm(range(args.sample), desc="Sample"):
            for j, seed in enumerate(tqdm.tqdm(config.SEED, desc="Seed", leave=False)):
                calime_explainer = CALIME.calime_explainer.CALimeExplainer(
                        graph,
                        generative_model,
                        X_train,
                        feature_names=ds.feature,
                        class_names=ds.label,
                        discretize_continuous=True)
                exp_inst, calime_data, calime_neighbor_gen_time = \
                        calime_explainer.explain_instance(
                                data_row=X_test[i],
                                predict_fn=best_opaque.predict_proba,
                                labels=[l for l in range(len(ds.label))],
                                num_features=n_feature)

                fidelity[i * len(config.SEED) + j] = list(exp_inst.score.values())

        np.save(f"{output_dir}/fidelity.npy", fidelity)
