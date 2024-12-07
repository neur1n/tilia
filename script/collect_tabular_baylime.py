#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

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
import sklearn.tree
import sklearn.utils
import tqdm

import config
import dataset
import util
import BayLIME
import BayLIME.lime_tabular


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
        # for idx in tqdm.tqdm(range(2)):
            df = [pd.DataFrame() for _ in range(len(ds.label))]

            for seed in tqdm.tqdm(config.SEED, desc="Seed", leave=False):
                explainer = BayLIME.lime_tabular.LimeTabularExplainer(
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
                        model_regressor="Bay_non_info_prior")
                # exp_inst.save_to_file(f"{output_dir}/lbl{y_test[idx]}_samp{idx}_seed{seed}.html")

                # if args.regressor is None:
                if True:
                    for l in range(len(ds.label)):
                        explanation: list[tuple] = exp_inst.as_list(label=l)

                        importance = [e[1] for e in explanation]
                        pos_i = [imp for imp in importance if imp > 0]
                        neg_i = [imp for imp in importance if imp < 0]

                        pos_q = None
                        neg_q = None

                        if len(pos_i) > 0:
                            # pos_q = np.percentile(pos_i, [50])
                            pos_q = np.quantile(pos_i, 0.5)

                        if len(neg_i) > 0:
                            # neg_q = np.percentile(neg_i, [50])
                            neg_q = np.quantile(neg_i, 0.5)

                        # NOTE: <feature>:<importance>:<binned importance>
                        # NOTE: Keeping `feature` for FP-Growth compatibility.
                        formatted = {e[0]: f"{e[0]}:{e[1]}:{util.bin_importance(e[1], pos_q, neg_q)}" for e in explanation}
                        new_row = pd.DataFrame([formatted], index=[seed])
                        df[l] = pd.concat([df[l], new_row], ignore_index=False)
                else:  # NOTE: For mean-zero-reference approach
                    explanation = [None for _ in range(len(ds.label))]  # type: ignore
                    importance = np.zeros((len(ds.label), len(ds.feature)))

                    for l in range(len(ds.label)):
                        explanation[l] = exp_inst.as_list(label=l)
                        for c, e in enumerate(explanation[l]):
                            importance[l, c] = e[1]
                    zero_ref = np.mean(importance, axis=0)
                    importance -= zero_ref

                    for l in range(len(ds.label)):
                        pos_i = [imp for imp in importance[l] if imp > 0]
                        neg_i = [imp for imp in importance[l] if imp < 0]

                        pos_q = None
                        neg_q = None

                        if len(pos_i) > 0:
                            pos_q = np.percentile(pos_i, [50])

                        if len(neg_i) > 0:
                            neg_q = np.percentile(neg_i, [50])

                        formatted = {}
                        for c, (f, _) in enumerate(explanation[l]):
                            formatted[f] = f"{f}:{importance[l, c]}:{util.bin_importance(importance[l, c], pos_q, neg_q)}"
                        new_row = pd.DataFrame([formatted], index=[seed])
                        df[l] = pd.concat([df[l], new_row], ignore_index=False)

            """
            NOTE: The feature importance used by linear models is different
                  from the one used by tree-based models. For linear models,
                  the importance is the coefficient of the feature. For tree-
                  based models, the importance is the gini importance.
                  Therefore, using raw values for evaluating the stability is
                  not appropriate. Instead, we use binned importances.
            """
            for l in range(len(ds.label)):
                mode = {}
                sum = {}
                binned_sum = {}
                ent = {}  # entropy
                aad = {}  # average absolute deviation
                var = {}  # variance

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
                    """ variance """
                    var[feature] = np.var(binned_importance)

                df_mode = pd.DataFrame(mode, index=["mode"])
                df_sum = pd.DataFrame(sum, index=["sum"])
                df_binned_sum = pd.DataFrame(binned_sum, index=["binned_sum"])
                df_ent = pd.DataFrame(ent, index=["ent"])
                df_aad = pd.DataFrame(aad, index=["aad"])
                df_var = pd.DataFrame(var, index=["var"])
                df[l] = pd.concat([df[l], df_mode, df_sum, df_binned_sum, df_ent, df_aad, df_var])

                file = f"{output_dir}/lbl{y_test[idx]}_samp{idx}_exp{ds.label[l]}.csv"
                df[l].to_csv(file, sep=config.DELIMITER, index=True)

            #     spark = pyspark.sql.SparkSession.builder.appName("LIME").getOrCreate()
            #     data = (spark.read.text(file).select(pyspark.sql.functions.split("value", delimiter).alias("items")))

            #     fp = pyspark.ml.fpm.FPGrowth(minSupport=0.3, minConfidence=0.8)
            #     fp_model = fp.fit(data)
            #     fp_model.setPredictionCol("prediction")
            #     # df = fp_model.associationRules.sort("antecedent", "consequent")
            #     spark_df = fp_model.associationRules.sort("support", ascending=False)

            #     file = f"../output/{ds.name}_lbl{y_test[idx]}_exp{l}_{type(regressor).__name__}_fp_{timestamp}.csv"
            #     spark_df.toPandas().head(10).to_csv(file, sep=delimiter)
