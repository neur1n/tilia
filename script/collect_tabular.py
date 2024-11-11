#!/usr/bin/env python3

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

import config
import dataset
import lime
import lime.lime_tabular
import util


dataset = [
        # dataset.Dataset("breast-cancer", "classification"),
        # dataset.Dataset("digits", "classification"),
        dataset.Dataset("iris", "classification"),
        ]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--regressor", default=None, required=False, help="Regressor, linear or tree")
    ap.add_argument("-t", "--timestamp", default=None, required=False, help="Timestamp")
    args = ap.parse_args()

    # NOTE: `None` is for linear.
    if args.regressor == "tree":
        config.REGRESSOR = args.regressor

    timestamp: str
    if args.timestamp:
        timestamp = args.timestamp
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d")

    bunch: sklearn.utils.Bunch
    X: sp.sparse.csr_matrix
    y: np.ndarray
    X_train: sp.sparse.csr_matrix
    y_train: np.ndarray
    X_test: sp.sparse.csr_matrix
    y_test: np.ndarray

    for ds in dataset:
        if ds.name == "breast-cancer":
            bunch = sklearn.datasets.load_breast_cancer()  # type: ignore
            X, y = bunch.data, bunch.target
            ds.label = bunch.target_names
            ds.feature = bunch.feature_names
        elif ds.name == "digits":
            bunch = sklearn.datasets.load_digits()  # type: ignore
            X, y = bunch.data, bunch.target
            ds.label = bunch.target_names
            ds.feature = bunch.feature_names
        elif ds.name == "iris":
            bunch = sklearn.datasets.load_iris()  # type: ignore
            X, y = bunch.data, bunch.target
            ds.label = bunch.target_names
            ds.feature = bunch.feature_names
        else:
            X, y = sklearn.datasets.load_svmlight_file(f"../dataset/{ds.name}")

        n_feature = min(100, X.shape[1])
        max_depth = n_feature if config.REGRESSOR == "tree" else 0

        X_train, X_test, y_train, y_test = \
                sklearn.model_selection.train_test_split(
                        X, y, train_size=0.8, random_state=config.SEED)  # type: ignore

        black_box = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
        black_box.fit(X_train, y_train)

        score = sklearn.metrics.accuracy_score(y_test, black_box.predict(X_test))

        exp = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                mode=ds.task,
                training_labels=y_train,
                feature_names=ds.feature,
                discretize_continuous=True,
                discretizer="quartile",
                class_names=ds.label,
                random_state=config.SEED,
                )

        for idx in range(len(X_test)):
        # for idx in range(2):
            output_dir = f"{config.ROOT_DIR}/output/{timestamp}/{ds.name}/{config.REGRESSOR}"
            os.makedirs(output_dir, exist_ok=True)

            print("\n============================================================")
            print(f"Explaining {idx}/{len(X_test) - 1} {ds.name} (ground truth: {ds.label[y_test[idx]]})...")
            exp_inst = exp.explain_instance(
                    data_row=X_test[idx],
                    predict_fn=black_box.predict_proba,
                    labels=[l for l in range(len(ds.label))],
                    num_features=n_feature,
                    model_regressor=config.REGRESSOR,
                    max_depth=max_depth,
                    )
            exp_inst.save_to_file(f"{output_dir}/lbl{y_test[idx]}_{idx}.html")

            print("\n============================================================")
            print("Binning the importances...")
            df = [pd.DataFrame() for _ in range(len(ds.label))]
            for r in range(config.REPEAT):
                exp_inst = exp.explain_instance(
                        data_row=X_test[idx],
                        predict_fn=black_box.predict_proba,
                        labels=[l for l in range(len(ds.label))],
                        num_features=n_feature,
                        model_regressor=config.REGRESSOR,
                        max_depth=max_depth,
                        )

                for l in range(len(ds.label)):
                    explanation: list[tuple] = exp_inst.as_list(label=l)

                    importance = [e[1] for e in explanation]
                    pos_i = [imp for imp in importance if imp > 0]
                    neg_i = [imp for imp in importance if imp < 0]

                    pos_q = None
                    neg_q = None

                    if len(pos_i) > 0:
                        pos_q = np.percentile(pos_i, [50])

                    if len(neg_i) > 0:
                        neg_q = np.percentile(neg_i, [50])

                    # binned = [(e[0], e[1], util.bin_importance(e[1], pos_q, neg_q)) for e in explanation]
                    # mapping_dict = {f: v for f, v in binned}
                    # updated_mapping_dict = {f: f"{f}:{v}" for f, v in mapping_dict.items()}
                    # new_row = pd.DataFrame([updated_mapping_dict])
                    # df[l] = pd.concat([df[l], new_row], ignore_index=False)

                    # NOTE: <feature>:<importance>:<binned importance>
                    # NOTE: Keeping `feature` for FP-Growth compatibility.
                    formatted = {e[0]: f"{e[0]}:{e[1]}:{util.bin_importance(e[1], pos_q, neg_q)}" for e in explanation}
                    new_row = pd.DataFrame([formatted])
                    df[l] = pd.concat([df[l], new_row], ignore_index=False)

            print("\n============================================================")
            print("Calculating accumulated importance and stability metrics...")
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
                    importance = []
                    binned_importance = []
                    for item in column:
                        imp, bimp = item.split(":")[1:]
                        importance.append(float(imp))
                        binned_importance.append(int(bimp))
                        # binned_importance.append(int(item.split(":")[1]))

                    sum[feature] = np.sum(importance)
                    binned_sum[feature] = np.sum(binned_importance)

                    """ entropy """
                    # cnt = collections.Counter(quantile)
                    # ent[feature] = scipy.stats.entropy(list(cnt.values()), base=2)
                    _, cnt = np.unique(binned_importance, return_counts=True)
                    ent[feature] = scipy.stats.entropy(cnt, base=2)
                    """ aad using mode """
                    # for q in cnt.keys():
                    #     aad[feature] += cnt[q] * abs(q - mode)
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

                file = f"{output_dir}/lbl{y_test[idx]}_exp{l}_idx{idx}.csv"
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
