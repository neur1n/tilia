#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.ml
import pyspark.sql
import pyspark.sql.functions
import scipy as sp
import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.svm
import sklearn.tree
import sklearn.utils
import typing

from dataset import Dataset
import lime
import lime.lime_tabular


def map_value(value, positive_quantile, negative_quantile):
    if value > 0:
        if value <= positive_quantile[0]:
            return "p1"
        else:
            return "p2"
    elif value < 0:
        if value <= negative_quantile[0]:
            return "n1"
        else:
            return "n2"
    else:
        return "ne"


def sort_explanation(explanation: typing.List[typing.Tuple[str, float]]):
    return sorted(explanation, key=lambda x: x[1], reverse=True)

seed = 42

# NOTE: Results from linear models are simliar.
regressor = sklearn.linear_model.Ridge(random_state=42)  # The default one.
# regressor = sklearn.linear_model.BayesianRidge()
# regressor = sklearn.linear_model.LinearRegression()
# regressor = sklearn.linear_model.SGDRegressor()
# regressor = sklearn.linear_model.TweedieRegressor(power=1)  # NOTE: Only 0 or 1 works.
# regressor = sklearn.svm.LinearSVR()

# NOTE: Results from decision trees are all positive values.
# TODO: I think this is where we need to tackle.
# regressor = sklearn.ensemble.AdaBoostRegressor()
# regressor = sklearn.ensemble.GradientBoostingRegressor()
regressor = sklearn.tree.DecisionTreeRegressor(max_depth=4, random_state=42)
# regressor = sklearn.tree.ExtraTreeRegressor()

# NOTE (2024-10-27): Not working yet.
# regressor = sklearn.linear_model.ARDRegression()
# regressor = sklearn.linear_model.ElasticNet()
# regressor = sklearn.linear_model.Lars()
# regressor = sklearn.linear_model.Lasso()
# regressor = sklearn.linear_model.LassoLars()
# regressor = sklearn.linear_model.OrthogonalMatchingPursuit()
# regressor = sklearn.linear_model.PassiveAggressiveRegressor()
# regressor = sklearn.linear_model.Perceptron()
# regressor = sklearn.svm.SVR()

dataset = [
        Dataset("iris", "classification"),
        # Dataset("a1a", "classification", ["-1", "+1"], False),
        # {"name": "a1a", "categorical": True},
        # {"name": "a9a", "categorical": True},
        # {"name": "breast-cancer", "categorical": True},
        # {"name": "w8a", "categorical": True},
        ]

repeat = 20
delimiter = ","


if __name__ == '__main__':
    pd.set_option('display.max_colwidth', None)

    X: sp.sparse.csr_matrix
    y: np.ndarray
    X_train: sp.sparse.csr_matrix
    y_train: np.ndarray
    X_test: sp.sparse.csr_matrix
    y_test: np.ndarray

    for d in dataset:
        """
        Explain
        """
        if d.name == "iris":
            bunch: sklearn.utils.Bunch = sklearn.datasets.load_iris()
            X, y = bunch.data, bunch.target
            d.label = bunch.target_names
            d.feature = bunch.feature_names
        else:
            X, y = sklearn.datasets.load_svmlight_file(f"../dataset/{d.name}")

        num_used_feature = min(10, X.shape[1])

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.8, random_state=seed)

        black_box = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
        black_box.fit(X_train, y_train)

        score = sklearn.metrics.accuracy_score(y_test, black_box.predict(X_test))

        explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                mode=d.task,
                training_labels=y_train,
                feature_names=d.feature,
                discretize_continuous=True,
                discretizer="quartile",
                class_names=d.label,
                random_state=seed,
                )

        # rng = np.random.default_rng(seed)
        # idx = rng.integers(0, X_test.shape[0])

        for idx in range(len(X_test)):
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")

            print("============================================================")
            print(f"Explaining {d.name} {idx}th instance of test set (ground truth: {d.label[y_test[idx]]})...")
            print("============================================================")

            exp_inst = explainer.explain_instance(
                    data_row=X_test[idx],
                    predict_fn=black_box.predict_proba,
                    labels=[l for l in range(len(d.label))],
                    num_features=num_used_feature,
                    model_regressor=regressor,
                    )
            exp_inst.save_to_file(f"../output/{d.name}_{y_test[idx]}_{type(regressor).__name__}_{timestamp}.html")

            """
            Find frequency pattern
            """
            df = [pd.DataFrame() for _ in range(len(d.label))]
            for r in range(repeat):
                exp_inst = explainer.explain_instance(
                        data_row=X_test[idx],
                        predict_fn=black_box.predict_proba,
                        labels=[l for l in range(len(d.label))],
                        num_features=num_used_feature,
                        model_regressor=regressor,
                        )

                for l in range(len(d.label)):
                    data = exp_inst.as_list(label=l)

                    value = [item[1] for item in data]
                    pos_v = [v for v in value if v > 0]
                    neg_v = [v for v in value if v < 0]

                    pos_q = None
                    neg_q = None

                    if len(pos_v) > 0:
                        pos_q = np.quantile(pos_v, 0.5)
                        # pos_q = np.percentile(pos_v, [50])

                    if len(neg_v) > 0:
                        neg_q = np.quantile(neg_v, 0.5)
                        # neg_q = np.percentile(neg_v, [50])

                    mapped_data = [(item[0], map_value(item[1], pos_v, neg_v)) for item in data]
                    mapping_dict = {f: v for f, v in mapped_data}
                    updated_mapping_dict = {f: f"{f}:{v}" for f, v in mapping_dict.items()}
                    new_row = pd.DataFrame([updated_mapping_dict])
                    df[l] = pd.concat([df[l], new_row], ignore_index=True)

            for l in range(len(d.label)):
                file = f"../output/{d.name}_{y_test[idx]}_{l}_{type(regressor).__name__}_mapped_{timestamp}.csv"
                df[l].to_csv(file, sep=delimiter, index=False)

                spark = pyspark.sql.SparkSession.builder.appName("LIME").getOrCreate()
                data = (spark.read.text(file).select(pyspark.sql.functions.split("value", delimiter).alias("items")))

                fp = pyspark.ml.fpm.FPGrowth(minSupport=0.2, minConfidence=0.3)
                fp_model = fp.fit(data)
                fp_model.setPredictionCol("prediction")
                # df = fp_model.associationRules.sort("antecedent", "consequent")
                spark_df = fp_model.associationRules.sort("support", ascending=False)

                file = f"../output/{d.name}_{y_test[idx]}_{l}_{type(regressor).__name__}_fp_{timestamp}.csv"
                spark_df.toPandas().head(10).to_csv(file, sep=delimiter)
