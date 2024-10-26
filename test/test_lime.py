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
import sklearn.metrics
import sklearn.model_selection
import sklearn.utils

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


seed = 42

dataset = [
        Dataset("iris", "classification"),
        # Dataset("a1a", "classification", ["-1", "+1"], False),
        # {"name": "a1a", "categorical": True},
        # {"name": "a9a", "categorical": True},
        # {"name": "breast-cancer", "categorical": True},
        # {"name": "w8a", "categorical": True},
        ]

repeat = 20
feature = 10
delimiter = ","
df = pd.DataFrame()


if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    X: sp.sparse.csr_matrix
    y: np.ndarray
    train_X: sp.sparse.csr_matrix
    train_y: np.ndarray
    test_X: sp.sparse.csr_matrix
    test_y: np.ndarray

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

        train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, y, train_size=0.8, random_state=seed)

        model = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
        model.fit(train_X, train_y)

        score = sklearn.metrics.accuracy_score(test_y, model.predict(test_X))

        explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=train_X,
                mode=d.task,
                training_labels=train_y,
                feature_names=d.feature,
                discretize_continuous=True,
                class_names=d.label,
                random_state=seed,
                )

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, test_X.shape[0])

        print("============================================================")
        print(f"Explaining {d.name} {idx}th instance of test set (ground truth: {d.label[test_y[idx]]})...")
        print("============================================================")

        exp_inst = explainer.explain_instance(
                data_row=test_X[idx],
                predict_fn=model.predict_proba,
                labels=[i for i in range(len(d.label))],
                num_features=min(10, X.shape[1]),
                )
        exp_inst.save_to_file(f"../output/{d.name}_{idx}_{timestamp}.html")

        """
        Find frequency pattern
        """
        for i in range(repeat):
            exp_inst = explainer.explain_instance(
                    data_row=test_X[idx],
                    predict_fn=model.predict_proba,
                    num_features=min(10, X.shape[1]),
                    )
            data = exp_inst.as_list()
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
            df = pd.concat([df, new_row], ignore_index=True)

        file = f"../output/{d.name}_{idx}_quantiled_{timestamp}.csv"
        df.to_csv(file, sep=delimiter, index=False)

        spark = pyspark.sql.SparkSession.builder.appName("LIME").getOrCreate()
        data = (spark.read.text(file).select(pyspark.sql.functions.split("value", delimiter).alias("items")))

        fp = pyspark.ml.fpm.FPGrowth(minSupport=0.2, minConfidence=0.3)
        model = fp.fit(data)
        model.setPredictionCol("prediction")
        df = model.associationRules.sort("antecedent", "consequent")

        file = f"../output/{d.name}_{idx}_fp_{timestamp}.csv"
        df.toPandas().to_csv(file, sep=delimiter)
