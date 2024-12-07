#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import sklearn.tree
import matplotlib.pyplot as plt


X, y = make_classification(random_state=0)

est_parameters = {
    "max_depth": range(2, 7)}

plt.figure(figsize=(18, 12))

def best_est(X, y):
    cv = KFold(2, shuffle=True, random_state=0)
    # default tree uses gini and best split
    est = DecisionTreeClassifier(random_state=0)
    grid_search = GridSearchCV(est, est_parameters, cv=cv)
    grid_search.fit(X, y)
    return grid_search.best_score_, grid_search.best_params_

for i in range(10):
    print(best_est(X, y))
    # why is the best tree different each time
    # unless I set the random_state in DecisionTreeClassifier() ?
