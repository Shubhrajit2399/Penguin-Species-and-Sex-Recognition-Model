#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 08:11:28 2025

@author: shubhrajit
"""

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

df=pd.read_csv("pengine_train_set.csv")
print(df.info())

df['combined_target'] = df['species'].astype(str) + '_' + df['sex'].astype(str)
counts = df['combined_target'].value_counts()
df = df[df['combined_target'].isin(counts[counts > 1].index)]

split=StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)

for train_index,test_index in split.split(df, df["combined_target"]):
    strat_train_set=df.loc[train_index]
    strat_test_set=df.loc[test_index]
    
df=strat_train_set.copy()

pengu_labels=df[["combined_target"]].copy()
pengu_features=df.drop(["species","sex","combined_target"],axis=1)

num_attribs=pengu_features.drop("island", axis=1, errors="Ignore").columns.tolist()
cat_attribs=["island"]

num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
cat_pipeline=Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])
full_pipeline=ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat",cat_pipeline,cat_attribs)
    ])

pengu_prepared=full_pipeline.fit_transform(pengu_features)

log_reg=LogisticRegression()
mul_target_class=MultiOutputClassifier(log_reg)
mul_target_class.fit(pengu_prepared, pengu_labels)
mul_output_accuracy=make_scorer(accuracy_score)
accuracy=cross_val_score(mul_target_class, pengu_prepared, pengu_labels, scoring=mul_output_accuracy, cv=10)
print("Logistic-Regression Accuracy thru cross-validation:")
print(pd.Series(accuracy).describe())

tree_class=DecisionTreeClassifier()
tree_class.fit(pengu_prepared, pengu_labels)
tree_accuracy=cross_val_score(tree_class, pengu_prepared, pengu_labels, scoring="accuracy", cv=10)
print("Decision-Tree Classifier Accuracy thru cross-validation:")
print(pd.Series(tree_accuracy).describe())

randm_frst=RandomForestClassifier()
randm_mul_op_class=MultiOutputClassifier(randm_frst)
randm_mul_op_class.fit(pengu_prepared, pengu_labels)
randm_frst_accuracy=cross_val_score(randm_mul_op_class, pengu_prepared, pengu_labels, scoring="accuracy", cv=10)
print("Random-Forest Classifier Accuracy thru cross-validation:")
print(pd.Series(randm_frst_accuracy).describe())

sv_class=SVC()
sv_mul_op_class=MultiOutputClassifier(sv_class)
sv_mul_op_class.fit(pengu_prepared, pengu_labels)
sv_class_accuracy=cross_val_score(sv_mul_op_class, pengu_prepared, pengu_labels, scoring="accuracy", cv=10)
print("Suuport-Vector Classifier Accuracy thru cross-validation:")
print(pd.Series(sv_class_accuracy).describe())