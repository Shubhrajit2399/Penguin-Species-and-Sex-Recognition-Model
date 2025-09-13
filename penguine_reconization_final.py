#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 10:18:34 2025

@author: shubhrajit
"""

import os
import joblib
import pandas as pd
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
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
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df=pd.read_csv("pengine_train_set.csv")
    df['combined_target'] = df['species'].astype(str) + '_' + df['sex'].astype(str)
    counts = df['combined_target'].value_counts()
    df = df[df['combined_target'].isin(counts[counts > 1].index)]
    #split=StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    #for train_index, test_index in split.split(df, df["combined_target"]):
    #    df.loc[test_index].to_csv("pre_test_set_penguine.csv",index=False)
    #    df=df.loc[train_index]
        
    pengu_labels=df[["combined_target"]].copy()
    pengu_features=df.drop(["species","sex","combined_target"],axis=1)
    
    num_attribs=pengu_features.drop("island", axis=1, errors="Ignore").columns.tolist()
    cat_attribs=["island"]
    
    pipeline=build_pipeline(num_attribs, cat_attribs)
    pengu_prepared=pipeline.fit_transform(pengu_features)
    
    model=RandomForestClassifier(random_state=42)
    model.fit(pengu_prepared, pengu_labels)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is now Trained.")
else:
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)
    input_data=pd.read_csv("penguine_test_set_input.csv")
    transformed_input=pipeline.transform(input_data)
    predictions=model.predict(transformed_input)
    input_data["combined_target"]=predictions
    input_data[['species', 'sex']] = input_data['combined_target'].str.split('_', expand=True)
    input_data.drop(columns=['combined_target'], inplace=True)
    input_data.to_csv("penguine_test_set_output.csv",index=False)
    print("Inference completed, result saved into penguine_test_set_output.csv")