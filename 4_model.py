#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from utils import process_data

# Load data
data = pd.read_csv(f"data/data_clean.csv")
# Format date
data['date'] = pd.to_datetime(data['date'])
# Define variables
features = ['netIncome', 'grossProfit', 'ebit', 'totalRevenue', 'costOfRevenue', 'totalOtherIncomeExpenseNet', 'otherCurrentLiab', 'totalAssets', 'commonStock', 'otherLiab', 'otherAssets', 'cash', 'propertyPlantEquipment', 'accountsPayable', 'capitalSurplus', 'changeToLiabilities', 'totalCashflowsFromInvestingActivities', 'netBorrowings', 'totalCashFromFinancingActivities', 'changeInCash', 'totalCashFromOperatingActivities', 'depreciation', 'changeToNetincome', 'capitalExpenditures', 'changeToOperatingActivities', 'market_cap', 'div_percent', 'ebitAbs', 'totalRevenueAbs', 'yield', 'sector', 'outperform', 'positive' ]
target = 'outperform_next'

# Date for validation and test sets
# date_valid = datetime(year=2020, month=1, day=1)
# date_test = datetime(year=2020, month=11, day=1)
date_valid = datetime(year=2019, month=1, day=1)
date_test = datetime(year=2020, month=6, day=1)

# Process data and delcare model
clf, X_train, y_train, X_valid, y_valid, X_test, y_test, _ = \
    process_data(data, date_valid, date_test, features, target)

# Fuse train and validation sets and random shuffle
X = np.append(X_train, X_valid, axis=0)
y = np.append(y_train, y_valid, axis=0)
indices = list(range(len(X)))
np.random.shuffle(indices)
n = int(np.floor(0.9*len(indices)))
t, v = indices[:n], indices[n:]
X_train, X_valid, y_train, y_valid = X[t], X[v], y[t], y[v]

# Training
clf.fit(X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=200,
        patience=100,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
        )

# save model
saving_path_name = "models/model"
saved_filepath = clf.save_model(saving_path_name)
# clf = TabNetClassifier()
# clf.load_model("models/model.zip")

# Test
test_auc = roc_auc_score(y_score=clf.predict_proba(X_test)[:, 1],
                         y_true=y_test)
valid_auc = roc_auc_score(y_score=clf.predict_proba(X_valid)[:, 1],
                          y_true=y_valid)
train_auc = roc_auc_score(y_score=clf.predict_proba(X_train)[:, 1],
                          y_true=y_train)
print("Testing AUC\n")
print(f"BEST TRAIN SCORE: {train_auc}")
print(f"BEST VALID SCORE: {valid_auc}")
print(f"BEST TEST SCORE: {test_auc}")

# Plot roc curve
fpr = dict()
tpr = dict()
fpr['train'], tpr['train'], _ = roc_curve(y_score=clf.predict_proba(X_train)[:, 1],
                                          y_true=y_train)
fpr['valid'], tpr['valid'], _ = roc_curve(y_score=clf.predict_proba(X_valid)[:, 1],
                                          y_true=y_valid)
fpr['test'], tpr['test'], _ = roc_curve(y_score=clf.predict_proba(X_test)[:, 1],
                                        y_true=y_test)
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
names = ['train', 'valid', 'test']
for name, color in zip(names, colors):
    plt.plot(fpr[name], tpr[name], color=color, lw=2,
             label='{0}'.format(name))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Feature importance
print("\nFeature importance AUC\n")
f = clf.feature_importances_
for i in np.argsort(f):
    print(f"{features[i]}: {f[i]}")
