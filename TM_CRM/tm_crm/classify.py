from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
from util import *

RANDOM_SEED = 99

# Read data
df = pd.read_csv("dataset.tsv", delimiter='\t')

# Replace Korean with English & Encode yes/no category
replace_dict = {
    "x4":{"저학력":"low", "중학력":"middle", "고학력":"high"},
    "x9":{"유선":"wired", '무선':"wireless"},
    "x5":{"yes":1, "no":0},
    "x7":{"yes":1, "no":0},
    "x8":{"yes":1, "no":0},
    "y":{"yes":1, "no":0},
    "x11":{"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, \
           "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    }

df.replace(replace_dict, inplace=True)


# Transform days and months into circular numeric data
days_in_month = 31
df['x10_sin_date'] = np.sin(2*np.pi*df['x10']/days_in_month)
df['x10_cos_date'] = np.cos(2*np.pi*df['x10']/days_in_month)
df.drop('x10', axis=1, inplace=True)

months_in_year = 12
df['x11_sin_month'] = np.sin(2*np.pi*df['x11']/months_in_year)
df['x11_cos_month'] = np.cos(2*np.pi*df['x11']/months_in_year)
df.drop('x11', axis=1, inplace=True)

# Make dummy variables for (multiple)categorical data
obj_df = df.select_dtypes(include=['object']).copy()
df = pd.get_dummies(data=df,
                    columns=list(obj_df),
                    drop_first = False)

df = df.astype(float)
df['y'] = df['y'].astype(int)
print(df.head())
print(df.dtypes)

# Modify Out-lier
# for x6
plotOutlier(df['x6'].sample(1000))
df.x6 = modify_max_outlier(df, col='x6', max=6000)

# for x12
#plotOutlier(df['x12'].sample(1000))
df.x12 = modify_max_outlier(df, col='x12', max=20)

# for x13
#plotOutlier(df['x13'].sample(1000))
df.x13 = modify_max_outlier(df, col='x13', max=370)

# for x14
#plotOutlier(df['x14'].sample(1000))
df.x14 = modify_max_outlier(df, col='x14', max=30)


y = df.y
X = df.drop('y', axis=1)

# Train:60%, Valid:20%, Test:20%
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                     test_size=0.2,
                                     train_size=0.8,
                                     random_state=RANDOM_SEED)

X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,
                                      test_size=0.25,
                                      train_size=0.75,
                                      random_state=RANDOM_SEED)

# Sanity-check that the data sets have been split representatively
for dataset in (y_train, y_valid, y_test):
    print('%i rows: %f yes rates' %(len(dataset), dataset.sum() / len(dataset)))


'''
#Out-lier removal
od_list = ['x1', 'x6', 'x12', 'x13', 'x14']
X_od = X_train[od_list]

if_model = IsolationForest(max_samples=200,
                           contamination=0.01,
                           random_state=RANDOM_SEED)
if_model.fit(X_od)
y_od_pred = if_model.predict(X_od)
print(Counter(y_od_pred))

outliers = []
for index, value in enumerate(y_od_pred):
    if(value==-1):
        outliers.append(index)

X_train.drop(X_train.index[outliers], inplace=True)
y_train.drop(y_train.index[outliers], inplace=True)
print("Number of train data after out-lier removal: %i" %(len(X_train)))
'''

'''
#Find the optimal number of random forest estimators
rf_auc = []
nTreeList = range(400, 1000, 50)
for iTrees in nTreeList:
    rf_model = RandomForestClassifier(n_estimators=iTrees,
                                    max_depth=None,
                                    max_features=16,
                                    bootstrap=False,
                                    oob_score=False,
                                    random_state=RANDOM_SEED)
    
    rf_model.fit(X=X_train, y=y_train)
    rf_pred = rf_model.predict_proba(X=X_valid)
    aucCal = roc_auc_score(y_valid, rf_pred[:,1])
    rf_auc.append(aucCal)

print("Training Done")
print(max(rf_auc))
print(rf_auc.index(max(rf_auc)))

plt.figure(1)
plt.plot(nTreeList, rf_auc, linewidth=2)
plt.xlabel('Number of Trees in Ensemble')
plt.ylabel('AUC')
plt.show()
'''

# Random forest model
rf_model = RandomForestClassifier(n_estimators=600,
                                max_depth=None,
                                max_features='auto',
                                min_samples_split=2,
                                min_samples_leaf=1,
                                bootstrap=True,
                                oob_score=False,
                                random_state=RANDOM_SEED)

rf_model.fit(X=X_train, y=y_train)
print("RF Training Done")

rf_pred = rf_model.predict_proba(X=X_valid)
aucCal = roc_auc_score(y_valid, rf_pred[:,1])
print("AUC - Random Forest: %f" %(aucCal))

print(accuracy_score(y_valid, rf_model.predict(X_valid)))


# Gradient boosting model
gb_model = GradientBoostingClassifier(n_estimators=600,
                                      max_depth=10,
                                      learning_rate=0.01,
                                      max_features=None)

gb_model.fit(X=X_train, y=y_train)
print("GB Training Done")

gb_auc = []
gb_aucBest = 0.0
gb_pred = gb_model.staged_decision_function(X_valid)
for p in gb_pred:
    aucCalc = roc_auc_score(y_valid, p)
    gb_auc.append(aucCalc)
    
    if aucCalc > gb_aucBest:
        gb_aucBest = aucCalc
        pBest = p

idxBest = gb_auc.index(max(gb_auc))

print("Best AUC - Gradient Boost: %f" %(gb_auc[idxBest]))
print("Number of Trees for Best AUC: %f" %(idxBest))
print(accuracy_score(y_valid, gb_model.predict(X_valid)))


# Variable importance for random forest
rf_fi = rf_model.feature_importances_
rf_fi = rf_fi / rf_fi.max()
featureNames = np.array([col for col in X_train])
rf_idxSorted = np.argsort(rf_fi)[::-1]

# Variable importance for gradient boosting
gb_fi = gb_model.feature_importances_
gb_fi = gb_fi / gb_fi.max()
gb_idxSorted = np.argsort(gb_fi)[::-1]

# Plot variable(feature) importance
f1, (fi_ax1, fi_ax2) = sns.plt.subplots(1, 2)
sns.barplot(x=rf_fi[rf_idxSorted], y=featureNames[rf_idxSorted], ax=fi_ax1)
fi_ax1.set_title("Variable Importance(RF)")
sns.barplot(x=gb_fi[gb_idxSorted], y=featureNames[gb_idxSorted], ax=fi_ax2)
fi_ax2.set_title("Variable Importance(GB)")
sns.plt.show()

# Plot ROC
rf_fpr, rf_tpr, rf_thresh = roc_curve(y_valid, list(rf_pred[:,1]))
gb_fpr, gb_tpr, gb_thresh = roc_curve(y_valid, list(pBest))
ctClass = [i*0.01 for i in range(101)]

f2 = plt.figure()
ax = plt.subplot(111)
plt.plot(rf_fpr, rf_tpr, linewidth=2, label="Random Forest")
plt.plot(gb_fpr, gb_tpr, linewidth=2, label="Gradient Boosting")
plt.plot(ctClass, ctClass, linestyle=':')
ax.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
sns.plt.show()

# Find the closest threshold point from the best
rf_best_thr = roc_best_cutoff(y_valid, rf_model.predict_proba(X_valid))
gb_best_thr = roc_best_cutoff(y_valid, gb_model.predict_proba(X_valid))

# Random Forest: Classification result and confusion matrix
rf_best_pred = pred_from_thresh(rf_model, X=X_valid, thr=rf_best_thr)
print(classification_report(y_valid, rf_best_pred))
plot_confusion_matrix(y_true=y_valid, y_pred=rf_best_pred, title="RF confusion matrix")

# Gradient Boosting: Classification result and confusion matrix
gb_best_pred = pred_from_thresh(gb_model, X=X_valid, thr=gb_best_thr)
print(classification_report(y_valid, gb_best_pred))
plot_confusion_matrix(y_true=y_valid, y_pred=gb_best_pred, title="GB confusion matrix")
