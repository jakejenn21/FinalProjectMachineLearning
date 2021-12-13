
import csv
import pprint
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# prepare input data
def prepare_inputs(X_train, X_test):

    enc = OrdinalEncoder()
    traindfOE = enc.fit_transform(X_train.astype(str))
    testdfOE = enc.fit_transform(X_test.astype(str))
    return traindfOE, testdfOE

def replace_missing(data):
    for col in data.columns:
        majority_class_index = np.argmax(np.unique(data[col], return_counts=True)[1])
        majority_value = np.unique(data[col])[majority_class_index]
        if majority_value == "?":
          newdata = data[data[col] != "?"]
          majority_class_index = np.argmax(np.unique(newdata[col], return_counts=True)[1])
          majority_value = np.unique(newdata[col])[majority_class_index]

        data[col] = np.where(data[col] == "?", majority_value, data[col])



print("\n\nXGB CLASSIFIER:\n\n")


# X Train
traindf = pd.read_csv("train_final.csv")

replace_missing(traindf)
X_train = traindf.iloc[:, :-1]

# Y Train
y_train = traindf.iloc[:,-1]

# X Test
testdf = pd.read_csv("test_final.csv")

plt.show()

replace_missing(testdf)
X_test = testdf.iloc[:, 1:]


X_train, X_test = prepare_inputs(X_train, X_test)

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, train_size=0.7, random_state=1234)

eval_set = [(X_train, y_train), (X_validation, y_validation)]

# XG Boost
model = xgb.XGBClassifier(
 learning_rate =0.01,
 n_estimators=2000, 
 max_depth=10,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,
 use_label_encoder=False)

model.fit(X_train,y_train,eval_metric=['auc'],eval_set=eval_set)

y_pred = model.predict(X_test)

y_pred_cv = model.predict(X_validation)

accuracy_cv = accuracy_score(y_validation, y_pred_cv)

print("\n\nCross Validation Accuracy: ", accuracy_cv, "\n\n")

rows = []
for i in testdf.index:
    row = tuple((testdf['ID'][i], y_pred[i]))
    rows.append(row)

outdf = pd.DataFrame(rows, columns=["ID","Prediction"])
outdf = outdf.set_index('ID')

outdf.to_csv("out.csv")




print("\n\nCATBOOST CLASSIFIER:\n\n")

# Train DF
traindf = pd.read_csv("train_final.csv")
replace_missing(traindf)

# X Train
X = traindf.drop(['income>50K'], axis=1)

# Y Train
y = traindf['income>50K']

# Test DF
testdf = pd.read_csv("test_final.csv")
replace_missing(testdf)

# X Test
X_test = testdf.drop(['ID'], axis=1)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)

categorical_features_indices = np.where(X.dtypes != np.float)[0]

#importing library and building model
from catboost import CatBoostRegressor
model=CatBoostClassifier(iterations=1450,
                           learning_rate=0.01,
                           depth=10,
                           loss_function='MultiClass')
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)

y_pred = model.predict(X_test)

y_pred = np.array(np.concatenate(y_pred).flat)

y_pred_cv = model.predict(X_validation)

accuracy_cv = accuracy_score(y_validation, y_pred_cv)

print("\n\nCross Validation Accuracy: ", accuracy_cv, "\n\n")

rows = []
for i in testdf.index:
    row = tuple((testdf['ID'][i], y_pred[i]))
    rows.append(row)

outdf = pd.DataFrame(rows, columns=["ID","Prediction"])
outdf = outdf.set_index('ID')

outdf.to_csv("out2.csv")

