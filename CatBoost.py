
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# Cat Boost

# X Train
traindf = pd.read_csv("in-csv/train_final.csv")
# X Test
testdf = pd.read_csv("in-csv/test_final.csv")
# X Sample
sampledf = pd.read_csv("in-csv/sample_final.csv")

#try dropping fnlwgt
traindf = traindf.drop('fnlwgt', axis=1)
testdf = testdf.drop('fnlwgt', axis=1)

X = traindf.drop('income>50K', axis=1)
y = traindf['income>50K']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

test_data = testdf.drop('ID', axis=1)




cat_feats = ['workclass', 'education', 'marital.status', 'occupation',
               'relationship', 'race', 'sex', 'native.country']

best_params = {
    'bagging_temperature': 0.8, 
    'depth': 5, 
    'iterations': 1000,
    'l2_leaf_reg': 30,
    'learning_rate': 0.05,
    'random_strength': 0.8
}

model_cat = CatBoostClassifier(**best_params,
                                loss_function='Logloss',
                                eval_metric='AUC', 
                                nan_mode='Min',
                                random_seed=42,
                                depth=10,
                                thread_count=4,
                                verbose=True)

model_cat.fit(X_train, y_train,
              eval_set=(X_test, y_test), 
              cat_features=cat_feats,
              verbose_eval=300, 
              use_best_model=True,
              plot=True)


print(X_test)

print(y_test)


#CatBoost predictions

test_data = testdf.drop('ID', axis=1).to_numpy()

cat_predictions = model_cat.predict(test_data)

cat_predictions_df = pd.DataFrame({'ID': sampledf['ID'], 
                                   'Prediction': cat_predictions})

print(cat_predictions_df.head())

#nn_predictions_df.head()

cat_predictions_df.to_csv('res_sub.csv', index=False)
                                


