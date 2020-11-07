# Doğrusal Olmayan Regresyon Modelleri
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle
# !pip install catboost
from catboost import CatBoostRegressor

# !pip install lightgbm
# conda install -c conda-forge lightgbm
from lightgbm import LGBMRegressor

# !pip install xgboost
from xgboost import XGBRegressor

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

# VERININ HAZIRLANMASI

# Kaydedilmiş Verinin Çağırılması
train_df = pd.read_pickle("Datasets/prepared_data/train_df.pkl")
test_df = pd.read_pickle("Datasets/prepared_data/test_df.pkl")
# Bariz hatalı 2 değişkenin çıkarılması:
all_data = [train_df, test_df]
test_id = test_df.copy()
drop_list = ["index", "Id"]
for data in all_data:
    data.drop(drop_list, axis=1, inplace=True)
# train & test ayrımını yapalım:
X = train_df.drop('SalePrice', axis=1)
y = np.ravel(train_df[["SalePrice"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
y_train = np.ravel(y_train)  # boyut ayarlaması


# CatBoost: Model & Tahmin
catb_model = CatBoostRegressor(verbose=False).fit(X_train, y_train)
y_pred = catb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Model Tuning
catb_params = {"iterations": [200, 500],
               "learning_rate": [0.01, 0.1],
               "depth": [3,6]}
catb_model = CatBoostRegressor()
catb_cv_model = GridSearchCV(catb_model,
                             catb_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)
catb_cv_model.best_params_ ={'depth': 3, 'iterations': 500, 'learning_rate': 0.1}
# Final Model
catb_tuned = CatBoostRegressor(**catb_cv_model.best_params_).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, y_pred))





# LightGBM: Model & Tahmin
lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Model Tuning
lgb_model = LGBMRegressor()
lgbm_params = {"learning_rate": [0.01, 0.001, 0.1, 0.5, 1],
               "n_estimators": [200, 500, 1000, 5000],
               "max_depth": [6, 8, 10, 15, 20],
               "colsample_bytree": [1, 0.8, 0.5, 0.4]}
lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)
lgbm_cv_model.best_params_ = {'colsample_bytree': 0.4, 'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 5000}
# Final Model
lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))




# GBM: Model & Tahmin
gbm_model = GradientBoostingRegressor().fit(X_train, y_train)
y_pred = gbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# # Model Tuning
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}
# gbm_params2 = {"learning_rate": [0.001, 0.1, 0.01, 0.05],
#               "max_depth": [3, 5, 8, 10,20,30],
#               "n_estimators": [200, 500, 1000, 1500, 5000],
#               "subsample": [1, 0.4, 0.5, 0.7],
#               "loss": ["ls", "lad", "quantile"]}
gbm_model = GradientBoostingRegressor()
gbm_cv_model = GridSearchCV(gbm_model,
                            gbm_params,
                            cv=10,
                            n_jobs=-1,
                            verbose=2).fit(X_train, y_train)
gbm_cv_model.best_params_={'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 0.5}
# Final Model
gbm_tuned = GradientBoostingRegressor(**gbm_cv_model.best_params_).fit(X_train, y_train)



# XGBoost: Model & Tahmin
xgb = XGBRegressor().fit(X_train, y_train)
y_pred = xgb.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Model Tuning
xgb_params2 = {"learning_rate": [0.1, 0.01, 0.5],
              "max_depth": [5, 8, 15, 20],
              "n_estimators": [100, 200, 500, 1000],
              "colsample_bytree": [0.4, 0.7, 1]}

xgb_cv_model = GridSearchCV(xgb, xgb_params2, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
xgb_cv_model.best_params_ = {'colsample_bytree': 0.4, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1000}
# Final Model
xgb_tuned = XGBRegressor(**xgb_cv_model.best_params_).fit(X_train, y_train)



# Random Forests: Model & Tahmin
rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
# Model Tuning
rf_params2 = {"max_depth": [3, 5, 8, 10, 15, None],
            "max_features": [5, 10, 15, 20, 50, 100],
            "n_estimators": [200, 500, 1000],
            "min_samples_split": [2, 5, 10, 20, 30, 50]}
rf_cv_model = GridSearchCV(rf_model, rf_params2, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_
# Final Model
rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)





# Tum Base Modeller

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor()),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

# Base modellerin test hataları
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    msg = "%s: (%f)" % (name, rmse)
    print(msg)








tuned_models = [("catb_tuned",catb_tuned),
                ("lgbm_tuned",lgbm_tuned),
                ("gbm_tuned",gbm_tuned),
                ("xgb_tuned",xgb_tuned),
                ("rf_tuned",rf_tuned)]

for name, model in tuned_models:
     model.fit(X_train, y_train) # tekrar fit etmeye gerek yok.
     y_pred = model.predict(X_test)
     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
     msg = "%s: (%f)" % (name, rmse)
     print(msg)




# train & test ayrımını yapalım:
X_train= train_df.drop('SalePrice', axis=1)
y_train = train_df["SalePrice"]
X_test = test_df.drop("SalePrice",axis=1)




xgb_tuned_model = rf_tuned
xgb_tuned_model.fit(X_train, y_train)
predictions =xgb_tuned_model.predict(X_test)


my_submission = pd.DataFrame({'Id': test_id.Id,'SalePrice': predictions})
my_submission.to_csv('submission.csv', index=False)


my_submission
