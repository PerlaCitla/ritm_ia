import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from typing import Dict, List, Tuple, Literal, Union, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
######################
from my_functions.models_functions import train_decision_tree,evaluate_and_visualize_tree,evaluate_model_performance,prepare_data_and_train_rf,prepare_data_and_train_xgb,train_and_optimize_xgb

df_music_master = pd.read_csv('music_master_final_model.csv')

# Filtrar solo columnas numéricas
df_num = df_music_master.select_dtypes(include=["number"])

# Separar la variable objetivo antes de estandarizar
target_column = 'target_success_30d'
y_target = df_num[target_column]
X_features = df_num.drop(columns=[target_column])

# Escalamiento estándar de las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Crear un DataFrame con las características escaladas
df_scaled_features = pd.DataFrame(X_scaled, columns=X_features.columns, index=X_features.index)

# Unir la variable objetivo de nuevo al DataFrame estandarizado
df_music_master = pd.concat([df_scaled_features, y_target], axis=1)
print("Unir la variable objetivo de nuevo al DataFrame estandarizado")
print(df_music_master.head())

df_ml = df_music_master.copy()

print(df_ml['target_success_30d'].value_counts())

dtree, X_train_dtc, X_test_dtc, y_train_dtc, y_test_dtc, features_dtc = train_decision_tree(df_ml)

y_pred_train_dtc, y_pred_test_dtc = evaluate_and_visualize_tree(dtree, X_train_dtc, X_test_dtc, features_dtc)

evaluate_model_performance(dtree, X_test_dtc, y_train_dtc, y_pred_train_dtc, y_test_dtc, y_pred_test_dtc,model_name="Decision Tree",filename="precision_recall_curve_decision_tree.png")


prepare_data_and_train_rf(df_ml, target_rfc='target_success_30d')

prepare_data_and_train_xgb(df_ml, target_xgb='target_success_30d')

train_and_optimize_xgb(df_ml, target_xgb='target_success_30d')