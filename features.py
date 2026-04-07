from typing import Dict, List, Tuple, Literal, Union, Optional
import pandas as pd
import numpy as np
import os

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


#################
from my_functions.feat_eng_functions import feature_engineering,create_target_variable

df_music_master = pd.read_csv('music_master_final_clean.csv')

df_music_master_model = feature_engineering(df_music_master)

# Ver las nuevas variables creadas
columnas_nuevas = ['v_release_month', 'v_release_dayofweek','v_is_weekend_release', 'c_title_length', 'v_is_collab','c_artist_name_length','c_num_artists','v_release_quarter']
print(df_music_master_model[columnas_nuevas].head())

# Aplicar la función al dataframe que ya tiene la ingeniería de variables
# OJO: Se pasa df_music_master_model en lugar de df_music_2025_imputed para usar todas las variables calculadas
df_music_master_model = create_target_variable(df_music_master_model)

# Mostrar el resultado
print(df_music_master_model[['t_title','t_artist_name', 'c_lastfm_listeners_per_day', 'c_estimated_30d_listeners', 'target_success_30d']].head(10))

print("Se tiene un total de {} filas y {} columnas después de la ingeniería de variables y creación del target.".format(df_music_master_model.shape[0], df_music_master_model.shape[1]))
#print(df_music_master_model.shape)

df_music_master_model.to_csv('music_master_final_model.csv', index=False)
print(f"Archivo 'music_master_final_model.csv' guardado exitosamente.")