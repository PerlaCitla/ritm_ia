from typing import Dict, List, Tuple, Literal, Union, Optional
import pandas as pd
import numpy as np
import os

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


#################
from my_functions.eda_functions import tipo_variable,find_completely_null_columns,rename_columns_by_type,identificar_outliers_iqr,remove_high_null_columns,get_columns_by_null_percentage,impute_missing_values,encontrar_correlaciones_perfectas,save_correlation_heatmap,eliminar_variables_unitarias,graficar_histogramas,graficar_barras_discretas


base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'inputs')

df_music_2025 = pd.read_csv(os.path.join(file_path, 'music_master_2025_final_clean_20260401.csv'))
df_music_2024 = pd.read_csv(os.path.join(file_path, 'music_master_2024_final_clean_20260401.csv'))
df_music_2023 = pd.read_csv(os.path.join(file_path, 'music_master_2023_final_clean_20260402.csv'))
df_music_2022 = pd.read_csv(os.path.join(file_path, 'music_master_2022_final_clean_20260402.csv'))
df_music_2021 = pd.read_csv(os.path.join(file_path, 'music_master_2021_final_clean_20260403.csv'))
df_music_2020 = pd.read_csv(os.path.join(file_path, 'music_master_2020_final_clean_20260403.csv'))
df_music_2019 = pd.read_csv(os.path.join(file_path, 'music_master_2019_final_clean_20260403.csv'))
df_music_2018 = pd.read_csv(os.path.join(file_path, 'music_master_2018_final_clean_20260403.csv'))
df_music_2017 = pd.read_csv(os.path.join(file_path, 'music_master_2017_final_clean_20260403.csv'))
df_music_2016 = pd.read_csv(os.path.join(file_path, 'music_master_2016_final_clean_20260403.csv'))
df_music_2015 = pd.read_csv(os.path.join(file_path, 'music_master_2015_final_clean_20260403.csv'))
df_music_2014 = pd.read_csv(os.path.join(file_path, 'music_master_2014_final_clean_20260404.csv'))
df_music_2013 = pd.read_csv(os.path.join(file_path, 'music_master_2013_final_clean_20260404.csv'))
df_music_2012 = pd.read_csv(os.path.join(file_path, 'music_master_2012_final_clean_20260404.csv'))
df_music_2011 = pd.read_csv(os.path.join(file_path, 'music_master_2011_final_clean_20260404.csv'))
df_music_2010 = pd.read_csv(os.path.join(file_path, 'music_master_2010_final_clean_20260404.csv'))
df_music_2009 = pd.read_csv(os.path.join(file_path, 'music_master_2009_final_clean_20260404.csv'))
df_music_2008 = pd.read_csv(os.path.join(file_path, 'music_master_2008_final_clean_20260405.csv'))
df_music_2007 = pd.read_csv(os.path.join(file_path, 'music_master_2007_final_clean_20260405.csv'))
df_music_2006 = pd.read_csv(os.path.join(file_path, 'music_master_2006_final_clean_20260405.csv'))
df_music_2005 = pd.read_csv(os.path.join(file_path, 'music_master_2005_final_clean_20260405.csv'))
df_music_2004 = pd.read_csv(os.path.join(file_path, 'music_master_2004_final_clean_20260406.csv'))

df_music_master = pd.concat([df_music_2025,df_music_2024,df_music_2023,df_music_2022,df_music_2021,
                             df_music_2020,df_music_2019,df_music_2018,df_music_2017,df_music_2016,
                             df_music_2015,df_music_2014,df_music_2013,df_music_2012,df_music_2011,
                             df_music_2010,df_music_2009,df_music_2008,df_music_2007,df_music_2006,
                             df_music_2005,df_music_2004])

df_music_master.reset_index(drop=True, inplace=True)


df_music_master.drop(columns=["trend_score_v0",
                          "score_popularity",
                          "score_users",
                          "score_sitewide",
                          "score_artist_strength",
                          "score_recency","lastfm_source_used"], inplace=True)

print(df_music_master.shape)
print(df_music_master.head())
# print("\n=== TIPOS DE DATOS ANTES DE RENOMBRAR ===")
# print(df_music_master.dtypes)

# resumen_df, discretas, continuas = tipo_variable(df_music_master)

# print(discretas)


# Limpieza de datos
## Verificar si hay filas duplicadas

if df_music_master[df_music_master.duplicated(keep=False)].shape[0] > 0:
    print("Hay filas duplicadas en el DataFrame.")
    df_music_master.drop_duplicates(inplace=True)
else:
    print("No hay filas duplicadas en el DataFrame.")


## Eliminación de Outliers

### Se muestran las columnas que son 100 nulas

nulas_completas = find_completely_null_columns(df_music_master)
print(f"Columnas completamente nulas: {nulas_completas}")

# Se eliminan las columnas completamente nulas
df_music_master.drop(columns=nulas_completas, inplace=True)
print("Se eliminaron las columnas con 100% nulos")

# Se identifican columnas numericas
columns_numericas = df_music_master.select_dtypes(include='number').columns

# Cambiamos el numbre las columnas
df_music_master = rename_columns_by_type(df_music_master)

columnas_numericas =  df_music_master.filter(like="c_").columns

indices_a_remover = identificar_outliers_iqr(df_music_master, columnas_numericas)
print(f"Se encontraron {len(indices_a_remover)} filas con outliers, las cuales serán eliminadas.")
df_music_master = df_music_master.drop(index=indices_a_remover)
print(f"Dimensiones del DataFrame después de eliminar outliers: {df_music_master.shape}")

## Detección y tratamiento de valores ausentes

### Se eliminan columnas con más del 50% de valores nulos
df_music_master = remove_high_null_columns(df_music_master, threshold=0.5)


### Se imputan las columnas con entre 0% y 20% de nulos
columns_with_nulls = get_columns_by_null_percentage(df_music_master, 0.0, 0.2)

print("Columnas a imputar:")
print(columns_with_nulls)

df_music_master_imputed = impute_missing_values(df_music_master,columns_to_impute=columns_with_nulls)

print("Missing values after imputation:")
print(df_music_master_imputed.isnull().sum() / df_music_master_imputed.shape[0])

print("Para los registros que no tienen información en las columnas 'c_lb_total_listen_count' y 'c_lb_total_user_count' se crea una nueva variable que indica si estas variables tenian infromación o no")


df_music_master_imputed["lb_total_listen_missing"] = df_music_master_imputed["c_lb_total_listen_count"].isna().astype(int)

df_music_master_imputed[["c_lb_total_listen_count", "c_lb_total_user_count"]] = df_music_master_imputed[["c_lb_total_listen_count", "c_lb_total_user_count"]].fillna(0)

print(df_music_master_imputed.isnull().sum() / df_music_master_imputed.shape[0])

print("Hasta este punto se han tratado los valores ausentes y los valores extremos")

### Deteccion de variables Altamente correlacionadas

print(encontrar_correlaciones_perfectas(df_music_master_imputed))

df_music_master_imputed.drop(columns=['c_log1p_lb_total_listen_count'], inplace=True)

print("Se eliminó la columna 'c_log1p_lb_total_listen_count' debido a su alta correlación con 'c_lb_total_listen_count'")

save_correlation_heatmap(df_music_master_imputed, output_dir="outputs/images", filename="correlation_heatmap.png")

### Eliminación de variables uniarias

df_music_master_imputed = eliminar_variables_unitarias(df_music_master_imputed)

save_correlation_heatmap(df_music_master_imputed, output_dir="outputs/images", filename="new_correlation_heatmap.png")

### Visualización de variables


df, discretas, continuas = tipo_variable(df_music_master_imputed)

graficar_histogramas(df_music_master_imputed, columnas=continuas)
graficar_barras_discretas(df_music_master_imputed, columnas=discretas)

df_music_master_imputed.to_csv('music_master_final_clean.csv', index=False)
print(f"Archivo 'music_master_final_clean.csv' guardado exitosamente. Dimensiones finales: {df_music_master_imputed.shape}")



