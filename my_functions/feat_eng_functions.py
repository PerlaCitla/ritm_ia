import pandas as pd
import numpy as np
import seaborn as sns
import os

def create_auxiliar_target_variable(df: pd.DataFrame) -> pd.DataFrame:
  df_target = df.copy()
  # Como c_lastfm_playcount fue eliminada, usaremos c_lastfm_listeners para estimar la tracción por día
  # y luego proyectarla a 30 días.
  df_target['c_lastfm_listeners_per_day'] = df_target['c_lastfm_listeners'] / df_target['c_days_since_release'].replace(0, 1)

  # 1. Estimar oyentes en los primeros 30 días (asumiendo ritmo lineal)
  df_target['c_estimated_30d_listeners'] = df_target['c_lastfm_listeners_per_day'] * 30

  return df_target

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica ingeniería de variables para extraer nuevas características
    basadas en engagement, temporalidad y texto.
    """
    df_fe = df.copy()
    
    # Debug: Ver qué columnas están disponibles
    print(f"DEBUG - Columnas disponibles en feature_engineering: {list(df_fe.columns)}")

    # Convertir fecha a datetime por si no lo está
    if 'd_first_release_date' in df_fe.columns:
        df_fe['d_first_release_date'] = pd.to_datetime(df_fe['d_first_release_date'], errors='coerce')
    else:
        print(f"WARNING: columna 'd_first_release_date' no encontrada")

    # Características Temporales
    if 'd_first_release_date' in df_fe.columns:
        df_fe['v_release_month'] = df_fe['d_first_release_date'].dt.month.fillna(0).astype(int)
        df_fe['v_release_dayofweek'] = df_fe['d_first_release_date'].dt.dayofweek.fillna(-1).astype(int)
        df_fe['v_is_weekend_release'] = df_fe['v_release_dayofweek'].isin([4, 5, 6]).astype(int)
        df_fe['v_release_quarter'] = df_fe['d_first_release_date'].dt.quarter
        df_fe['v_day_of_year'] = df_fe['d_first_release_date'].dt.dayofyear.fillna(0).astype(int)
        df_fe['v_is_holiday_season'] = df_fe['v_release_month'].isin([11, 12, 6, 7]).astype(int)
    else:
        # Crear valores por defecto si no existe la columna de fecha
        df_fe['v_release_month'] = 0
        df_fe['v_release_dayofweek'] = -1
        df_fe['v_is_weekend_release'] = 0
        df_fe['v_release_quarter'] = 0
        df_fe['v_day_of_year'] = 0
        df_fe['v_is_holiday_season'] = 0

    # Características de Texto - con validación de columnas
    if 't_title' in df_fe.columns:
        df_fe['c_title_length'] = df_fe['t_title'].astype(str).apply(len)
        df_fe['c_title_word_count'] = df_fe['t_title'].astype(str).apply(lambda x: len(x.split()))
        df_fe['v_title_has_parentheses'] = df_fe['t_title'].astype(str).str.contains(r'\(|\)', regex=True).astype(int)
    else:
        print(f"WARNING: columna 't_title' no encontrada")
        df_fe['c_title_length'] = 0
        df_fe['c_title_word_count'] = 0
        df_fe['v_title_has_parentheses'] = 0

    if 't_artist_name' in df_fe.columns:
        df_fe['c_artist_name_length'] = df_fe['t_artist_name'].astype(str).apply(len)
    else:
        print(f"WARNING: columna 't_artist_name' no encontrada")
        df_fe['c_artist_name_length'] = 0

    if 't_title' in df_fe.columns and 't_artist_name' in df_fe.columns:
        df_fe['v_is_eponymous'] = (df_fe['t_title'].astype(str).str.lower() == df_fe['t_artist_name'].astype(str).str.lower()).astype(int)
    else:
        df_fe['v_is_eponymous'] = 0

    # Detectar si es una colaboración basada en el título o el nombre del artista
    collab_pattern = r'feat\.|ft\.|&| x '
    if 't_title' in df_fe.columns and 't_artist_name' in df_fe.columns:
        df_fe['v_is_collab'] = (
            df_fe['t_title'].astype(str).str.contains(collab_pattern, case=False, na=False) |
            df_fe['t_artist_name'].astype(str).str.contains(collab_pattern, case=False, na=False)
        ).astype(int)
    else:
        df_fe['v_is_collab'] = 0

    # Count-based features from delimited string columns
    if 't_artist_ids' in df_fe.columns:
        df_fe['c_num_artists'] = df_fe['t_artist_ids'].astype(str).apply(lambda x: len(x.split('|')) if pd.notna(x) else 0)
    else:
        print(f"WARNING: columna 't_artist_ids' no encontrada")
        df_fe['c_num_artists'] = 0

    return df_fe


def create_target_variable(df: pd.DataFrame, threshold_percentile: float = 0.75) -> pd.DataFrame:
    """
    Crea una variable objetivo binaria para predecir si un lanzamiento
    es un 'éxito' basado en la estimación de sus primeros 30 días.
    """
    df_target = df.copy()

    # 2. Definir el umbral de éxito basado en un percentil (ej. Top 25%)
    umbral_exito = df_target['c_estimated_30d_listeners'].quantile(threshold_percentile)

    # 3. Crear la variable binaria (1 = Éxito, 0 = No Éxito)
    df_target['target_success_30d'] = (df_target['c_estimated_30d_listeners'] >= umbral_exito).astype(int)

    print(f"Umbral de éxito (Percentil {threshold_percentile*100}%): {umbral_exito:,.2f} oyentes estimados en 30 días.")
    print("\nDistribución de la variable objetivo:")
    print(df_target['target_success_30d'].value_counts(normalize=True) * 100)

    return df_target
