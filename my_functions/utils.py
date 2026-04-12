import os
import calendar
import joblib
import pandas as pd
import numpy as np
import requests
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, timedelta
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler

import urllib.request
import streamlit as st
from io import BytesIO

from dotenv import load_dotenv
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

base_path_inputs = os.path.join(os.path.dirname(__file__), "..", "inputs")


from prompts import stronger_prompt

from my_functions.eda_functions import impute_missing_values, rename_columns_by_type
from my_functions.feat_eng_functions import feature_engineering, create_auxiliar_target_variable
from my_functions.models_functions import concatenate_features, create_embeddings_with_word2vec
from musicbrainz_client import (
    collect_recent_musicbrainz_dataset,browse_release_groups_by_artist,
    search_artist_by_name)
from listenbrainz_client import (get_sitewide_artist_activity, 
                                 parse_sitewide_artist_activity, 
                                 parse_top_release_groups,get_sitewide_top_release_groups,
                                 get_release_group_popularity, get_release_group_metadata)
from lastfm_client import enrich_release_groups_with_lastfm, build_country_track_signal, add_single_chart_signals          

def get_recent_releases_data(n_releases: int = 10, days_back: int = 30) -> pd.DataFrame:
    """
    Extrae un número n_releases de lanzamientos recientes, los enriquece con
    ListenBrainz y Last.fm, calcula un score de tendencia y devuelve el dataset maestro limpio.
    """
    # Define la fecha de fin como la fecha actual.
    end_date = date.today()
    # Define la fecha de inicio como 'days_back' días antes de la fecha actual.
    start_date = end_date - timedelta(days=days_back)

    print(f"\n{'='*50}")
    print(f"Extrayendo {n_releases} datos para: {start_date.isoformat()} a {end_date.isoformat()}")
    print(f"{'='*50}")

    # Extraer registros para el periodo indicado
    # Ajustamos dinámicamente max_search_pages si n_releases es grande
    max_pages = max(3, (n_releases // 100) + 1)

    df_release_groups_master_recent, df_releases_master_recent = collect_recent_musicbrainz_dataset(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        primary_types=("album", "single", "ep"),
        max_search_pages=max_pages,
        max_release_groups=n_releases
    )

    if df_release_groups_master_recent.empty:
        print("No se encontraron lanzamientos para el periodo especificado.")
        return pd.DataFrame()

    print("\n*** EXTRACCIÓN DE MUSICBRAINZ COMPLETADA ***")

    # ============================================================
    # Enriquecimiento del Dataset reciente con ListenBrainz
    # ============================================================
    # 1. Preparar la tabla principal y normalizar tipos
    rg_recent = df_release_groups_master_recent.copy()
    rg_recent["release_group_id"] = rg_recent["release_group_id"].astype(str)
    rg_recent["artist_name"] = rg_recent["artist_name"].astype(str).str.strip()

    # 2. Extraer los MBIDs únicos para buscar en la API
    mbids_recent = rg_recent["release_group_id"].dropna().unique().tolist()
    print(f"Consultando popularidad en ListenBrainz para {len(mbids_recent)} release groups...")

    # 3. Obtener popularidad y metadatos
    df_lb_pop_recent = get_release_group_popularity(mbids_recent, chunk_size=100)

    print("Obteniendo metadatos (tags, artistas)...")
    df_lb_meta_recent = get_release_group_metadata(mbids_recent, inc="artist tag release", chunk_size=20)

    print("Obteniendo señales sitewide globales y de artistas...")
    payload_sw = get_sitewide_top_release_groups(stats_range="month", count=300, offset=0)
    df_lb_sitewide_recent = parse_top_release_groups(payload_sw)

    payload_art = get_sitewide_artist_activity(stats_range="month")
    df_lb_artist_recent = parse_sitewide_artist_activity(payload_art)

    # 4. Hacer el cruce (merge) de la data
    print("Cruzando datos y consolidando el dataframe...")
    df_release_groups_recent_lb = (
        rg_recent
        .merge(df_lb_pop_recent, left_on="release_group_id", right_on="release_group_mbid", how="left")
        .merge(df_lb_meta_recent[["release_group_mbid", "artist_name_meta", "tags_meta"]],
               left_on="release_group_id", right_on="release_group_mbid", how="left", suffixes=("", "_meta"))
        .merge(df_lb_sitewide_recent[["release_group_mbid", "listen_count"]],
               left_on="release_group_id", right_on="release_group_mbid", how="left", suffixes=("", "_sitewide"))
        .rename(columns={"listen_count": "lb_sitewide_listen_count_month"})
        .merge(df_lb_artist_recent[["artist_name", "artist_listen_count"]],
               on="artist_name", how="left")
        .rename(columns={"artist_listen_count": "artist_catalog_strength"})
    )

    # 5. Limpiar columnas duplicadas
    cols_to_drop = ["release_group_mbid", "release_group_mbid_meta", "release_group_mbid_sitewide"]
    cols_to_drop = [c for c in cols_to_drop if c in df_release_groups_recent_lb.columns]
    df_release_groups_recent_lb = df_release_groups_recent_lb.drop(columns=cols_to_drop, errors="ignore")

    print("\n*** Enriquecimiento con ListenBrainz completado ***")

    # ============================================================
    # Enriquecimiento del Dataset reciente con Last.fm
    # ============================================================
    print("Iniciando extracción de Last.fm. Esto puede tardar varios minutos...")
    df_lastfm_features_recent = enrich_release_groups_with_lastfm(
        df_release_groups=df_release_groups_recent_lb,
        max_rows=None
    )

    print("\nExtracción completada. Generando señales de charts por país...")
    countries = ["United States", "United Kingdom", "Germany", "France", "Spain", "Mexico"]
    df_country_tracks_recent = build_country_track_signal(countries=countries, page=1, limit=100)

    df_country_signal_recent = add_single_chart_signals(
        df_release_groups=df_release_groups_recent_lb,
        df_country_tracks=df_country_tracks_recent
    )

    print("Señales generadas exitosamente.")

    # ============================================================
    # Fusión Maestra (MusicBrainz + ListenBrainz + Last.fm)
    # ============================================================
    df_base_recent = df_release_groups_recent_lb.rename(columns={
        "total_listen_count": "lb_total_listen_count",
        "total_user_count": "lb_total_user_count"
    })

    df_master_recent = (
        df_base_recent
        .merge(
            df_lastfm_features_recent[[
                "release_group_id", "lastfm_source_used", "lastfm_listeners",
                "lastfm_playcount", "lastfm_tags_item", "lastfm_album_name",
                "lastfm_track_name", "lastfm_url", "lastfm_error"
            ]],
            on="release_group_id",
            how="left"
        )
        .merge(
            df_country_signal_recent[[
                "release_group_id", "lastfm_country_chart_hits",
                "lastfm_best_country_rank", "lastfm_total_country_playcount",
                "lastfm_countries"
            ]],
            on="release_group_id",
            how="left"
        )
    )

    # Preparación de variables numéricas y fechas
    def fix_mb_date(d):
        d = str(d).strip()
        if d in ('None', 'nan', ''):
            return np.nan
        if len(d) == 4:
            return f"{d}-01-01"
        elif len(d) == 7:
            return f"{d}-01"
        return d

    df_master_recent["first_release_date"] = df_master_recent["first_release_date"].apply(fix_mb_date)
    df_master_recent["first_release_date"] = pd.to_datetime(df_master_recent["first_release_date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    df_master_recent["days_since_release"] = (today - df_master_recent["first_release_date"]).dt.days

    num_cols = ["lb_total_listen_count", "lb_total_user_count", "lb_sitewide_listen_count_month", "artist_catalog_strength"]
    for c in num_cols:
        if c in df_master_recent.columns:
            df_master_recent[c] = pd.to_numeric(df_master_recent[c], errors="coerce")
            df_master_recent[f"log1p_{c}"] = df_master_recent[c].fillna(0).apply(lambda x: np.log1p(x) if x >= 0 else 0)
        else:
            df_master_recent[f"log1p_{c}"] = 0.0

    # Cálculo del Trend Score v0
    def minmax_safe(s):
        if s is None or s.empty: return pd.Series([])
        s = s.fillna(0)
        if s.max() == s.min():
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    df_master_recent["score_popularity"] = minmax_safe(df_master_recent.get("log1p_lb_total_listen_count", pd.Series([0]*len(df_master_recent))))
    df_master_recent["score_users"] = minmax_safe(df_master_recent.get("log1p_lb_total_user_count", pd.Series([0]*len(df_master_recent))))
    df_master_recent["score_sitewide"] = minmax_safe(df_master_recent.get("log1p_lb_sitewide_listen_count_month", pd.Series([0]*len(df_master_recent))))
    df_master_recent["score_artist_strength"] = minmax_safe(df_master_recent.get("log1p_artist_catalog_strength", pd.Series([0]*len(df_master_recent))))
    df_master_recent["score_recency"] = 1 - minmax_safe(df_master_recent["days_since_release"].clip(lower=0, upper=365))

    # Ponderación del score
    df_master_recent["trend_score_v0"] = (
        0.35 * df_master_recent["score_popularity"] +
        0.20 * df_master_recent["score_users"] +
        0.25 * df_master_recent["score_sitewide"] +
        0.10 * df_master_recent["score_artist_strength"] +
        0.10 * df_master_recent["score_recency"]
    ) * 100

    # Ordenar por el score y limpiar duplicados
    df_master_recent = df_master_recent.sort_values("trend_score_v0", ascending=False).reset_index(drop=True)

    df_master_recent_clean = df_master_recent.drop_duplicates(
        subset=["release_group_id"],
        keep="first"
    ).reset_index(drop=True)

    print(f"\n¡Proceso completado! Dimensiones del Master reciente Enriquecido: {df_master_recent_clean.shape}")

    return df_master_recent_clean

def predict_recent_releases(df_master_recent_clean, base_path=base_path_inputs):
    """
    Recibe un dataframe con lanzamientos recientes, aplica los modelos
    (Word2Vec, PCA, KMeans y XGBoost por cluster) y devuelve el dataframe
    con el cluster asignado y las predicciones de éxito.
    """
    print("0. Cargando modelos desde el disco...")
    model = Word2Vec.load(os.path.join(base_path, 'word2vec_model.bin'))
    pca_3 = joblib.load(os.path.join(base_path, 'pca_3_model.joblib'))
    kmeans_3d = joblib.load(os.path.join(base_path, 'kmeans_3d_model.joblib'))
    xgboost_model_cluster0 = joblib.load(os.path.join(base_path, 'xgboost_model_cluster0.joblib'))
    xgboost_model_cluster1 = joblib.load(os.path.join(base_path, 'xgboost_model_cluster1.joblib'))
    xgboost_model_cluster2 = joblib.load(os.path.join(base_path, 'xgboost_model_cluster2.joblib'))
    xgboost_model_cluster3 = joblib.load(os.path.join(base_path, 'xgboost_model_cluster3.joblib'))

    df_results = df_master_recent_clean.copy()
    df_results.drop(columns=["trend_score_v0",
                              "score_popularity",
                              "score_users",
                              "score_sitewide",
                              "score_artist_strength",
                              "score_recency"], inplace=True, errors='ignore')

    print("1. Aplicando preprocesamiento (renombrado, ingeniería e imputación)...")
    df_recent_prep = rename_columns_by_type(df_results)

    # Usando load_path para evitar el data leakage
    imputation_file = os.path.join(base_path, 'imputation_values.joblib')
    df_recent_imp = impute_missing_values(df_recent_prep, load_path=imputation_file)

    # Corregido: Pasar df_recent_imp a feature_engineering
    df_recent_imp = feature_engineering(df_recent_imp)

    df_recent_imp = create_auxiliar_target_variable(df_recent_imp)

    #Leer el dataframe original para saber con cuales columnas debo quedarme al final
    df_music_master_model = pd.read_csv(os.path.join(base_path, "df_music_master_model.csv"))
    columnas_finales = df_music_master_model.columns.tolist()
    if "target_success_30d" in columnas_finales:
        columnas_finales.remove("target_success_30d")

    # Asegurar que existan todas las columnas antes de filtrar
    for col in columnas_finales:
        if col not in df_recent_imp.columns:
            df_recent_imp[col] = np.nan

    df_recent_imp = df_recent_imp[columnas_finales]

    print("2. Concatenando texto y generando embeddings...")
    df_recent_text = concatenate_features(df_recent_imp)
    recent_embeddings, _ = create_embeddings_with_word2vec(
        df=df_recent_text,
        text_column='concatenated_features',
        existing_model=model
    )

    print("3. Cargando scalers, escalando embeddings y asignando clusters...")
    scaler = joblib.load(os.path.join(base_path, 'scaler_embeddings.joblib'))
    scaler_struct = joblib.load(os.path.join(base_path, 'scaler_struct.joblib'))

    recent_embeddings_scaled = scaler.transform(recent_embeddings)
    recent_pca = pca_3.transform(recent_embeddings_scaled)
    recent_clusters = kmeans_3d.predict(recent_pca)

    print("4. Preparando matrices combinadas (Embeddings + Struct)...")
    # Definir explicitamente las columnas estructuradas usadas en el entrenamiento
    structured_cols = [
        'c_lb_total_listen_count', 'c_lb_total_user_count', 'c_days_since_release',
        'c_log1p_lb_total_user_count', 'v_release_month', 'v_release_dayofweek',
        'v_is_weekend_release', 'v_release_quarter', 'v_day_of_year',
        'v_is_holiday_season', 'c_title_length', 'c_artist_name_length',
        'c_title_word_count', 'v_title_has_parentheses', 'v_is_eponymous',
        'v_is_collab', 'c_num_artists'
    ]

    # Asegurar que todas las columnas estructuradas esperadas existan en el nuevo set
    for col in structured_cols:
        if col not in df_recent_imp.columns:
            df_recent_imp[col] = 0

    X_recent_struct = df_recent_imp[structured_cols].fillna(0)
    X_recent_struct_scaled = scaler_struct.transform(X_recent_struct)

    # Combinar 100 features de embeddings + 17 features estructurados
    X_recent_combined = np.hstack((recent_embeddings_scaled, X_recent_struct_scaled))

    print("5. Realizando predicciones con XGBoost por cluster...")
    predictions = []
    probabilities = []

    for i, cluster in enumerate(recent_clusters):
        features = X_recent_combined[i].reshape(1, -1)

        # Seleccionar el modelo de acuerdo al cluster
        if cluster == 0:
            xgb_model = xgboost_model_cluster0
        elif cluster == 1:
            xgb_model = xgboost_model_cluster1
        elif cluster == 2:
            xgb_model = xgboost_model_cluster2
        else:
            xgb_model = xgboost_model_cluster3

        pred = xgb_model.predict(features)[0]
        prob = xgb_model.predict_proba(features)[0][1]

        predictions.append(pred)
        probabilities.append(prob)

    # 6. Unir resultados al dataframe para visualización
    df_results['predicted_cluster'] = recent_clusters
    df_results['predicted_success_30d'] = predictions
    df_results['success_probability'] = np.round(probabilities, 4)

    print("\n*** ¡INFERENCIA COMPLETADA! ***")
    return df_results

class MusicFreshReleaseInsights(BaseModel):
    """
    Output estructurado con insights clave extraídos del rendimiento de un lanzamiento musical
    con base en los datos de MusicBrainz, ListenBrainz, Last.fm y predicciones del modelo ML.
    """
    artist_name: str = Field(description="Nombre del artista")
    release_title: str = Field(description="Título del lanzamiento (álbum, EP, single)")
    primary_type: Optional[str] = Field(description="Tipo de lanzamiento: Album, Single, EP, etc.")
    predicted_cluster: Optional[int] = Field(description="Clúster asignado por el modelo predictivo (0, 1, 2 o 3)")
    predicted_success_30d: Optional[int] = Field(description="Predicción de éxito a 30 días (1 = Éxito, 0 = No Éxito)")
    success_probability: Optional[float] = Field(description="Probabilidad estimada de éxito (0.0 a 1.0)")
    momentum_evaluation: Optional[str] = Field(description="Evaluación general: Viral / Estable / Creciente / Bajo radar")
    executive_summary: str = Field(description="Resumen analítico del desempeño del lanzamiento en español (párrafo de 3 líneas)")
    top_genres: List[str] = Field(default_factory=list, description="Géneros musicales principales identificados")
    key_tags: List[str] = Field(default_factory=list, description="Etiquetas (tags) descriptivas del lanzamiento, máximo 5")
    global_reach: List[str] = Field(default_factory=list, description="Lista de países donde el lanzamiento tiene impacto o presencia en charts")
    numeric_highlights: List[str] = Field(default_factory=list, description="Cifras clave destacadas (ej. total de oyentes, reproducciones en Last.fm)")
    strengths: List[str] = Field(default_factory=list, description="Puntos fuertes o tracción positiva detectada en los datos, en español")
    weaknesses: List[str] = Field(default_factory=list, description="Áreas de oportunidad o señales de bajo rendimiento, en español")



def get_music_predictions_insights(client, df_predictions: pd.DataFrame, model_name: str = "gpt-5.4-mini") -> dict:
    """
    Obtiene insights clave de los lanzamientos recientes utilizando un modelo de OpenAI
    basándose en las predicciones del modelo ML.

    Args:
        client: Cliente de OpenAI inicializado.
        df_predictions: DataFrame con los lanzamientos recientes y predicciones de ML.
        model_name: El nombre del modelo de OpenAI a utilizar.

    Returns:
        Un objeto diccionario derivado de MusicFreshReleaseInsights con los insights extraídos.
    """
    # Convertimos los datos del dataframe a formato JSON para el prompt
    data_text = df_predictions.to_json(orient="records", force_ascii=False, date_format="iso")

    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": "Eres un experto Analista en Tendencias musicales y Data Science. Devuelve SOLO un JSON válido que siga exactamente el esquema de MusicFreshReleaseInsights. Salidas en español. Presta especial atención a la probabilidad de éxito (success_probability) y al clúster (predicted_cluster) para emitir tu evaluación."},
            {"role": "user", "content": f"Analiza los siguientes datos de los lanzamientos recientes y sus predicciones de Machine Learning:\n{data_text}"},
        ],
        response_format=MusicFreshReleaseInsights,
    )
    insights = response.choices[0].message.parsed
    return insights.model_dump()

def generate_insights_list(client, df_predictions: pd.DataFrame) -> list:
    """
    Itera sobre cada fila del dataframe de predicciones para obtener los insights
    individuales de todos los lanzamientos.
    """
    all_insights = []

    for index, row in df_predictions.iterrows():
        print(f"Generando insights para: {row['artist_name']} - {row['title']}...")

        # Convertimos la fila actual en un DataFrame de un solo registro para reutilizar tu función
        single_row_df = pd.DataFrame([row])

        # Obtenemos el insight individual
        insight = get_music_predictions_insights(client=client, df_predictions=single_row_df)
        all_insights.append(insight)

    return all_insights

def get_all_insights_fresh(n_releases: int, days_back: int):
  os.environ['OPENAI_API_KEY']=OPENAI_API_KEY
  client = OpenAI(api_key=OPENAI_API_KEY)

  df_master_recent_clean = get_recent_releases_data(n_releases=n_releases, days_back=days_back)
  df_predictions = predict_recent_releases(df_master_recent_clean)
  all_insights = generate_insights_list(client=client, df_predictions=df_predictions)
  return all_insights


def get_artist_master_data(artist_name: str) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de extracción y enriquecimiento (MusicBrainz,
    ListenBrainz y Last.fm) para un artista dado y devuelve el DataFrame maestro limpio.
    """
    print(f"🔍 Buscando al artista: {artist_name}...")
    artist_candidates = search_artist_by_name(artist_name, limit=5)

    if artist_candidates.empty:
        print("No se encontraron resultados para el artista.")
        return pd.DataFrame()

    # 1. Tomar el ID del primer candidato
    artist_id = artist_candidates.iloc[0]["artist_id"]
    real_name = artist_candidates.iloc[0]["name"]
    print(f"✅ Artista seleccionado: {real_name} ({artist_id})")

    # 2. Obtener los release groups del artista
    df_artist_rg = browse_release_groups_by_artist(artist_id=artist_id)
    if df_artist_rg.empty:
        print("No se encontraron release groups para este artista.")
        return pd.DataFrame()

    print(f"💿 Enriqueciendo {len(df_artist_rg)} release groups...")
    rg_artist = df_artist_rg.copy()
    rg_artist["release_group_id"] = rg_artist["release_group_id"].astype(str)
    rg_artist["artist_name"] = rg_artist["artist_name"].astype(str).str.strip()
    mbids_artist = rg_artist["release_group_id"].dropna().unique().tolist()

    # 3. Consultar ListenBrainz
    print("🧠 Consultando ListenBrainz...")
    df_lb_pop_art = get_release_group_popularity(mbids_artist, chunk_size=100)
    df_lb_meta_art = get_release_group_metadata(mbids_artist, inc="artist tag release", chunk_size=20)

    payload_sw_art = get_sitewide_top_release_groups(stats_range="month", count=300, offset=0)
    df_lb_sitewide_art = parse_top_release_groups(payload_sw_art)

    payload_art_act = get_sitewide_artist_activity(stats_range="month")
    df_lb_artist_act = parse_sitewide_artist_activity(payload_art_act)

    df_artist_lb = (
        rg_artist
        .merge(df_lb_pop_art, left_on="release_group_id", right_on="release_group_mbid", how="left")
        .merge(df_lb_meta_art[["release_group_mbid", "artist_name_meta", "tags_meta"]], left_on="release_group_id", right_on="release_group_mbid", how="left", suffixes=("", "_meta"))
        .merge(df_lb_sitewide_art[["release_group_mbid", "listen_count"]], left_on="release_group_id", right_on="release_group_mbid", how="left", suffixes=("", "_sitewide"))
        .rename(columns={"listen_count": "lb_sitewide_listen_count_month"})
        .merge(df_lb_artist_act[["artist_name", "artist_listen_count"]], on="artist_name", how="left")
        .rename(columns={"artist_listen_count": "artist_catalog_strength"})
    )
    cols_drop = [c for c in ["release_group_mbid", "release_group_mbid_meta", "release_group_mbid_sitewide"] if c in df_artist_lb.columns]
    df_artist_lb = df_artist_lb.drop(columns=cols_drop, errors="ignore")

    # 4. Consultar Last.fm
    print("📻 Consultando Last.fm...")
    df_lastfm_features_art = enrich_release_groups_with_lastfm(df_artist_lb, max_rows=None)
    countries = ["United States", "United Kingdom", "Germany", "France", "Spain", "Mexico"]
    df_country_tracks_art = build_country_track_signal(countries=countries, page=1, limit=100)
    df_country_signal_art = add_single_chart_signals(df_artist_lb, df_country_tracks_art)

    # 5. Consolidando Master y Calculando Scores
    print("⚙️ Consolidando datos y calculando trend scores...")
    df_base_art = df_artist_lb.rename(columns={"total_listen_count": "lb_total_listen_count", "total_user_count": "lb_total_user_count"})

    df_master_artist = (
        df_base_art
        .merge(
            df_lastfm_features_art[[
                "release_group_id", "lastfm_source_used", "lastfm_listeners",
                "lastfm_playcount", "lastfm_tags_item", "lastfm_album_name",
                "lastfm_track_name", "lastfm_url", "lastfm_error"
            ]], on="release_group_id", how="left"
        )
        .merge(
            df_country_signal_art[[
                "release_group_id", "lastfm_country_chart_hits",
                "lastfm_best_country_rank", "lastfm_total_country_playcount",
                "lastfm_countries"
            ]], on="release_group_id", how="left"
        )
    )

    def fix_mb_date(d):
        d = str(d).strip()
        if d in ('None', 'nan', ''): return np.nan
        if len(d) == 4: return f"{d}-01-01"
        elif len(d) == 7: return f"{d}-01"
        return d

    df_master_artist["first_release_date"] = df_master_artist["first_release_date"].apply(fix_mb_date)
    df_master_artist["first_release_date"] = pd.to_datetime(df_master_artist["first_release_date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    df_master_artist["days_since_release"] = (today - df_master_artist["first_release_date"]).dt.days

    num_cols = ["lb_total_listen_count", "lb_total_user_count", "lb_sitewide_listen_count_month", "artist_catalog_strength"]
    for c in num_cols:
        if c in df_master_artist.columns:
            df_master_artist[c] = pd.to_numeric(df_master_artist[c], errors="coerce")
            df_master_artist[f"log1p_{c}"] = df_master_artist[c].fillna(0).apply(lambda x: np.log1p(x) if x >= 0 else 0)
        else:
            df_master_artist[f"log1p_{c}"] = 0.0

    def minmax_safe(s):
        if s is None or s.empty: return pd.Series([])
        s = s.fillna(0)
        if s.max() == s.min(): return pd.Series([0.0] * len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    df_master_artist["score_popularity"] = minmax_safe(df_master_artist.get("log1p_lb_total_listen_count", pd.Series([0]*len(df_master_artist))))
    df_master_artist["score_users"] = minmax_safe(df_master_artist.get("log1p_lb_total_user_count", pd.Series([0]*len(df_master_artist))))
    df_master_artist["score_sitewide"] = minmax_safe(df_master_artist.get("log1p_lb_sitewide_listen_count_month", pd.Series([0]*len(df_master_artist))))
    df_master_artist["score_artist_strength"] = minmax_safe(df_master_artist.get("log1p_artist_catalog_strength", pd.Series([0]*len(df_master_artist))))
    df_master_artist["score_recency"] = 1 - minmax_safe(df_master_artist["days_since_release"].clip(lower=0, upper=365))

    df_master_artist["trend_score_v0"] = (
        0.35 * df_master_artist["score_popularity"] +
        0.20 * df_master_artist["score_users"] +
        0.25 * df_master_artist["score_sitewide"] +
        0.10 * df_master_artist["score_artist_strength"] +
        0.10 * df_master_artist["score_recency"]
    ) * 100

    df_master_artist = df_master_artist.sort_values("trend_score_v0", ascending=False).reset_index(drop=True)
    df_master_artist_clean = df_master_artist.drop_duplicates(subset=["release_group_id"], keep="first").reset_index(drop=True)

    print("\n🚀 Pipeline completado exitosamente.")
    return df_master_artist_clean

class MusicArtistInsights(BaseModel):
    """
    Output estructurado con insights clave extraídos del rendimiento de un lanzamiento musical
    con base en los datos de MusicBrainz, ListenBrainz y Last.fm.
    """
    artist_name: str = Field(description="Nombre del artista")
    release_title: str = Field(description="Título del lanzamiento (álbum, EP, single)")
    primary_type: Optional[str] = Field(description="Tipo de lanzamiento: Album, Single, EP, etc.")
    # trend_score: Optional[float] = Field(description="Puntuación de tendencia o momentum (basado en trend_score_v0)")
    momentum_evaluation: Optional[str] = Field(description="Evaluación general: Viral / Estable / Creciente / Bajo radar")
    executive_summary: str = Field(description="Resumen analítico del desempeño del lanzamiento en español (párrafo de 3 líneas)")
    top_genres: List[str] = Field(default_factory=list, description="Géneros musicales principales identificados")
    key_tags: List[str] = Field(default_factory=list, description="Etiquetas (tags) descriptivas del lanzamiento, máximo 5")
    global_reach: List[str] = Field(default_factory=list, description="Lista de países donde el lanzamiento tiene impacto o presencia en charts")
    numeric_highlights: List[str] = Field(default_factory=list, description="Cifras clave destacadas (ej. total de oyentes, reproducciones en Last.fm)")
    strengths: List[str] = Field(default_factory=list, description="Puntos fuertes o tracción positiva detectada en los datos, en español")
    weaknesses: List[str] = Field(default_factory=list, description="Áreas de oportunidad o señales de bajo rendimiento, en español")

def get_artist_catalog_insights(client, df_artist: pd.DataFrame, model_name: str = "gpt-5.4-mini") -> dict:
    """
    Obtiene insights clave del catálogo de un artista utilizando un modelo de OpenAI.

    Args:
        client: Cliente de OpenAI inicializado.
        df_artist: DataFrame con los datos del artista (ej. df_master_artist_clean).
        model_name: El nombre del modelo de OpenAI a utilizar.

    Returns:
        Un objeto diccionario derivado de MusicArtistInsights con los insights extraídos.
    """
    # Convertimos los datos del dataframe a formato JSON para el prompt
    data_text = df_artist.to_json(orient="records", force_ascii=False, date_format="iso")

    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": "Eres un experto Analista en Tendencias musicales. Devuelve SOLO un JSON válido que siga exactamente el esquema de MusicArtistInsights. Salidas en español."},
            {"role": "user", "content": f"Analiza los siguientes datos del catálogo del artista:\n{data_text}"},
        ],
        response_format=MusicArtistInsights,
    )
    insights = response.choices[0].message.parsed
    return insights.model_dump()

def get_insights_artist(artist_name: str):
  df = get_artist_master_data(artist_name)

  os.environ['OPENAI_API_KEY']=OPENAI_API_KEY
  client = OpenAI(api_key=OPENAI_API_KEY)

  response = get_artist_catalog_insights(client, df)
  return response #regrea un json


clusters_info = {
    "cluster_0": {
        "nombre": "Mainstream Estándar",
        "tamano": 17609,
        "perfil": "Grupo mayoritario con rendimiento histórico sólido y el mayor promedio de oyentes en Last.fm (~100.9)."
    },
    "cluster_1": {
        "nombre": "La Masa Promedio",
        "tamano": 15113,
        "perfil": "Segundo segmento más grande con métricas consistentes y buena estimación de tracción a 30 días (~13.88 oyentes)."
    },
    "cluster_2": {
        "nombre": "Nicho Activo",
        "tamano": 3441,
        "perfil": "Grupo más pequeño pero estable, con base de fans leal y métricas robustas (~100.3 oyentes históricos)."
    },
    "cluster_3": {
        "nombre": "Long Tail de Impacto Rápido",
        "tamano": 1849,
        "perfil": "El grupo más pequeño, con menor promedio histórico (~95.53) pero mayor tracción a 30 días (~17.83)."
    }
}

def get_cluster_insights(cluster_id: str):
    key = f"cluster_{cluster_id}"
    if key in clusters_info:
        return {"cluster": cluster_id, "perfil": clusters_info[key]}
    else:
        return {"clusters": clusters_info}