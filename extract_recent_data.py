# extract_recent_data.py
# Script para extraer datos de lanzamientos musicales de los últimos 30 días usando MusicBrainz.

import calendar
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Importar funciones necesarias de los módulos locales
from musicbrainz_client import collect_recent_musicbrainz_dataset
from listenbrainz_client import get_release_group_popularity, get_release_group_metadata, get_sitewide_top_release_groups, parse_top_release_groups, get_sitewide_artist_activity, parse_sitewide_artist_activity
from lastfm_client import enrich_release_groups_with_lastfm, build_country_track_signal, add_single_chart_signals

def main():
    # Define la fecha de fin como la fecha actual.
    end_date = date.today()
    # Define la fecha de inicio como 30 días antes de la fecha actual.
    start_date = end_date - timedelta(days=30)

    print(f"\n{'='*50}")
    print(f"Extrayendo datos para: {start_date.isoformat()} a {end_date.isoformat()}")
    print(f"{'='*50}")

    # Extraer registros para los últimos 30 días
    # Usamos max_search_pages para obtener más resultados y max_release_groups para el número total
    df_release_groups_master_recent, df_releases_master_recent = collect_recent_musicbrainz_dataset(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        primary_types=("album", "single", "ep"),
        max_search_pages=3,
        max_release_groups=10  # Ajusta este valor si necesitas más o menos resultados
    )

    print("\n*** EXTRACCIÓN COMPLETADA ***")
    print(f"Total Release Groups recientes: {df_release_groups_master_recent.shape[0]} registros")
    print(f"Total Releases (ediciones) recientes: {df_releases_master_recent.shape[0]} registros")

    # Vista previa
    print("\nRelease Groups recientes (primeras filas):")
    print(df_release_groups_master_recent.head())

    print("\nReleases recientes (primeras filas):")
    print(df_releases_master_recent.head())

    # Guardar en csv
    # df_release_groups_master_recent.to_csv('df_release_groups_master_recent.csv', index=False)
    # df_releases_master_recent.to_csv('df_releases_master_recent.csv', index=False)

    # print("\nArchivos CSV guardados: 'df_release_groups_master_recent.csv' y 'df_releases_master_recent.csv'")


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

    # 3. Obtener popularidad y metadatos (esperará automáticamente si toca el rate-limit)
    df_lb_pop_recent = get_release_group_popularity(mbids_recent, chunk_size=100)

    print("Obteniendo metadatos (tags, artistas)...")
    df_lb_meta_recent = get_release_group_metadata(mbids_recent, inc="artist tag release", chunk_size=20)

    # NUEVO: Obtener señales globales frescas para no depender de la memoria de celdas anteriores
    print("Obteniendo señales sitewide globales y de artistas...")
    payload_sw = get_sitewide_top_release_groups(stats_range="month", count=300, offset=0)
    df_lb_sitewide_recent = parse_top_release_groups(payload_sw)

    payload_art = get_sitewide_artist_activity(stats_range="month") # Cambiado a month para ser más relevante a 30 días
    df_lb_artist_recent = parse_sitewide_artist_activity(payload_art)

    # 4. Hacer el cruce (merge) de la data
    print("Cruzando datos y consolidando el dataframe...")
    df_release_groups_recent_lb = (
        rg_recent
        .merge(
            df_lb_pop_recent,
            left_on="release_group_id",
            right_on="release_group_mbid",
            how="left"
        )
        .merge(
            df_lb_meta_recent[["release_group_mbid", "artist_name_meta", "tags_meta"]],
            left_on="release_group_id",
            right_on="release_group_mbid",
            how="left",
            suffixes=("", "_meta")
        )
        .merge(
            df_lb_sitewide_recent[["release_group_mbid", "listen_count"]],
            left_on="release_group_id",
            right_on="release_group_mbid",
            how="left",
            suffixes=("", "_sitewide")
        )
        .rename(columns={"listen_count": "lb_sitewide_listen_count_month"})
        .merge(
            df_lb_artist_recent[["artist_name", "artist_listen_count"]],
            on="artist_name",
            how="left"
        )
        .rename(columns={"artist_listen_count": "artist_catalog_strength"})
    )

    # 5. Limpiar columnas duplicadas que resultan de los merges
    cols_to_drop = ["release_group_mbid", "release_group_mbid_meta", "release_group_mbid_sitewide"]
    cols_to_drop = [c for c in cols_to_drop if c in df_release_groups_recent_lb.columns]
    df_release_groups_recent_lb = df_release_groups_recent_lb.drop(columns=cols_to_drop, errors="ignore")
        
    print("\n*** Enriquecimiento con ListenBrainz completado ***")
    print(f"Dimensiones del dataset: {df_release_groups_recent_lb.shape}")
    print(df_release_groups_recent_lb.head())

    # 1. Extraer features de Last.fm fila por fila
    print("Iniciando extracción de Last.fm. Esto puede tardar varios minutos...")
    # (Nota: Si ya tienes df_lastfm_features_recent en memoria por una corrida anterior y no quieres esperar, puedes comentar esta función)
    df_lastfm_features_recent = enrich_release_groups_with_lastfm(
        df_release_groups=df_release_groups_recent_lb,
        max_rows=None # Procesamos todo el dataset reciente
    )

    print("\nExtracción completada. Generando señales de charts por país...")
    # NUEVO: Obtener señales geográficas frescas de Last.fm (no de memoria)
    countries = ["United States", "United Kingdom", "Germany", "France", "Spain", "Mexico"]
    df_country_tracks_recent = build_country_track_signal(countries=countries, page=1, limit=100)

    # 2. Agregar señales geográficas al dataset reciente
    df_country_signal_recent = add_single_chart_signals(
        df_release_groups=df_release_groups_recent_lb,
        df_country_tracks=df_country_tracks_recent
    )

    print("Señales generadas exitosamente.")

    # 3. Fusión Maestra (MusicBrainz + ListenBrainz + Last.fm)
    # Primero renombramos las columnas de popularidad de ListenBrainz para que coincidan con la fórmula
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

    # 4. Preparación de variables numéricas y fechas
    # Corrección de fechas parciales de MusicBrainz (ej. "2021" -> "2021-01-01")
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
            df_master_recent[f"log1p_{c}"] = df_master_recent[c].fillna(0).apply(lambda x: np.log1p(x))
        else:
            df_master_recent[f"log1p_{c}"] = 0.0

    # 5. Cálculo del Trend Score v0
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

    # Ordenar por el score y limpiar
    df_master_recent = df_master_recent.sort_values("trend_score_v0", ascending=False).reset_index(drop=True)

    # Guardar resultado final
    df_master_recent.to_csv("music_master_recent_final.csv", index=False)
    print(f"\n¡Proceso completado! Dimensiones del Master reciente Enriquecido: {df_master_recent.shape}")
    print("Archivo guardado como: 'music_master_recent_final.csv'")

    print(df_master_recent[[
        "release_group_id", "artist_name", "title", "primary_type",
        "first_release_date", "lastfm_listeners", "lb_total_listen_count",
        "trend_score_v0"
    ]].head())


    # ============================================================
    # Diagnóstico de Fechas Nulas (NaT)
    # ============================================================

    # Tomamos los datos crudos recién salidos de la API de MusicBrainz
    df_diagnostic = df_release_groups_master_recent[['release_group_id', 'artist_name', 'title', 'first_release_date']].copy()

    # Simulamos la conversión que hace el master
    df_diagnostic['first_release_date_parsed'] = pd.to_datetime(df_diagnostic['first_release_date'], errors='coerce')

    # Contamos cuántos vienen nulos originalmente y cuántos quedan como NaT
    original_missing = df_diagnostic['first_release_date'].isna().sum()
    parsed_nat = df_diagnostic['first_release_date_parsed'].isna().sum()

    print(f"Total de registros recientes: {len(df_diagnostic)}")
    print(f"Fechas que ya venían vacías desde MusicBrainz: {original_missing}")
    print(f"Total de NaT tras la conversión: {parsed_nat}")
    print("\nEjemplos de registros con NaT (fechas ausentes):")

    # Mostramos una muestra de los registros defectuosos
    print(df_diagnostic[df_diagnostic['first_release_date_parsed'].isna()].head(10))

    # ============================================================
    # Limpieza de Duplicados (Un registro por Lanzamiento)
    # ============================================================

    # Eliminamos duplicados basados en el ID único del grupo de lanzamiento.
    # Como ya ordenamos por trend_score_v0 descendente, keep='first' mantiene el mejor.
    df_master_recent_clean = df_master_recent.drop_duplicates(
        subset=["release_group_id"],
        keep="first"
    ).reset_index(drop=True)

    # Sobrescribimos el CSV final con los datos limpios
    df_master_recent_clean.to_csv("music_master_recent_final.csv", index=False)

    print(f"Dimensiones ANTES de limpiar: {df_master_recent.shape}")
    print(f"Dimensiones DESPUÉS de limpiar: {df_master_recent_clean.shape}")
    print("\nArchivo 'music_master_recent_final.csv' actualizado con éxito.\n")

    # Mostramos los primeros resultados para confirmar
    print(df_master_recent_clean[[
        "release_group_id", "artist_name", "title",
        "primary_type", "first_release_date", "trend_score_v0"
    ]].head(10))

    #Guardamos el csv limpio
    df_master_recent_clean.to_csv("music_master_recent_final.csv", index=False)
    print("Archivo 'music_master_recent_final.csv' guardado con los datos limpios y sin duplicados.")

if __name__ == "__main__":
    main()