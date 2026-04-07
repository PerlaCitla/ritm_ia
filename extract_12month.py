import calendar
import pandas as pd
import numpy as np

from  musicbrainz_client import collect_recent_musicbrainz_dataset
from lastfm_client import enrich_release_groups_with_lastfm,build_country_track_signal,add_single_chart_signals
from listenbrainz_client import get_release_group_popularity, get_release_group_metadata, get_sitewide_top_release_groups, parse_top_release_groups, get_sitewide_artist_activity, parse_sitewide_artist_activity





# Listas para almacenar los DataFrames de cada mes
all_release_groups_2021 = []
all_releases_2021 = []

year = 2021

for month in range(1, 13):
    # 1. Calcular el último día del mes (ej. 28, 30, 31)
    _, last_day = calendar.monthrange(year, month)

    # 2. Formatear fechas en ISO (YYYY-MM-DD)
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{last_day:02d}"

    print(f"\n{'='*50}")
    print(f"Extrayendo datos para: {start_date} a {end_date}")
    print(f"{'='*50}")

    # 3. Extraer 100 registros para ese mes en particular
    # Usamos max_search_pages=1 y limit=100 internamente en la API para traer los más relevantes
    df_rg_month, df_rel_month = collect_recent_musicbrainz_dataset(
        start_date=start_date,
        end_date=end_date,
        primary_types=("album", "single", "ep"),
        max_search_pages=1,
        max_release_groups=1
    )

    # 4. Guardar en las listas
    if not df_rg_month.empty:
        all_release_groups_2021.append(df_rg_month)
    if not df_rel_month.empty:
        all_releases_2021.append(df_rel_month)

# 5. Concatenar todo en un solo DataFrame maestro de MusicBrainz
df_release_groups_master_2021 = pd.concat(all_release_groups_2021, ignore_index=True)
df_releases_master_2021 = pd.concat(all_releases_2021, ignore_index=True)

print("\n*** EXTRACCIÓN COMPLETADA ***")
print(f"Total Release Groups 2021: {df_release_groups_master_2021.shape[0]} registros")
print(f"Total Releases (ediciones) 2021: {df_releases_master_2021.shape[0]} registros")

# Vista previa
print(df_release_groups_master_2021.head())


# ============================================================
# Enriquecimiento del Dataset 2021 con ListenBrainz
# ============================================================

# 1. Preparar la tabla principal y normalizar tipos
rg_2021 = df_release_groups_master_2021.copy()
rg_2021["release_group_id"] = rg_2021["release_group_id"].astype(str)
rg_2021["artist_name"] = rg_2021["artist_name"].astype(str).str.strip()

# 2. Extraer los MBIDs únicos para buscar en la API
mbids_2021 = rg_2021["release_group_id"].dropna().unique().tolist()
print(f"Consultando popularidad en ListenBrainz para {len(mbids_2021)} release groups...")

# 3. Obtener popularidad y metadatos (esperará automáticamente si toca el rate-limit)
df_lb_pop_2021 = get_release_group_popularity(mbids_2021, chunk_size=100)

print("Obteniendo metadatos (tags, artistas)...")
df_lb_meta_2021 = get_release_group_metadata(mbids_2021, inc="artist tag release", chunk_size=20)

# NUEVO: Obtener señales globales frescas para no depender de la memoria de celdas anteriores
print("Obteniendo señales sitewide globales y de artistas...")
payload_sw = get_sitewide_top_release_groups(stats_range="year", count=300, offset=0)
df_lb_sitewide_2021 = parse_top_release_groups(payload_sw)

payload_art = get_sitewide_artist_activity(stats_range="year")
df_lb_artist_2021 = parse_sitewide_artist_activity(payload_art)

# 4. Hacer el cruce (merge) de la data
print("Cruzando datos y consolidando el dataframe...")
df_release_groups_2021_lb = (
    rg_2021
    .merge(
        df_lb_pop_2021,
        left_on="release_group_id",
        right_on="release_group_mbid",
        how="left"
    )
    .merge(
        df_lb_meta_2021[["release_group_mbid", "artist_name_meta", "tags_meta"]],
        left_on="release_group_id",
        right_on="release_group_mbid",
        how="left",
        suffixes=("", "_meta")
    )
    .merge(
        df_lb_sitewide_2021[["release_group_mbid", "listen_count"]],
        left_on="release_group_id",
        right_on="release_group_mbid",
        how="left",
        suffixes=("", "_sitewide")
    )
    .rename(columns={"listen_count": "lb_sitewide_listen_count_month"})
    .merge(
        df_lb_artist_2021[["artist_name", "artist_listen_count"]],
        on="artist_name",
        how="left"
    )
    .rename(columns={"artist_listen_count": "artist_catalog_strength"})
)

# 5. Limpiar columnas duplicadas que resultan de los merges
cols_to_drop = ["release_group_mbid", "release_group_mbid_meta", "release_group_mbid_sitewide"]
cols_to_drop = [c for c in cols_to_drop if c in df_release_groups_2021_lb.columns]
df_release_groups_2021_lb = df_release_groups_2021_lb.drop(columns=cols_to_drop, errors="ignore")

print("\n*** Enriquecimiento con ListenBrainz completado ***")
print(f"Dimensiones del dataset: {df_release_groups_2021_lb.shape}")
print(df_release_groups_2021_lb.head())

# 1. Extraer features de Last.fm fila por fila
print("Iniciando extracción de Last.fm. Esto puede tardar varios minutos...")
# (Nota: Si ya tienes df_lastfm_features_2021 en memoria por una corrida anterior y no quieres esperar, puedes comentar esta función)
df_lastfm_features_2021 = enrich_release_groups_with_lastfm(
    df_release_groups=df_release_groups_2021_lb,
    max_rows=None # Procesamos todo el dataset de 2021
)

print("\nExtracción completada. Generando señales de charts por país...")
# NUEVO: Obtener señales geográficas frescas de Last.fm (no de memoria)
countries = ["United States", "United Kingdom", "Germany", "France", "Spain", "Mexico"]
df_country_tracks_2021 = build_country_track_signal(countries=countries, page=1, limit=100)

# 2. Agregar señales geográficas al dataset de 2021
df_country_signal_2021 = add_single_chart_signals(
    df_release_groups=df_release_groups_2021_lb,
    df_country_tracks=df_country_tracks_2021
)

print("Señales generadas exitosamente.")


# 3. Fusión Maestra (MusicBrainz + ListenBrainz + Last.fm)
# Primero renombramos las columnas de popularidad de ListenBrainz para que coincidan con la fórmula
df_base_2021 = df_release_groups_2021_lb.rename(columns={
    "total_listen_count": "lb_total_listen_count",
    "total_user_count": "lb_total_user_count"
})

df_master_2021 = (
    df_base_2021
    .merge(
        df_lastfm_features_2021[[
            "release_group_id", "lastfm_source_used", "lastfm_listeners",
            "lastfm_playcount", "lastfm_tags_item", "lastfm_album_name",
            "lastfm_track_name", "lastfm_url", "lastfm_error"
        ]],
        on="release_group_id",
        how="left"
    )
    .merge(
        df_country_signal_2021[[
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

df_master_2021["first_release_date"] = df_master_2021["first_release_date"].apply(fix_mb_date)
df_master_2021["first_release_date"] = pd.to_datetime(df_master_2021["first_release_date"], errors="coerce")
today = pd.Timestamp.today().normalize()
df_master_2021["days_since_release"] = (today - df_master_2021["first_release_date"]).dt.days

num_cols = ["lb_total_listen_count", "lb_total_user_count", "lb_sitewide_listen_count_month", "artist_catalog_strength"]
for c in num_cols:
    if c in df_master_2021.columns:
        df_master_2021[c] = pd.to_numeric(df_master_2021[c], errors="coerce")
        df_master_2021[f"log1p_{c}"] = df_master_2021[c].fillna(0).apply(lambda x: np.log1p(x))
    else:
        df_master_2021[f"log1p_{c}"] = 0.0

# 5. Cálculo del Trend Score v0
def minmax_safe(s):
    if s is None or s.empty: return pd.Series([])
    s = s.fillna(0)
    if s.max() == s.min():
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

df_master_2021["score_popularity"] = minmax_safe(df_master_2021.get("log1p_lb_total_listen_count", pd.Series([0]*len(df_master_2021))))
df_master_2021["score_users"] = minmax_safe(df_master_2021.get("log1p_lb_total_user_count", pd.Series([0]*len(df_master_2021))))
df_master_2021["score_sitewide"] = minmax_safe(df_master_2021.get("log1p_lb_sitewide_listen_count_month", pd.Series([0]*len(df_master_2021))))
df_master_2021["score_artist_strength"] = minmax_safe(df_master_2021.get("log1p_artist_catalog_strength", pd.Series([0]*len(df_master_2021))))
df_master_2021["score_recency"] = 1 - minmax_safe(df_master_2021["days_since_release"].clip(lower=0, upper=365))

# Ponderación del score
df_master_2021["trend_score_v0"] = (
    0.35 * df_master_2021["score_popularity"] +
    0.20 * df_master_2021["score_users"] +
    0.25 * df_master_2021["score_sitewide"] +
    0.10 * df_master_2021["score_artist_strength"] +
    0.10 * df_master_2021["score_recency"]
) * 100

# Ordenar por el score y limpiar
df_master_2021 = df_master_2021.sort_values("trend_score_v0", ascending=False).reset_index(drop=True)

# Guardar resultado final
# df_master_2021.to_csv("music_master_2021_final.csv", index=False)
print(f"\n¡Proceso completado! Dimensiones del Master 2021 Enriquecido: {df_master_2021.shape}")
# print("Archivo guardado como: 'music_master_2021_final.csv'")

print(df_master_2021[[
    "release_group_id", "artist_name", "title", "primary_type",
    "first_release_date", "lastfm_listeners", "lb_total_listen_count",
    "trend_score_v0"
]].head(15))

# ============================================================
# Limpieza de Duplicados (Un registro por Lanzamiento)
# ============================================================

# Eliminamos duplicados basados en el ID único del grupo de lanzamiento.
# Como ya ordenamos por trend_score_v0 descendente, keep='first' mantiene el mejor.
df_master_2021_clean = df_master_2021.drop_duplicates(
    subset=["release_group_id"],
    keep="first"
).reset_index(drop=True)

# Sobrescribimos el CSV final con los datos limpios
df_master_2021_clean.to_csv("music_master_2021_final.csv", index=False)

print(f"Dimensiones ANTES de limpiar: {df_master_2021.shape}")
print(f"Dimensiones DESPUÉS de limpiar: {df_master_2021_clean.shape}")
print("\nArchivo 'music_master_2021_final.csv' actualizado con éxito.\n")

# Mostramos los primeros resultados para confirmar
print(df_master_2021_clean[[
    "release_group_id", "artist_name", "title",
    "primary_type", "first_release_date", "trend_score_v0"
]].head(10))