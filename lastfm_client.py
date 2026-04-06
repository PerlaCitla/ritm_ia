# ============================================================
# CELDA 1 — Imports y configuración
# ============================================================
import os
import time
import requests
import pandas as pd
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv(override=True)
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

# Define la URL base para la API de Last.fm.
BASE_URL_LASTFM = "https://ws.audioscrobbler.com/2.0/"



# Crea una sesión de requests para Last.fm para mantener la configuración entre peticiones.
lastfm_session = requests.Session()
# Configura las cabeceras de la sesión para aceptar respuestas JSON
# y establecer un User-Agent específico para la aplicación.
lastfm_session.headers.update({
    "Accept": "application/json",
    "User-Agent": "radar-lanzamientos-diplomado/0.1.0 (data.citla@gmail.com) Last.fm Client"
})

# Define el intervalo mínimo entre peticiones para respetar el rate limit de la API de Last.fm.
MIN_INTERVAL = 0.25  # pausa suave entre requests
# Variable global para registrar el momento de la última petición, utilizada para el rate limiting.
_last_request_ts = 0.0

# ============================================================
# CELDA 2 — Cliente Last.fm
# ============================================================

def _sleep_if_needed(min_interval: float = MIN_INTERVAL):
    """
    Espera si la última petición fue hace muy poco tiempo para respetar el rate limit de Last.fm.
    Esta función asegura que no se exceda el número permitido de peticiones por unidad de tiempo
    a la API de Last.fm, tal como lo define MIN_INTERVAL.
    """
    global _last_request_ts
    elapsed = time.time() - _last_request_ts
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

def ensure_list(x):
    """
    Asegura que un valor sea una lista. Si es None, devuelve una lista vacía.
    Si ya es una lista, la devuelve tal cual. Si es otro tipo, lo envuelve en una lista.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def safe_int(x):
    """
    Intenta convertir un valor a entero de forma segura. Si falla la conversión,
    devuelve None en lugar de lanzar una excepción.
    """
    try:
        return int(x)
    except (TypeError, ValueError):
        return None

def lastfm_get(method: str, params: Optional[dict] = None, retries: int = 5) -> dict:
    """
    Realiza una petición GET a la API de Last.fm.
    Maneja:
    - Errores HTTP (ej. 429 Too Many Requests)
    - Errores JSON específicos de Last.fm (ej. código 29 para rate limit)

    Args:
        method (str): El método de la API de Last.fm a llamar (ej. "chart.getTopTracks").
        params (Optional[dict]): Parámetros adicionales para la petición.
        retries (int): Número máximo de intentos en caso de errores recuperables.

    Returns:
        dict: La respuesta de la API de Last.fm en formato JSON.

    Raises:
        RuntimeError: Si la petición falla después de todos los reintentos o por un error no recuperable.
    """
    global _last_request_ts

    params = dict(params or {})
    params.update({
        "method": method,
        "api_key": LASTFM_API_KEY,
        "format": "json",
    })

    last_error = None

    for attempt in range(retries):
        _sleep_if_needed()
        response = lastfm_session.get(BASE_URL_LASTFM, params=params, timeout=60) # Usa lastfm_session y BASE_URL_LASTFM
        _last_request_ts = time.time()

        # Algunos errores de Last.fm vienen como JSON con status 200
        try:
            data = response.json()
        except Exception:
            data = None

        if response.status_code == 200 and isinstance(data, dict) and "error" not in data:
            return data

        # Error lógico de Last.fm
        if isinstance(data, dict) and "error" in data:
            error_code = data.get("error")
            error_msg = data.get("message", "Error desconocido")
            last_error = f"Last.fm error {error_code}: {error_msg}"

            # Rate limit o errores temporales (códigos específicos de Last.fm)
            if error_code in (11, 16, 29):
                wait_s = 2 ** attempt
                print(f"[retry] {last_error}. Esperando {wait_s}s...")
                time.sleep(wait_s)
                continue

            raise RuntimeError(last_error)

        # Errores HTTP de rate limit / temporales
        if response.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** attempt
            print(f"[retry] HTTP {response.status_code} en {method}. Esperando {wait}s...")
            time.sleep(wait)
            continue

        raise RuntimeError(
            f"Error HTTP {response.status_code} en método {method}\n"
            f"Params: {params}\n"
            f"Respuesta: {response.text[:500]}"
        )

    raise RuntimeError(last_error or f"No se pudo completar {method} tras {retries} intentos.")

# ============================================================
# CELDA 3 — Global top tracks
# ============================================================

def get_chart_top_tracks(page: int = 1, limit: int = 100) -> dict:
    """
    Obtiene las pistas más populares a nivel global de la API de Last.fm.

    Args:
        page (int): El número de página de resultados a solicitar.
        limit (int): El número máximo de pistas a devolver por página.

    Returns:
        dict: La respuesta cruda de la API de Last.fm con las pistas principales.
    """
    return lastfm_get(
        "chart.getTopTracks",
        params={
            "page": page,
            "limit": limit
        }
    )

def parse_chart_top_tracks(payload: dict) -> pd.DataFrame:
    """
    Parsea la respuesta de la API de Last.fm para las pistas globales más populares
    en un DataFrame de pandas.

    Args:
        payload (dict): El diccionario de respuesta de la API que contiene los datos de las pistas.

    Returns:
        pd.DataFrame: Un DataFrame con las pistas principales parseadas y sus metadatos.
    """
    tracks = ensure_list(payload.get("tracks", {}).get("track"))
    meta = payload.get("tracks", {})

    rows = []
    for item in tracks:
        artist = item.get("artist", {}) if isinstance(item.get("artist"), dict) else {}
        attrs = item.get("@attr", {}) if isinstance(item.get("@attr"), dict) else {}

        rows.append({
            "scope": "global",
            "country": None,
            "page": safe_int(meta.get("@attr", {}).get("page")) if isinstance(meta.get("@attr"), dict) else None,
            "per_page": safe_int(meta.get("@attr", {}).get("perPage")) if isinstance(meta.get("@attr"), dict) else None,
            "total_pages": safe_int(meta.get("@attr", {}).get("totalPages")) if isinstance(meta.get("@attr"), dict) else None,
            "total": safe_int(meta.get("@attr", {}).get("total")) if isinstance(meta.get("@attr"), dict) else None,
            "rank": safe_int(attrs.get("rank")),
            "track_name": item.get("name"),
            "track_mbid": item.get("mbid"),
            "artist_name": artist.get("name"),
            "artist_mbid": artist.get("mbid"),
            "playcount": safe_int(item.get("playcount")),
            "listeners": safe_int(item.get("listeners")),
            "url": item.get("url"),
            "streamable": item.get("streamable"),
        })

    return pd.DataFrame(rows)

# ============================================================
# CELDA 5 — Top tracks por país
# ============================================================

def get_geo_top_tracks(country: str, page: int = 1, limit: int = 100, location: Optional[str] = None) -> dict:
    """
    Obtiene las pistas más populares en un país específico de la API de Last.fm.
    Permite filtrar por país, paginación y, opcionalmente, por una ubicación dentro del país.

    Args:
        country (str): El nombre del país del que se quieren obtener las pistas.
        page (int): El número de página de resultados a solicitar.
        limit (int): El número máximo de pistas a devolver por página.
        location (Optional[str]): Una ubicación específica dentro del país (ej. una ciudad).

    Returns:
        dict: La respuesta cruda de la API de Last.fm con las pistas principales para la región.
    """
    params = {
        "country": country,
        "page": page,
        "limit": limit
    }
    if location:
        params["location"] = location

    payload = lastfm_get("geo.getTopTracks", params=params)
    print(f"DEBUG: Payload for {country}: {payload}") # Added debug print
    return payload

def parse_geo_top_tracks(payload: dict) -> pd.DataFrame:
    """
    Parsea la respuesta de la API de Last.fm para las pistas principales por país
    en un DataFrame de pandas.

    Args:
        payload (dict): El diccionario de respuesta de la API que contiene los datos de las pistas.

    Returns:
        pd.DataFrame: Un DataFrame con las pistas principales por país parseadas y sus metadatos.
    """
    # Corrected: The key is 'tracks', not 'toptracks'
    tracks_data = payload.get("tracks", {})
    tracks = ensure_list(tracks_data.get("track"))
    attr = tracks_data.get("@attr", {}) if isinstance(tracks_data.get("@attr"), dict) else {}

    rows = []
    for item in tracks:
        artist = item.get("artist", {}) if isinstance(item.get("artist"), dict) else {}
        attrs = item.get("@attr", {}) if isinstance(item.get("@attr"), dict) else {}

        rows.append({
            "scope": "country",
            "country": attr.get("country"),
            "page": safe_int(attr.get("page")),
            "per_page": safe_int(attr.get("perPage")),
            "total_pages": safe_int(attr.get("totalPages")),
            "total": safe_int(attr.get("total")),
            "rank": safe_int(attrs.get("rank")),
            "track_name": item.get("name"),
            "track_mbid": item.get("mbid"),
            "artist_name": artist.get("name"),
            "artist_mbid": artist.get("mbid"),
            "playcount": safe_int(item.get("playcount")),
            "listeners": safe_int(item.get("listeners")),
            "url": item.get("url"),
        })

    return pd.DataFrame(rows)

# ============================================================
# CELDA 7 — Top tags de artista
# ============================================================

def get_artist_top_tags(artist: Optional[str] = None, mbid: Optional[str] = None, autocorrect: int = 1) -> dict:
    """
    Obtiene los tags más populares asociados a un artista desde la API de Last.fm.
    Requiere el nombre del artista o su MusicBrainz ID (MBID).

    Args:
        artist (Optional[str]): El nombre del artista. Si se proporciona, se usa para la búsqueda.
        mbid (Optional[str]): El MusicBrainz ID del artista. Si se proporciona, tiene preferencia sobre `artist`.
        autocorrect (int): Si es 1, Last.fm intentará corregir errores tipográficos en el nombre del artista.

    Returns:
        dict: La respuesta cruda de la API de Last.fm con los tags principales.

    Raises:
        ValueError: Si no se proporciona ni el nombre del artista ni su MBID.
    """
    if not artist and not mbid:
        raise ValueError("Debes pasar artist o mbid.") # Lanza un error si faltan los parámetros requeridos.

    params = {"autocorrect": autocorrect}
    if artist:
        params["artist"] = artist
    if mbid:
        params["mbid"] = mbid

    return lastfm_get("artist.getTopTags", params=params)

def parse_artist_top_tags(payload: dict, artist_name_input: Optional[str] = None, artist_mbid_input: Optional[str] = None) -> pd.DataFrame:
    """
    Parsea la respuesta de la API de Last.fm para los tags principales de un artista
    en un DataFrame de pandas.

    Args:
        payload (dict): El diccionario de respuesta de la API que contiene los datos de los tags.
        artist_name_input (Optional[str]): El nombre del artista original usado en la consulta.
        artist_mbid_input (Optional[str]): El MBID del artista original usado en la consulta.

    Returns:
        pd.DataFrame: Un DataFrame con los tags principales parseados y sus metadatos.
    """
    toptags = payload.get("toptags", {}) # Extrae la sección 'toptags' del payload.
    tags = ensure_list(toptags.get("tag")) # Asegura que la lista de tags sea un iterable.

    rows = [] # Lista para almacenar los diccionarios de filas para el DataFrame.
    for tag in tags:
        rows.append({
            "artist_name_input": artist_name_input,       # Nombre del artista proporcionado en la entrada.
            "artist_mbid_input": artist_mbid_input,       # MBID del artista proporcionado en la entrada.
            "artist_name_response": toptags.get("@attr", {}).get("artist") if isinstance(toptags.get("@attr"), dict) else None, # Nombre del artista en la respuesta de Last.fm.
            "tag_name": tag.get("name"),                 # Nombre del tag.
            "tag_url": tag.get("url"),                   # URL del tag en Last.fm.
            "tag_count": safe_int(tag.get("count")),     # Conteo de veces que el tag ha sido usado (popularidad).
        })

    return pd.DataFrame(rows) # Retorna el DataFrame construido.


# ============================================================
# CELDA 9 — Album info
# ============================================================

def get_album_info(
    artist: Optional[str] = None,
    album: Optional[str] = None,
    mbid: Optional[str] = None,
    autocorrect: int = 1
) -> dict:
    """
    Obtiene información detallada de un álbum desde la API de Last.fm.
    Requiere el MusicBrainz ID (MBID) del álbum o el nombre del artista y el título del álbum.

    Args:
        artist (Optional[str]): El nombre del artista del álbum.
        album (Optional[str]): El título del álbum.
        mbid (Optional[str]): El MusicBrainz ID del álbum. Si se proporciona, tiene preferencia.
        autocorrect (int): Si es 1, Last.fm intentará corregir errores tipográficos en el nombre del artista o álbum.

    Returns:
        dict: La respuesta cruda de la API de Last.fm con la información del álbum.

    Raises:
        ValueError: Si no se proporciona ni el MBID ni el par artista+álbum.
    """
    if not mbid and not (artist and album):
        raise ValueError("Debes pasar mbid o bien artist + album.")

    params = {"autocorrect": autocorrect}
    if mbid:
        params["mbid"] = mbid
    else:
        params["artist"] = artist
        params["album"] = album

    return lastfm_get("album.getInfo", params=params)

def parse_album_info(payload: dict, artist_input: Optional[str] = None, album_input: Optional[str] = None) -> pd.DataFrame:
    """
    Parsea la respuesta de la API de Last.fm con la información de un álbum
    en un DataFrame de pandas.

    Args:
        payload (dict): El diccionario de respuesta de la API que contiene los datos del álbum.
        artist_input (Optional[str]): El nombre del artista original usado en la consulta.
        album_input (Optional[str]): El nombre del álbum original usado en la consulta.

    Returns:
        pd.DataFrame: Un DataFrame con la información detallada del álbum, incluyendo tags y pistas.
    """
    album = payload.get("album", {})
    tags = ensure_list(album.get("tags", {}).get("tag")) if isinstance(album.get("tags"), dict) else []
    tracks = ensure_list(album.get("tracks", {}).get("track")) if isinstance(album.get("tracks"), dict) else []

    tag_names = [t.get("name") for t in tags if isinstance(t, dict)]
    track_names = [t.get("name") for t in tracks if isinstance(t, dict)]

    rows = [{
        "artist_input": artist_input, # Nombre del artista proporcionado en la entrada.
        "album_input": album_input,   # Nombre del álbum proporcionado en la entrada.
        "album_name": album.get("name"),             # Nombre del álbum en la respuesta.
        "artist_name": album.get("artist"),           # Nombre del artista del álbum en la respuesta.
        "album_mbid": album.get("mbid"),             # MBID del álbum.
        "url": album.get("url"),                     # URL del álbum en Last.fm.
        "releasedate": album.get("releasedate"),       # Fecha de lanzamiento del álbum.
        "listeners": safe_int(album.get("listeners")), # Número de oyentes.
        "playcount": safe_int(album.get("playcount")),
        "tags": "|".join([x for x in tag_names if x]) if tag_names else None, # Tags del álbum separados por '|'.
        "track_count": len(track_names) if track_names else None, # Conteo de pistas del álbum.
        "tracks": "|".join([x for x in track_names if x]) if track_names else None, # Nombres de las pistas separadas por '|'.
    }]

    return pd.DataFrame(rows)


# ============================================================
# CELDA 11 — Track info
# ============================================================

def get_track_info(
    artist: Optional[str] = None,
    track: Optional[str] = None,
    mbid: Optional[str] = None,
    autocorrect: int = 1
) -> dict:
    """
    Obtiene información detallada de una pista desde la API de Last.fm.
    Requiere el MusicBrainz ID (MBID) de la pista o el nombre del artista y el título de la pista.

    Args:
        artist (Optional[str]): El nombre del artista de la pista.
        track (Optional[str]): El título de la pista.
        mbid (Optional[str]): El MusicBrainz ID de la pista. Si se proporciona, tiene preferencia.
        autocorrect (int): Si es 1, Last.fm intentará corregir errores tipográficos en el nombre del artista o pista.

    Returns:
        dict: La respuesta cruda de la API de Last.fm con la información de la pista.

    Raises:
        ValueError: Si no se proporciona ni el MBID ni el par artista+pista.
    """
    if not mbid and not (artist and track):
        raise ValueError("Debes pasar mbid o bien artist + track.")

    params = {"autocorrect": autocorrect}
    if mbid:
        params["mbid"] = mbid
    else:
        params["artist"] = artist
        params["track"] = track

    return lastfm_get("track.getInfo", params=params)

def parse_track_info(payload: dict, artist_input: Optional[str] = None, track_input: Optional[str] = None) -> pd.DataFrame:
    """
    Parsea la respuesta de la API de Last.fm con la información de una pista
    en un DataFrame de pandas.

    Args:
        payload (dict): El diccionario de respuesta de la API que contiene los datos de la pista.
        artist_input (Optional[str]): El nombre del artista original usado en la consulta.
        track_input (Optional[str]): El nombre de la pista original usado en la consulta.

    Returns:
        pd.DataFrame: Un DataFrame con la información detallada de la pista, incluyendo tags y metadatos.
    """
    track = payload.get("track", {})
    artist = track.get("artist", {}) if isinstance(track.get("artist"), dict) else {}
    album = track.get("album", {}) if isinstance(track.get("album"), dict) else {}
    tags = ensure_list(track.get("toptags", {}).get("tag")) if isinstance(track.get("toptags"), dict) else []

    tag_names = [t.get("name") for t in tags if isinstance(t, dict)]

    rows = [{
        "artist_input": artist_input, # Nombre del artista proporcionado en la entrada.
        "track_input": track_input,   # Nombre de la pista proporcionado en la entrada.
        "track_name": track.get("name"),             # Nombre de la pista en la respuesta.
        "track_mbid": track.get("mbid"),             # MBID de la pista.
        "artist_name": artist.get("name"),           # Nombre del artista de la pista.
        "artist_mbid": artist.get("mbid"),           # MBID del artista.
        "album_name": album.get("title"),            # Título del álbum asociado a la pista.
        "album_mbid": album.get("mbid"),             # MBID del álbum asociado.
        "url": track.get("url"),                     # URL de la pista en Last.fm.
        "duration_ms": safe_int(track.get("duration")), # Duración de la pista en milisegundos.
        "listeners": safe_int(track.get("listeners")), # Número de oyentes de la pista.
        "playcount": safe_int(track.get("playcount")), # Conteo de reproducciones de la pista.
        "tags": "|".join([x for x in tag_names if x]) if tag_names else None, # Tags de la pista separados por '|'.
    }]

    return pd.DataFrame(rows)


# ============================================================
# CELDA 13 — Enriquecer release groups con Last.fm
# ============================================================

def enrich_release_group_row_with_lastfm(row: pd.Series) -> dict:
    """
    Enriquece una fila de un DataFrame de 'release groups' de MusicBrainz con datos de Last.fm.
    Dependiendo del tipo primario de lanzamiento (álbum, EP, single), busca información
    relevante en Last.fm (oyentes, reproducciones, tags) utilizando las funciones get_album_info
    o get_track_info.

    Args:
        row (pd.Series): Una fila (representando un 'release group') del DataFrame de MusicBrainz.

    Returns:
        dict: Un diccionario con los datos originales del 'release group' y los campos enriquecidos
              de Last.fm, incluyendo posibles errores si la consulta falla.
    """
    artist_name = row.get("artist_name")
    title = row.get("title")
    primary_type = str(row.get("primary_type") or "").lower()

    result = {
        "release_group_id": row.get("release_group_id"),
        "artist_name": artist_name,
        "title": title,
        "primary_type": primary_type,
        "lastfm_source_used": None,
        "lastfm_listeners": None,
        "lastfm_playcount": None,
        "lastfm_tags_item": None,
        "lastfm_album_name": None,
        "lastfm_track_name": None,
        "lastfm_url": None,
    }

    try:
        if primary_type in ("album", "ep"):
            # Si es un álbum o EP, busca información del álbum en Last.fm
            payload = get_album_info(artist=artist_name, album=title, autocorrect=1)
            df_tmp = parse_album_info(payload, artist_input=artist_name, album_input=title)
            if not df_tmp.empty:
                rec = df_tmp.iloc[0].to_dict()
                result.update({
                    "lastfm_source_used": "album.getInfo",
                    "lastfm_listeners": rec.get("listeners"),
                    "lastfm_playcount": rec.get("playcount"),
                    "lastfm_tags_item": rec.get("tags"),
                    "lastfm_album_name": rec.get("album_name"),
                    "lastfm_url": rec.get("url"),
                })

        elif primary_type == "single":
            # Si es un single, busca información de la pista en Last.fm
            payload = get_track_info(artist=artist_name, track=title, autocorrect=1)
            df_tmp = parse_track_info(payload, artist_input=artist_name, track_input=title)
            if not df_tmp.empty:
                rec = df_tmp.iloc[0].to_dict()
                result.update({
                    "lastfm_source_used": "track.getInfo",
                    "lastfm_listeners": rec.get("listeners"),
                    "lastfm_playcount": rec.get("playcount"),
                    "lastfm_tags_item": rec.get("tags"),
                    "lastfm_track_name": rec.get("track_name"),
                    "lastfm_url": rec.get("url"),
                })

        else:
            # Fallback: intentar buscar como álbum si el tipo primario no es claro
            try:
                payload = get_album_info(artist=artist_name, album=title, autocorrect=1)
                df_tmp = parse_album_info(payload, artist_input=artist_name, album_input=title)
                if not df_tmp.empty:
                    rec = df_tmp.iloc[0].to_dict()
                    result.update({
                        "lastfm_source_used": "album.getInfo_fallback",
                        "lastfm_listeners": rec.get("listeners"),
                        "lastfm_playcount": rec.get("playcount"),
                        "lastfm_tags_item": rec.get("tags"),
                        "lastfm_album_name": rec.get("album_name"),
                        "lastfm_url": rec.get("url"),
                    })
            except Exception:
                pass

    except Exception as e:
        # Registra cualquier error que ocurra durante la consulta a Last.fm
        result["lastfm_error"] = str(e)

    return result

def enrich_release_groups_with_lastfm(df_release_groups: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Enriquece un DataFrame completo de 'release groups' de MusicBrainz con datos de Last.fm.
    Itera sobre cada fila del DataFrame y aplica la función `enrich_release_group_row_with_lastfm`.

    Args:
        df_release_groups (pd.DataFrame): DataFrame de 'release groups' de MusicBrainz.
        max_rows (Optional[int]): Número máximo de filas a procesar para pruebas o rendimiento.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con las filas enriquecidas con datos de Last.fm.
    """
    work_df = df_release_groups.copy()
    if max_rows is not None:
        work_df = work_df.head(max_rows).copy() # Limita el DataFrame si se especifica max_rows

    rows = []
    total = len(work_df)

    for i, (_, row) in enumerate(work_df.iterrows(), start=1):
        print(f"[{i}/{total}] {row.get('artist_name')} — {row.get('title')}")
        rows.append(enrich_release_group_row_with_lastfm(row))

    return pd.DataFrame(rows) # Retorna el DataFrame final con los datos enriquecidos.


# ============================================================
# CELDA 15 — Top tags por artista para el catálogo
# ============================================================

def get_artist_top_tags_flat(artist_name: str) -> pd.DataFrame:
    """
    Obtiene los tags principales de un artista de Last.fm y los devuelve en un DataFrame.
    Maneja excepciones para que la función sea robusta y devuelva un DataFrame vacío o
    con información de error si la consulta falla.

    Args:
        artist_name (str): El nombre del artista a consultar.

    Returns:
        pd.DataFrame: Un DataFrame con los tags principales del artista, o un DataFrame
                      con información de error si la consulta falla.
    """
    try:
        payload = get_artist_top_tags(artist=artist_name, autocorrect=1)
        df = parse_artist_top_tags(payload, artist_name_input=artist_name)
        return df
    except Exception as e:
        return pd.DataFrame([{
            "artist_name_input": artist_name,
            "artist_mbid_input": None,
            "artist_name_response": None,
            "tag_name": None,
            "tag_url": None,
            "tag_count": None,
            "error": str(e)
        }])

def collect_artist_tags_for_catalog(df_release_groups: pd.DataFrame, top_n_tags: int = 10) -> pd.DataFrame:
    """
    Recolecta los tags principales de Last.fm para todos los artistas únicos presentes
    en un DataFrame de 'release groups'. Procesa cada artista y combina los resultados.

    Args:
        df_release_groups (pd.DataFrame): DataFrame que contiene los 'release groups',
                                        incluyendo una columna 'artist_name'.
        top_n_tags (int): El número máximo de tags principales a recuperar por cada artista.

    Returns:
        pd.DataFrame: Un DataFrame consolidado con los tags principales para todos los artistas únicos.
    """
    unique_artists = (
        df_release_groups["artist_name"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    frames = []
    total = len(unique_artists)

    for i, artist_name in enumerate(unique_artists, start=1):
        print(f"[{i}/{total}] tags: {artist_name}")
        df_tags = get_artist_top_tags_flat(artist_name)
        if not df_tags.empty:
            # Ordena por 'tag_count' y toma los 'top_n_tags' más populares
            df_tags = df_tags.sort_values("tag_count", ascending=False, na_position="last").head(top_n_tags)
        frames.append(df_tags)

    # Concatena todos los DataFrames de tags y devuelve el resultado
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================
# CELDA 17 — Utilidades de matching para singles
# ============================================================

import re
import unicodedata

def normalize_text(s: Optional[str]) -> Optional[str]:
    """
    Normaliza una cadena de texto para facilitar la comparación.
    Convierte a minúsculas, elimina acentos y caracteres no alfanuméricos,
    y reduce múltiples espacios a uno solo.

    Args:
        s (Optional[str]): La cadena de texto a normalizar.

    Returns:
        Optional[str]: La cadena normalizada, o None si la entrada fue None.
    """
    if s is None:
        return None
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s

def build_country_track_signal(countries: List[str], page: int = 1, limit: int = 100) -> pd.DataFrame:
    """
    Construye un DataFrame de las pistas principales por país, obteniendo datos de Last.fm.
    Normaliza los nombres de artistas y pistas para su posterior matching.

    Args:
        countries (List[str]): Una lista de nombres de países para buscar pistas.
        page (int): El número de página de resultados a solicitar por país.
        limit (int): El número máximo de pistas a devolver por página por país.

    Returns:
        pd.DataFrame: Un DataFrame consolidado con las pistas principales por país, incluyendo
                      columnas de nombres normalizados.
    """
    frames = []

    for country in countries:
        payload = get_geo_top_tracks(country=country, page=page, limit=limit)
        df_country = parse_geo_top_tracks(payload)

        # Asegura que las columnas 'artist_name_norm' y 'track_name_norm' siempre estén presentes
        # incluso si df_country estaba vacío inicialmente. Inicialízalas como Series vacías de tipo string
        # si las columnas originales ('artist_name', 'track_name') no están presentes o df_country está vacío.
        if not df_country.empty and "artist_name" in df_country.columns:
            df_country["artist_name_norm"] = df_country["artist_name"].apply(normalize_text)
        else:
            df_country["artist_name_norm"] = pd.Series(dtype='str', index=df_country.index)

        if not df_country.empty and "track_name" in df_country.columns:
            df_country["track_name_norm"] = df_country["track_name"].apply(normalize_text)
        else:
            df_country["track_name_norm"] = pd.Series(dtype='str', index=df_country.index)

        frames.append(df_country)

    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        # Si no se añadió ningún frame (por ejemplo, la lista de países estaba vacía),
        # devuelve un DataFrame vacío con las columnas normalizadas esperadas.
        # Otras columnas que `parse_geo_top_tracks` devolvería también deben estar aquí para completitud.
        # Basado en `parse_geo_top_tracks`, las columnas serían:
        expected_geo_columns = [
            "scope", "country", "page", "per_page", "total_pages", "total", "rank",
            "track_name", "track_mbid", "artist_name", "artist_mbid", "playcount",
            "listeners", "url"
        ]
        all_columns = expected_geo_columns + ["artist_name_norm", "track_name_norm"]
        return pd.DataFrame(columns=all_columns)

def add_single_chart_signals(df_release_groups: pd.DataFrame, df_country_tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Añade señales de popularidad en las listas de éxitos por país a los singles en el DataFrame
    de 'release groups' de MusicBrainz. Realiza un matching basado en nombres normalizados
    de artista y título de pista.

    Args:
        df_release_groups (pd.DataFrame): DataFrame de 'release groups' de MusicBrainz.
        df_country_tracks (pd.DataFrame): DataFrame con las pistas principales por país, incluyendo
                                        nombres normalizados de artista y pista.

    Returns:
        pd.DataFrame: El DataFrame original de 'release groups' enriquecido con las señales
                      de las listas de éxitos por país (hits, rank, playcount, países).
    """
    df = df_release_groups.copy()

    df["artist_name_norm"] = df["artist_name"].apply(normalize_text)
    df["title_norm"] = df["title"].apply(normalize_text)
    df["primary_type_norm"] = df["primary_type"].astype(str).str.lower()

    singles = df[df["primary_type_norm"] == "single"].copy()

    print("DEBUG: df_country_tracks head after normalization in add_single_chart_signals:")
    print(df_country_tracks[['artist_name_norm', 'track_name_norm']].head())
    print("DEBUG: singles head with normalized columns:")
    print(singles[['artist_name', 'title', 'artist_name_norm', 'title_norm']].head())

    # Si df_country_tracks está vacío, no se pueden hacer coincidencias, así que se devuelve temprano con columnas de señal vacías
    if df_country_tracks.empty:
        df["lastfm_country_chart_hits"] = None
        df["lastfm_best_country_rank"] = None
        df["lastfm_total_country_playcount"] = None
        df["lastfm_countries"] = None
        return df

    merged = singles.merge(
        df_country_tracks,
        left_on=["artist_name_norm", "title_norm"],
        right_on=["artist_name_norm", "track_name_norm"],
        how="left",
        suffixes=("", "_lastfm")
    )

    print("DEBUG: Merged DataFrame head:")
    print(merged.head())

    # Solo procede con la agregación si 'merged' tiene datos
    if merged.empty:
        df["lastfm_country_chart_hits"] = None
        df["lastfm_best_country_rank"] = None
        df["lastfm_total_country_playcount"] = None
        df["lastfm_countries"] = None
        return df

    agg = (
        merged.groupby("release_group_id", dropna=False)
        .agg(
            lastfm_country_chart_hits=("country", lambda x: x.notna().sum()),
            lastfm_best_country_rank=("rank", "min"),
            lastfm_total_country_playcount=("playcount", "sum"),
            lastfm_countries=("country", lambda x: "|".join(sorted(set([v for v in x.dropna().astype(str)]))) if x.notna().any() else None),
        )
        .reset_index()
    )

    out = df_release_groups.merge(agg, on="release_group_id", how="left")
    return out
