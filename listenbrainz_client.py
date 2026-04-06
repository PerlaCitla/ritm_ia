# ============================================================
# CELDA 1 — Imports y configuración
# ============================================================

import time
import requests
import pandas as pd
from datetime import date, timedelta
from typing import Optional, List, Dict, Any

from musicbrainz_client import search_release_groups
from musicbrainz_client import get_release_group_details


BASE_URL = "https://api.listenbrainz.org" # Define la URL base para la API de ListenBrainz.

session = requests.Session() # Crea una sesión de requests para mantener la configuración entre peticiones.
session.headers.update({
    "Accept": "application/json" # Configura la cabecera para aceptar respuestas JSON.
})

# Define los rangos de tiempo permitidos para ciertas consultas a la API.
ALLOWED_RANGES = [
    "this_week", "this_month", "this_year",
    "week", "month", "quarter", "year",
    "half_yearly", "all_time"
]


# ============================================================
# CELDA 2 — Cliente público ListenBrainz
# ============================================================

def _sleep_from_ratelimit_headers(response: requests.Response, fallback_seconds: float = 2.0):
    """
    Espera si es necesario según las cabeceras de `rate-limit` de la respuesta de ListenBrainz.
    Si se ha alcanzado el límite de peticiones (código 429 o X-RateLimit-Remaining: 0),
    espera el tiempo indicado en 'X-RateLimit-Reset-In' o un tiempo de retroceso exponencial.

    Args:
        response (requests.Response): Objeto de respuesta de la petición HTTP.
        fallback_seconds (float): Tiempo de espera por defecto si las cabeceras no están disponibles o son inválidas.
    """
    reset_in = response.headers.get("X-RateLimit-Reset-In")
    remaining = response.headers.get("X-RateLimit-Remaining")

    try:
        remaining_val = int(remaining) if remaining is not None else None
    except ValueError:
        remaining_val = None

    try:
        reset_in_val = float(reset_in) if reset_in is not None else fallback_seconds
    except ValueError:
        reset_in_val = fallback_seconds

    if remaining_val == 0 or response.status_code == 429:
        wait_s = max(reset_in_val, fallback_seconds)
        print(f"[rate-limit] Esperando {wait_s:.1f}s...")
        time.sleep(wait_s)

def lb_get(endpoint: str, params: Optional[dict] = None, retries: int = 5) -> Optional[dict]:
    """
    Realiza una petición GET a la API pública de ListenBrainz, manejando rate limits y reintentos.

    Args:
        endpoint (str): El path específico de la API (ej. "/1/explore/fresh-releases/").
        params (Optional[dict]): Parámetros de la consulta GET.
        retries (int): Número máximo de intentos en caso de errores recuperables (rate limit, errores de servidor).

    Returns:
        Optional[dict]: La respuesta de la API en formato JSON, o None si el status es 204 (No Content).

    Raises:
        RuntimeError: Si la petición falla después de todos los reintentos o por un error no recuperable.
    """
    url = f"{BASE_URL}{endpoint}"
    params = params or {}

    last_response = None
    for attempt in range(retries):
        response = session.get(url, params=params, timeout=60)
        last_response = response

        if response.status_code == 200:
            return response.json()

        if response.status_code == 204:
            return None

        if response.status_code == 429:
            _sleep_from_ratelimit_headers(response, fallback_seconds=2 ** attempt)
            continue

        if response.status_code >= 500:
            wait_s = 2 ** attempt
            print(f"[retry] Error {response.status_code} en {endpoint}. Esperando {wait_s}s...")
            time.sleep(wait_s)
            continue

        raise RuntimeError(
            f"Error {response.status_code} en {url}\n"
            f"Params: {params}\n"
            f"Respuesta: {response.text[:500]}"
        )

    raise RuntimeError(
        f"No se pudo completar la petición tras {retries} intentos. "
        f"Último status: {last_response.status_code if last_response else 'sin respuesta'}"
    )

# Se usa para solicitar grandes envios de datos
def lb_post(endpoint: str, json_payload: dict, retries: int = 5) -> Optional[Any]:
    """
    Realiza una petición POST a la API pública de ListenBrainz, manejando rate limits y reintentos.

    Args:
        endpoint (str): El path específico de la API.
        json_payload (dict): El cuerpo de la petición POST en formato JSON.
        retries (int): Número máximo de intentos.

    Returns:
        Optional[Any]: La respuesta de la API en formato JSON, o None si el status es 204.

    Raises:
        RuntimeError: Si la petición falla después de todos los reintentos o por un error no recuperable.
    """
    url = f"{BASE_URL}{endpoint}"

    last_response = None
    for attempt in range(retries):
        response = session.post(url, json=json_payload, timeout=60)
        last_response = response

        if response.status_code == 200:
            return response.json()

        if response.status_code == 204:
            return None

        if response.status_code == 429:
            _sleep_from_ratelimit_headers(response, fallback_seconds=2 ** attempt)
            continue

        if response.status_code >= 500:
            wait_s = 2 ** attempt
            print(f"[retry] Error {response.status_code} en {endpoint}. Esperando {wait_s}s...")
            time.sleep(wait_s)
            continue

        raise RuntimeError(
            f"Error {response.status_code} en {url}\n"
            f"Payload: {json_payload}\n"
            f"Respuesta: {response.text[:500]}"
        )

    raise RuntimeError(
        f"No se pudo completar la petición tras {retries} intentos. "
        f"Último status: {last_response.status_code if last_response else 'sin respuesta'}"
    )

def list_to_pipe(values: List) -> Optional[str]:
    """
    Convierte una lista de valores en una cadena separada por '|', eliminando duplicados y nulos/vacíos.

    Args:
        values (List): Lista de elementos.

    Returns:
        Optional[str]: Cadena con los valores separados por '|' o None si la lista resultante está vacía.
    """
    if values is None:
        return None
    if not isinstance(values, list):
        return values
    clean = [str(v).strip() for v in values if v not in (None, "", [], {})]
    clean = list(dict.fromkeys(clean))
    return "|".join(clean) if clean else None

def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    """
    Divide una lista grande en sublistas (chunks) de un tamaño especificado.

    Args:
        items (List[str]): La lista a dividir.
        chunk_size (int): El tamaño máximo de cada sublista.

    Returns:
        List[List[str]]: Una lista de listas, donde cada sublista es un chunk de la original.
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


# ============================================================
# CELDA 3 — Fresh releases públicos
# ============================================================

def get_fresh_releases(
    release_date: Optional[str] = None,
    days: int = 30,
    sort: str = "release_date",
    past: bool = True,
    future: bool = False
) -> list:
    """
    Obtiene una lista de lanzamientos 'frescos' (fresh releases) de la API de ListenBrainz.
    Permite filtrar por fecha de lanzamiento, un rango de días alrededor de esa fecha y el orden.

    Args:
        release_date (Optional[str]): Fecha pivote para la búsqueda en formato YYYY-MM-DD. Si es None, usa la fecha actual.
        days (int): Número de días para buscar lanzamientos antes (y/o después) de la `release_date`.
        sort (str): Criterio de ordenación de los resultados (ej. "release_date").
        past (bool): Si es True, incluye lanzamientos anteriores a la fecha pivote.
        future (bool): Si es True, incluye lanzamientos posteriores a la fecha pivote.

    Returns:
        list: Una lista de diccionarios, donde cada diccionario representa un 'fresh release'.
    """
    # Si no se proporciona una fecha de lanzamiento, se usa la fecha actual.
    if release_date is None:
        release_date = date.today().isoformat()

    # Prepara los parámetros para la petición GET a la API de ListenBrainz.
    params = {
        "release_date": release_date,
        "days": days,
        "sort": sort,
        "past": str(past).lower(),   # Convierte el booleano a string "true" o "false".
        "future": str(future).lower(), # Convierte el booleano a string "true" o "false".
    }

    # Realiza la petición GET al endpoint de fresh releases.
    data = lb_get("/1/explore/fresh-releases/", params=params)

    # Si la respuesta no contiene datos, devuelve una lista vacía.
    if not data:
        return []

    # El endpoint generalmente devuelve una lista directamente.
    if isinstance(data, list):
        return data

    # Fallback por si la estructura de la respuesta cambia y los datos vienen envueltos en un 'payload'.
    if isinstance(data, dict) and "payload" in data:
        return data["payload"].get("releases", [])

    return []

def parse_fresh_releases(rows: list) -> pd.DataFrame:
    """
    Parsea una lista de diccionarios de 'fresh releases' de ListenBrainz a un DataFrame de pandas.
    Extrae y normaliza campos clave para facilitar el análisis.

    Args:
        rows (list): Lista de diccionarios de 'fresh releases' obtenidos de la API.

    Returns:
        pd.DataFrame: Un DataFrame con los lanzamientos frescos parseados y ordenados.
    """
    parsed = [] # Lista para almacenar los diccionarios de lanzamientos parseados.
    for item in rows:
        parsed.append({
            "artist_credit_name": item.get("artist_credit_name"), # Nombre del artista o crédito de artistas.
            "artist_mbids": list_to_pipe(item.get("artist_mbids", [])), # IDs de los artistas separados por '|'.
            "release_date": item.get("release_date"),               # Fecha de lanzamiento.
            "release_group_mbid": item.get("release_group_mbid"),   # MBID del 'release group'.
            "release_group_primary_type": item.get("release_group_primary_type"), # Tipo primario del 'release group'.
            "release_group_secondary_type": item.get("release_group_secondary_type"), # Tipo secundario del 'release group'.
            "release_mbid": item.get("release_mbid"),               # MBID de la 'release' específica.
            "release_name": item.get("release_name"),               # Nombre de la 'release'.
            "caa_id": item.get("caa_id"),                           # ID de Cover Art Archive.
            "caa_release_mbid": item.get("caa_release_mbid"),       # MBID de la 'release' en Cover Art Archive.
            "confidence": item.get("confidence"),                   # Nivel de confianza del lanzamiento.
            "release_tags": list_to_pipe(item.get("release_tags", [])) if isinstance(item.get("release_tags"), list) else item.get("release_tags"), # Tags de la 'release' separados por '|'.
        })
    df = pd.DataFrame(parsed) # Crea un DataFrame a partir de los datos parseados.
    # Si el DataFrame no está vacío, lo ordena por fecha, nombre de artista y nombre de lanzamiento.
    if not df.empty:
        df = df.sort_values(["release_date", "artist_credit_name", "release_name"], ascending=[False, True, True]).reset_index(drop=True)
    return df


# ============================================================
# CELDA 5 — Top release groups sitewide
# ============================================================

def get_sitewide_top_release_groups(
    stats_range: str = "month",
    count: int = 100,
    offset: int = 0
) -> Optional[dict]:
    """
    Obtiene los principales grupos de lanzamientos a nivel de sitio (sitewide) de ListenBrainz.
    Permite filtrar por rango de tiempo y paginación.

    Args:
        stats_range (str): El rango de tiempo para la consulta (ej. "month", "week").
                           Debe ser uno de los valores definidos en ALLOWED_RANGES.
        count (int): El número máximo de resultados a devolver por petición (límite).
        offset (int): El desplazamiento para la paginación de los resultados.

    Returns:
        Optional[dict]: Un diccionario con la respuesta de la API de ListenBrainz
                        conteniendo los grupos de lanzamientos principales, o None si la petición falla.

    Raises:
        ValueError: Si el rango de estadísticas proporcionado no es válido.
    """
    if stats_range not in ALLOWED_RANGES:
        raise ValueError(f"range inválido: {stats_range}")

    return lb_get(
        "/1/stats/sitewide/release-groups", # Endpoint para obtener grupos de lanzamientos sitewide
        params={
            "range": stats_range,
            "count": count,
            "offset": offset,
        }
    )
# de la respuesta dela api se regresa información, de la cual hay infromación de valor (payload)
def parse_top_release_groups(payload: dict) -> pd.DataFrame:
    """
    Parsea el payload de la respuesta de la API de ListenBrainz para los principales
    grupos de lanzamientos a nivel de sitio en un DataFrame de pandas.

    Args:
        payload (dict): El diccionario de respuesta de la API que contiene los datos de los grupos de lanzamientos.

    Returns:
        pd.DataFrame: Un DataFrame con los grupos de lanzamientos principales parseados y sus metadatos.
    """
    rg_list = payload.get("payload", {}).get("release_groups", []) # Extrae la lista de grupos de lanzamientos
    meta = payload.get("payload", {}) # Extrae los metadatos generales de la respuesta

    rows = [] # Lista para almacenar los diccionarios de filas para el DataFrame
    for item in rg_list:
        rows.append({
            "range": meta.get("range"), # Rango de tiempo de la consulta
            "from_ts": meta.get("from_ts"), # Timestamp de inicio del rango
            "to_ts": meta.get("to_ts"),     # Timestamp de fin del rango
            "last_updated": meta.get("last_updated"), # Última actualización de los datos
            "artist_name": item.get("artist_name"), # Nombre del artista
            "artist_mbids": list_to_pipe(item.get("artist_mbids", [])), # IDs de MusicBrainz del artista
            "release_group_name": item.get("release_group_name"), # Nombre del grupo de lanzamiento
            "release_group_mbid": item.get("release_group_mbid"), # ID de MusicBrainz del grupo de lanzamiento
            "listen_count": item.get("listen_count"), # Conteo de escuchas del grupo de lanzamiento
            "caa_id": item.get("caa_id"),             # ID de Cover Art Archive
            "caa_release_mbid": item.get("caa_release_mbid"), # ID de MusicBrainz de la release en Cover Art Archive

        })

    return pd.DataFrame(rows) # Retorna el DataFrame construido

def get_first_release_date_by_release_group_name(release_group_name: str) -> Optional[str]:
    """
    Busca un 'release group' por su nombre y devuelve su 'first-release-date'
    utilizando la API de MusicBrainz.

    Args:
        release_group_name (str): El nombre del 'release group' a buscar.

    Returns:
        Optional[str]: La fecha de primer lanzamiento del 'release group' en formato YYYY-MM-DD,
                       o None si no se encuentra el 'release group' o la fecha.
    """
    # Usar la función search_release_groups con una query específica para el nombre
    # La API de MusicBrainz permite buscar por 'releasegroup' en el endpoint 'release-group'.
    query = f'releasegroup:"{release_group_name}"'

    # Limitar la búsqueda a 1 resultado por página y solo 1 página,
    # asumiendo que el primer resultado será el más relevante o exacto.
    search_results = search_release_groups(query=query, limit=1, max_pages=1)

    if not search_results:
        print(f"No se encontró ningún release group con el nombre: {release_group_name}")
        return None

    # Tomar el primer resultado encontrado
    first_rg = search_results[0]
    rg_id = first_rg.get("id")

    if not rg_id:
        print(f"No se encontró ID para el release group: {release_group_name}")
        return None

    # Obtener los detalles completos del release group para asegurar la fecha de lanzamiento.
    # Aunque search_results puede contener 'first-release-date', obtener los detalles
    # asegura la consistencia y flexibilidad si se necesitan más campos en el futuro.
    rg_details = get_release_group_details(rg_id)

    return rg_details.get("first-release-date")


# ============================================================
# CELDA 7 — Sitewide artist activity
# ============================================================

def get_sitewide_artist_activity(stats_range: str = "month") -> Optional[dict]:
    """
    Obtiene la actividad de los artistas a nivel de sitio (sitewide) de ListenBrainz.
    Permite filtrar por un rango de tiempo específico.

    Args:
        stats_range (str): El rango de tiempo para la consulta (ej. "month", "week").
                           Debe ser uno de los valores definidos en ALLOWED_RANGES.

    Returns:
        Optional[dict]: Un diccionario con la respuesta de la API de ListenBrainz
                        conteniendo la actividad de los artistas, o None si la petición falla.

    Raises:
        ValueError: Si el rango de estadísticas proporcionado no es válido.
    """
    if stats_range not in ALLOWED_RANGES:
        raise ValueError(f"range inválido: {stats_range}")

    return lb_get(
        "/1/stats/sitewide/artist-activity", # Endpoint para obtener la actividad de artistas sitewide
        params={"range": stats_range}
    )

def parse_sitewide_artist_activity(payload: dict) -> pd.DataFrame:
    """
    Parsea el payload de la respuesta de la API de ListenBrainz para la actividad
    de artistas a nivel de sitio en un DataFrame de pandas.

    Args:
        payload (dict): El diccionario de respuesta de la API que contiene los datos de actividad.

    Returns:
        pd.DataFrame: Un DataFrame con la actividad de los artistas parseada.
    """
    rows = [] # Lista para almacenar los diccionarios de filas para el DataFrame
    artist_activity = payload.get("payload", {}).get("artist_activity", []) # Extrae la lista de actividad de artistas
    meta = payload.get("payload", {}) # Extrae los metadatos generales de la respuesta

    for artist in artist_activity:
        base = {
            "range": meta.get("range"), # Rango de tiempo de la consulta
            "from_ts": meta.get("from_ts"), # Timestamp de inicio del rango
            "to_ts": meta.get("to_ts"),     # Timestamp de fin del rango
            "last_updated": meta.get("last_updated"), # Última actualización de los datos
            "artist_name": artist.get("artist_name") or artist.get("name"), # Nombre del artista
            "artist_mbid": artist.get("artist_mbid"), # ID de MusicBrainz del artista
            "artist_listen_count": artist.get("listen_count"), # Conteo de escuchas del artista
        }

        albums = artist.get("albums", []) or [] # Extrae los álbumes asociados al artista
        if not albums:
            # Si no hay álbumes, añade una fila con campos nulos para el álbum
            rows.append({
                **base,
                "release_group_name": None,
                "release_group_mbid": None,
                "release_group_listen_count": None
            })
        else:
            # Si hay álbumes, añade una fila por cada álbum
            for album in albums:
                rows.append({
                    **base,
                    "release_group_name": album.get("name"), # Nombre del grupo de lanzamiento
                    "release_group_mbid": album.get("release_group_mbid"), # ID de MusicBrainz del grupo de lanzamiento
                    "release_group_listen_count": album.get("listen_count"), # Conteo de escuchas del grupo de lanzamiento
                })

    return pd.DataFrame(rows) # Retorna el DataFrame construido

# ============================================================
# CELDA 9 — Popularidad por release group MBID
# ============================================================

def get_release_group_popularity_notuse(
    release_mbids: List[str],
    chunk_size: int = 100
) -> pd.DataFrame:
    """
    Obtiene la popularidad de una lista de MBIDs de 'release groups' de ListenBrainz.
    Consulta la API en lotes (chunks) para manejar grandes cantidades de MBIDs.

    Args:
        release_mbids (List[str]): Una lista de MBIDs de 'release groups'.
        chunk_size (int): El número de MBIDs a enviar en cada petición a la API.

    Returns:
        pd.DataFrame: Un DataFrame con la popularidad de los 'release groups',
                      incluyendo el conteo total de escuchas y de usuarios.
                      Devuelve un DataFrame vacío si no hay datos.
    """
    all_rows = []

    # Elimina duplicados y valores nulos de la lista de MBIDs, manteniendo el orden.
    unique_mbids = [x for x in dict.fromkeys(release_mbids) if x]

    # Itera sobre los MBIDs en lotes para realizar las peticiones a la API.
    for chunk in chunk_list(unique_mbids, chunk_size):
        # Prepara el payload JSON para la petición POST.
        payload = {
            "release_mbids": chunk
        }
        # Realiza la petición POST a la API de popularidad de 'release groups'.
        data = lb_post("/1/popularity/release", json_payload=payload)

        # Si no hay datos en la respuesta del lote, continúa con el siguiente.
        if data is None:
            continue

        # Procesa cada fila de datos recibida y la añade a la lista general.
        for row in data:
            all_rows.append({
                "release_mbid": row.get("release_mbid"),
                "total_listen_count": row.get("total_listen_count"),
                "total_user_count": row.get("total_user_count"),
            })


    # Crea un DataFrame a partir de todos los datos recolectados.
    df = pd.DataFrame(all_rows)
    # Si el DataFrame no está vacío, elimina posibles duplicados por MBID y reinicia el índice.
    if not df.empty:
        df = df.drop_duplicates(subset=["release_mbid"]).reset_index(drop=True)
    return df

# Se modificó para que trajera info del release, para ver si hay algo de info, pero parece que no
# Antes era release_grupo

# ============================================================
# CELDA 9 — Popularidad por release group MBID
# ============================================================

def get_release_group_popularity(
    release_group_mbids: List[str],
    chunk_size: int = 100
) -> pd.DataFrame:
    all_rows = []

    unique_mbids = [x for x in dict.fromkeys(release_group_mbids) if x]

    for chunk in chunk_list(unique_mbids, chunk_size):
        payload = {
            "release_group_mbids": chunk
        }
        data = lb_post("/1/popularity/release-group", json_payload=payload)

        if data is None:
            continue

        for row in data:
            all_rows.append({
                "release_group_mbid": row.get("release_group_mbid"),
                "total_listen_count": row.get("total_listen_count"),
                "total_user_count": row.get("total_user_count"),
            })

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["release_group_mbid"]).reset_index(drop=True)
    return df

# ============================================================
# CELDA 11 — Metadata pública de release groups
# ============================================================

def get_release_group_metadata(
    release_group_mbids: List[str],
    inc: str = "artist tag release",
    chunk_size: int = 25
) -> pd.DataFrame:
    """
    Obtiene metadatos públicos detallados para una lista de MBIDs de 'release groups' de ListenBrainz.
    Incluye información como el artista y las etiquetas asociadas, y maneja la consulta en lotes.

    Args:
        release_group_mbids (List[str]): Una lista de MBIDs de 'release groups' para los que se desea obtener metadatos.
        inc (str): Parámetros a incluir en la respuesta (ej. "artist tag release").
        chunk_size (int): El número de MBIDs a enviar en cada petición a la API.

    Returns:
        pd.DataFrame: Un DataFrame con los metadatos de los 'release groups', incluyendo
                      el nombre del artista, etiquetas y los metadatos crudos.
    """
    all_rows = []

    # Elimina duplicados y valores nulos de la lista de MBIDs, manteniendo el orden.
    unique_mbids = [x for x in dict.fromkeys(release_group_mbids) if x]

    # Itera sobre los MBIDs en lotes para realizar las peticiones a la API.
    for chunk in chunk_list(unique_mbids, chunk_size):
        # Prepara los parámetros para la petición GET, incluyendo los MBIDs y los elementos a incluir.
        params = {
            "release_group_mbids": ",".join(chunk),
            "inc": inc,
        }
        # Realiza la petición GET a la API de metadatos de 'release group'.
        data = lb_get("/1/metadata/release_group/", params=params)

        # Si no hay datos en la respuesta del lote, continúa con el siguiente.
        if not data:
            continue

        # La respuesta viene indexada por MBID, se procesa cada entrada.
        for rg_mbid, meta in data.items():
            # Extrae la información del artista si está disponible.
            artist_block = meta.get("artist", {}) if isinstance(meta, dict) else {}
            artist_name = artist_block.get("name")

            # Extrae las etiquetas asociadas al 'release group'.
            tags = []
            tag_block = meta.get("tag", {}) if isinstance(meta, dict) else {}
            rg_tags = tag_block.get("release_group", []) if isinstance(tag_block, dict) else []
            for t in rg_tags:
                if isinstance(t, dict) and t.get("tag"):
                    tags.append(t["tag"])

            # Añade la fila con los metadatos extraídos.
            all_rows.append({
                "release_group_mbid": rg_mbid,
                "artist_name_meta": artist_name,
                "tags_meta": list_to_pipe(tags),
                "raw_metadata": meta,
            })

    return pd.DataFrame(all_rows)

# ============================================================
# CELDA 13 — Dataset maestro ListenBrainz público
# ============================================================

def build_public_listenbrainz_master(
    release_date: Optional[str] = None,
    days: int = 30,
    stats_range: str = "month"
) -> pd.DataFrame:
    """
    Construye un DataFrame maestro combinando diferentes fuentes de datos públicos de ListenBrainz:
    lanzamientos frescos, popularidad de release groups, metadatos y los principales release groups a nivel de sitio.

    Args:
        release_date (Optional[str]): Fecha pivote para la búsqueda de lanzamientos frescos en formato YYYY-MM-DD.
                                      Si es None, se usa la fecha actual.
        days (int): Número de días para buscar lanzamientos frescos antes de la `release_date`.
        stats_range (str): Rango de tiempo para obtener los 'sitewide top release groups' (ej. "month").

    Returns:
        pd.DataFrame: Un DataFrame maestro con información enriquecida de ListenBrainz.
                      Devuelve un DataFrame vacío si no se encuentran lanzamientos frescos.
    """
    # 1) Obtener lanzamientos frescos (fresh releases)
    fresh_rows = get_fresh_releases(
        release_date=release_date,
        days=days,
        sort="release_date",
        past=True,
        future=False
    )
    df_fresh_local = parse_fresh_releases(fresh_rows)

    if df_fresh_local.empty:
        return df_fresh_local

    # 2) Obtener la popularidad de esos release groups
    mbids = df_fresh_local["release_group_mbid"].dropna().tolist()
    df_pop = get_release_group_popularity(mbids, chunk_size=100)

    # 3) Obtener metadatos extra (artista, tags) para los release groups
    df_meta = get_release_group_metadata(mbids, inc="artist tag release", chunk_size=20)

    # 4) Obtener los release groups más populares a nivel de sitio (señal agregada)
    sitewide_payload = get_sitewide_top_release_groups(
        stats_range=stats_range,
        count=200,
        offset=0
    )
    df_sitewide_local = parse_top_release_groups(sitewide_payload)

    # 5) Realizar el merge final de todos los DataFrames obtenidos
    df_master = (
        df_fresh_local
        .merge(df_pop, on="release_group_mbid", how="left")
        .merge(df_meta[["release_group_mbid", "artist_name_meta", "tags_meta"]], on="release_group_mbid", how="left")
        .merge(
            df_sitewide_local[["release_group_mbid", "listen_count"]],
            on="release_group_mbid",
            how="left",
            suffixes=("", "_sitewide")
        )
    )

    # Renombrar la columna de conteo de escuchas del sitewide para mayor claridad
    if "listen_count" in df_master.columns:
        df_master = df_master.rename(columns={"listen_count": f"sitewide_listen_count_{stats_range}"})

    # Añadir columnas derivadas mínimas para indicar la presencia de señales
    df_master["has_sitewide_signal"] = df_master[f"sitewide_listen_count_{stats_range}"].notna().astype(int)
    df_master["has_popularity_signal"] = df_master["total_listen_count"].notna().astype(int)

    return df_master