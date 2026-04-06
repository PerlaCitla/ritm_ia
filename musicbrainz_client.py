# ============================================================
# CELDA 1 — Imports y configuración
# ============================================================

import time
import requests
import pandas as pd
from datetime import date, timedelta
from typing import Optional, List, Tuple

BASE_URL_MUSICBRAINZ = "https://musicbrainz.org/ws/2" # Renamed BASE_URL

# IMPORTANTE: cambia este contacto por tu correo o GitHub
APP_NAME = "radar-lanzamientos-diplomado"
APP_VERSION = "0.1.0"
CONTACT = "data.citla@gmail.com"

USER_AGENT = f"{APP_NAME}/{APP_VERSION} ({CONTACT})"
MIN_INTERVAL = 1.1  # para respetar ~1 request/segundo

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
})

_last_request_ts = 0.0

# ============================================================
# CELDA 2 — Cliente MusicBrainz con rate limit y retry
# ============================================================

def _sleep_if_needed(min_interval: float = MIN_INTERVAL):
    """
    Espera si la última petición fue hace muy poco tiempo para respetar el rate limit.
    Esta función asegura que no se exceda el número permitido de peticiones por unidad de tiempo
    a la API de MusicBrainz, tal como lo define MIN_INTERVAL.
    """
    global _last_request_ts  # Declara que se usará la variable global _last_request_ts
    elapsed = time.time() - _last_request_ts  # Calcula el tiempo transcurrido desde la última petición
    if elapsed < min_interval:  # Si el tiempo transcurrido es menor que el intervalo mínimo
        time.sleep(min_interval - elapsed)  # Espera el tiempo restante para alcanzar el intervalo mínimo

def mb_get(endpoint: str, params: Optional[dict] = None, retries: int = 5) -> dict:
    """
    Realiza una petición GET a la API de MusicBrainz (/ws/2) en formato JSON, respetando el rate limit
    y con mecanismos de reintento para errores temporales.

    Args:
        endpoint (str): La parte específica de la URL de la API (ej. "release-group").
        params (Optional[dict]): Un diccionario de parámetros para la consulta GET.
        retries (int): El número máximo de intentos para la petición en caso de errores recuperables.

    Returns:
        dict: La respuesta de la API de MusicBrainz en formato JSON.

    Raises:
        RuntimeError: Si la petición falla después de todos los reintentos o por un error no recuperable.
    """
    global _last_request_ts  # Declara que se usará la variable global _last_request_ts

    url = f"{BASE_URL_MUSICBRAINZ}/{endpoint}"  # Construye la URL completa de la petición
    params = dict(params or {})  # Crea una copia mutable de los parámetros o un diccionario vacío si es None
    params["fmt"] = "json"  # Asegura que la respuesta se solicite en formato JSON

    response = None  # Inicializa la variable de respuesta
    for attempt in range(retries):  # Itera a través del número de intentos definidos
        _sleep_if_needed()  # Llama a la función para respetar el rate limit antes de cada petición

        response = session.get(url, params=params, timeout=60)  # Realiza la petición GET usando la sesión global
        _last_request_ts = time.time()  # Actualiza el tiempo de la última petición

        if response.status_code == 200:  # Si la petición fue exitosa (código 200 OK)
            return response.json()  # Devuelve la respuesta en formato JSON

        # Manejo de backoff para throttling (429 Too Many Requests) o servidor ocupado (503 Service Unavailable)
        if response.status_code in (429, 503):
            wait = 2 ** attempt  # Calcula el tiempo de espera exponencial
            print(f"[retry] {response.status_code} en {endpoint}. Esperando {wait}s...")  # Imprime un mensaje de reintento
            time.sleep(wait)  # Espera el tiempo calculado antes del siguiente intento
            continue  # Pasa a la siguiente iteración del bucle de reintentos

        # Si el código de estado es otro error no recuperable, lanza una excepción
        raise RuntimeError(
            f"Error {response.status_code} en {url}\n"
            f"Params: {params}\n"
            f"Respuesta: {response.text[:500]}"
        )

    # Si se agotan todos los reintentos y la petición no fue exitosa, lanza una excepción
    raise RuntimeError(
        f"No se pudo completar la petición tras {retries} intentos.\n"
        f"Último status: {response.status_code if response else 'sin respuesta'}"
    )


# ============================================================
# CELDA 3 — Utilidades de parsing / normalización
# ============================================================

def list_to_pipe(values: List) -> Optional[str]:
    """
    Convierte una lista de valores en una cadena separada por '|',
    eliminando duplicados y valores nulos/vacíos.

    Args:
        values (List): Una lista de elementos a convertir.

    Returns:
        Optional[str]: Una cadena con los valores limpios separados por '|',
                       o None si la lista resultante está vacía.
    """
    # Filtra valores nulos, cadenas vacías, listas vacías o diccionarios vacíos
    clean = [str(v).strip() for v in values if v not in (None, "", [], {})]
    # Elimina duplicados manteniendo el orden original (usando un truco con dict.fromkeys)
    clean = list(dict.fromkeys(clean))
    # Une los valores limpios con '|' si la lista no está vacía, de lo contrario devuelve None
    return "|".join(clean) if clean else None

def artist_credit_to_text(artist_credit: list) -> Tuple[Optional[str], Optional[str]]:
    """
    Convierte el formato 'artist-credit' de MusicBrainz a:
    - artist_name: El nombre visible del artista o de los artistas (incluyendo 'joinphrases').
    - artist_ids: Una cadena de IDs de artistas separados por '|'.

    Args:
        artist_credit (list): La lista 'artist-credit' de la respuesta de MusicBrainz.

    Returns:
        Tuple[Optional[str], Optional[str]]: Una tupla que contiene el nombre del artista
                                             y sus IDs, ambos como cadenas o None.
    """
    if not artist_credit:
        return None, None

    names = []  # Lista para construir el nombre visible del artista
    ids = []    # Lista para recolectar los IDs de los artistas

    for part in artist_credit:
        if isinstance(part, str):
            # Si una parte es una cadena, se asume que es un nombre directo o parte de él
            names.append(part)
            continue

        if isinstance(part, dict):
            # Si es un diccionario, puede contener el nombre del artista, un 'joinphrase' o el ID.
            display_name = part.get("name") or part.get("artist", {}).get("name", "")
            if display_name:
                names.append(display_name)

            joinphrase = part.get("joinphrase", "")
            if joinphrase:
                names.append(joinphrase)

            artist_obj = part.get("artist", {})
            artist_id = artist_obj.get("id")
            if artist_id:
                ids.append(artist_id)

    # Une todas las partes del nombre y elimina espacios extra, o None si queda vacío
    artist_name = "".join(names).strip() or None
    # Une los IDs únicos separados por '|' o None si no hay IDs
    artist_ids = "|".join(dict.fromkeys(ids)) if ids else None
    return artist_name, artist_ids

def normalize_release_group(rg: dict) -> dict:
    """
    Normaliza un diccionario de 'release group' de MusicBrainz a un formato más plano.
    Extrae y renombra campos clave para facilitar el análisis.

    Args:
        rg (dict): Diccionario que representa un 'release group' de MusicBrainz.

    Returns:
        dict: Diccionario normalizado con campos seleccionados del 'release group'.
    """
    # Extrae el nombre del artista y los IDs utilizando la función auxiliar
    artist_name, artist_ids = artist_credit_to_text(rg.get("artist-credit", []))

    # Extrae los nombres de los géneros, asegurándose de que sean diccionarios válidos
    genres = [g.get("name") for g in rg.get("genres", []) if isinstance(g, dict)]
    # Extrae los nombres de las etiquetas (tags), asegurándose de que sean diccionarios válidos
    tags = [t.get("name") for t in rg.get("tags", []) if isinstance(t, dict)]

    return {
        "release_group_id": rg.get("id"),
        "title": rg.get("title"),
        "primary_type": rg.get("primary-type"),
        "secondary_types": list_to_pipe(rg.get("secondary-types", [])), # Normaliza tipos secundarios a cadena
        "first_release_date": rg.get("first-release-date"),
        "artist_name": artist_name,
        "artist_ids": artist_ids,
        "disambiguation": rg.get("disambiguation"),
        "genres": list_to_pipe(genres), # Normaliza géneros a cadena
        "tags": list_to_pipe(tags),     # Normaliza tags a cadena
        "score": rg.get("score"),  # Puntuación de relevancia (aparece en búsquedas, no siempre en lookups)
    }

def normalize_release(rel: dict, fallback_release_group_id: Optional[str] = None) -> dict:
    """
    Normaliza un diccionario de 'release' (edición concreta) de MusicBrainz a un formato más plano.
    Incluye lógica para manejar etiquetas (labels) y formatos de medios (media formats).

    Args:
        rel (dict): Diccionario que representa una 'release' de MusicBrainz.
        fallback_release_group_id (Optional[str]): Un ID de 'release group' para usar si no se encuentra uno en la 'release'.

    Returns:
        dict: Diccionario normalizado con campos seleccionados de la 'release'.
    """
    # Extrae el nombre del artista y los IDs utilizando la función auxiliar
    artist_name, artist_ids = artist_credit_to_text(rel.get("artist-credit", []))

    # Intenta obtener el 'release group' asociado, si existe y es un diccionario
    rg = rel.get("release-group", {}) if isinstance(rel.get("release-group"), dict) else {}

    label_names = []       # Lista para recolectar nombres de sellos discográficos
    catalog_numbers = []   # Lista para recolectar números de catálogo
    for item in rel.get("label-info", []) or []: # Itera sobre la información de sellos
        label = item.get("label", {}) or {}     # Obtiene el diccionario del sello
        if label.get("name"):
            label_names.append(label["name"]) # Añade el nombre del sello
        if item.get("catalog-number"):
            catalog_numbers.append(item["catalog-number"]) # Añade el número de catálogo

    media_formats = []     # Lista para recolectar formatos de medios (CD, Vinyl, Digital, etc.)
    track_counts = []      # Lista para recolectar el número de pistas por medio
    for m in rel.get("media", []) or []: # Itera sobre la información de medios
        if m.get("format"):
            media_formats.append(m["format"]) # Añade el formato
        if isinstance(m.get("track-count"), int):
            track_counts.append(m["track-count"]) # Añade el conteo de pistas

    return {
        "release_id": rel.get("id"),
        "release_group_id": rg.get("id") or fallback_release_group_id, # Usa fallback si no hay ID de RG
        "release_group_title": rg.get("title"),
        "release_title": rel.get("title"),
        "artist_name": artist_name,
        "artist_ids": artist_ids,
        "status": rel.get("status"),
        "quality": rel.get("quality"),
        "date": rel.get("date"),
        "country": rel.get("country"),
        "barcode": rel.get("barcode"),
        "packaging": rel.get("packaging"),
        "label_names": list_to_pipe(label_names),     # Normaliza nombres de sellos
        "catalog_numbers": list_to_pipe(catalog_numbers), # Normaliza números de catálogo
        "media_formats": list_to_pipe(media_formats), # Normaliza formatos de medios
        "track_count_total": sum(track_counts) if track_counts else None, # Suma total de pistas
    }

# ============================================================
# CELDA 4 — Búsqueda de release groups recientes
# ============================================================

def build_release_group_query(
    start_date: str,
    end_date: str,
    primary_types=("album", "single", "ep"),
    official_only: bool = True
) -> str:
    """
    Arma una query Lucene para buscar 'release groups' en MusicBrainz.
    Esta función construye una cadena de consulta que se puede usar en la API
    para filtrar por rango de fechas, tipos primarios de lanzamiento y estado.

    Args:
        start_date (str): Fecha de inicio del rango de búsqueda en formato YYYY-MM-DD.
        end_date (str): Fecha de fin del rango de búsqueda en formato YYYY-MM-DD.
        primary_types (tuple): Tupla de cadenas que representan los tipos primarios
                               de lanzamientos a incluir (ej. "album", "single", "ep").
        official_only (bool): Si es True, solo incluye lanzamientos con estado 'official'.

    Returns:
        str: La cadena de consulta Lucene construida.

    Ejemplo de query generada:
    firstreleasedate:[2026-02-01 TO 2026-03-01] AND
    (primarytype:album OR primarytype:single OR primarytype:ep) AND
    status:official
    """
    # Construye la parte de la consulta para los tipos primarios, uniendo con ' OR '
    type_expr = " OR ".join(f"primarytype:{t.lower()}" for t in primary_types)
    # Combina el rango de fechas con la expresión de tipos primarios
    query = f"firstreleasedate:[{start_date} TO {end_date}] AND ({type_expr})"

    # Si se requiere, añade la condición para que solo sean lanzamientos oficiales
    if official_only:
        query += " AND status:official"

    return query

def search_release_groups(query: str, limit: int = 100, max_pages: int = 3) -> list:
    """
    Busca 'release groups' en MusicBrainz usando la query Lucene y maneja la paginación.

    Args:
        query (str): La cadena de consulta Lucene generada por `build_release_group_query`.
        limit (int): El número máximo de resultados por página (por defecto 100).
        max_pages (int): El número máximo de páginas a recuperar.

    Returns:
        list: Una lista de diccionarios, donde cada diccionario representa un 'release group'.
    """
    results = [] # Lista para almacenar todos los 'release groups' encontrados

    # Itera a través del número máximo de páginas para recuperar resultados
    for page in range(max_pages):
        offset = page * limit # Calcula el offset para la paginación
        payload = mb_get(
            "release-group", # Endpoint para buscar 'release groups'
            {
                "query": query,   # La query Lucene
                "limit": limit,   # Límite de resultados por petición
                "offset": offset, # Offset para la paginación
            },
        )
        batch = payload.get("release-groups", []) # Extrae la lista de 'release groups' de la respuesta
        print(f"Página {page + 1}: {len(batch)} release groups") # Imprime el progreso

        results.extend(batch) # Añade los resultados de la página actual a la lista general

        # Si el número de resultados en el lote es menor que el límite, significa que no hay más páginas
        if len(batch) < limit:
            break # Sale del bucle de paginación

    return results


# ============================================================
# CELDA 5 — Lookup de release group y browse de releases
# ============================================================

def get_release_group_details(release_group_id: str) -> dict:
    """
    Trae detalle del release group con géneros, tags y artist-credit.

    Args:
        release_group_id (str): El MBID (MusicBrainz ID) del 'release group' a consultar.

    Returns:
        dict: Un diccionario con los detalles completos del 'release group'.
    """
    return mb_get(
        f"release-group/{release_group_id}",
        {
            "inc": "artist-credits+genres+tags",  # Incluye créditos de artistas, géneros y tags en la respuesta
        },
    )

def browse_releases_for_release_group(release_group_id: str, limit: int = 100) -> list:
    """
    Baja TODAS las releases concretas (ediciones) de un 'release group'.
    Realiza paginación automática para obtener todos los resultados.

    Args:
        release_group_id (str): El MBID del 'release group' del cual se quieren obtener las 'releases'.
        limit (int): El número máximo de 'releases' por página en cada petición (por defecto 100).

    Returns:
        list: Una lista de diccionarios, donde cada diccionario representa una 'release' concreta.
    """
    releases = []  # Lista para almacenar todas las 'releases' encontradas
    offset = 0     # Inicializa el offset para la paginación

    while True:
        payload = mb_get(
            "release",  # Endpoint para buscar 'releases'
            {
                "release-group": release_group_id,  # Filtra por el 'release group' especificado
                "limit": limit,
                "offset": offset,
                "inc": "artist-credits+labels+media+release-groups",  # Incluye créditos de artista, sellos, medios y 'release groups' relacionados
            },
        )
        batch = payload.get("releases", [])  # Extrae la lista de 'releases' de la respuesta
        releases.extend(batch)  # Añade las 'releases' de la página actual a la lista general

        # Si el número de resultados en el lote es menor que el límite, significa que no hay más páginas
        if len(batch) < limit:
            break  # Sale del bucle de paginación

        offset += limit  # Incrementa el offset para la siguiente página

    return releases

# ============================================================
# CELDA 6 — Pipeline principal de extracción
# ============================================================

def collect_recent_musicbrainz_dataset(
    start_date: str,
    end_date: str,
    primary_types=("album", "single", "ep"),
    max_search_pages: int = 1,
    max_release_groups: int = 20,
):
    """
    Devuelve:
    - df_release_groups: una fila por lanzamiento lógico
    - df_releases: una fila por edición concreta
    """
    query = build_release_group_query(
        start_date=start_date,
        end_date=end_date,
        primary_types=primary_types,
        official_only=True,
    )

    print("Query MusicBrainz:")
    print(query)
    print("-" * 80)

    search_results = search_release_groups(
        query=query,
        limit=100,
        max_pages=max_search_pages,
    )

    # empieza pequeño para no tardar demasiado en Colab
    search_results = search_results[:max_release_groups]

    release_group_rows = []
    release_rows = []

    total = len(search_results)
    for idx, rg in enumerate(search_results, start=1):
        rgid = rg["id"]
        title = rg.get("title")

        print(f"[{idx}/{total}] {title} :: {rgid}")

        # detalle del release group
        rg_details = get_release_group_details(rgid)
        release_group_rows.append(normalize_release_group(rg_details))

        # releases asociadas
        releases = browse_releases_for_release_group(rgid, limit=100)
        for rel in releases:
            release_rows.append(normalize_release(rel, fallback_release_group_id=rgid))

    df_release_groups = pd.DataFrame(release_group_rows)
    df_releases = pd.DataFrame(release_rows)

    if not df_release_groups.empty and "first_release_date" in df_release_groups.columns:
        df_release_groups = (
            df_release_groups
            .sort_values(
                ["first_release_date", "artist_name", "title"],
                ascending=[False, True, True],
                na_position="last",
            )
            .reset_index(drop=True)
        )

    if not df_releases.empty and "date" in df_releases.columns:
        df_releases = (
            df_releases
            .sort_values(
                ["date", "artist_name", "release_title"],
                ascending=[False, True, True],
                na_position="last",
            )
            .reset_index(drop=True)
        )

    return df_release_groups, df_releases

# ============================================================
# CELDA 9 — OPCIONAL: buscar un artista y bajar sus release groups
# ============================================================

def search_artist_by_name(name: str, limit: int = 5) -> pd.DataFrame:
    """
    Busca artistas por nombre en MusicBrainz utilizando la API.
    Devuelve un DataFrame con los artistas encontrados y su información clave.

    Args:
        name (str): Nombre del artista a buscar.
        limit (int): Número máximo de resultados de artistas a devolver.

    Returns:
        pd.DataFrame: DataFrame que contiene la información de los artistas encontrados,
                      incluyendo ID, nombre, país, tipo y puntuación de relevancia.
    """
    # Realiza la petición GET a la API de MusicBrainz para buscar artistas.
    payload = mb_get(
        "artist",
        {
            "query": f'artist:"{name}"', # La query busca por nombre de artista exacto.
            "limit": limit,                # Limita el número de resultados.
        },
    )

    rows = []
    # Itera sobre los artistas encontrados en la respuesta y extrae la información relevante.
    for a in payload.get("artists", []):
        rows.append({
            "artist_id": a.get("id"),           # ID del artista en MusicBrainz.
            "name": a.get("name"),             # Nombre principal del artista.
            "sort_name": a.get("sort-name"),     # Nombre del artista para ordenar.
            "country": a.get("country"),         # País de origen del artista.
            "type": a.get("type"),               # Tipo de artista (persona, grupo, etc.).
            "disambiguation": a.get("disambiguation"), # Texto de desambiguación si existe.
            "score": a.get("score"),             # Puntuación de relevancia de la búsqueda.
        })

    return pd.DataFrame(rows) # Retorna los resultados como un DataFrame.

def browse_release_groups_by_artist(
    artist_id: str,
    types=("album", "single", "ep"),
    max_pages: int = 2
) -> pd.DataFrame:
    """
    Navega por los 'release groups' (álbumes, singles, EPs) asociados a un artista específico en MusicBrainz.
    Realiza la paginación para obtener todos los resultados hasta `max_pages`.

    Args:
        artist_id (str): ID (MBID) del artista en MusicBrainz.
        types (tuple): Tupla de cadenas que representan los tipos de 'release groups' a incluir
                       (ej. "album", "single", "ep").
        max_pages (int): Número máximo de páginas a consultar para 'release groups' del artista.

    Returns:
        pd.DataFrame: DataFrame con los 'release groups' normalizados del artista,
                      ordenados por fecha de lanzamiento más reciente.
    """
    rows = []
    offset = 0
    # Une los tipos de lanzamiento con '|' para usarlos como filtro en la API.
    type_filter = "|".join(types)

    # Itera a través de las páginas para recolectar todos los 'release groups'.
    for _ in range(max_pages):
        # Realiza la petición GET para buscar 'release groups' de un artista.
        payload = mb_get(
            "release-group",
            {
                "artist": artist_id, # Filtra por el ID del artista.
                "type": type_filter, # Filtra por los tipos de lanzamiento.
                "limit": 100,        # Número de resultados por petición.
                "offset": offset,    # Offset para la paginación.
                "inc": "artist-credits", # Incluye los créditos de artistas en la respuesta.
            },
        )

        batch = payload.get("release-groups", []) # Extrae la lista de 'release groups' de la respuesta.
        # Normaliza cada 'release group' y lo añade a la lista general.
        rows.extend(normalize_release_group(x) for x in batch)

        # Si el número de resultados en el lote es menor que el límite, no hay más páginas.
        if len(batch) < 100:
            break

        offset += 100 # Incrementa el offset para la siguiente página.

    df = pd.DataFrame(rows)
    # Ordena el DataFrame de 'release groups' si no está vacío y contiene la columna de fecha.
    if not df.empty and "first_release_date" in df.columns:
        df = df.sort_values("first_release_date", ascending=False, na_position="last").reset_index(drop=True)
    return df
