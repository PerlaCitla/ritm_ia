import json

from my_functions.utils import (
    get_all_insights_fresh,
    get_insights_artist,
    get_cluster_insights,
    get_recent_comparisons,
)

get_all_insights_fresh_json = {
    "name": "get_all_insights_fresh",
    "description": "Usa esta herramienta (tool) para obtener insights y predicciones de éxito de los últimos X lanzamientos musicales publicados en los últimos Y días.",
    "parameters": {
        "type": "object",
        "properties": {
            "n_releases": {
                "type": "integer",
                "description": "El número de lanzamientos recientes a extraer y analizar. Si el usuario no especifica una cantidad exacta, usa 3 por defecto. Se recomienda un máximo de 10."
            },
            "days_back": {
                "type": "integer",
                "description": "El número de días hacia atrás desde hoy para buscar dichos lanzamientos. Si el usuario no especifica los días, usa 3 por defecto. Se recomienda un máximo de 30 días."
            }
        },
        "required": ["n_releases", "days_back"],
        "additionalProperties": False
    }
}

get_insights_artist_json = {
    "name": "get_insights_artist",
    "description": "Usa esta herramienta (tool) para obtener insights clave del catálogo y lanzamientos de un artista musical",
    "parameters": {
        "type": "object",
        "properties": {
            "artist_name": {
                "type": "string",
                "description": "El nombre del artista musical a analizar."
            }
        },
        "required": ["artist_name"],
        "additionalProperties": False
    }
}

get_cluster_insights_json = {
    "name": "get_cluster_insights",
    "description": "Obtiene el perfilamiento, estadísticas y data storytelling de los clústeres musicales (0, 1, 2, 3) o de todos ('todos').",
    "parameters": {
        "type": "object",
        "properties": {
            "cluster_id": {
                "type": "string",
                "description": "El ID del clúster a explorar (0, 1, 2, 3) o 'todos'."
            }
        },
        "required": ["cluster_id"],
        "additionalProperties": False
    }
}
# 3. Definir la Tool JSON
get_recent_comparisons_json = {
    "name": "get_recent_comparisons",
    "description": "Usa esta herramienta (tool) para obtener una comparativa de los lanzamientos recientes contra los artistas top (éxito y no éxito) del dataset de entrenamiento. Ideal para el CASO D.",
    "parameters": {
        "type": "object",
         "properties": {
            "artist_name": {
                "type": "string",
                "description": "El nombre del artista musical que se va a comparar."
            }
        },
        "required": ["artist_name"],
        "additionalProperties": False
    }
}


tools = [
    {"type": "function", "function": get_all_insights_fresh_json},
    {"type": "function", "function": get_insights_artist_json},
    {"type": "function", "function": get_cluster_insights_json},
    {"type": "function", "function": get_recent_comparisons_json}
]

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
    return results
