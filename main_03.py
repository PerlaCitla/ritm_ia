import os
import glob
import time
import threading
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from prompts import stronger_prompt

# Configuración de página (debe ser lo primero después de imports)
st.set_page_config(
    page_title="Ritm-IA",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== DESCARGAR RECURSOS NLTK ====================
# Necesario para Streamlit Cloud (evita LookupError con punkt tokenizer)
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Cargar variables de entorno localmente
load_dotenv(override=True)

# Prioridad: secrets de Streamlit Cloud > variables de entorno locales
try:
    OPENAI_API_KEY = st.secrets["openai_api_key"]
except (KeyError, FileNotFoundError):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("⚠️ Error: OPENAI_API_KEY no está configurada. Verifica tu archivo .env o los secrets en Streamlit Cloud.")
    st.stop()

client_openai = OpenAI(api_key=OPENAI_API_KEY)

from my_functions.tooling import handle_tool_calls, tools
from my_functions.config_st import set_bg_hack,set_animated_background,set_bg_gif

# ==================== FUNCIONES AUXILIARES ====================
def stream_assistant_answer(client, model, conversation):
    """
    Llama al modelo con stream=True y pinta la respuesta progresivamente.
    Devuelve el texto completo generado.
    """
    full_response = ""
    placeholder = st.empty()

    stream = client.chat.completions.create(
        model=model,
        messages=conversation,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            full_response += delta.content
            placeholder.markdown(full_response)

    return full_response


MUSIC_FUN_FACTS = [
    "El tema mas reproducido de la historia en plataformas de streaming es 'Blinding Lights' de The Weeknd.",
    "Spotify reporta mas de 100,000 canciones nuevas subidas cada dia por artistas de todo el mundo.",
    "El reggaeton y el regional mexicano han tenido uno de los mayores crecimientos globales en escucha reciente.",
    "Escuchar musica puede mejorar el estado de animo en pocos minutos, especialmente con canciones conocidas.",
    "Las playlists editoriales pueden acelerar el descubrimiento de un artista emergente en cuestion de horas.",
    "La colaboracion entre artistas de distintos paises suele aumentar el alcance internacional de una cancion.",
    "El vinilo ha vuelto a crecer en ventas en varios mercados, incluso en la era del streaming.",
    "La cancion 'Asereje' de Las Ketchup fue un fenomeno global en la decada de los 2000.",
    "Mozart compuso mas de 600 obras antes de morir a los 35 anos.",
    "El primer videoclip emitido en MTV fue 'Video Killed the Radio Star' de The Buggles.",
    "El tempo promedio de muchos hits pop suele estar entre 90 y 120 BPM.",
    "Muchas canciones exitosas usan progresiones armonicas simples para ser mas memorables.",
    "El algoritmo de recomendacion aprende de saltos, repeticiones y tiempo de escucha de cada usuario.",
    "La musica en vivo suele incrementar de forma notable las reproducciones en streaming de un artista.",
    "El K-pop se ha consolidado como una fuerza global gracias a fandoms digitales muy organizados.",
    "Bandas sonoras de peliculas y videojuegos impulsan el descubrimiento de artistas en nuevas audiencias.",
]


def run_with_fun_facts(
    task_fn,
    placeholder,
    context_msg,
    threshold_seconds=5,
    min_visible_seconds=7.5,
):
    """
    Ejecuta una tarea bloqueante en segundo plano y, si supera `threshold_seconds`,
    muestra datos curiosos rotativos para reducir la sensacion de espera.
    """
    state = {"result": None, "error": None, "done": False}

    def _runner():
        try:
            state["result"] = task_fn()
        except Exception as err:
            state["error"] = err
        finally:
            state["done"] = True

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()

    fact_idx = 0
    shown = False
    first_shown_at = None
    last_switch = -999.0
    start = time.time()

    while not state["done"]:
        elapsed = time.time() - start
        if elapsed >= threshold_seconds and (elapsed - last_switch) >= 4:
            fact = MUSIC_FUN_FACTS[fact_idx % len(MUSIC_FUN_FACTS)]
            placeholder.info(
                f"⏳ {context_msg}\n\n"
                f"🎵 **Dato curioso:** {fact}"
            )
            shown = True
            if first_shown_at is None:
                first_shown_at = time.time()
            fact_idx += 1
            last_switch = elapsed
        time.sleep(0.25)

    if shown:
        visible_for = time.time() - first_shown_at if first_shown_at is not None else 0
        if visible_for < min_visible_seconds:
            time.sleep(min_visible_seconds - visible_for)
        placeholder.empty()

    if state["error"] is not None:
        raise state["error"]

    return state["result"]

# ==================== CONFIGURACIÓN ====================

# Definir la ruta a la carpeta 'inputs' (sin ".." porque inputs está en el mismo nivel que main_03.py)
path_background = os.path.join(os.path.dirname(__file__), "inputs")

# Buscar archivos .gif dentro de la carpeta
gif_files = glob.glob(os.path.join(path_background, "*.gif"))

print(f"DEBUG: Ruta base para inputs: {gif_files[0] if gif_files else 'No se encontraron GIFs'}")

model_openai = "gpt-5.4-mini"

#set_bg_hack("#aba5ff")
#set_animated_background()
#set_bg_gif(gif_files[0]) #if gif_files else None

st.title("🎶 RitmIA ✨")
st.caption("📈 Music Trend Intelligence 🤖")

# ==================== INICIALIZACIÓN SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": """¡Hola! ✨ Soy RitmIA 🎵, tu asistente IA especializado en tendencias musicales. Descubre el futuro de la industria con tres funciones clave:


- 🎯 Analizar lanzamientos recientes y predecir éxito de canciones
- 🎤 Explorar datos de artistas y su catálogo musical
- 🔗 Comparar perfiles de clústeres musicales

¿Por dónde quieres empezar?"""
    }]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ==================== CAJA DE TEXTO PRINCIPAL ====================
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

typed_prompt = st.chat_input("Escribe tu mensaje aquí...")
prompt = typed_prompt or st.session_state.pending_prompt

# ==================== PREGUNTAS SUGERIDAS ====================
st.divider()
st.markdown("**💡 Sugerencias:**")

suggested_questions = [
    "🎯 Analiza los últimos 10 lanzamientos musicales",
    "🎤 Cuéntame sobre el catálogo de Zoé",
    "🔗 ¿Qué características definen el clúster 0?",
]

cols = st.columns(2)
for idx, question in enumerate(suggested_questions):
    col = cols[idx % 2]
    if col.button(question, use_container_width=True, key=f"suggested_{idx}"):
        st.session_state.pending_prompt = question.split(" ", 1)[1]  # Remover emoji
        st.rerun()

st.divider()

# ==================== PROCESAR MENSAJE ====================
if prompt:
    st.session_state.pending_prompt = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)
    conversation = [{"role": "system", "content": stronger_prompt}]
    conversation.extend({"role": m["role"], "content": m["content"]} for m in st.session_state.messages)

    with st.chat_message("assistant", avatar="🎵"):
        done = False
        wait_placeholder = st.empty()
        
        while not done:
            response = run_with_fun_facts(
                task_fn=lambda: client_openai.chat.completions.create(
                    model=model_openai,
                    messages=conversation,
                    tools=tools,
                ),
                placeholder=wait_placeholder,
                context_msg="Procesando tu solicitud y consultando el modelo...",
            )
            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason

            if finish_reason == "tool_calls" and message.tool_calls:
                tool_calls = message.tool_calls
                # Serializar tool_calls para compatibilidad
                tool_calls_serialized = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                        "type": tc.type,
                    }
                    for tc in tool_calls
                ]
                results = run_with_fun_facts(
                    task_fn=lambda: handle_tool_calls(tool_calls),
                    placeholder=wait_placeholder,
                    context_msg="Descargando y analizando datos musicales...",
                )
                safe_content = message.content or ""
                if safe_content:
                    st.session_state.messages.append({"role": message.role, "content": safe_content})
                conversation.append(
                    {
                        "role": message.role,
                        "content": safe_content,
                        "tool_calls": tool_calls_serialized,
                    }
                )
                conversation.extend(results)
            else:
                done = True

        wait_placeholder.empty()
        
        # Streaming de respuesta con visualización progresiva
        response_final = stream_assistant_answer(
            client=client_openai,
            model=model_openai,
            conversation=conversation,
        )

    st.session_state.messages.append({"role": "assistant", "content": response_final})
    # Fuerza refresco de UI para que las sugerencias se mantengan visibles
    st.rerun()
