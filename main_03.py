import os
import glob
import time
import threading
import json
import base64
import re
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI, APIConnectionError, APITimeoutError
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

client_openai = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=60.0,
    max_retries=2,
)

from my_functions.tooling import handle_tool_calls,tools
from my_functions.config_st import set_bg_hack,set_animated_background,set_bg_gif



# def add_animated_background_with_video():
#     # Estilos CSS para el fondo animado
#     st.markdown(
#         """
#         <style>
#             /* 1. Seleccionar el contenedor principal de Streamlit */
#             .stApp {
#                 background: linear-gradient(315deg, #F8FFF0 3%, #F0FFFF 38%, #F7F0FF 68%, #FFF0F1 98%);
#                 background-size: 400% 400%; /* Agrandar el fondo para animarlo */
#                 animation: gradient 30s ease infinite; /* Redefinir la animación con más tiempo (30 segundos) */
#                 background-attachment: fixed;
#             }

#             /* 2. Definir la animación del movimiento */
#             @keyframes gradient {
#                 0% { background-position: 0% 0%; }
#                 50% { background-position: 100% 100%; }
#                 100% { background-position: 0% 0%; }
#             }

#             /* 3. Video banner: estado inicial (centrado) */
#             .fixed-top-video {
#                 position: fixed;
#                 top: 0.6rem;
#                 left: 50%;
#                 transform: translateX(-50%);
#                 width: min(620px, calc(100vw - 2rem));
#                 height: 132px;
#                 z-index: 9998;
#                 background: transparent;
#                 border-radius: 14px;
#                 display: flex;
#                 align-items: center;
#                 justify-content: center;
#                 transition: all 320ms ease;
#             }

#             .fixed-top-video video {
#                 width: 92%;
#                 height: 110px;
#                 object-fit: contain;
#                 border-radius: 12px;
#                 box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
#                 transition: all 320ms ease;
#             }

#             /* 3b. Al hacer scroll: encabezado izquierdo más pequeño */
#             body.ritmia-scrolled .fixed-top-video,
#             html.ritmia-scrolled .fixed-top-video,
#             .stApp.ritmia-scrolled .fixed-top-video {
#                 top: 0.45rem;
#                 left: 0.9rem;
#                 transform: none;
#                 width: min(310px, calc(100vw - 1.8rem));
#                 height: 74px;
#                 background: #eceff1;
#                 border-radius: 10px;
#                 justify-content: flex-start;
#                 padding-left: 0.55rem;
#             }

#             body.ritmia-scrolled .fixed-top-video video,
#             html.ritmia-scrolled .fixed-top-video video,
#             .stApp.ritmia-scrolled .fixed-top-video video {
#                 width: 140px;
#                 height: 56px;
#                 border-radius: 8px;
#                 box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
#             }

#             /* 4. Empujar contenido para que no quede debajo del video fijo */
#             .block-container {
#                 padding-top: 10.2rem !important;
#             }

#             /* Opcional: Hacer el header transparente */
#             [data-testid="stHeader"] {
#                 background-color: rgba(0,0,0,0);
#             }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     # Renderizar video desde ruta relativa (compatible local + Streamlit Cloud)
#     video_path = os.path.join(os.path.dirname(__file__), "inputs", "video_ritmia.mp4")
#     if os.path.exists(video_path):
#         with open(video_path, "rb") as video_file:
#             video_bytes = video_file.read()
#         video_b64 = base64.b64encode(video_bytes).decode("utf-8")
#         video_html = f"""
#         <div class="fixed-top-video">
#             <video autoplay loop muted playsinline preload="auto">
#                 <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
#             </video>
#         </div>
#         """
#         st.markdown(video_html, unsafe_allow_html=True)
#         components.html(
#             """
#             <script>
#               (function () {
#                 const getContexts = () => {
#                   const contexts = [{ w: window, d: document }];
#                   try {
#                     if (window.parent && window.parent !== window && window.parent.document) {
#                       contexts.push({ w: window.parent, d: window.parent.document });
#                     }
#                   } catch (_) {}
#                   return contexts;
#                 };

#                 const getScrollTop = (ctx) => {
#                   const d = ctx.d;
#                   const w = ctx.w;
#                   const appContainer = d.querySelector('[data-testid="stAppViewContainer"]');
#                   const mainSection = d.querySelector('section.main');
#                   const fromApp = appContainer ? appContainer.scrollTop : 0;
#                   const fromMain = mainSection ? mainSection.scrollTop : 0;
#                   const fromWindow = w.scrollY || d.documentElement.scrollTop || d.body.scrollTop || 0;
#                   return Math.max(fromApp, fromMain, fromWindow);
#                 };

#                 const setCompactClass = (d, on) => {
#                   if (d.body) d.body.classList.toggle("ritmia-scrolled", on);
#                   if (d.documentElement) d.documentElement.classList.toggle("ritmia-scrolled", on);
#                   const app = d.querySelector('.stApp');
#                   if (app) app.classList.toggle("ritmia-scrolled", on);
#                 };

#                 const applyState = () => {
#                   const contexts = getContexts();
#                   const shouldCompact = contexts.some((ctx) => getScrollTop(ctx) > 90);
#                   contexts.forEach((ctx) => setCompactClass(ctx.d, shouldCompact));
#                 };

#                 const bindOncePerContext = () => {
#                   const contexts = getContexts();
#                   contexts.forEach((ctx) => {
#                     if (ctx.w.__ritmiaScrollHandlerBound) return;
#                     ctx.w.addEventListener("scroll", applyState, { passive: true });
#                     ctx.w.addEventListener("resize", applyState);
#                     const appContainer = ctx.d.querySelector('[data-testid="stAppViewContainer"]');
#                     const mainSection = ctx.d.querySelector('section.main');
#                     if (appContainer) appContainer.addEventListener("scroll", applyState, { passive: true });
#                     if (mainSection) mainSection.addEventListener("scroll", applyState, { passive: true });
#                     ctx.w.__ritmiaScrollHandlerBound = true;
#                   });
#                 };

#                 bindOncePerContext();
#                 applyState();
#               })();
#             </script>
#             """,
#             height=0,
#         )
#     else:
#         st.info("No se encontró el archivo `inputs/video_ritmia.mp4`.")

# # Llamar a la función
# add_animated_background_with_video()


def add_animated_background_with_video():
    # Estilos CSS para el fondo animado y video fijo
    st.markdown(
        """
        <style>
            /* Fondo animado */
            .stApp {
                background: linear-gradient(315deg, #F8FFF0 3%, #F0FFFF 38%, #F7F0FF 68%, #FFF0F1 98%);
                background-size: 400% 400%;
                animation: gradient 30s ease infinite;
                background-attachment: fixed;
            }

            @keyframes gradient {
                0% { background-position: 0% 0%; }
                50% { background-position: 100% 100%; }
                100% { background-position: 0% 0%; }
            }

            /* Video siempre fijo en la parte izquierda */
            .fixed-top-video {
                position: fixed;
                top: 0.6rem;
                left: 0.9rem;
                width: min(310px, calc(100vw - 1.8rem));
                height: 100px;
                z-index: 9998;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: flex-start;
                padding: 0; /* sin padding */
                overflow: hidden; /* para que el video se ajuste al contenedor */
            }

            .fixed-top-video video {
                width: 80%;
                height: 80%;
                object-fit: cover; /* el video llena todo el cuadro */
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
            }

            /* Empujar contenido para que no quede debajo del video fijo */
            .block-container {
                padding-top: 6rem !important;
            }

            /* Header transparente */
            [data-testid="stHeader"] {
                background-color: rgba(0,0,0,0);
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Renderizar video desde ruta relativa
    video_path = os.path.join(os.path.dirname(__file__), "inputs", "video_ritmia.mp4")
    if os.path.exists(video_path):
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        video_html = f"""
        <div class="fixed-top-video">
            <video autoplay loop muted playsinline preload="auto">
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
        </div>
        """
        st.markdown(video_html, unsafe_allow_html=True)
    else:
        st.info("No se encontró el archivo `inputs/video_ritmia.mp4`.")


add_animated_background_with_video()

def add_footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: white;
            color: black;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            z-index: 9999;
        }
        </style>
        <div class="footer">
            <p> RitmIA ✨ Music Trend Intelligence | © 2026</p>
        </div>
        """,
        unsafe_allow_html=True
    )

add_footer()


# ==================== FUNCIONES AUXILIARES ====================

def with_openai_retry(fn, max_attempts=3, base_delay=1.5):
    """
    Ejecuta llamadas a OpenAI con reintentos ante fallos transitorios de red.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except (APIConnectionError, APITimeoutError) as err:
            last_error = err
            if attempt == max_attempts:
                raise
            time.sleep(base_delay * attempt)
    if last_error is not None:
        raise last_error

def stream_assistant_answer(client, model, conversation):
    """
    Llama al modelo con stream=True y pinta la respuesta progresivamente.
    Devuelve el texto completo generado.
    """
    full_response = ""
    placeholder = st.empty()

    stream = with_openai_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=conversation,
            stream=True,
        )
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            full_response += delta.content
            placeholder.markdown(full_response)

    return full_response


MUSIC_FUN_FACTS = [
    "🎼 El tema mas reproducido de la historia en plataformas de streaming es 'Blinding Lights' de The Weeknd.",
    "🟢 Spotify reporta mas de 100,000 canciones nuevas subidas cada dia por artistas de todo el mundo.",
    "🎶 El reggaeton y el regional mexicano han tenido uno de los mayores crecimientos globales en escucha reciente.",
    "🎧 Escuchar musica puede mejorar el estado de animo en pocos minutos, especialmente con canciones conocidas.",
    "🎚️ Las playlists editoriales pueden acelerar el descubrimiento de un artista emergente en cuestion de horas.",
    "🎛️ La colaboracion entre artistas de distintos paises suele aumentar el alcance internacional de una cancion.",
    "💽 El vinilo ha vuelto a crecer en ventas en varios mercados, incluso en la era del streaming.",
    "🏆 La cancion 'Asereje' de Las Ketchup fue un fenomeno global en la decada de los 2000.",
    "🎹 Mozart compuso mas de 600 obras antes de morir a los 35 años.",
    "🎸 El primer videoclip emitido en MTV fue 'Video Killed the Radio Star' de The Buggles.",
    "🎶 El tempo promedio de muchos hits pop suele estar entre 90 y 120 BPM.",
    "🪗 Muchas canciones exitosas usan progresiones armonicas simples para ser mas memorables.",
    "🔄 El algoritmo de recomendacion aprende de saltos, repeticiones y tiempo de escucha de cada usuario.",
    "🥁 La musica en vivo suele incrementar de forma notable las reproducciones en streaming de un artista.",
    "🪘 El K-pop se ha consolidado como una fuerza global gracias a fandoms digitales muy organizados.",
    "🎺 Bandas sonoras de peliculas y videojuegos impulsan el descubrimiento de artistas en nuevas audiencias.",
]


def run_with_fun_facts(
    task_fn,
    placeholder,
    context_msg,
    threshold_seconds=5,
    min_visible_seconds=8.5,
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


def extract_suggested_questions_from_response(response_text):
    """Extrae preguntas sugeridas ya escritas por el modelo para mostrarlas como botones."""
    if not response_text:
        return []

    suggestions = []
    seen = set()
    valid_prefixes = (
        "saber más acerca de",
        "saber mas acerca de",
        "que significa el",
        "qué significa el",
        "explorar a",
        "comparar a este artista",
        "comparar este artista",
        "comparar",
    )

    lines = response_text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Solo considerar bullets/listas del texto generado.
        if re.match(r"^[-*]\s+", line):
            candidate = re.sub(r"^[-*]\s+", "", line).strip()
        elif re.match(r"^\d+[.)]\s+", line):
            candidate = re.sub(r"^\d+[.)]\s+", "", line).strip()
        else:
            continue

        # Limpieza de comillas y markdown básico.
        candidate = candidate.strip('"“”').replace("**", "").strip()
        # Normalizar signos de apertura para detectar preguntas como "¿Qué significa el Cluster 1?".
        normalized_candidate = candidate.lstrip("¿¡").strip()
        lowered = normalized_candidate.lower()
        if lowered.startswith(valid_prefixes) and candidate not in seen:
            seen.add(candidate)
            suggestions.append(candidate)
        if len(suggestions) >= 3:
            break

    return suggestions


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

# st.title("🎶 RitmIA ✨")
# st.caption("📈 Music Trend Intelligence 🤖")

# ==================== INICIALIZACIÓN SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": """¡Hola! ✨ Soy RitmIA 🎵, tu asistente IA especializado en tendencias musicales. 

Comencemos analizando **últimos lanzamientos** para descubrir las tendencias actuales del mercado 🎶"""
    }]
    st.session_state.conversation_stage = "initial_launches"

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ==================== CAJA DE TEXTO PRINCIPAL ====================
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "dynamic_suggestions" not in st.session_state:
    st.session_state.dynamic_suggestions = []

typed_prompt = st.chat_input("Escribe tu mensaje aquí...")
prompt = typed_prompt or st.session_state.pending_prompt

# ==================== PREGUNTAS SUGERIDAS ====================
is_fresh_chat = len(st.session_state.messages) <= 1
if is_fresh_chat:
    st.divider()
    st.markdown("**💡 Elige una opción:**")

    suggested_questions = [
        "Dame los últimos 3 lanzamientos",
        "Dame los últimos 5 lanzamientos de los pasados 3 días",
    ]

    for idx, question in enumerate(suggested_questions):
        if st.button(question, use_container_width=False, key=f"suggested_{idx}"):
            st.session_state.pending_prompt = question
            st.rerun()

if st.session_state.dynamic_suggestions:
    st.divider()
    st.markdown("**➡️ Sugerencias según tu último análisis:**")
    for idx, question in enumerate(st.session_state.dynamic_suggestions):
        if st.button(question, use_container_width=False, key=f"dynamic_suggested_{idx}"):
            st.session_state.pending_prompt = question
            st.rerun()

# ==================== PROCESAR MENSAJE ====================
if prompt:
    # Validar si estamos en fase inicial y validar el contenido
    is_initial_stage = st.session_state.conversation_stage == "initial_launches"
    
    if is_initial_stage:
        # Keywords válidos para la fase inicial
        valid_keywords = ["últimos", "lanzamientos", "recientes", "últimas"]
        prompt_lower = prompt.lower()
        is_valid = any(keyword in prompt_lower for keyword in valid_keywords)
        
        if not is_valid:
            # Rechazar solicitud y mostrar sugerencias nuevamente
            st.chat_message("user", avatar="👤").write(prompt)
            with st.chat_message("assistant", avatar="🎵"):
                st.warning("⚠️ Por favor, comencemos analizando los **últimos lanzamientos** musicales. Elige una de estas opciones:")
                st.info("- Dame los últimos 3 lanzamientos\n- Dame los últimos 5 lanzamientos de los pasados 3 días")
            st.session_state.pending_prompt = None
            st.stop()
        else:
            # Cambiar a siguiente fase después de primera interacción válida
            st.session_state.conversation_stage = "full_access"
    
    st.session_state.pending_prompt = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)
    conversation = [{"role": "system", "content": stronger_prompt}]
    conversation.extend({"role": m["role"], "content": m["content"]} for m in st.session_state.messages)

    with st.chat_message("assistant", avatar="🎵"):
        done = False
        request_failed = False
        wait_placeholder = st.empty()
        last_tool_name = None
        last_tool_args = {}
        
        while not done:
            try:
                response = run_with_fun_facts(
                    task_fn=lambda: with_openai_retry(
                        lambda: client_openai.chat.completions.create(
                            model=model_openai,
                            messages=conversation,
                            tools=tools,
                        )
                    ),
                    placeholder=wait_placeholder,
                    context_msg="Procesando tu solicitud y consultando el modelo...",
                )
            except (APIConnectionError, APITimeoutError):
                wait_placeholder.empty()
                st.error("⚠️ No pude conectar con OpenAI en este momento. Revisa tu red y vuelve a intentar.")
                request_failed = True
                done = True
                break
            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason

            if finish_reason == "tool_calls" and message.tool_calls:
                tool_calls = message.tool_calls
                for tc in tool_calls:
                    try:
                        parsed_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = {}
                    last_tool_name = tc.function.name
                    last_tool_args = parsed_args
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
        if request_failed:
            response_final = (
                "⚠️ Tuvimos un problema de conexión temporal con OpenAI. "
                "Intenta de nuevo en unos segundos."
            )
        else:
            # Streaming de respuesta con visualización progresiva
            try:
                response_final = stream_assistant_answer(
                    client=client_openai,
                    model=model_openai,
                    conversation=conversation,
                )
            except (APIConnectionError, APITimeoutError):
                response_final = (
                    "⚠️ Hubo un corte de conexión mientras generaba la respuesta. "
                    "Intenta reenviar tu última pregunta."
                )
            followup_suggestions = extract_suggested_questions_from_response(response_final)
            st.session_state.dynamic_suggestions = followup_suggestions
            response_final = response_final.strip()

    st.session_state.messages.append({"role": "assistant", "content": response_final})
    # Fuerza refresco de UI para que las sugerencias se mantengan visibles
    st.rerun()
