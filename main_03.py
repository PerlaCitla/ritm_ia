import os
import glob
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

st.title("🎶 Ritm-IA ✨")
st.caption("📈 Tendencias musicales con IA 🤖")

# ==================== INICIALIZACIÓN SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¿En qué te puedo ayudar?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:= st.chat_input("Escribe tu mensaje aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)
    conversation = [{"role": "system", "content": stronger_prompt}]
    conversation.extend({"role": m["role"], "content": m["content"]} for m in st.session_state.messages)

    with st.chat_message("assistant", avatar="🎵"):
        done = False
        
        while not done:
            response = client_openai.chat.completions.create(
                model=model_openai, 
                messages=conversation, 
                tools=tools
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
                results = handle_tool_calls(tool_calls)
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
        
        # Streaming de respuesta con visualización progresiva
        response_final = stream_assistant_answer(
            client=client_openai,
            model=model_openai,
            conversation=conversation,
        )

    st.session_state.messages.append({"role": "assistant", "content": response_final})