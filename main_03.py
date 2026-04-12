import os
import glob
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from prompts import stronger_prompt

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)

from my_functions.tooling import handle_tool_calls, tools
from my_functions.config_st import set_bg_hack,set_animated_background,set_bg_gif


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
        response_content = None
        
        while not done:
            response = client_openai.chat.completions.create(
                model=model_openai, 
                messages=conversation, 
                tools=tools
            )
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = handle_tool_calls(tool_calls)
                conversation.append(message)
                conversation.extend(results)
            else:
                done = True
                response_content = response.choices[0].message.content
        
        st.write(response_content)

    st.session_state.messages.append({"role": "assistant", "content": response_content})