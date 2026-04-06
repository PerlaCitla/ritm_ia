import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from prompts import stronger_prompt

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)


model_openai = "gpt-5.4-mini"


st.title("🎶 Ritm-IA ✨")
st.caption("📈 Tendencias musicales con IA 🤖")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¿En qué te puedo ayudar?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:= st.chat_input("Escribe tu mensaje aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)
    conversation = [{"role": "assistant", "content": stronger_prompt}]
    conversation.extend({"role": m["role"], "content": m["content"]} for m in st.session_state.messages)

    with st.chat_message("assistant", avatar="🎵"):
        stream = client_openai.chat.completions.create(model= model_openai, messages=conversation, stream=True)
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})