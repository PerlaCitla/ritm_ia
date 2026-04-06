import streamlit as st

st.title('🎶 Ritm-IA ✨')
st.caption('📈 Tendencias musicales con IA 🤖')

prompt = st.chat_input('¿En que te puedo ayudar?')

if prompt:
    st.write("El usuario ha enviado el siguiente prompt: ", prompt)