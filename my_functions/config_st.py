import streamlit as st

def set_bg_hack(main_bg):
    '''
    Una función para establecer una imagen o color de fondo
    '''
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {main_bg};
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def set_animated_background():
    # CSS para el fondo animado
    page_bg_css = """
    <style>
    .stApp {
        background: linear-gradient(45deg, #3a5683 10%, #0E1117 45%, #0E1117 55%, #3a5683 90%);
        background-size: 200% 200%;
        animation: gradientAnimation 10s ease infinite;
        background-attachment: fixed;
    }

    @keyframes gradientAnimation {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    </style>
    """
    st.markdown(page_bg_css, unsafe_allow_html=True)
import base64


def set_bg_gif(gif_path):
    # Leer el GIF local
    with open(gif_path, "rb") as f:
        data = f.read()
    
    # Convertir a Base64
    bin_str = base64.b64encode(data).decode()
    
    # CSS para establecer el fondo
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/gif;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    # Inyectar el CSS
    st.markdown(page_bg_img, unsafe_allow_html=True)