import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
import re
import unicodedata
from stylecloud import gen_stylecloud
from PIL import Image
import matplotlib.pyplot as plt
import os

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize



def concatenate_features(df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """
    Concatena todas las columnas (excepto la columna objetivo si se especifica)
    en una nueva columna llamada 'concatenated_features'.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        target_column (str, optional): Nombre de la columna objetivo a excluir
                                       de la concatenación. Por defecto es None.

    Returns:
        pd.DataFrame: DataFrame con la nueva columna 'concatenated_features'.
    """
    df_copy = df.copy()
    features_to_concatenate = df_copy.columns.tolist()

    if target_column and target_column in features_to_concatenate:
        features_to_concatenate.remove(target_column)

    # Convertir todas las columnas de características a string para la concatenación
    df_copy['concatenated_features'] = df_copy[features_to_concatenate].astype(str).agg(' '.join, axis=1).str.lower()
    return df_copy

# def _patch_textsize(self, text, font=None, *args, **kwargs):
#     left, top, right, bottom = self.textbbox((0, 0), text, font=font, *args, **kwargs)
#     return right - left, bottom - top

# # Parcheamos la librería dinámicamente
# ImageDraw.ImageDraw.textsize = _patch_textsize

def get_wordcloud(text, icon="fas fa-music", background_color=None, output_dir="output/models", image_name="wordcloud.png"):
    # Crear el directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Construir la ruta completa
    output_path = os.path.join(output_dir, image_name)
    
    # https://fontawesome.com/icons/alicorn?s=solid
    gen_stylecloud(text=text, icon_name=icon, background_color=background_color, output_name=output_path)
    
    return Image.open(output_path)


def clean_text(text, pattern="[^a-zA-Z0-9 ]"):
    cleaned_text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
    cleaned_text = re.sub(pattern, " ", cleaned_text.decode("utf-8"), flags=re.UNICODE)
    cleaned_text = u' '.join(cleaned_text.lower().split())
    return cleaned_text


def create_embeddings_with_word2vec(df: pd.DataFrame, text_column: str, vector_size: int = 100, window: int = 5, min_count: int = 1, workers: int = 4) -> np.ndarray:
    """
    Genera embeddings para una columna de texto utilizando Word2Vec.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        text_column (str): Nombre de la columna que contiene el texto concatenado.
        vector_size (int, optional): Dimensionalidad de los vectores de palabras. Por defecto es 100.
        window (int, optional): Distancia máxima entre la palabra actual y la predicha. Por defecto es 5.
        min_count (int, optional): Ignora todas las palabras con frecuencia total inferior a este. Por defecto es 1.
        workers (int, optional): Número de hilos de trabajo para entrenar el modelo. Por defecto es 4.

    Returns:
        np.ndarray: Un array NumPy con los embeddings para cada registro del DataFrame.
    """
    # 1. Tokenizar el texto
    tokenized_sentences = [word_tokenize(text) for text in df[text_column].astype(str)]

    # 2. Entrenar el modelo Word2Vec
    # Asegúrate de que las sentencias tokenizadas no estén vacías
    cleaned_tokenized_sentences = [s for s in tokenized_sentences if s]
    if not cleaned_tokenized_sentences:
        print("Advertencia: No hay frases válidas para entrenar Word2Vec. Retornando un array vacío.")
        return np.array([])

    word2vec_model = Word2Vec(
        sentences=cleaned_tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )

    # 3. Función para obtener el vector de una frase (promedio de vectores de palabras)
    def get_sentence_vector(sentence_tokens, model, vector_size):
        vectors = []
        for word in sentence_tokens:
            if word in model.wv:
                vectors.append(model.wv[word])
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(vector_size) # Retorna un vector de ceros si no hay palabras en el vocabulario

    # 4. Generar embeddings para cada fila del DataFrame original
    embeddings = np.array([
        get_sentence_vector(word_tokenize(text), word2vec_model, vector_size)
        for text in df[text_column].astype(str)
    ])

    return embeddings

def plot_pca_variance(embeddings, n_components=20, subcarpeta="outputs/dim_reduc"):
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)

    plt.figure()
    plt.grid(True)
    plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
    plt.xlabel('Nummero de componentes')
    plt.ylabel('Varianza Explicada')

    # Crear la subcarpeta si no existe y guardar la imagen
    os.makedirs(subcarpeta, exist_ok=True)
    ruta_guardado = os.path.join(subcarpeta, "pca_variance.png")
    plt.savefig(ruta_guardado)
    print(f"Gráfico guardado en: {ruta_guardado}")

    # plt.show()

    return pca