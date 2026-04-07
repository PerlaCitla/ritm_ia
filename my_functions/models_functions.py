from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA


import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import unicodedata
import pandas as pd
import numpy as np
import seaborn as sns
import re
import os

from sklearn.cluster import KMeans, DBSCAN
from yellowbrick.cluster import KElbowVisualizer

from stylecloud import gen_stylecloud
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

from keras.layers import Input, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import plotly.io as pio



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

def plot_kmeans_elbow(df, k_range=(1, 15), subcarpeta="outputs/dim_reduc"):
    # Crear la subcarpeta si no existe
    os.makedirs(subcarpeta, exist_ok=True)
    ruta_guardado = os.path.join(subcarpeta, "elbow_method.png")

    model = KMeans(init='k-means++', max_iter=50, n_init=10)
    
    # Parche para compatibilidad entre scikit-learn >= 1.6 y Yellowbrick
    if getattr(model, "_estimator_type", None) != "clusterer":
        model._estimator_type = "clusterer"
        
    visualizer = KElbowVisualizer(model, k=k_range)
    visualizer.fit(df)

    # Guardar el gráfico
    visualizer.show(outpath=ruta_guardado)
    print(f"Gráfico guardado en: {ruta_guardado}")

    return visualizer


def plot_pca_kmeans_3d(df, embeddings, k=4, subcarpeta="outputs/dim_reduc"):
    df_copy = df.copy()
    
    # 1. Reducción de Dimensionalidad con PCA a 3 componentes
    pca_3d = PCA(n_components=3)
    reduced_embeddings_3d = pca_3d.fit_transform(embeddings)

    # 2. Clustering con K-Means
    kmeans_pca = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans_pca.fit_predict(reduced_embeddings_3d)

    # Añadimos los resultados al DataFrame
    df_copy['pca_component_1'] = reduced_embeddings_3d[:, 0]
    df_copy['pca_component_2'] = reduced_embeddings_3d[:, 1]
    df_copy['pca_component_3'] = reduced_embeddings_3d[:, 2]
    df_copy['cluster'] = clusters

    # 3. Visualización de los Clusters en 3D
    fig_3d = px.scatter_3d(
        df_copy,
        x='pca_component_1',
        y='pca_component_2',
        z='pca_component_3',
        color='cluster',
        title='Clusters de K-Means en Embeddings Reducidos por PCA (3D)',
        hover_data=['target_success_30d'] if 'target_success_30d' in df_copy.columns else None
    )
    
    # Crear la subcarpeta si no existe y guardar el gráfico interactivo
    os.makedirs(subcarpeta, exist_ok=True)
    ruta_guardado = os.path.join(subcarpeta, "clusters_pca_3d.html")
    fig_3d.write_html(ruta_guardado)
    print(f"Gráfico interactivo guardado en: {ruta_guardado}")
    
    # Mostrar figura (funciona bien en local y Jupyter)
    # fig_3d.show()
    
    return df_copy, kmeans_pca, pca_3d

def plot_cluster_feature_distribution(df: pd.DataFrame,
                                      feature_column: str,
                                      cluster_column: str = 'clusters',
                                      is_binary: bool = False,
                                      palette_name: str = "Oranges",
                                      figsize: tuple = (15, 3),
                                      show_labels: bool = True,
                                      treat_as_category: bool = True,
                                      subcarpeta: str = "outputs/dim_reduc") -> None:
    """
    Genera gráficos de barras que muestran la distribución de una columna
    (`feature_column`) para cada cluster (`cluster_column`).

    Ahora soporta variables numéricas que deben tratarse como categorías
    (ej. años, códigos, etc.) convirtiéndolas a string/categorical.

    Parámetros:
    -----------
    df : pd.DataFrame
        El DataFrame de entrada que contiene los datos.
    feature_column : str
        El nombre de la columna a graficar.
    cluster_column : str, opcional
        El nombre de la columna que contiene las etiquetas de los clusters. Por defecto es 'cluster'.
    is_binary : bool, opcional
        Si la `feature_column` es binaria (contiene solo 0 y 1). Si es True,
        asegura que ambos 0 y 1 estén presentes en los conteos. Por defecto es False.
    palette_name : str, opcional
        Nombre de la paleta de colores de seaborn a usar. Por defecto es 'Oranges'.
    figsize : tuple, opcional
        Tamaño total de la figura. Por defecto es (15, 3).
    show_labels : bool, opcional
        Si se deben mostrar etiquetas numéricas en las barras. Por defecto es True.
    treat_as_category : bool, opcional
        Si se debe forzar a tratar la columna como categórica (ej. años, códigos).
    subcarpeta : str, opcional
        Carpeta donde se guardarán las imágenes generadas. Por defecto es 'outputs/dim_reduc'.
    """
    unique_clusters = sorted(df[cluster_column].dropna().unique())
    fig, axes = plt.subplots(nrows=1, ncols=len(unique_clusters), figsize=figsize)

    if len(unique_clusters) == 1:
        axes = [axes]

    for i, cluster in enumerate(unique_clusters):
        cluster_data = df[df[cluster_column] == cluster]

        if is_binary:
            counts = cluster_data[feature_column].dropna().value_counts().reindex([0, 1], fill_value=0)
        else:
            counts = cluster_data[feature_column].dropna().value_counts()


        counts_df = counts.reset_index()
        counts_df.columns = ['Value', 'Count']

        # Forzar categórico si se pide
        if treat_as_category:
            counts_df['Value'] = counts_df['Value'].astype(str)
            # Escape dollar signs in 'Value' column
            counts_df['Value'] = counts_df['Value'].str.replace('$', '\\$', regex=False)


        sns.barplot(x='Count', y='Value', data=counts_df,
                    ax=axes[i], palette=palette_name, hue='Value', legend=False)

        if show_labels:
            for p in axes[i].patches:
                width = p.get_width()
                y_pos = p.get_y() + p.get_height() / 2
                axes[i].text(width + axes[i].get_xlim()[1] * 0.02, y_pos,
                             f'{int(width)}', va='center')

        axes[i].set_xlabel("Count")
        axes[i].set_title(f"Cluster {cluster}")

        max_count = counts_df['Count'].max()
        axes[i].set_xlim([0, max_count * 1.2])
        axes[i].set_ylim([-0.5, len(counts_df) - 0.5])

    fig.tight_layout()
    fig.suptitle(feature_column)
    
    # Crear la subcarpeta si no existe y guardar la imagen
    os.makedirs(subcarpeta, exist_ok=True)
    ruta_guardado = os.path.join(subcarpeta, f"dist_{feature_column}.png")
    plt.savefig(ruta_guardado, bbox_inches='tight')
    print(f"Gráfico guardado en: {ruta_guardado}")
    
    # plt.show()

def train_cluster_model(cluster_id, X_train_c, y_train_c, class_weight_c, epochs=50, batch_size=1024):
    """
    Entrena un modelo Sequential de Keras para un cluster específico,
    grafica el historial de entrenamiento y guarda la imagen.
    """
    # Define the model
    model = Sequential()
    model.add(Input(shape=(X_train_c.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Create a specific ModelCheckpoint
    os.makedirs('models', exist_ok=True)
    checkpoint_path = f'models/model_c{cluster_id}_best.keras'
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 monitor='val_accuracy',
                                 verbose=1)

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, min_delta=0.0001)

    # Train the model
    print(f"Training model for Cluster {cluster_id}...")
    history = model.fit(
        X_train_c,
        y_train_c,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        verbose=1,
        class_weight=class_weight_c
    )

    # Convert history to DataFrame
    results = pd.DataFrame(history.history)

    # Visualization with Matplotlib to easily save as PNG
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Accuracy
    axes[0].plot(results[['accuracy', 'val_accuracy']])
    axes[0].set_title(f'Cluster {cluster_id} - Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Validation'], loc='lower right')
    
    # Plot Loss
    axes[1].plot(results[['loss', 'val_loss']])
    axes[1].set_title(f'Cluster {cluster_id} - Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Train', 'Validation'], loc='upper right')

    # Save the plot in outputs/dim_reduc
    output_dir = "outputs/dim_reduc"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"history_model_c{cluster_id}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\nGráfico de historial guardado en: {plot_path}")

    #plt.show()

    return model, history