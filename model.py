# Misc
from warnings import filterwarnings

# Data Wrangling
import re
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Literal, Union, Optional

# Data visualization
#from PIL import Image
from PIL import ImageDraw
# import cufflinks as cf
from stylecloud import gen_stylecloud

import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Modeling
from keras import metrics
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import SpatialDropout1D
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Preprocessing
import nltk
import unicodedata
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Model performance
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

# # Environment setup
# cf.go_offline()
# nltk.download("all")
# filterwarnings("ignore")


from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from sklearn.model_selection import train_test_split


#################################
from my_functions.models_functions import concatenate_features,get_wordcloud,clean_text,create_embeddings_with_word2vec,plot_pca_variance


df_music_master = pd.read_csv("music_master_final_model.csv")
df_music_master.drop(columns=["t_lastfm_source_used"], inplace=True)

# Filtrar solo columnas numéricas
df_num = df_music_master.select_dtypes(include=["number"])

# Separar la variable objetivo antes de estandarizar
target_column = 'target_success_30d'
y_target = df_num[target_column]
X_features = df_num.drop(columns=[target_column])

# Escalamiento estándar de las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Crear un DataFrame con las características escaladas
df_scaled_features = pd.DataFrame(X_scaled, columns=X_features.columns, index=X_features.index)

# Unir la variable objetivo de nuevo al DataFrame estandarizado
df_music_master_ss = pd.concat([df_scaled_features, y_target], axis=1)


chi2,p = calculate_bartlett_sphericity(df_music_master_ss)
print("Prueba de Esfericidad de Bartlett")
print("Valor de Chi : ",chi2)
print("P - value : ",p)


## Dividir el dataset en train y test

# Dividimos el dataset master en trainy test



# Asumiendo que 'target_success_30d' es tu variable objetivo (y ya está en df_music_master)
# X serán las características e y será la variable objetivo
X = df_music_master.drop('target_success_30d', axis=1)
y = df_music_master['target_success_30d']

# Realizamos la división en conjuntos de entrenamiento y prueba
# test_size: proporción del dataset que se usará para prueba (ej. 0.2 para 20%)
# random_state: para reproducibilidad de los resultados
# stratify: importante si tu variable objetivo está desbalanceada, asegura que la proporción de clases sea similar en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

print(f"Dimensiones de Entrenamiento X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Dimensiones de Testeo, X_test: {X_test.shape} y_test: {y_test.shape}")


print("Reutilizamos el nombre del dataset para los valores de train")
df_music_master = pd.concat([X_train, y_train], axis=1)
print(df_music_master.shape)


print("Concatenación de características para embeddings/clustering")
df_music_master_processed = concatenate_features(df_music_master, target_column='target_success_30d')

print(df_music_master_processed[['concatenated_features', 'target_success_30d']].head())


## Nube de palabras de las características concatenadas

# Ejemplo de uso de la nueva función
texto_ejemplo = " ".join(df_music_master_processed['concatenated_features'].dropna().sample(frac=0.01))


def _patch_textsize(self, text, font=None, *args, **kwargs):
    left, top, right, bottom = self.textbbox((0, 0), text, font=font, *args, **kwargs)
    return right - left, bottom - top

# Parcheamos la librería dinámicamente
ImageDraw.ImageDraw.textsize = _patch_textsize

# Llamando a la función con los nuevos parámetros
img = get_wordcloud(
    text=texto_ejemplo, 
    icon="fas fa-music", 
    background_color="black",
    output_dir="outputs/mis_nubes", 
    image_name="nube_palabras.png"
)


# 1. Eliminamos la URL específica que mencionas
df_music_master_processed['concatenated_features'] = df_music_master_processed['concatenated_features'].str.replace('https://www.last.fm/music/', ' ', regex=False)

# 2. De forma opcional y preventiva, eliminamos cualquier otra URL (que empiece con http o https)
df_music_master_processed['concatenated_features'] = df_music_master_processed['concatenated_features'].str.replace(r'http\S+', ' ', regex=True)

# Mostramos un pequeño ejemplo para verificar que se haya limpiado
#df_music_master_processed['concatenated_features'].head(3)

# Eliminar IDs tipo UUID (ej. 2b9174b9-7b01-4ceb-8e67-8c0aa5909e00)
uuid_pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
df_music_master_processed['concatenated_features'] = df_music_master_processed['concatenated_features'].str.replace(uuid_pattern, ' ', regex=True)

# Verificar los cambios
print(df_music_master_processed['concatenated_features'].head(3))

stop_words = [clean_text(x) for x in stopwords.words("english")]

df_music_master_processed['concatenated_features'] = df_music_master_processed['concatenated_features'].map(lambda sentence: " ".join([word for word in sentence.split() if word not in stop_words]))

# Ejemplo de uso de la nueva función
texto_procesado = " ".join(df_music_master_processed['concatenated_features'].dropna().sample(frac=0.01))

# Eliminar palabras que contienen números (como E3, E7, fragmentos alfanuméricos, etc.)
texto_procesado = re.sub(r'\b\w*\d\w*\b', ' ', texto_procesado)
# Eliminar cualquier palabra de 1 o 2 letras
texto_procesado = re.sub(r'\b[a-zA-Z]{1,2}\b', ' ', texto_procesado)
# Limpiar espacios extra
texto_procesado = re.sub(r'\s+', ' ', texto_procesado).strip()

# Llamando a la función con los nuevos parámetros
img = get_wordcloud(
    text=texto_procesado, 
    icon="fas fa-music", 
    background_color="white",
    output_dir="outputs/mis_nubes", 
    image_name="clean_nube_palabras.png"
)

print(df_music_master_processed[['t_title','t_artist_name','concatenated_features']].head(3))

df_music_master_processed["words"] = df_music_master_processed["concatenated_features"].str.split()

print(df_music_master_processed["words"].str.len().describe(percentiles=[.95]))

## Embedings

MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 50

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df_music_master_processed["concatenated_features"].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Generar embeddings para el DataFrame
train_embeddings = create_embeddings_with_word2vec(df_music_master_processed, 'concatenated_features')
print("Dimensiones de los embeddings generados: ", train_embeddings.shape)

# Reduccion de dimensiones

# Se usan los embeddings anteriormente realizados

pca_99 = plot_pca_variance(train_embeddings, n_components=20)

varianza_explicada_acumulada = np.cumsum(pca_99.explained_variance_ratio_ * 100)
print(f'La varianza explicada correspondiente a usar 2 componentes es: {varianza_explicada_acumulada[2]}')

varianza_explicada_acumulada = np.cumsum(pca_99.explained_variance_ratio_ * 100)
print(f'La varianza explicada correspondiente a usar 3 componentes es: {varianza_explicada_acumulada[3]}')

print("Se utiliza 3 componentes para mantener una varianza explicada acumulada del 81%")
pca_3 = PCA(n_components=3)
pca_3.fit(train_embeddings)

df_pca_3 = pd.DataFrame(pca_3.transform(train_embeddings), columns=['PC1', 'PC2', 'PC3'])

# Definicion de clusters
# CODO - KMeans


