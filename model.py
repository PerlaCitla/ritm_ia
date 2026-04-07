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

# # Preprocessing
# import nltk
# import unicodedata
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.sequence import pad_sequences

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
from my_functions.models_functions import concatenate_features,get_wordcloud,clean_text


df_music_master = pd.read_csv("music_master_final_model.csv")

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
    background_color="white",
    output_dir="outputs/mis_nubes", 
    image_name="nube_prueba.png"
)