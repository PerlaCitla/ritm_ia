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
import joblib

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

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight

# Model performance
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import KernelPCA, PCA

# # Environment setup
# cf.go_offline()
# nltk.download("all")
# filterwarnings("ignore")


from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from sklearn.model_selection import train_test_split


#################################
from my_functions.models_functions import concatenate_features,get_wordcloud,clean_text,create_embeddings_with_word2vec,plot_pca_variance,plot_kmeans_elbow,plot_pca_kmeans_3d,plot_cluster_feature_distribution,train_cluster_model
from my_functions.eda_functions import tipo_variable

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

print("Guardamos X_test")
X_test.to_csv("X_test.csv", index=False)

print("Guardamos y_test")
y_test.to_csv("y_test.csv", index=False)

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

# Se guarda este modelo en la carpeta models para usarlo posteriormente en la reducción de dimensiones del dataset de testeo
joblib.dump(pca_3, "models/pca_3_components.pkl")

df_pca_3 = pd.DataFrame(pca_3.transform(train_embeddings), columns=['PC1', 'PC2', 'PC3'])

# Definicion de clusters
# CODO - KMeans

visualizer = plot_kmeans_elbow(df_pca_3)

print("Se elige k=4 para el clustering, ya que es el punto donde la curva comienza a estabilizarse (codo)")

df_music_master_reducedimentions = df_music_master_processed.copy()
df_music_master_reducedimentions, kmeans_pca, pca_3d = plot_pca_kmeans_3d(
    df_music_master_reducedimentions, 
    train_embeddings, 
    k=4)

# Se guarda Kmeans
joblib.dump(kmeans_pca, "models/kmeans_pca_4_clusters.pkl")

print("Debido a poder de computo, se queda con la reducción de dimensiones de PCA a 3 componentes y 4 clusters definidos por KMeans")


cols = [col for col in df_music_master_processed.columns if col.startswith('c_')]

print("Las columnas que son numericas para ver su distribución por cluster son: ", cols)

df_music_master['cluster'] = df_music_master_reducedimentions['cluster']

print(df_music_master.groupby('cluster')[cols].describe())

df_music_master.groupby('cluster')[cols].describe().to_csv("outputs/dim_reduc/cluster_descriptions.csv")

df, discretas, continuas = tipo_variable(df_music_master)
discretas.remove('t_release_group_id')
discretas.remove('t_artist_ids')
discretas.remove('t_lastfm_url')
discretas.remove('t_title')
discretas.remove('t_artist_name')
discretas.remove('t_artist_name_meta')
#discretas.remove('lastfm_tags_item')
discretas.remove('t_lastfm_album_name')
discretas.remove('d_first_release_date')

for i in discretas:
  plot_cluster_feature_distribution(df=df_music_master,cluster_column='cluster',feature_column=i,is_binary=False,treat_as_category=True)

print("El tamaño de cada clusters es:")
print(df_music_master['cluster'].value_counts())


## Modelado de RRN

mask_c0 = df_music_master_reducedimentions['cluster'] == 0
X_train_c0 = train_embeddings[mask_c0]
y_train_c0 = df_music_master_reducedimentions.loc[mask_c0, 'target_success_30d']

print(f"Shape of X_train_c0: {X_train_c0.shape}")
print(f"Shape of y_train_c0: {y_train_c0.shape}")


weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_c0),
    y=y_train_c0
)
class_weight_c0 = dict(zip(np.unique(y_train_c0), weights))
print(class_weight_c0)

# Cluster 0:
model_c0, history_c0 = train_cluster_model(
    cluster_id=0, 
    X_train_c=X_train_c0, 
    y_train_c=y_train_c0, 
    class_weight_c=class_weight_c0,
    epochs=50
)
print(model_c0.summary())

## Cluster 1:

mask_c1 = df_music_master_reducedimentions['cluster'] == 1
X_train_c1 = train_embeddings[mask_c1]
y_train_c1 = df_music_master_reducedimentions.loc[mask_c1, 'target_success_30d']

print(f"Shape of X_train_c1: {X_train_c1.shape}")
print(f"Shape of y_train_c1: {y_train_c1.shape}")

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_c1),
    y=y_train_c1
)
class_weight_c1 = dict(zip(np.unique(y_train_c1), weights))
print(class_weight_c1)

model_c1, history_c1 = train_cluster_model(
    cluster_id=1,
    X_train_c=X_train_c1, 
    y_train_c=y_train_c1, 
    class_weight_c=class_weight_c1,
    epochs=50
)
print(model_c1.summary())

# Mocel C2:
mask_c2 = df_music_master_reducedimentions['cluster'] == 2
X_train_c2 = train_embeddings[mask_c2]
y_train_c2 = df_music_master_reducedimentions.loc[mask_c2, 'target_success_30d']

print(f"Shape of X_train_c2: {X_train_c2.shape}")
print(f"Shape of y_train_c2: {y_train_c2.shape}")

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_c2),
    y=y_train_c2
)
class_weight_c2 = dict(zip(np.unique(y_train_c2), weights))
print(class_weight_c2)

model_c2, history_c2 = train_cluster_model(
    cluster_id=2,
    X_train_c=X_train_c2, 
    y_train_c=y_train_c2, 
    class_weight_c=class_weight_c2,
    epochs=50
)
print(model_c2.summary())


# Model C3:

mask_c3 = df_music_master_reducedimentions['cluster'] == 3
X_train_c3 = train_embeddings[mask_c3]
y_train_c3 = df_music_master_reducedimentions.loc[mask_c3, 'target_success_30d']

print(f"Shape of X_train_c3: {X_train_c3.shape}")
print(f"Shape of y_train_c3: {y_train_c3.shape}")

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_c3),
    y=y_train_c3
)
class_weight_c3 = dict(zip(np.unique(y_train_c3), weights))
print(class_weight_c3)

model_c3, history_c3 = train_cluster_model(
    cluster_id=3,
    X_train_c=X_train_c3, 
    y_train_c=y_train_c3, 
    class_weight_c=class_weight_c3,
    epochs=50
)
print(model_c3.summary())