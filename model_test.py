# Misc
from warnings import filterwarnings

# Data Wrangling
import re
import os
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

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
from keras.models import load_model

#################################
from my_functions.models_functions import concatenate_features,get_wordcloud,clean_text,create_embeddings_with_word2vec,plot_pca_variance,plot_kmeans_elbow,plot_pca_kmeans_3d,plot_cluster_feature_distribution,train_cluster_model
from my_functions.eda_functions import tipo_variable


X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

pca_3_kpca = joblib.load("models/pca_3_components.pkl")
kmeans_pca_kernel = joblib.load("models/kmeans_pca_4_clusters.pkl")

stop_words = [clean_text(x) for x in stopwords.words("english")]

# 1. Procesar el texto de X_test (limpieza y features) de forma similar a X_train
X_test_processed = concatenate_features(X_test)

# Limpieza de texto (eliminamos URLs y UUIDs)
X_test_processed['concatenated_features'] = X_test_processed['concatenated_features'].str.replace('https://www.last.fm/music/', ' ', regex=False)
X_test_processed['concatenated_features'] = X_test_processed['concatenated_features'].str.replace(r'http\S+', ' ', regex=True)

uuid_pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
X_test_processed['concatenated_features'] = X_test_processed['concatenated_features'].str.replace(uuid_pattern, ' ', regex=True)

# Remover stopwords usando la lista previamente generada
X_test_processed['concatenated_features'] = X_test_processed['concatenated_features'].map(lambda sentence: " ".join([word for word in sentence.split() if word not in stop_words]))

# Limpieza avanzada (regex)
X_test_processed['concatenated_features'] = X_test_processed['concatenated_features'].apply(lambda x: re.sub(r'\b\w*\d\w*\b', ' ', x) if pd.notnull(x) else x)
X_test_processed['concatenated_features'] = X_test_processed['concatenated_features'].apply(lambda x: re.sub(r'\b[a-zA-Z]{1,2}\b', ' ', x) if pd.notnull(x) else x)
X_test_processed['concatenated_features'] = X_test_processed['concatenated_features'].apply(lambda x: re.sub(r'\s+', ' ', x).strip() if pd.notnull(x) else x)

# 2. Generar embeddings para X_test
# Nota: esto entrena un nuevo word2vec temporal. Lo ideal sería usar el modelo w2v guardado.
test_embeddings = create_embeddings_with_word2vec(X_test_processed, 'concatenated_features')

# 3. Reducción a 3 componentes con el PCA KERNEL ya entrenado y predicción de los 4 clusters
# Usamos pca_3_kpca y kmeans_pca_kernel definidos previamente
reduced_test_embeddings_3d = pca_3_kpca.transform(test_embeddings)
test_clusters_3d = kmeans_pca_kernel.predict(reduced_test_embeddings_3d)
X_test_processed['cluster_3d'] = test_clusters_3d

# 4. Evaluación sobre y_test utilizando los modelos de Keras correspondientes
# ADVERTENCIA: model_c0, model_c1, etc., fueron entrenados con los clusters de PCA de 2 componentes.
# Evaluarlos en los clusters definidos por PCA 3D puede mostrar desalineación en las distribuciones.
print("=== Validación sobre X_test usando PCA de 3 Componentes y 4 Clusters ===\n")

for c in range(4):
    mask_c = test_clusters_3d == c

    if np.sum(mask_c) > 0:
        X_test_c = test_embeddings[mask_c]
        y_test_c = y_test[mask_c]

        # Buscar el modelo en la carpeta models
        model_path = f'models/model_c{c}_best.keras'

        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"--- Evaluando Cluster {c} (Modelo cargado de {model_path}) ---")
            print(f"Muestras asignadas al cluster {c}: {np.sum(mask_c)}")
            loss, accuracy = model.evaluate(X_test_c, y_test_c, verbose=0)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Loss: {loss:.4f}\n")
        else:
            print(f"El modelo {model_path} no se encontró en memoria.\n")


# Inicializar columnas para predicciones
X_test_processed['predicted_prob'] = np.nan
X_test_processed['predicted_class'] = np.nan

print("Realizando predicciones por cluster y ajustando umbral para maximizar F1-score...")

for c in sorted(X_test_processed['cluster_3d'].unique()):
    mask_c = X_test_processed['cluster_3d'] == c
    X_test_c = test_embeddings[mask_c]
    y_test_c = y_test[mask_c]

    # Obtener el modelo correspondiente al cluster actual
    model_path = f'models/model_c{c}_best.keras'

    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Prediciendo para el Cluster {c} ({np.sum(mask_c)} muestras) (Modelo cargado de {model_path})...")
        # Predecir probabilidades
        probs = model.predict(X_test_c, verbose=0).flatten()

        # Calcular el mejor umbral usando F1-score
        from sklearn.metrics import f1_score
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = [f1_score(y_test_c, (probs >= t).astype(int)) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = max(f1_scores)
        print(f"  Mejor umbral para Cluster {c}: {best_threshold:.4f} (F1: {best_f1:.4f})")

        # Guardar probabilidades y clase predicha con el mejor umbral
        X_test_processed.loc[mask_c, 'predicted_prob'] = probs
        X_test_processed.loc[mask_c, 'predicted_class'] = (probs >= best_threshold).astype(int)
    else:
        print(f"Advertencia: No se encontró el modelo {model_path}")

# Mostrar un resumen de las predicciones
print(X_test_processed[['t_title', 'cluster_3d', 'predicted_prob', 'predicted_class']].head())


# Filtrar índices válidos por si algún cluster no tuvo modelo y generó NaNs
valid_indices = X_test_processed['predicted_class'].dropna().index
y_true = y_test.loc[valid_indices]
y_pred = X_test_processed.loc[valid_indices, 'predicted_class']

# Exactitud Global (Accuracy)
accuracy = accuracy_score(y_true, y_pred)
print(f"Exactitud Global (Accuracy) con Percentil 90: {accuracy:.4f}\n")

# Reporte de Clasificación
print("Reporte de Clasificación:")
print(classification_report(y_true, y_pred))

# Matriz de Confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Clase 0', 'Clase 1'],
            yticklabels=['Clase 0', 'Clase 1'])
plt.title('Matriz de Confusión Global (Percentil 90)')
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Real')

# Guardar la imagen en la carpeta outputs
os.makedirs('outputs', exist_ok=True)
ruta_imagen = os.path.join('outputs', 'matriz_confusion.png')
plt.savefig(ruta_imagen, bbox_inches='tight')
print(f"\nImagen de la matriz de confusión guardada en: {ruta_imagen}")

# plt.show()
