import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
import unicodedata
from stylecloud import gen_stylecloud
from PIL import Image
import os



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