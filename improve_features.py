"""
Script para combinar embeddings de texto con features numéricas que NO causen data leakage.
Esto reemplazará el pipeline actual solo-embeddings con uno híbrido.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Features a EVITAR (causa data leakage / derivadas del target)
FEATURES_TO_DROP = [
    'target_success_30d',
    'c_estimated_30d_listeners',    # Derivada de la que se calcula el target
    'c_lastfm_listeners_per_day',   # Derivada
    'c_estimated_30d_playcount',    # Derivada del target
    'c_lastfm_plays_per_day',       # Derivada
    'c_trend_score_v0',             # Target de regresión
    'c_score_popularity',           # Derivada
    'c_score_users',                # Derivada
    'c_score_recency',              # Derivada
    'c_lb_total_user_count',        # Derivada
    'c_lb_total_listen_count',      # Derivada
    'c_estimated_30d_lsiteners',    # Typo: derivada
    'c_lastfm_listeners',           # POTENCIAL data leakage
    'c_lastfm_playcount',           # POTENCIAL data leakage
]

def get_numeric_features_safe(df: pd.DataFrame, target_success_col: str = 'target_success_30d') -> np.ndarray:
    """
    Extrae features numéricas seguras (sin data leakage).
    
    Selecciona todas las columnas numéricas excepto las que pueden causar leakage.
    Normaliza las features usando StandardScaler.
    
    Returns:
        np.ndarray: Array de shape (n_samples, n_features) normalizado
    """
    # Seleccionar solo numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filtrar columnas problemáticas
    safe_cols = [col for col in numeric_cols if col not in FEATURES_TO_DROP and col != target_success_col]
    
    print(f"Features numéricas seguras seleccionadas ({len(safe_cols)}):")
    print(safe_cols[:10], "..." if len(safe_cols) > 10 else "")
    
    X_numeric = df[safe_cols].fillna(0)
    
    # Normalizar
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    
    return X_numeric_scaled, safe_cols, scaler


def combine_embeddings_with_numeric_features(embeddings: np.ndarray,
                                             numeric_features: np.ndarray,
                                             w2vec_weight: float = 0.5,
                                             numeric_weight: float = 0.5) -> np.ndarray:
    """
    Combina embeddings de Word2Vec con features numéricas.
    
    Args:
        embeddings: Array (n_samples, 100) de Word2Vec
        numeric_features: Array (n_samples, m) de features numéricas normalizadas
        w2vec_weight: Peso para los embeddings (default 0.5)
        numeric_weight: Peso para features numéricas (default 0.5)
    
    Returns:
        Array (n_samples, 100 + min(m, 100)) con features combinadas
    """
    # Si numeric_features tiene muchas dimensiones, usar PCA
    if numeric_features.shape[1] > 100:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=100)
        numeric_features_reduced = pca.fit_transform(numeric_features)
        print(f"Features numéricas reducidas a 100 componentes (varianza explicada: {pca.explained_variance_ratio_.sum():.2%})")
    else:
        numeric_features_reduced = numeric_features
    
    # Normalizar ambos a escala similar
    embeddings_scaled = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-8) * w2vec_weight
    numeric_scaled = (numeric_features_reduced - numeric_features_reduced.mean(axis=0)) / (numeric_features_reduced.std(axis=0) + 1e-8) * numeric_weight
    
    # Combinar (pad con ceros si es necesario)
    if numeric_scaled.shape[1] < embeddings_scaled.shape[1]:
        numeric_scaled = np.hstack([numeric_scaled, np.zeros((numeric_scaled.shape[0], embeddings_scaled.shape[1] - numeric_scaled.shape[1]))])
    elif numeric_scaled.shape[1] > embeddings_scaled.shape[1]:
        embeddings_scaled = np.hstack([embeddings_scaled, np.zeros((embeddings_scaled.shape[0], numeric_scaled.shape[1] - embeddings_scaled.shape[1]))])
    
    combined = embeddings_scaled + numeric_scaled
    
    print(f"Embeddings combinados: shape {combined.shape}")
    return combined


if __name__ == "__main__":
    # Test
    df = pd.read_csv("music_master_final_model.csv")
    X_numeric, safe_cols, scaler = get_numeric_features_safe(df)
    print(f"\nX_numeric shape: {X_numeric.shape}")
    
    # Simular embeddings
    embeddings = np.random.randn(X_numeric.shape[0], 100)
    combined = combine_embeddings_with_numeric_features(embeddings, X_numeric)
    print(f"Combined shape: {combined.shape}")
