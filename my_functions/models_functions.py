from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import numpy as np
import matplotlib.pyplot as plt
import os

def train_decision_tree(df_ml, target_dtc='target_success_30d', max_depth=5, test_size=0.2, random_state=42):
    """
    Entrena un DecisionTreeClassifier evitando fuga de datos.
    """
    # 2. Prevenir fuga de datos para este problema
    cols_to_drop_dtc = [
        target_dtc,
        'c_estimated_30d_listeners',   # Esta es la causa más probable de la fuga de datos
        'c_lastfm_listeners_per_day',  # También es una métrica derivada que podría causar fuga
        'c_estimated_30d_playcount',   # Variable base con la que se calculó el target
        'c_lastfm_plays_per_day',      # Altamente correlacionada con la estimación 30d
        'c_trend_score_v0',            # Target de regresión
        'c_score_popularity',
        'c_score_users',
        'c_score_recency',
        'c_lb_total_user_count',
        'c_lb_total_listen_count',
        'c_estimated_30d_lsiteners',   # Variable base con la que se calculó el target
        'c_lastfm_listeners',
        'c_lastfm_playcount'
    ]

    # Asegurarse de que solo se intenten eliminar columnas que realmente existen en df_ml
    existing_cols_to_drop_dtc = [col for col in cols_to_drop_dtc if col in df_ml.columns]

    # Obtener todas las columnas numéricas
    numeric_features = df_ml.select_dtypes(include=['number']).columns.tolist()

    # Definir features_dtc excluyendo las columnas identificadas para evitar fuga de datos
    features_dtc = [col for col in numeric_features if col not in existing_cols_to_drop_dtc]

    X_dtc = df_ml[features_dtc]
    y_dtc = df_ml[target_dtc]

    # 3. Dividir datos (Train/Test)
    X_train_dtc, X_test_dtc, y_train_dtc, y_test_dtc = train_test_split(
        X_dtc, y_dtc, test_size=test_size, random_state=random_state, stratify=y_dtc
    )

    print(f"Variables predictoras usadas: {len(features_dtc)}")
    print(features_dtc)

    # 4. Entrenar el Árbol de Decisión
    # max_depth para permitir al árbol aprender más patrones y potencialmente mejorar la precisión
    dtree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, class_weight='balanced')
    dtree.fit(X_train_dtc, y_train_dtc)

    print("\nModelo de Clasificación entrenado exitosamente.")
    
    return dtree, X_train_dtc, X_test_dtc, y_train_dtc, y_test_dtc, features_dtc

def visualize_tree(dtree, features, filename="decision_tree.png"):
    # Crear la subcarpeta si no existe
    os.makedirs('outputs/models', exist_ok=True)
    
    # Visualizar el Árbol
    plt.figure(figsize=(20, 10))
    plot_tree(
        dtree,
        feature_names=features,
        class_names=['No Éxito', 'Éxito'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title('Reglas de Decisión para predecir el Éxito de un Lanzamiento', fontsize=16)
    
    # Guardar la imagen en la ruta especificada
    filepath = os.path.join('outputs/models', filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Imagen guardada exitosamente en: {filepath}")
    
    #plt.show()

def evaluate_and_visualize_tree(dtree, X_train, X_test, features, filename="decision_tree.png"):
    # 1. Predice sobre los datos de entrenamiento y validación
    y_pred_train = dtree.predict(X_train)
    y_pred_test = dtree.predict(X_test)

    # 2. Llamar a la nueva función de visualización
    visualize_tree(dtree, features, filename)

    return y_pred_train, y_pred_test


def evaluate_model_performance(model, X_test, y_train, y_pred_train, y_test, y_pred_test, model_name="Model", filename="precision_recall_curve.png"):
    # 4. Evalúa el modelo
    print(f"Métricas de ajuste ({model_name})")

    print("\nEntrenamiento:")
    print("Accuracy :", accuracy_score(y_train, y_pred_train))
    print("Precision:", precision_score(y_train, y_pred_train))
    print("Recall   :", recall_score(y_train, y_pred_train))
    print("F1       :", f1_score(y_train, y_pred_train))

    print("\nValidación:")
    print("Accuracy :", accuracy_score(y_test, y_pred_test))
    print("Precision:", precision_score(y_test, y_pred_test))
    print("Recall   :", recall_score(y_test, y_pred_test))
    print("F1       :", f1_score(y_test, y_pred_test))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred_test))

    # Curva precision-recall validación
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)
        precisions, recalls, trhs = precision_recall_curve(y_test, y_scores[:,1])

        plt.figure(figsize=(10,6))
        plt.plot(trhs, precisions[:-1], 'b--', label='Precisión')
        plt.plot(trhs, recalls[:-1], 'g--', label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Métrica')
        plt.legend(loc='upper left')
        plt.ylim([0,1])
        
        # Guardar la imagen en la subcarpeta outputs/models
        os.makedirs('outputs/models', exist_ok=True)
        filepath = os.path.join('outputs/models', filename)
        plt.savefig(filepath, bbox_inches='tight')
        print(f"\nCurva Precision-Recall guardada exitosamente en: {filepath}")
        
        # plt.show()
    else:
        print("El modelo no soporta predict_proba para graficar la curva precision-recall.")

def prepare_data_and_train_rf(df_ml, target_rfc='target_success_30d'):
    # 2. Prevenir fuga de datos para este problema
    cols_to_drop_rfc = [
        target_rfc,
        'c_estimated_30d_listeners',   # Esta es la causa más probable de la fuga de datos
        'c_lastfm_listeners_per_day',  # También es una métrica derivada que podría causar fuga
        'c_estimated_30d_playcount',   # Variable base con la que se calculó el target
        'c_lastfm_plays_per_day',      # Altamente correlacionada con la estimación 30d
        'c_trend_score_v0',            # Target de regresión
        'c_score_popularity',
        'c_score_users',
        'c_score_recency',
        'c_lb_total_user_count',
        'c_lb_total_listen_count',
        'c_estimated_30d_lsiteners',   # Variable base con la que se calculó el target
        'c_lastfm_listeners',
        'c_lastfm_playcount'
    ]

    # Asegurarse de que solo se intenten eliminar columnas que realmente existan en df_ml
    existing_cols_to_drop_rfc = [col for col in cols_to_drop_rfc if col in df_ml.columns]

    # Obtener todas las columnas numéricas
    numeric_features = df_ml.select_dtypes(include=['number']).columns.tolist()

    # Definir features_rfc excluyendo las columnas identificadas para evitar fuga de datos
    features_rfc = [col for col in numeric_features if col not in existing_cols_to_drop_rfc]

    X_rfc = df_ml[features_rfc]
    y_rfc = df_ml[target_rfc]

    # 3. Dividir datos (Train/Test)
    X_train_rfc, X_test_rfc, y_train_rfc, y_test_rfc = train_test_split(
        X_rfc, y_rfc, test_size=0.2, random_state=42, stratify=y_rfc
    )

    print(f"Variables predictoras usadas: {len(features_rfc)}")

    # 4. Entrenar el Random Forest
    # max_depth=5 para permitir al árbol aprender más patrones y potencialmente mejorar la precisión
    randomforestcl = RandomForestClassifier(max_depth=5, random_state=42, class_weight='balanced')
    randomforestcl.fit(X_train_rfc, y_train_rfc)

    # 3. Predice sobre los datos de entrenamiento y validación
    y_pred_train_rfc = randomforestcl.predict(X_train_rfc)
    y_pred_test_rfc = randomforestcl.predict(X_test_rfc)

    evaluate_model_performance(randomforestcl, X_test_rfc, y_train_rfc, y_pred_train_rfc, y_test_rfc, y_pred_test_rfc, model_name="Random Forest", filename="precision_recall_curve_random_forest.png") 


def prepare_data_and_train_xgb(df_ml, target_xgb='target_success_30d'):
    # 2. Prevenir fuga de datos para este problema
    cols_to_drop_xgb = [
        target_xgb,
        'c_estimated_30d_listeners',   # Esta es la causa más probable de la fuga de datos
        'c_lastfm_listeners_per_day',  # También es una métrica derivada que podría causar fuga
        'c_estimated_30d_playcount',   # Variable base con la que se calculó el target
        'c_lastfm_plays_per_day',      # Altamente correlacionada con la estimación 30d
        'c_trend_score_v0',            # Target de regresión
        'c_score_popularity',
        'c_score_users',
        'c_score_recency',
        'c_lb_total_user_count',
        'c_lb_total_listen_count',
        'c_estimated_30d_lsiteners',   # Variable base con la que se calculó el target
        'c_lastfm_listeners',
        'c_lastfm_playcount'
    ]

    # Asegurarse de que solo se intenten eliminar columnas que realmente existen en df_ml
    existing_cols_to_drop_xgb = [col for col in cols_to_drop_xgb if col in df_ml.columns]

    # Obtener todas las columnas numéricas
    numeric_features = df_ml.select_dtypes(include=['number']).columns.tolist()

    # Definir features_xgb excluyendo las columnas identificadas para evitar fuga de datos
    features_xgb = [col for col in numeric_features if col not in existing_cols_to_drop_xgb]

    X_xgb = df_ml[features_xgb]
    y_xgb = df_ml[target_xgb]

    # 3. Dividir datos (Train/Test)
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
        X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
    )

    print(f"Variables predictoras usadas: {len(features_xgb)}")
    print(features_xgb)

    # 4. Entrenar el modelo XGBoost
    xgboostc = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        scale_pos_weight=(len(y_train_xgb) - sum(y_train_xgb)) / sum(y_train_xgb),
        random_state=42,
        eval_metric='logloss'
    )

    xgboostc.fit(X_train_xgb, y_train_xgb)

    # 3. Predice sobre los datos de entrenamiento y validación
    y_pred_train_xgb = xgboostc.predict(X_train_xgb)
    y_pred_test_xgb = xgboostc.predict(X_test_xgb)

    evaluate_model_performance(xgboostc, X_test_xgb, y_train_xgb, y_pred_train_xgb, y_test_xgb, y_pred_test_xgb, model_name="XGBoost", filename="precision_recall_curve_xgboost.png")