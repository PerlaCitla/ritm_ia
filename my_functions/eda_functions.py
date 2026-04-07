import pandas as pd
import numpy as np


def tipo_variable(df: pd.DataFrame, umbral_unicos: int = 5) -> pd.DataFrame:
    """Clasifica las variables de un DataFrame como continuas o discretas
    y muestra el número de valores únicos por columna.

    Se considera una variable como:
    - Discreta: si es de tipo entero, categórico o tiene pocos valores únicos.
    - Continua: si es de tipo flotante y tiene muchos valores únicos.

    Args:
        df (pd.DataFrame): DataFrame con las variables a clasificar.
        umbral_unicos (int, optional): Número máximo de valores únicos para considerar
            una variable como discreta. Por defecto es 15.

    Returns:
        pd.DataFrame: DataFrame con el nombre de cada columna como índice y dos columnas:
            - 'tipo_variable': 'continua' o 'discreta'
            - 'n_valores_unicos': número de valores únicos en la columna
    """
    resultados = []
    discretas = []
    continuas = []

    for col in df.columns:
        tipo_dato = df[col].dtype
        n_unicos = df[col].nunique()

        if pd.api.types.is_numeric_dtype(tipo_dato):
            if pd.api.types.is_object_dtype(tipo_dato) or n_unicos <= umbral_unicos:
                tipo = 'discreta'
                discretas.append(col)
            else:
                tipo = 'continua'
                continuas.append(col)
        else:
            tipo = 'discreta'
            discretas.append(col)

        resultados.append({
            'columna': col,
            'tipo_variable': tipo,
            'n_valores_unicos': n_unicos
        })

    resumen_df = pd.DataFrame(resultados).set_index('columna')
    return resumen_df, discretas, continuas

def find_completely_null_columns(df: pd.DataFrame) -> list:
    """
    Calculates the percentage of null values for each column in a DataFrame
    and returns a list of columns that are entirely null.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list of column names that are 100% null.
    """
    # Calculate the percentage of nulls for each column
    null_percentage = df.isnull().sum() / df.shape[0]

    # Identify columns that have 100% null values
    columns_with_all_nulls = null_percentage[null_percentage == 1].index.tolist()

    return columns_with_all_nulls

def rename_columns_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns in a DataFrame based on their inferred type and specific keywords.

    - Numeric columns get 'c_' prefix.
    - Date columns get 'd_' prefix (even if they start as object and are convertible).
    - Text columns (comments, descriptions, URLs, names, etc.) get 't_' prefix.
    - Categorical columns (low cardinality object) get 'v_' prefix.
    - Other columns remain unchanged.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with columns renamed according to their types.
    """
    df_copy = df.copy()
    new_columns = []

    # Threshold for considering an object column as categorical based on unique values ratio
    LOW_CARDINALITY_THRESHOLD_RATIO = 0.05
    # Keywords to identify text columns
    TEXT_KEYWORDS = ['comment', 'description', 'url', 'name', 'about', 'overview']
    # Keywords to prioritize date conversion for object columns
    # Removed 'review' as it can falsely identify non-date columns like 'reviewer_name'
    DATE_KEYWORDS = ['date', 'scraped', 'since', 'calendar_updated']

    for col in df_copy.columns:
        current_series = df_copy[col]
        col_name_lower = col.lower()
        new_col_name = col  # Default to original name

        # Priority 1: Numeric columns
        if pd.api.types.is_numeric_dtype(current_series):
            new_col_name = f'c_{col}'
        # Priority 2: Already datetime columns
        elif pd.api.types.is_datetime64_any_dtype(current_series):
            new_col_name = f'd_{col}'
        # Priority 3: Object columns - further classification (including potential date conversion)
        elif pd.api.types.is_object_dtype(current_series):
            converted_to_date = False
            # Attempt to convert to datetime if column name suggests it
            if any(keyword in col_name_lower for keyword in DATE_KEYWORDS):
                temp_date_series = pd.to_datetime(current_series, errors='coerce')
                # Check if conversion was successful for at least some non-null values
                if not temp_date_series.isnull().all():
                    df_copy[col] = temp_date_series  # Update the column in the dataframe copy
                    new_col_name = f'd_{col}'
                    converted_to_date = True

            if not converted_to_date:
                # If not a date keyword or date conversion failed, then check for text or categorical
                # Check for text columns first (by keyword in name)
                if any(keyword in col_name_lower for keyword in TEXT_KEYWORDS):
                    new_col_name = f't_{col}'
                # Next, check for categorical columns (by low cardinality)
                elif current_series.nunique() / len(df_copy) < LOW_CARDINALITY_THRESHOLD_RATIO:
                    new_col_name = f'v_{col}'

        new_columns.append(new_col_name)

    df_copy.columns = new_columns
    return df_copy

def identificar_outliers_iqr(df, columnas):
    """
    Identifica los índices de las filas que contienen outliers 
    basado en el método IQR para las columnas especificadas.
    """
    indices_outliers = set()
    
    for col in columnas:
        # Solo considerar valores no nulos para calcular los cuartiles
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir los límites
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Encontrar los índices de los outliers en esta columna
        outliers_col = df[(df[col] < limite_inferior) | (df[col] > limite_superior)].index
        indices_outliers.update(outliers_col)
        
    return list(indices_outliers)

def remove_high_null_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Elimina las columnas de un DataFrame que tienen un porcentaje de nulos
    mayor al umbral especificado.
    """
    # Se saca el porcentaje de nulos
    null_percentage = df.isnull().sum() / df.shape[0]

    # Se guarda las columnas que tienen un porcentaje de nulos mayor al umbral
    columns_to_drop = null_percentage[null_percentage > threshold].index.tolist()
    
    print(f"Columnas eliminadas (>{threshold*100}% nulos): {columns_to_drop}")

    # Retorna el dataframe sin esas columnas
    return df.drop(columns=columns_to_drop)

def get_columns_by_null_percentage(df: pd.DataFrame, min_pct: float = 0.0, max_pct: float = 0.2) -> List[str]:
    """
    Obtiene una lista de columnas cuyo porcentaje de nulos está dentro de un rango específico.
    """
    null_percentage = df.isnull().sum() / df.shape[0]
    return null_percentage[(null_percentage > min_pct) & (null_percentage < max_pct)].index.tolist()

def impute_missing_values(df: pd.DataFrame, columns_to_impute: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Imputes missing values in a DataFrame based on column type prefixes and an optional list of columns.
    If 'columns_to_impute' is not provided, all columns with missing values are considered.
    - 'c_': Numeric columns imputed with the mean.
    - 'v_': Categorical columns imputed with the mode.
    - 'd_': Datetime columns imputed with the mode.
    - 't_': Text columns imputed with 'missing'.
    Columns that are entirely NaN/NaT will not be imputed, assuming they were handled previously.

    Args:
        df (pd.DataFrame): The input DataFrame with columns prefixed by type.
        columns_to_impute (Optional[List[str]]): A list of column names to specifically impute.
                                                  If None, all columns with missing values are considered.

    Returns:
        pd.DataFrame: A new DataFrame with specified missing values imputed.
    """
    df_imputed = df.copy()

    # Determine which columns to process
    if columns_to_impute is not None:
        cols_to_process = [col for col in columns_to_impute if col in df_imputed.columns]
        if len(cols_to_process) != len(columns_to_impute):
            missing_cols = set(columns_to_impute) - set(cols_to_process)
            print(f"Warning: The following columns specified in 'columns_to_impute' were not found in the DataFrame: {missing_cols}")
    else:
        # Process all columns if no specific list is given
        # Filter to only columns that actually have nulls, to optimize
        cols_to_process = [col for col in df_imputed.columns if df_imputed[col].isnull().any()]

    for col in cols_to_process:
        # Skip imputation if the column is entirely null/NaT (should have been dropped, but for robustness)
        if df_imputed[col].isnull().all():
            print(f"Info: Column '{col}' is entirely null/NaT and will not be imputed.")
            continue

        # Only proceed if there are partial NaNs (already checked `isnull().any()` for `cols_to_process`)
        if col.startswith('c_'):  # Numeric columns
            if pd.api.types.is_numeric_dtype(df_imputed[col]):
                df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)
            else:
                print(f"Warning: Column {col} prefixed as numeric but not numeric dtype. Skipping mean imputation.")
        elif col.startswith('v_') or col.startswith('d_'):  # Categorical or Datetime columns
            mode_val = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else None
            if mode_val is not None:
                df_imputed[col].fillna(mode_val, inplace=True)
            else:
                print(f"Warning: Column {col} has no mode to impute. Skipping mode imputation.")
        elif col.startswith('t_'):  # Text columns
            df_imputed[col].fillna('missing', inplace=True)
        else:
            print(f"Column {col} does not have a recognized type prefix (c_, v_, d_, t_). Skipping imputation for this column.")
    return df_imputed