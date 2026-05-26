# -*- coding: utf-8 -*-
"""
Signal Lab: Pipeline de Ingeniería de Datos y Modelado Predictivo
Origen: Refactorización limpia del núcleo analítico para la ENSANUT 2024
Diseñado bajo estándares de producción de software.
"""

import os
import io
import re
import zipfile
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Componentes de Machine Learning e Infraestructura de Modelos
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, 
                             confusion_matrix, silhouette_score, calinski_harabasz_score)

# Configuración del motor de logging del sistema
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnsanutDataPipeline:
    """
    Clase Core encargada de la Ingesta, Limpieza, Alineación Estructural
    y preparación final de las matrices analíticas de la ENSANUT.
    """
    def __init__(self, ruta_datos: str = "data"):
        self.ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.directorio_datos = os.path.join(self.ruta_base, ruta_datos)
        self.archivos = {
            'antropometria': 'Cuestionario de antropometría y tensión arterial_2024.csv.zip',
            'salud_adultos': 'Cuestionario de salud de adultos (20 años o más).zip',
            'alimentos': 'Frecuencia de consumo de alimentos de adolescentes y adultos (12 años o más).zip',
            'hemoglobina': 'sangre_hemoglobina_ensanut2024_w.csv'
        }

    def cargar_datos_robusto(self, nombre_archivo: str) -> pd.DataFrame:
        """
        Ingesta flujos de datos estructurados manejando descompresión dinámica,
        detección de encodings latinos y firmas corruptas de archivos de Excel falsos.
        """
        ruta_completa = os.path.join(self.directorio_datos, nombre_archivo)
        if not os.path.exists(ruta_completa):
            logging.error(f"Falta archivo crítico de infraestructura: {nombre_archivo}")
            return pd.DataFrame()

        try:
            if nombre_archivo.endswith('.zip'):
                with zipfile.ZipFile(ruta_completa, 'r') as z:
                    # Estrategia de optimización: buscar el archivo de datos real por tamaño masivo
                    info_archivos = sorted(z.infolist(), key=lambda x: x.file_size, reverse=True)
                    archivo_interno = info_archivos[0].filename
                    
                    with z.open(archivo_interno) as f:
                        contenido = f.read()
                        if contenido.startswith(b'PK'): # Firma binaria de estructura Microsoft Office Excel
                            df = pd.read_excel(io.BytesIO(contenido))
                        else:
                            df = pd.read_csv(io.BytesIO(contenido), encoding='latin-1', sep=None, engine='python', on_bad_lines='skip')
            else:
                df = pd.read_csv(ruta_completa, encoding='latin-1', sep=None, engine='python', on_bad_lines='skip')

            # Normalización agresiva de cabeceras (Eliminación de BOM tokens y saltos de línea)
            df.columns = [str(c).strip().upper().replace('Ï»¿', '').replace('ï»¿', '') for c in df.columns]
            df.columns = [re.sub(r'[^A-Z0-9_]', '', c) for c in df.columns]
            return df
            
        except Exception as e:
            logging.error(f"Fallo en la ingesta del archivo {nombre_archivo}: {str(e)}")
            return pd.DataFrame()

    def alinear_encabezados_desplazados(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrije fallas de desalineación de renglones cuando las cabeceras bajan por errores de exportación."""
        for i in range(min(5, len(df))):
            fila = df.iloc[i].astype(str).str.upper()
            if 'UPM' in fila.values or 'FOLIO_I' in fila.values:
                df.columns = df.iloc[i].str.strip().str.upper()
                df = df.iloc[i+1:].reset_index(drop=True)
                break
        return df

    @staticmethod
    def estandarizar_clave_folio(df: pd.DataFrame) -> pd.DataFrame:
        """Garantiza e inyecta llaves primarias unificadas homogeneizando tipos de datos por Hash."""
        df = df.loc[:, df.columns.notna()]
        if 'FOLIO_I' in df.columns:
            df['FOLIO_I'] = df['FOLIO_I'].astype(str).str.strip()
        elif all(k in df.columns for k in ['UPM', 'VIV_SEL', 'NUM_REN']):
            df['FOLIO_I'] = (df['UPM'].astype(str) +
                             df['VIV_SEL'].astype(str).str.zfill(2) +
                             df['NUM_REN'].astype(str).str.zfill(2))
        return df

    @staticmethod
    def limpiar_series_numericas(df: pd.DataFrame, nombre_columna: str) -> pd.Series:
        """Limpia cadenas numéricas corruptas con comas decimales y resuelve colisiones de columnas idénticas."""
        if nombre_columna in df.columns:
            subset = df[nombre_columna]
            serie = subset.iloc[:, 0] if isinstance(subset, pd.DataFrame) else subset
            s_limpia = serie.astype(str).str.replace(',', '.').str.strip()
            return pd.to_numeric(s_limpia, errors='coerce')
        return pd.Series(np.nan, index=df.index)

    def construir_matriz_maestra(self) -> pd.DataFrame:
        """Orquesta el ciclo ETL completo e implementa el script correctivo de rescate de folios."""
        logging.info("Iniciando Ingesta y Limpieza de Componentes Analíticos...")
        
        df_antropo = self.cargar_datos_robusto(self.archivos['antropometria'])
        df_salud = self.alinear_encabezados_desplazados(self.cargar_datos_robusto(self.archivos['salud_adultos']))
        df_alimentos = self.cargar_datos_robusto(self.archivos['alimentos'])
        df_hemo = self.cargar_datos_robusto(self.archivos['hemoglobina'])

        # Integración de Folios Homogéneos
        df_antropo = self.estandarizar_clave_folio(df_antropo)
        df_salud = self.estandarizar_clave_folio(df_salud)
        df_alimentos = self.estandarizar_clave_folio(df_alimentos)
        df_hemo = self.estandarizar_clave_folio(df_hemo)

        logging.info("Ejecutando cruce de vectores de infraestructura...")
        try:
            # Integración de Infraestructura Primaria
            df_unido = pd.merge(df_salud, df_antropo, on='FOLIO_I', how='inner', suffixes=('', '_ANTRO'))
            df_unido = pd.merge(df_unido, df_alimentos, on='FOLIO_I', how='left', suffixes=('', '_ALIM'))
            df_unido = pd.merge(df_unido, df_hemo, on='FOLIO_I', how='left', suffixes=('', '_HEMO'))
        except Exception as e:
            logging.warning(f"Merge estricto fallido: {e}. Activando algoritmo de rescate por recorte estructural.")
            df_antropo['FOLIO_VIVIENDA'] = df_antropo['FOLIO_I'].str[:13]
            df_hemo['FOLIO_VIVIENDA'] = df_hemo['FOLIO_I'].str[:13]
            df_unido = pd.merge(df_antropo, df_hemo, on='FOLIO_VIVIENDA', how='inner', suffixes=('', '_HEMO'))

        logging.info(f"Unión consolidada. Registros estables: {len(df_unido)}")
        
        # Extracción y Normalización de Variables Críticas Biológicas
        df_modelo = df_unido.copy()
        df_modelo['imc'] = self.limpiar_series_numericas(df_modelo, 'IMC')
        df_modelo['sistolica'] = self.limpiar_series_numericas(df_modelo, 'AN27_01S')
        df_modelo['diastolica'] = self.limpiar_series_numericas(df_modelo, 'AN27_01D')
        df_modelo['diabetes'] = (self.limpiar_series_numericas(df_modelo, 'H0302') == 1).astype(int)
        
        # Búsqueda adaptativa de Edad
        df_modelo['edad'] = self.limpiar_series_numericas(df_modelo, 'H0303')
        if df_modelo['edad'].isna().sum() > len(df_modelo) * 0.5:
            df_modelo['edad'] = self.limpiar_series_numericas(df_modelo, 'MESES') / 12

        # Búsqueda adaptativa de Sexo (1: Hombre, 2: Mujer)
        df_modelo['sexo'] = self.limpiar_series_numericas(df_modelo, 'SEXO').fillna(self.limpiar_series_numericas(df_modelo, 'H0301')).fillna(1)

        # Filtros de Calidad Biológica y Eliminación de Códigos de Error (999/800+)
        df_modelo = df_modelo[df_modelo['imc'].between(10, 80)].dropna(subset=['imc', 'diabetes']).copy()
        df_modelo.loc[df_modelo['sistolica'] > 300, 'sistolica'] = np.nan
        df_modelo.loc[df_modelo['diastolica'] > 200, 'diastolica'] = np.nan

        # Depuración de duplicados por Folio Integrador Único
        if 'FOLIO_I' in df_modelo.columns:
            df_modelo = df_modelo.drop_duplicates(subset=['FOLIO_I'])

        # Feature Engineering Avanzado (Indicadores de Presión)
        df_modelo['presion_pulso'] = df_modelo['sistolica'] - df_modelo['diastolica']
        df_modelo['riesgo_edad_imc'] = df_modelo['edad'] * df_modelo['imc']

        # Almacenamiento Profesional del Dataset Preparado
        ruta_salida = os.path.join(self.directorio_datos, 'base_preparada_samsung.csv')
        df_modelo[['FOLIO_I', 'sexo', 'edad', 'imc', 'sistolica', 'diastolica', 'presion_pulso', 'riesgo_edad_imc', 'diabetes']].to_csv(ruta_salida, index=False)
        logging.info(f"Matriz Analítica Guardada de forma exitosa en: {ruta_salida}")
        
        return df_modelo


def entrenar_modelos_clasificacion(df: pd.DataFrame):
    """
    Orquesta el entrenamiento competitivo entre clasificadores probabilísticos,
    búsqueda de hiperparámetros por malla e impresión de métricas en producción.
    """
    features = ['sexo', 'edad', 'imc', 'sistolica', 'diastolica', 'presion_pulso', 'riesgo_edad_imc']
    X = df[features]
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    # Configuración de Pipelines Robustos con Imputación y Escalado Automático
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['edad', 'imc', 'sistolica', 'diastolica']),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), ['sexo'])
    ])

    # --- TORNEO MODELO 1: RANDOM FOREST OPTIMIZADO ---
    pipeline_rf = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=0, n_jobs=-1, class_weight='balanced'))])
    param_grid_rf = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    
    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=cv_stratified, scoring='f1', n_jobs=-1)
    logging.info("Optimizando hiperparámetros de Random Forest...")
    grid_rf.fit(X_train, y_train)
    
    # --- EVALUACIÓN GENERAL DE RENDIMIENTO ---
    best_model = grid_rf.best_estimator_
    y_pred = best_model.predict(X_test)
    logging.info("\n" + "="*40 + "\nREPORTE DE CLASIFICACIÓN FINAL (RF)\n" + "="*40)
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))


def segmentar_poblacion_kmeans(df: pd.DataFrame):
    """Ejecuta segmentación analítica no supervisada (Clustering) y genera reportes de cohesión."""
    features = ["edad", "imc", "sexo", "sistolica", "diastolica"]
    X_cluster = df[features].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    score_s = silhouette_score(X_scaled, labels)
    logging.info(f"Validación de Cohesión No Supervisada (Silhouette Score): {score_s:.4f}")


if __name__ == "__main__":
    # Instanciar orquestador e iniciar ejecución del motor analítico de producción
    pipeline = EnsanutDataPipeline(ruta_datos="data")
    df_maestro = pipeline.construir_matriz_maestra()
    
    if not df_maestro.empty:
        entrenar_modelos_clasificacion(df_maestro)
        segmentar_poblacion_kmeans(df_maestro)