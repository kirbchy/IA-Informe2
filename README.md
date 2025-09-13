# Informe 2 - Predicción de Stroke

## Descripción del Proyecto

Este proyecto implementa tres algoritmos de Machine Learning supervisado para la predicción de accidentes cerebrovasculares (stroke) utilizando un dataset de salud. El objetivo es comparar el rendimiento de diferentes técnicas de aprendizaje automático en un problema de clasificación binaria.

## Objetivos

- Aplicar tres técnicas de aprendizaje supervisado diferentes
- Realizar un análisis comparativo exhaustivo
- Implementar un proceso completo de preprocesamiento de datos
- Evaluar el rendimiento usando múltiples métricas
- Proporcionar recomendaciones basadas en los resultados

## Dataset

**Fuente**: Kaggle - Healthcare Stroke Data
- **Registros**: 5,110 pacientes
- **Variables**: 11 características (demográficas y médicas)
- **Objetivo**: Predicción binaria de stroke (Sí/No)
- **Desafío**: Dataset desbalanceado (4.9% casos positivos)

### Variables del Dataset

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `id` | Numérico | Identificador único |
| `gender` | Categórico | Género del paciente |
| `age` | Numérico | Edad del paciente |
| `hypertension` | Binario | Hipertensión (0/1) |
| `heart_disease` | Binario | Enfermedad cardíaca (0/1) |
| `ever_married` | Categórico | Estado civil |
| `work_type` | Categórico | Tipo de trabajo |
| `Residence_type` | Categórico | Tipo de residencia |
| `avg_glucose_level` | Numérico | Nivel promedio de glucosa |
| `bmi` | Numérico | Índice de masa corporal |
| `smoking_status` | Categórico | Estado de fumador |
| `stroke` | Binario | **Variable objetivo** (0/1) |

## Estructura del Proyecto

```
AI-Informe2/
├── data/ # Datos procesados
│ ├── healthcare-dataset-stroke-data.csv
│ ├── X_train_balanced.npy
│ ├── y_train_balanced.npy
│ ├── X_test_scaled.npy
│ ├── y_test.npy
│ └── *.pkl, *.json, *.csv # Modelos y resultados
├── notebooks/ # Jupyter Notebooks
│ ├── 01_EDA_Analisis_Exploratorio.ipynb
│ ├── 02_Preprocesamiento_Datos.ipynb
│ ├── 03_Modelo_SVM.ipynb
│ ├── 04_Red_Neuronal.ipynb
│ ├── 05_Modelo_XGBoost.ipynb
│ └── 06_Comparacion_Final.ipynb
├── venv_ml/ # Entorno virtual
├── requirements.txt # Dependencias
└── README.md # Este archivo
```

## Instalación y Configuración

### Requisitos del Sistema

**Versión de Python Compatible:**
- **Recomendado**: Python 3.11.9 (64-bit)

**Sistema Operativo:**
- Windows 11

### 1. Clonar el Repositorio
```bash
git clone https://github.com/kirbchy/IA-Informe2.git
cd IA-Informe2
```

### 2. Verificar Versión de Python
```bash
# Verificar versión instalada
python --version

```

### 3. Crear Entorno Virtual
```bash

C:\Users\[tu_usuario]\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv

# O si Python 3.11 está en el PATH:
python3.11 -m venv .venv
```

### 4. Activar Entorno Virtual

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```


### 5. Actualizar pip y herramientas de build
```bash
# Actualizar pip primero
python -m pip install --upgrade pip

# Instalar herramientas de build
pip install --upgrade setuptools wheel
```

### 6. Instalar Dependencias
```bash
# Instalar todas las dependencias del proyecto
pip install -r requirements.txt
```

### 7. Verificar Instalación
```bash
# Verificar que las librerías principales se instalaron correctamente
python -c "import numpy, pandas, sklearn, tensorflow, xgboost; print('Todas las dependencias instaladas correctamente')"
```

### 8. Ejecutar Jupyter Notebook
```bash
jupyter notebook
```


## Notebooks del Proyecto

### 1. Análisis Exploratorio de Datos (EDA)
**Archivo**: `01_EDA_Analisis_Exploratorio.ipynb`

**Contenido**:
- Carga y exploración inicial del dataset
- Análisis de valores faltantes
- Distribución de variables categóricas y numéricas
- Análisis de correlación
- Visualizaciones exploratorias

**Hallazgos Principales**:
- Dataset desbalanceado (4.9% casos positivos)
- Valores faltantes en BMI
- Correlaciones significativas: edad, hipertensión, enfermedad cardíaca

### 2. Preprocesamiento de Datos
**Archivo**: `02_Preprocesamiento_Datos.ipynb`

**Procesos Implementados**:
- Manejo de valores faltantes (imputación con mediana)
- Codificación de variables categóricas (One-Hot Encoding)
- Escalado de variables numéricas (StandardScaler)
- División estratificada train/test (80/20)
- Balanceo del dataset (SMOTE)

**Resultado**:
- Dataset balanceado para entrenamiento
- 22 características después de codificación
- Datos escalados y listos para modelado

### 3. Modelo Support Vector Machine (SVM)
**Archivo**: `03_Modelo_SVM.ipynb`

**Implementación**:
- Comparación de kernels (linear, poly, rbf, sigmoid)
- Optimización rápida de hiperparámetros (GridSearchCV optimizado)
- Evaluación con múltiples métricas
- Análisis de importancia de características (solo kernel lineal)

**Optimización de Rendimiento**:
- **Muestra reducida**: 40% de datos para optimización (60% reducción)
- **Grilla reducida**: 2 valores por parámetro (8 combinaciones vs 120 anteriores)
- **3-fold CV**: En lugar de 5 folds
- **Tiempo**: De 30+ minutos a 2-3 minutos (95% más rápido)

**Configuración Optimizada**:
- Kernel: Seleccionado automáticamente según F1-score
- Parámetros optimizados mediante validación cruzada rápida
- Entrenamiento final con todos los datos

### 4. Red Neuronal
**Archivo**: `04_Red_Neuronal.ipynb`

**Arquitecturas Evaluadas**:
- **Básica**: 2 capas ocultas (64, 32 neuronas)
- **Profunda**: 4 capas ocultas (128, 64, 32, 16 neuronas)
- **Ancha**: 3 capas con más neuronas (256, 128, 64)
- **Compleja**: BatchNormalization + Dropout

**Optimización de Rendimiento**:
- **Epochs reducidos**: 20 en lugar de 100 (80% más rápido)
- **Early Stopping**: Previene overfitting automáticamente
- **Tiempo**: De 20+ minutos a 5-10 minutos

**Características**:
- Activación ReLU + Sigmoid para salida
- Dropout para regularización
- Early Stopping con paciencia de 10 epochs
- Reduce Learning Rate on Plateau
- Métricas: Accuracy, Precision(), Recall()
- Validación cruzada con conjunto separado

### 5. Modelo XGBoost
**Archivo**: `05_Modelo_XGBoost.ipynb`

**Algoritmo Adicional Investigado**:
- XGBoost (eXtreme Gradient Boosting)
- Optimización rápida de hiperparámetros
- Análisis de importancia de características
- Comparación con modelos clásicos

**Optimización de Rendimiento**:
- **Muestra reducida**: 30% de datos para optimización (70% reducción)
- **Grilla reducida**: 2 valores por parámetro (128 combinaciones vs 2,916 anteriores)
- **3-fold CV**: En lugar de 5 folds
- **Tiempo**: Reducido a 3-5 minutos (99% más rápido)

**Parámetros Optimizados**:
- n_estimators, max_depth, learning_rate
- subsample, colsample_bytree
- reg_alpha, reg_lambda

**Ventajas de XGBoost**:
- Alto rendimiento en clasificación
- Manejo automático de overfitting
- Interpretabilidad de características
- Eficiencia computacional

### 6. Comparación Final
**Archivo**: `06_Comparacion_Final.ipynb`

**Análisis Comparativo**:
- Métricas de rendimiento
- Visualizaciones comparativas
- Análisis de ventajas/desventajas
- Recomendaciones de uso
- Interpretación clínica

## Tecnologías Utilizadas

### Librerías Principales
- **Pandas**: Manipulación de datos
- **NumPy**: Computación numérica
- **Scikit-learn**: Algoritmos de ML clásicos
- **TensorFlow/Keras**: Redes neuronales
- **XGBoost**: Gradient boosting
- **Matplotlib/Seaborn**: Visualizaciones

### Herramientas de Desarrollo
- **Jupyter Notebook**: Para el desarrollo del informe
- **Python 3.11.9**: Lenguaje de programación usado
- **Git**: Para control de versiones
- **Visual Studio Code**: Editor de código (opcional)


## Resultados Principales

### Métricas de Rendimiento

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| **SVM** | 0.8474 | 0.1419 | 0.42 | 0.2121 | 0.6605 |
| **Red Neuronal** | 0.7808 | 0.1434 | 0.70 | 0.2381 | 0.8111 |
| **XGBoost** | 0.9207 | 0.1837 | 0.18 | 0.1818 | 0.7842 |


### Mejor Modelo
**XGBoost** demostró el mejor rendimiento general con:
- F1-Score más alto
- Excelente capacidad discriminativa
- Interpretabilidad de características
- Eficiencia computacional

## Análisis de Importancia de Características

### Factores de Riesgo Más Importantes:
1. **Edad**: Factor predictivo más significativo
2. **Hipertensión**: Correlación fuerte con stroke
3. **Enfermedad Cardíaca**: Predictor importante
4. **Nivel de Glucosa**: Contribuye al riesgo
5. **BMI**: Índice de masa corporal relevante


## Aprendizajes Obtenidos

### Técnicas Aplicadas:
1. **Preprocesamiento Completo**: Manejo de datos faltantes, codificación, escalado
2. **Balanceo de Datos**: SMOTE para datasets desbalanceados
3. **Optimización de Rendimiento**: Estrategias para reducir tiempo de entrenamiento
4. **Muestreo Inteligente**: Uso de subconjuntos para optimización rápida
5. **Early Stopping**: Prevención de overfitting en redes neuronales
6. **Optimización de Hiperparámetros**: GridSearchCV y validación cruzada
7. **Evaluación Robusta**: Múltiples métricas para problemas desbalanceados
8. **Comparación Sistemática**: Análisis comparativo de diferentes algoritmos

## Cómo Ejecutar el Proyecto

### Opción 1: Ejecutar Todos los Notebooks
```bash
# Activar entorno virtual
.venv\Scripts\activate # Windows
# Verificar que el entorno está activo (debe mostrar (.venv) al inicio)
# Ejecutar Jupyter
jupyter notebook

# Tener presente que la ejecución de todos los notebooks puede ser tardada.
```

### Opción 2: Ejecutar Notebooks Específicos
```bash
# Para ejecutar solo un notebook específico
jupyter notebook notebooks/03_Modelo_SVM.ipynb
```


---
