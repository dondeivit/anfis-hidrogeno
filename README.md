📘 ANFIS para Predicción de Producción de Hidrógeno Verde

Este proyecto implementa un modelo ANFIS (Adaptive Neuro-Fuzzy Inference System) para predecir la producción de hidrógeno verde mediante electrólisis del agua.
El modelo utiliza como entradas:

Tiempo de electrólisis (min)

Voltaje aplicado (V)

Cantidad de catalizador (µg)

Y produce como salida:

Hidrógeno generado (mL / mg / unidad experimental)

El sistema se entrena usando:

70% de datos reales (entrenamiento)

30% de datos reales (validación no vista)

Datos sintéticos filtrados generados mediante modelos de mezcla gaussiana (GMM)

El modelo ANFIS se entrena mediante el método híbrido de Jang (mínimos cuadrados + backpropagation), con funciones de pertenencia gaussianas ajustadas automáticamente.

📁 Estructura del Proyecto
MODELO/
├── data/                         # Datos reales y sintéticos (CSV)
│   ├── datos_reales_entrenamiento.csv
│   ├── datos_reales_validacion.csv
│   └── datos_sinteticos_filtrados.csv
│
├── Images/                       # Gráficas generadas (MFs, resultados, etc.)
│
├── membership/                   # Sistema de funciones de pertenencia
│   └── __init__.py
│
├── models/                       # Modelos entrenados y escaladores (.pkl)
│   ├── anfis_model.pkl
│   ├── scaler_X.pkl
│   └── scaler_y.pkl
│
├── anfis.py                      # Implementación del modelo ANFIS
├── Modelo_Proyecto.ipynb         # Notebook principal (entrenamiento + resultados)
├── Metricas.ipynb                # Análisis de métricas del modelo
├── PSO_INPUT.ipynb               # Ejemplo de optimización con PSO (opcional)
│
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Este documento

🔧 Instalación Paso a Paso

Sigue estos pasos para ejecutar el proyecto correctamente.

1️⃣ Clonar el repositorio
git clone https://github.com/TU_USUARIO/TU_REPO.git
cd TU_REPO

2️⃣ Crear un entorno virtual
🔹 Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

🔹 Linux / Mac:
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Instalar dependencias
pip install -r requirements.txt


Esto instalará todas las librerías necesarias:

numpy

pandas

matplotlib

scikit-fuzzy

scikit-learn

scipy

pyswarm

joblib

seaborn

🚀 Entrenamiento del Modelo ANFIS

El flujo completo de entrenamiento está en:

📌 Modelo_Proyecto.ipynb

En ese notebook se realiza:

Carga de datos reales y sintéticos

Normalización mediante MinMaxScaler

Definición de funciones de pertenencia iniciales

Entrenamiento del modelo ANFIS

Ajuste automático de MFs mediante backpropagation

Validación con datos reales no vistos

Graficado de MFs iniciales vs entrenadas

Guardado del modelo entrenado

🔬 Ejecución Rápida en Script (opcional)
from joblib import dump
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from anfis import ANFIS, predict

# 1. Cargar datos
df = pd.read_csv("data/datos_reales_entrenamiento.csv")

# 2. Separar X e y
X = df[['tiempo', 'voltaje', 'catalizador']].values
y = df[['hidrogeno']].values

# 3. Normalizar
scaler_X = MinMaxScaler().fit(X)
scaler_y = MinMaxScaler().fit(y)
X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y)

# 4. Definir funciones de pertenencia iniciales (ver notebook)

# 5. Entrenar
anfis_model = ANFIS(X_scaled, y_scaled.flatten(), mfc)
anfis_model.trainHybridJangOffLine(epochs=30, k=0.01, initialGamma=1000)

# 6. Guardar modelo
dump(anfis_model, "models/anfis_model.pkl")
dump(scaler_X,   "models/scaler_X.pkl")
dump(scaler_y,   "models/scaler_y.pkl")

📈 Validación del Modelo
from sklearn.metrics import mean_squared_error, r2_score

y_pred_scaled = predict(anfis_model, X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2   = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)


Valores esperados:

RMSE ≈ 4

R² ≈ 0.94

🧠 Uso del Modelo Entrenado
from joblib import load
from anfis import predict
import numpy as np

# Cargar modelo
anfis_model = load("models/anfis_model.pkl")
scaler_X = load("models/scaler_X.pkl")
scaler_y = load("models/scaler_y.pkl")

# Ejemplo de predicción
X_new = np.array([[15, 2.8, 10]])   # tiempo, voltaje, catalizador
X_new_scaled = scaler_X.transform(X_new)
y_pred_scaled = predict(anfis_model, X_new_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

print("Predicción de hidrógeno:", y_pred[0])

📝 Notas Importantes

Las funciones de pertenencia se ajustan durante el entrenamiento.

Los parámetros mean y sigma son limitados para evitar inestabilidades.

Los datos de entrada son normalizados en [0, 1] antes de ser usados en ANFIS.

El modelo puede combinarse con PSO u otras metaheurísticas para optimización.

📬 Contacto

Puedes abrir un Issue en GitHub o contactarme directamente si necesitas ayuda o tienes sugerencias de mejora.