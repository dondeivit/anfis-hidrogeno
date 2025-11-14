ANFIS para Predicción de Producción de Hidrógeno Verde
=======================================================

Este proyecto implementa un sistema ANFIS (Adaptive Neuro-Fuzzy Inference System) para modelar y predecir la producción de hidrógeno generada mediante electrólisis del agua.

El modelo utiliza tres variables de entrada principales: tiempo de electrólisis, voltaje aplicado y cantidad de catalizador.

El propósito del sistema es estimar con precisión la producción de hidrógeno a partir de datos experimentales reales, complementados con datos sintéticos generados mediante modelos de mezcla gaussiana (GMM). Además, se integra un proceso de optimización basado en PSO (Particle Swarm Optimization) para buscar la combinación de entradas que maximiza la producción de hidrógeno según el modelo entrenado.

Estructura del proyecto
=======================

```text
MODELO/
├── data/                       # Datos reales y sintéticos (CSV)
├── Images/                     # Gráficas generadas
├── membership/                 # Implementación de funciones de pertenencia
├── models/                     # Modelos entrenados y escaladores (.pkl)
│   ├── anfis_model.pkl         # Modelo ANFIS entrenado
│   ├── scaler_X.pkl
│   └── scaler_y.pkl
├── anfis.py                    # Definicion de ANFIS
├── Modelo_Proyecto.ipynb       # Notebook principal
├── Metricas.ipynb              # Metricas del modelo para no cargarlo nuevamente
├── PSO_INPUT.ipynb             # Implementacion PSO para encontrar inputs que maximizan H2
├── requirements.txt            # Dependencias
└── README.md                   # Este archivo
```
Instalación
===========

Para ejecutar este proyecto, primero clonar el repositorio:

    git clone https://github.com/dondeivit/anfis-hidrogeno.git
    cd anfis-hidrogeno

Crear un entorno virtual:

Windows (PowerShell):

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

Linux / Mac:

    python3 -m venv .venv
    source .venv/bin/activate

Instalar dependencias (desde requirements.txt):

    pip install -r requirements.txt

Esto instalará automáticamente todas las librerías necesarias para entrenar el modelo ANFIS, generar gráficas, normalizar datos y ejecutar el notebook de optimización con PSO.

Ejecución
=========

El flujo completo del modelo se encuentra en el archivo:

    Modelo_Proyecto.ipynb

Este notebook permite:

    Entrenar el modelo ANFIS
    Visualizar funciones de pertenencia iniciales y entrenadas
    Validar el modelo con datos reales
    Guardar el modelo entrenado
    Generar métricas y gráficas de desempeño

Optimización con PSO
====================

El notebook:

    PSO_INPUT.ipynb

Utiliza el modelo ANFIS entrenado para buscar:

    La combinación de tiempo, voltaje y catalizador
    que maximiza la producción de hidrógeno predicha por el modelo.

Notas
=====
```text
  Los modelos entrenados se guardan en la carpeta models/.
  Las dependencias necesarias están en requirements.txt.
```
Dependencias
============
```text
Python
numpy
pandas
matplotlib
scikit-fuzzy
scikit-learn
joblib
```
