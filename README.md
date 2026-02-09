# 🦴 Microproyecto MLOps — Clasificación de Patologías de Columna Vertebral

## 📌 Descripción del Proyecto

Este microproyecto tiene como objetivo desarrollar un **sistema de Machine Learning con enfoque MLOps** para la **clasificación automática de imágenes médicas** en tres categorías:

* 🟢 **Sana**
* 🟡 **Escoliosis**
* 🔴 **Espondilolistesis**

El sistema permitirá entrenar, evaluar, versionar y desplegar un modelo de clasificación utilizando buenas prácticas de ingeniería de datos, MLOps y despliegue de modelos.

---

## 🎯 Objetivo

Construir un **pipeline completo de MLOps** que incluya:

* Versionamiento de datos con **DVC**
* Seguimiento de experimentos con **MLflow**
* Entrenamiento de modelos de clasificación
* Exposición del modelo mediante una **API REST con FastAPI**
* Contenerización con **Docker**

---

## 🧠 Descripción del Modelo

El sistema entrenará un modelo de **clasificación multiclase**, usando CNN el cual recibirá una imagen de columna vertebral y retornará una predicción entre:

* `Healthy`
* `Scoliosis`
* `Spondylolisthesis`

Inicialmente se usará un modelo base (baseline) con redes neuronales convolucionales (CNN), el cual será iterado y optimizado.

---

## 🏗️ Arquitectura General

```
Usuario → API (FastAPI) → Modelo ML → Predicción
                          ↑
                    MLflow + DVC
```

---

## 📂 Estructura del Proyecto

```
mlops-spine-classification/
│
├── .dvc/
│   └── .gitignore
│   └── config
│
├── data/
│   ├── raw/            # Datos originales
│   ├── processed/      # Datos procesados
│
├── deployment/
│   ├── fastap_app/
│   │   └── main.py
│   │   └── schemas.py
│   └── README.md
│
├── models/
│
├── notebooks/
│   └── 01_exploration.ipynb
│   └── 02_preprocessing.ipynb
│   └── 03_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   └── load_data.py
│   │   └── preprocess.py
│   │
│   ├── evaluation/
│   │   └── metric.py
│   │   └── visualize.py
│   │
│   ├── inference/
│   │   └── predictor.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   ├── utils/
│   │   └── logger.py
│   └── __init__.py
├── models/
├── experiments/
│
├── dvc.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔁 Pipeline del Proyecto

1. Ingesta de datos
2. Preprocesamiento de imágenes
3. Entrenamiento del modelo
4. Registro de experimentos en MLflow
5. Versionamiento de datasets con DVC
6. Despliegue del modelo vía API REST

---

## 🚀 Ejecución Rápida

### Crear entorno

```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Entrenar modelo

```bash
python src/models/train.py
```

### Levantar API

```bash
uvicorn src.api.main:app --reload
```

---

## 📌 Tecnologías

* Python 3.10+
* Scikit-learn / TensorFlow / PyTorch
* MLflow
* DVC
* FastAPI
* Docker

---

## 👨‍💻 Autores

**Javier Garcia, Juan Vallarino, Patricio Romeo, Diana Rojas, Ivan Eslava**
Microproyecto académico — Machine Learning + MLOps

---

## 📜 Licencia

Proyecto académico sin fines comerciales.
