# ![Air Quality](https://img.shields.io/badge/AQI-Predictor-blue) Pearls-AQI-Predictor

![Python Version](https://img.shields.io/badge/python-3.12-green)
![License](https://img.shields.io/badge/license-MIT-blue)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-90%25-yellow)

**Predicting Air Quality Index (AQI) using Machine Learning and real-time weather data**

![Demo GIF](https://via.placeholder.com/600x200.png?text=Streamlit+Dashboard+Demo+GIF)
*Replace this placeholder with your actual demo GIF or screenshot*

Pearls-AQI-Predictor is a Python project that predicts AQI for cities using weather and pollutant data. It integrates **Hopsworks Feature Store**, provides a **FastAPI backend**, and a **Streamlit dashboard** for visualization. The project leverages **XGBoost** for modeling, implements **explainability tools**, and is designed for real-time predictions and easy deployment.

---

## 🌟 Features

* Real-time AQI prediction using **OpenWeather API** and pollutant data.
* Stores processed features in **Hopsworks Feature Store**.
* ML model using **XGBoost**, with preprocessing via **scikit-learn**.
* Explainable predictions using **SHAP** and **LIME**.
* **FastAPI** backend serving prediction and health endpoints.
* Interactive **Streamlit dashboard** with visualizations using **Plotly**, **Matplotlib**, and **Seaborn**.
* Easy orchestration with **start_services.ps1** and config management via `.env` and `config.yaml`.

---

## 🚀 Tech Stack

| Layer                          | Tools / Libraries                                                           |
| ------------------------------ | --------------------------------------------------------------------------- |
| **Web / API / UI**             | FastAPI, Uvicorn, Streamlit, Gradio (experimental)                          |
| **Machine Learning**           | XGBoost, scikit-learn, joblib, SHAP, LIME, statsmodels, pandas, NumPy       |
| **Data & Feature Store**       | Hopsworks / HSML, OpenWeather API, PyArrow, Confluent Kafka, requests/httpx |
| **Visualization**              | Plotly, Matplotlib, Seaborn                                                 |
| **Server / Validation**        | pydantic, aiofiles, python-multipart, python-dotenv, PyYAML                 |
| **Development / Testing**      | pytest, pytest-cov, tqdm                                                    |
| **Orchestration / Deployment** | start_services.ps1, train_pipeline.py, run_training.py                      |

---

## 📈 Workflow

<details>
<summary>Click to expand workflow diagram</summary>

```
                        ┌────────────────────────┐
                        │     OpenWeather API    │
                        │ (Weather & Pollution)  │
                        └──────────┬─────────────┘
                                   │
                                   ▼
                        ┌────────────────────────┐
                        │  Data Ingestion Script │
                        │  (api_client / loader) │
                        └──────────┬─────────────┘
                                   │
                                   ▼
                   ┌────────────────────────────────────┐
                   │   Hopsworks Feature Store (HSML)   │
                   │  • Upload processed weather data    │
                   │  • Manage feature groups            │
                   │  • Store historical features        │
                   └──────────┬──────────────────────────┘
                              │
                              ▼
                   ┌────────────────────────────────────┐
                   │     Model Training Pipeline         │
                   │  • Load features from Hopsworks     │
                   │  • Preprocess with scikit-learn     │
                   │  • Train & evaluate XGBoost model   │
                   │  • Explain with SHAP/LIME           │
                   │  • Save model with joblib           │
                   │  • Register model in Hopsworks      │
                   └──────────┬──────────────────────────┘
                              │
                              ▼
                   ┌────────────────────────────────────┐
                   │      FastAPI Backend (main.py)      │
                   │  • Load trained model               │
                   │  • Serve prediction & health APIs   │
                   │  • Run on Uvicorn ASGI server       │
                   └──────────┬──────────────────────────┘
                              │
                              ▼
                   ┌────────────────────────────────────┐
                   │   Streamlit Dashboard (app.py)      │
                   │  • User-friendly UI for predictions │
                   │  • Visualize charts & predictions   │
                   │  • Connects to FastAPI endpoints    │
                   └──────────┬──────────────────────────┘
                              │
                              ▼
                   ┌────────────────────────────────────┐
                   │   start_services.ps1 Script         │
                   │  • Launches FastAPI & Streamlit     │
                   │  • Loads .env / config.yaml         │
                   │  • Activates environment            │
                   └────────────────────────────────────┘
```

</details>

---

## 💾 Installation

```bash
# Clone repository
git clone https://github.com/fizzah09/Pearls-AQI-Predictor.git
cd Pearls-AQI-Predictor

# Create environment
conda create -n aqi python=3.12
conda activate aqi

# Install dependencies
pip install -r requirements.txt
```

Set environment variables:

```bash
cp .env.example .env
# Update API keys and Hopsworks credentials
```

---

## ⚡ Usage

1. **Start services**

```bash
./start_services.ps1
```

2. **API Endpoints**

* `GET /health` – Health check
* `POST /predict` – Predict AQI

3. **Dashboard**

* Open browser at `http://localhost:8501`

---

## 🗂 Project Structure

```
├── src/
│   ├── api_client.py
│   ├── data_loader_hopswork.py
│   ├── model_registry.py
│   ├── train_pipeline.py
│   └── ...
├── app.py
├── main.py
├── start_services.ps1
├── config.yaml
├── requirements.txt
├── requirements-ci.txt
└── README.md
```

---

## 🧪 Testing

```bash
pytest --cov=src
```

---

## ⚙️ Configuration

* **config.yaml** – Feature groups, schedule intervals, Hopsworks & OpenWeather API settings
* **.env** – API keys and secrets

---

## 📄 License

MIT License © Fizzah Abdullah

