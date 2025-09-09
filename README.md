# ML Model Optimization & Caching API (WIP)

An experimental project to explore **hyperparameter optimization with Optuna** and **model caching with Redis**, wrapped in a **FastAPI** service and containerized with **Docker**.  

## Goals
- Learn and implement **Optuna** for ML model hyperparameter tuning.  
- Explore **Redis caching** to speed up inference requests.  
- Build a clean, **FastAPI-based API** to serve predictions.  
- Practice **Dockerizing** multi-service applications.  

---

## Tech Stack
- **Machine Learning:** PyTorch or TensorFlow, XGBoost  
- **Hyperparameter Optimization:** Optuna (+ optuna-dashboard)  
- **Caching:** Redis  
- **API Framework:** FastAPI  
- **Containerization:** Docker  

---

## ⚡ Features
- Upload an image with **FastAPI**
- Store uploaded images in **Redis**
- Run inference using a **PyTorch model**
- Return prediction results as JSON
- Portable with **Docker + docker-compose**

---

## Installation (Local)

### 1. Clone the repository
```bash
git clone https://github.com/sopacha/ml-optuna-redis-api.git
cd ml-optuna-redis-api.git
```

### 2. Install dependencies
```bash 
pip install -r requirements.txt
```

### 3. Start Redis
```bash
redis-server
```

---

## Usage (Local)

### 1. Run the API server

```bash
uvicorn app.main:app --reload
```

### 2. Test the API
Visit http://127.0.0.1:8000/docs

---

## Run with Docker

### 1. Build and run with docker-compose

```bash
docker-compose up --build
```

### 2. Test the API
Visit http://127.0.0.1:8000/docs
