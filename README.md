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

## Planned Features

| Feature                         | Description |
|--------------------------------|-------------|
| **Model Training & Optimization** | Train a small CNN or XGBoost model, tune with Optuna, and save the best trial to Redis. |
| **Prediction Endpoint**        | `/predict` endpoint takes input data (image or tabular), checks Redis cache by hash, and returns results. |
| **Optimization Dashboard** (optional) | Use `optuna-dashboard` to visualize trial history and optimization progress. |
| **Dockerized Setup**           | One container for API, one for Redis. |


## 📂 Project Status
**Work in Progress** — Currently setting up repository structure and environment. Implementation steps will be added incrementally.  