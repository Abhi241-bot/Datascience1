---
title: Real-Time Fraud Detection
emoji: 🛡️
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Real-Time Fraud Detection — Live Demo

This Hugging Face Space serves an XGBoost fraud-scoring API (`POST /predict`,
Swagger at `/docs`) together with a live Evidently drift dashboard
(`/monitoring`). Click **"Simulate a drift event"** on the home page, then open
the drift report to watch it flip from "no drift" to "drift detected".

> This is the deployable single-container demo. The full closed-loop system
> (MLflow registry + automated drift→retrain→promote) runs locally with
> `docker compose up` — see the project repository.
