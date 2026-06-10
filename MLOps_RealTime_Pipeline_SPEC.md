# Real-Time MLOps Inference & Monitoring Pipeline — Build Spec

---

## 0. Project Goal (read first)

Build an **end-to-end production ML system** that does NOT just train a model — it **owns a model in production**. The system must demonstrate the full closed loop:

**train → track → register → serve (real-time API) → monitor for drift → auto-trigger retraining → promote better model → serve new version**

This "closed loop" is the entire point. A recruiter clicking the live demo should see (1) a working prediction endpoint and (2) a live monitoring dashboard that detects data drift and shows a retraining event.

### Domain choice
Use **real-time energy demand forecasting** OR **e-commerce transaction fraud risk scoring**. Default to **fraud risk scoring** (binary classification, clearer drift story, easy to simulate streaming transactions). Pick a public dataset:
- Fraud: Kaggle "Credit Card Fraud Detection" (ULB) or IEEE-CIS Fraud Detection.
- If energy: the "Hourly Energy Consumption" (PJM) dataset.

Handle the class imbalance honestly (fraud is ~0.17% positive): report **precision, recall, PR-AUC, and F1** — NEVER accuracy alone. Use class weights or scale_pos_weight, not naive SMOTE on the full set.

---

## 1. Tech Stack (use exactly these — do not substitute)

| Layer | Tool | Why |
|---|---|---|
| Model | XGBoost (+ scikit-learn preprocessing) | Strong tabular baseline, fast |
| Experiment tracking + model registry | **MLflow** | Logs runs, stores artifacts, registers prod model |
| Data/model versioning | **DVC** (optional, only if time allows) | Versions datasets/models |
| Inference API | **FastAPI** + Pydantic + Uvicorn | 2026 standard, async, auto docs |
| Drift monitoring | **Evidently AI** | Data drift, target drift, concept drift |
| Dashboard | Evidently's HTML reports served via FastAPI, OR a small Streamlit page | The "real-time" recruiter hook |
| Containerization | **Docker** + docker-compose | One-command run |
| CI/CD + retraining trigger | **GitHub Actions** | Automates tests + scheduled/triggered retrain |
| Language | Python 3.11 | — |

---

## 2. Repository Structure

```
mlops-realtime-pipeline/
├── README.md                  # recruiter-grade, written LAST
├── requirements.txt
├── docker-compose.yml         # spins up: api, mlflow server, monitoring
├── Dockerfile.api
├── Dockerfile.train
├── .github/workflows/
│   ├── ci.yml                 # lint + tests on every push
│   └── retrain.yml            # triggered/scheduled retraining
├── data/
│   ├── raw/                   # downloaded dataset (gitignored, pulled via script)
│   └── reference/             # reference window for drift comparison
├── src/
│   ├── config.py              # all paths, thresholds, params in ONE place
│   ├── data/
│   │   ├── download.py        # programmatic dataset download
│   │   └── preprocess.py      # cleaning, feature engineering, train/val/test split
│   ├── train/
│   │   ├── train.py           # trains model, logs everything to MLflow
│   │   └── evaluate.py        # PR-AUC, precision, recall, F1, confusion matrix
│   ├── registry/
│   │   └── promote.py         # promotes best run to "Production" in MLflow registry
│   ├── serve/
│   │   ├── app.py             # FastAPI: /predict, /health, /model-info
│   │   └── schemas.py         # Pydantic request/response models
│   ├── monitor/
│   │   ├── drift.py           # Evidently drift computation
│   │   └── dashboard.py       # serves/refreshes the monitoring report
│   └── simulate/
│       └── stream.py          # simulates live traffic + an injected drift event
├── tests/
│   ├── test_preprocess.py
│   ├── test_api.py            # tests /predict returns valid schema
│   └── test_drift.py
└── notebooks/
    └── 01_eda.ipynb           # quick EDA only
```

---

## 3. Build Phases (do in order; confirm acceptance criteria each time)

### Phase 1 — Data & baseline model
- `download.py`: pull dataset programmatically (Kaggle API or direct URL); save to `data/raw/`.
- `preprocess.py`: clean, engineer features, create a time-ordered train/val/test split, and save a **reference window** (`data/reference/`) — this is what drift is measured against later.
- `train.py`: train XGBoost; log params, metrics, model, and the preprocessing pipeline as MLflow artifacts.
- `evaluate.py`: compute **PR-AUC, precision, recall, F1, confusion matrix**; log to MLflow.
- **✅ Acceptance:** `mlflow ui` shows at least one run with all metrics; the trained model artifact is logged. PR-AUC is reported (not just accuracy).

### Phase 2 — Model registry + promotion
- `promote.py`: query MLflow for the best run by PR-AUC, register it in the MLflow Model Registry, and transition it to stage `Production`.
- **✅ Acceptance:** Registry shows one model in `Production`; promotion is reproducible by re-running the script.

### Phase 3 — Real-time inference API
- `schemas.py`: Pydantic models for a single transaction request and the prediction response (`{fraud_probability, label, model_version}`).
- `app.py`: FastAPI app that loads the `Production` model from the registry at startup; endpoints `/health`, `/model-info` (returns current version), `/predict`.
- **✅ Acceptance:** `POST /predict` with a sample transaction returns a valid response; `/docs` (auto Swagger) works; the served version matches the registry's Production version.

### Phase 4 — Drift monitoring + dashboard (the recruiter hook)
- `drift.py`: use Evidently to compare the reference window vs a live window; compute data drift, target drift (when labels arrive), and prediction drift.
- `dashboard.py`: generate the Evidently HTML report and serve it at `/monitoring` (or a small Streamlit page). It should refresh as new live data arrives.
- `stream.py`: simulate live traffic by replaying held-out data through `/predict`, and **inject a deliberate drift event partway through** (e.g., shift a feature distribution) so the dashboard visibly flips to "drift detected."
- **✅ Acceptance:** Running the simulator shows the monitoring dashboard go from "no drift" → "drift detected" live.

### Phase 5 — The closed loop (retraining trigger)
- Logic: when `drift.py` flags drift above the threshold in `config.py`, it triggers retraining (`train.py`), which logs a new run; `promote.py` promotes it ONLY if PR-AUC beats the current Production model; the API picks up the new version.
- `retrain.yml`: a GitHub Actions workflow that runs this on a schedule (cron) AND can be triggered manually (`workflow_dispatch`). Locally, a simple script wires the same loop.
- **✅ Acceptance:** A documented end-to-end run where injected drift leads to a retrain, a new registered version, and the API serving it — with a visible event log/printout proving each step fired.

### Phase 6 — Containerize, test, document
- `docker-compose.yml`: services for `api`, `mlflow`, and `monitoring`, all reachable on localhost. `docker compose up` must work from a clean clone.
- `tests/`: the three test files above must pass; `ci.yml` runs lint (ruff) + pytest on every push.
- `README.md` (write last): see Phase 7.
- **✅ Acceptance:** Fresh clone → `docker compose up` → all services reachable → tests green in CI.

### Phase 7 — README & deployment (the part recruiters actually read)
The README must contain, in this order:
1. **One-paragraph problem statement** + a one-line result (e.g., "Detects fraud in real time with PR-AUC 0.8X; auto-retrains when data drifts").
2. **Architecture diagram** (Mermaid is fine) showing the closed loop.
3. **Live demo link** + a 30–60s GIF of the drift→retrain→promote flow.
4. **Metrics table** with PR-AUC, precision, recall, F1, and the class balance stated honestly.
5. **One-command run** (`docker compose up`) and a quickstart.
6. **Tech stack** and **what each component does**.

**Deployment:** Deploy the API + monitoring to a free tier — **Hugging Face Spaces (Docker SDK)** or **Render/Railway free tier**. Keep the MLflow server local or on the same free host. The clickable demo is mandatory — a project recruiters can't click is worth a fraction of one they can.

---

## 4. Hard Constraints / Guardrails
- **No accuracy-only reporting.** Always PR-AUC + precision + recall + F1, with class balance stated.
- **No Kubernetes, no paid cloud required.** Must run locally with docker-compose.
- **One config file** (`src/config.py`) holds all thresholds, paths, and hyperparameters — no magic numbers scattered in code.
- **The closed loop is non-negotiable.** If time runs short, cut DVC and the Streamlit polish — but the drift→retrain→promote→serve loop MUST work and be demoable.
- **Keep it runnable from a clean clone.** Test this before declaring done.

## 5. Definition of Done
- [ ] `docker compose up` works from a fresh clone
- [ ] `/predict` returns valid predictions in real time
- [ ] `/monitoring` dashboard shows live drift detection
- [ ] Injected drift triggers retraining → new model registered → promoted only if better → served
- [ ] CI (lint + tests) green on GitHub
- [ ] README with architecture diagram, honest metrics, live demo link, and a demo GIF
- [ ] Deployed to a free tier with a clickable URL
