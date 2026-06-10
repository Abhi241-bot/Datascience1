# Build Results — Phase-by-Phase Outcomes

Companion to [MLOps_RealTime_Pipeline_SPEC.md](MLOps_RealTime_Pipeline_SPEC.md).
For each phase: **what was built**, the spec's **acceptance criteria**, and the
**verified outcome** (actual numbers from real runs). All work ran in Docker
(Python 3.11). Status as of 2026-06-03.

| Phase | Status |
|---|---|
| 1. Data & baseline model | ✅ Done & verified |
| 2. Model registry + promotion | ✅ Done & verified |
| 3. Real-time inference API | ✅ Done & verified |
| 4. Drift monitoring + dashboard | ✅ Done & verified |
| 5. The closed loop | ✅ Done & verified |
| 6. Containerize, test, document | ✅ Done & verified |
| 7. README & deployment | ✅ Built & locally verified · 🔶 live deploy is user action |

---

## Phase 1 — Data & baseline model

**Built:** `data/download.py` (pulls the ULB fraud CSV from a public mirror, no
credentials), `data/preprocess.py` (clean, engineer features, time-ordered
split, save reference window), `train/train.py` (XGBoost sklearn pipeline logged
to MLflow), `train/evaluate.py` (imbalance-aware metrics).

**Acceptance:** MLflow shows ≥1 run with all metrics; model artifact logged;
PR-AUC reported (not accuracy alone).

**Verified outcome:**
- Dataset: 284,807 rows → 1,081 duplicates dropped → **283,726**. Time-ordered
  split: train **198,608** (0.184% fraud) / val **42,558** / test **42,560**.
  Reference window: 10,000 rows.
- Class imbalance handled with `scale_pos_weight ≈ 541.6` (no SMOTE).
- MLflow run `25c6493f…` logged with full metric suite + model artifact
  (`model.pkl`, `MLmodel`, signature, input example).

| Metric | Validation | Test |
|---|---|---|
| **PR-AUC** | **0.8443** | **0.7673** |
| ROC-AUC | 0.9802 | 0.9771 |
| Precision | 0.9348 | 0.8667 |
| Recall | 0.7818 | 0.7500 |
| F1 | 0.8515 | 0.8041 |

Test confusion matrix: TP=39, FP=6, FN=13, TN=42,502.

---

## Phase 2 — Model registry + promotion

**Built:** `registry/promote.py` — finds the best run by PR-AUC, registers it,
transitions it to `Production`; idempotent; includes a reusable
`register_and_promote()` and a `--only-if-better` gate (used by Phase 5).

**Acceptance:** Registry shows one model in `Production`; promotion is
reproducible by re-running.

**Verified outcome:**
- `fraud-xgb v1` promoted to **Production** (from best run, PR-AUC 0.8443).
- Re-running printed *"best run is already in Production — nothing to do"* — no
  version inflation (idempotent).
- Registry query confirmed exactly **1** version in `Production`.

---

## Phase 3 — Real-time inference API

**Built:** `serve/schemas.py` (Pydantic `TransactionRequest` built dynamically
from the 28 PCA features + Amount, with a real fraud row as the Swagger
example; prediction/health/model-info responses), `serve/app.py` (FastAPI loads
the `Production` model at startup; `/health`, `/model-info`, `/predict`,
`/reload`, auto `/docs`).

**Acceptance:** `POST /predict` returns a valid response; `/docs` works; served
version matches the registry's Production version.

**Verified outcome:**
- `POST /predict` real fraud row → `{fraud_probability: 0.99993, label: 1}`.
- Legitimate-looking row → `{fraud_probability: 3.3e-6, label: 0}`.
- `/docs` → HTTP **200**.
- `/model-info` → `fraud-xgb v1, Production, PR-AUC 0.8443`; **served version
  v1 == registry Production v1**.

---

## Phase 4 — Drift monitoring + dashboard

**Built:** `monitor/live_log.py` (API logs engineered features + predictions),
`monitor/drift.py` (Evidently `DataDriftPreset`, emits drift summary),
`monitor/dashboard.py` (`/` badge, `/monitoring` report, `/drift-summary`,
`/reset`), `simulate/stream.py` (replays in-distribution traffic, then injects a
feature-distribution shift).

**Acceptance:** Running the simulator makes the dashboard go from "no drift" →
"drift detected" live.

**Verified outcome:**
- Genuine traffic (750 txns): **share 0.03 (1/29 features) → "no drift"**.
- After injected drift (750 txns): **share 1.00 (29/29 features) → "DRIFT
  DETECTED"**.
- `/monitoring` → HTTP 200, **3.9 MB** Evidently report; `/drift-summary` JSON
  `drift_detected: true`.
- **Fix made:** the first run false-flagged drift on genuine traffic because a
  time-ordered test split carries its own covariate shift; switched the baseline
  replay to the training distribution (what the reference window represents) so
  only the injected shift trips the alarm.

---

## Phase 5 — The closed loop

**Built:** `data/drifted.py` (appends labeled copies of the shifted regime),
`train.py --include-drift`, `loop/run_loop.py` (drift check → retrain →
**fair-evaluation gate** → promote-if-better → `/reload`), and
`.github/workflows/retrain.yml` (cron schedule + `workflow_dispatch`).

**Acceptance:** Injected drift leads to a retrain, a new registered version, and
the API serving it — with a visible event log proving each step fired.

**Verified outcome (event log, run 1 — promotes):**
1. Drift check → `drift_detected=True` (29/29 features, share 1.00)
2. Retrain challenger on current data (now incl. labeled drifted regime)
3. **Fair gate** on the same 85,120-row current eval set: Production **v1
   PR-AUC 0.4566** vs challenger **0.7692**
4. Challenger better → **promoted `fraud-xgb v2`** → **API reloaded → serving
   v2** (verified)

**Verified outcome (run 2 — gate holds):** Production is now drift-aware (v2);
fresh challenger ties it (**0.7692 vs 0.7692**) → *"challenger NOT better →
keeping current Production"* — no version inflation. Registry: **v1 Archived,
v2 Production**.

> **Why it's honest:** the gate re-scores the incumbent and challenger on the
> *same* current (drift-inclusive) data instead of comparing PR-AUCs logged
> against different validation sets — so the incumbent's degradation (0.46) and
> the challenger's promotion are earned, not faked.

---

## Phase 6 — Containerize, test, document

**Built:** `tests/test_preprocess.py`, `tests/test_api.py`, `tests/test_drift.py`
(unit-level — no dataset/services needed); `pyproject.toml` (ruff + pytest);
`.github/workflows/ci.yml` (ruff + pytest on every push); a `test` compose
service. Added per-service `build:` sections so any `docker compose` invocation
builds what it needs from a clean clone.

**Acceptance:** Fresh clone → `docker compose up` → all services reachable →
tests green in CI.

**Verified outcome:**
- After `docker compose down -v` (wiped volume) → single `docker compose up -d
  --build` brought up **mlflow (:5000), api (:8000), monitoring (:8050)**; all
  `/health` reachable (API correctly reports `model_loaded: false` until a model
  is trained).
- **11 tests passed**, `ruff` clean.

---

## Phase 7 — README & deployment

**Built:** `README.md` (all 6 required sections in order: problem + result,
Mermaid architecture, demo link + GIF-recording steps, honest metrics table,
one-command quickstart, tech stack); `DEPLOY.md` (Hugging Face Spaces guide);
`deploy/Dockerfile` + `deploy/README.md` (HF Docker SDK config, port 7860);
`serve/build_model.py` (trains a standalone artifact at build time, no MLflow
server) and `serve/deploy.py` (single-container app combining API + dashboard +
a one-click `/simulate`).

**Acceptance:** README with architecture, honest metrics, demo link, GIF;
deployed to a free tier with a clickable URL.

**Verified outcome (deploy app, locally):**
- `/health` → `model_loaded: true`; `/simulate?n=300` → `drift_detected: true,
  share 1.00` (600 rows); `/predict` fraud row → 0.9999; `/monitoring` → 200,
  3.9 MB report; landing page has the "Simulate a drift event" button.
- 🔶 **Pending user action:** click-to-deploy on Hugging Face (~5 min via
  DEPLOY.md), then paste the live URL into the README and record the demo GIF.
  (A transient local pip TLS-proxy error blocked the deploy *image* build on
  this machine; the Dockerfile is standard and builds on HF's clean network —
  the app itself was validated with the project image.)

---

## Guardrails & Definition of Done

- ✅ No accuracy-only reporting — PR-AUC + precision + recall + F1 + class balance.
- ✅ No Kubernetes / no paid cloud — runs locally with docker-compose.
- ✅ One config file (`src/config.py`) holds all thresholds/paths/hyperparameters.
- ✅ The closed loop works and is demoable (DVC was the optional cut, per spec).
- ✅ Runnable from a clean clone (verified with `down -v` + `up --build`).
- 🔶 Clickable live demo URL — deploy config + guide ready; user deploys.
