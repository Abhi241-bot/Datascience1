# Deployment Guide — Hugging Face Spaces (Docker SDK)

The public demo is a **single self-contained container** that trains the model
at build time and serves the inference API + drift dashboard on port `7860`.
It needs no MLflow server and no external storage — ideal for a free Space.

> The full closed-loop system (MLflow registry + automated drift→retrain→promote)
> is the local `docker compose up` experience. The Space is the clickable
> recruiter-facing slice: `/predict`, `/docs`, `/monitoring`, and a one-click
> "Simulate a drift event".

## What gets deployed

| File | Role in the Space |
|---|---|
| `deploy/Dockerfile` → repo-root `Dockerfile` | Builds the image; trains the model during build |
| `deploy/README.md` → repo-root `README.md` | Hugging Face Space config (frontmatter: `sdk: docker`, `app_port: 7860`) |
| `src/` | Application code |
| `requirements.txt` | Dependencies |

## Steps

1. **Create the Space**
   - Go to https://huggingface.co/new-space
   - Name it (e.g. `fraud-detection-realtime`), choose **Docker** as the SDK,
     and **Blank** template. Free hardware (CPU basic) is sufficient.

2. **Add the files** (either push with git or upload via the web UI). The Space
   repo must contain, at its root:
   - `Dockerfile`  ← copy of `deploy/Dockerfile`
   - `README.md`   ← copy of `deploy/README.md` (keep the frontmatter)
   - `requirements.txt`
   - the `src/` directory

   Using git:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/fraud-detection-realtime
   cd fraud-detection-realtime
   cp /path/to/project/deploy/Dockerfile ./Dockerfile
   cp /path/to/project/deploy/README.md  ./README.md
   cp /path/to/project/requirements.txt  ./requirements.txt
   cp -r /path/to/project/src            ./src
   git add . && git commit -m "Deploy fraud detection demo" && git push
   ```

3. **Wait for the build.** The build downloads the dataset (~143 MB) and trains
   XGBoost — first build takes a few minutes. Watch the build logs in the Space.

4. **Open the Space.** You should see the landing page with a drift badge.
   - Click **"▶ Simulate a drift event"**, then open the **live drift report** —
     it flips to "drift detected".
   - Open **`/docs`** to try `POST /predict` interactively.

5. **Put the URL in the README.** Copy your Space URL
   (`https://huggingface.co/spaces/<user>/<space>`) into the project
   `README.md` "Live demo" section.

## Local verification (optional)

The deploy app was validated locally with the project image:
```powershell
# Build the model artifact
docker run --rm -v ${PWD}/src:/app/src -v ${PWD}/data:/app/data `
  -v ${PWD}/artifacts:/app/artifacts -w /app -e GIT_PYTHON_REFRESH=quiet `
  mlops-pipeline:latest python -m src.serve.build_model

# Serve it
docker run --rm -p 7860:7860 -v ${PWD}/src:/app/src -v ${PWD}/data:/app/data `
  -v ${PWD}/artifacts:/app/artifacts -w /app -e GIT_PYTHON_REFRESH=quiet `
  mlops-pipeline:latest uvicorn src.serve.deploy:app --host 0.0.0.0 --port 7860
```
Then visit http://localhost:7860.

## Alternative: Render / Railway

The same `deploy/Dockerfile` works on Render or Railway:
- Create a new **Web Service** from the repo, set the Dockerfile path to
  `deploy/Dockerfile`, and expose port `7860` (or set `--port $PORT` in the
  start command and let the platform inject `$PORT`).
- Free tiers sleep when idle; the first request after idle will be slow.
