# Archived Deploy Configurations

The canonical deploy target for this backend is **Google Cloud Run**, using the
top-level `Dockerfile`. The files in this directory are kept for historical
reference only and are **not** used by the production deploy.

| File | Original target | Why archived |
|------|-----------------|--------------|
| `Procfile` | Heroku-style worker definition | Cloud Run uses Dockerfile CMD |
| `railway.json` | Railway.app deploy spec | Migrated off Railway |
| `nixpacks.toml` | Railway/Nixpacks build config | Same as above |
| `runpod_3ports.sh` | RunPod GPU runtime | Vision/embedding moved to managed services |
| `runtime.txt` | Heroku Python pin | Dockerfile sets Python version |
| `RUNPOD_SETUP.md` | RunPod setup guide | Stale |
| `requirements.runpod.txt` | RunPod-only ML stack (torch, transformers, rembg) | Heavy deps not used in Cloud Run image |
| `requirements.railway.txt` | Subset for Railway builds | Drifted from `requirements.txt` |

If you ever need to reactivate one of these, copy it back to the repo root and
update CI accordingly. **Do not run them in parallel with the Cloud Run
deploy** — runtime drift causes silent feature degradation (e.g.,
`google-genai` missing from Railway requirements).
