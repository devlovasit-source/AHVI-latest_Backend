# AHVI Backend — Cloud Run deploy recipe

> Run these from `cloudbackend-main/` after `git pull origin main`.
> Requires `gcloud` CLI authenticated against the AHVI GCP project.

## 1. One-time setup

```bash
# install gcloud (Windows): https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project ahvi-485510
gcloud config set run/region asia-south1     # pick a region close to users
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com vertexai.googleapis.com
```

## 2. Env vars

Copy your local `.env` values into Cloud Run as either a YAML file
or `--set-env-vars`. Recommended: create `deploy/env.yaml` (gitignored)
with the live values:

```yaml
AUTH_REQUIRED: "true"
RATE_LIMIT_ENABLED: "true"
RATE_LIMIT_REQUIRE_REDIS: "false"
RATE_LIMIT_FAIL_CLOSED: "false"
CORS_ALLOWED_ORIGINS: "*"

APPWRITE_ENDPOINT: "https://fra.cloud.appwrite.io/v1"
APPWRITE_PROJECT_ID: "69958f25003190519213"
APPWRITE_DATABASE_ID: "69958fe40017ccd03111"
APPWRITE_API_KEY: "REPLACE_WITH_SECRET"
APPWRITE_COLLECTION_OUTFITS: "outfits"
APPWRITE_COLLECTION_WARDROBE_STYLE_METADATA: "wardrobe_style_metadata"

AI_PROVIDER: "gemini"
GEMINI_MODEL: "gemini-2.0-flash-001"
GOOGLE_CLOUD_PROJECT: "ahvi-485510"
GOOGLE_CLOUD_LOCATION: "global"

# Agent layer flags (turn on per release confidence)
ENABLE_AGENT_STYLE_ORCHESTRATOR: "1"
AGENT_STYLE_ORCHESTRATOR_MODEL: "gemini-3.5-flash"
AGENT_STYLE_ORCHESTRATOR_TIMEOUT_SECONDS: "12"
ENABLE_AGENT_METADATA_VALIDATOR: "1"
AGENT_METADATA_VALIDATOR_MODEL: "gemini-3.5-flash"
AGENT_METADATA_VALIDATOR_TIMEOUT_SECONDS: "12"
AGENT_METADATA_LOW_CONFIDENCE_THRESHOLD: "0.55"

# Token budgets
AHVI_LLM_TOKENS_QUICK_CHAT: "500"
AHVI_LLM_TOKENS_STYLE_ADVICE: "900"
AHVI_LLM_TOKENS_OUTFIT_EXPLANATION: "1200"
AHVI_LLM_TOKENS_BOARD_EXPLANATION: "1400"
AHVI_LLM_TOKENS_CLARIFICATION: "350"
AHVI_LLM_RETRY_ON_TRUNCATION: "1"
```

Better: put `APPWRITE_API_KEY` in Secret Manager and mount with
`--set-secrets`.

## 3. Source-based deploy (uses Cloud Build under the hood)

```bash
gcloud run deploy ahvi-backend \
  --source . \
  --region asia-south1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 40 \
  --min-instances 0 \
  --max-instances 10 \
  --env-vars-file deploy/env.yaml \
  --service-account ahvi-backend-runtime@ahvi-485510.iam.gserviceaccount.com
```

Cloud Run will build from the existing `Dockerfile`, push to
Artifact Registry, and roll out a new revision.

## 4. Smoke check after deploy

```bash
SERVICE_URL=$(gcloud run services describe ahvi-backend --region asia-south1 --format='value(status.url)')
curl -fsS "$SERVICE_URL/health"
```

Tail logs while testing:

```bash
gcloud beta run services logs tail ahvi-backend --region asia-south1
```

## 5. Roll back

```bash
gcloud run services update-traffic ahvi-backend --to-revisions=PREVIOUS_REVISION_ID=100 --region asia-south1
```

## 6. Scheduled Account Purge Job (Cloud Scheduler + Cloud Run Job)

To enforce the 45-day account deletion lifecycle and execute permanent hard purges (Cloudflare R2, Qdrant vectors, Appwrite DB, and Appwrite Auth), deploy a scheduled Cloud Run Job.

### A. Create the Cloud Run Job

Deploy the job targeting the purge script using the same runtime image and environment variables:

```bash
# Enable required Google Cloud services
gcloud services enable cloudscheduler.googleapis.com run.googleapis.com

# Create the Cloud Run Job
gcloud run jobs create ahvi-purge-expired-accounts \
  --source . \
  --region asia-south1 \
  --command python \
  --args scripts/purge_expired_accounts.py \
  --memory 2Gi \
  --cpu 1 \
  --max-retries 1 \
  --task-timeout 600 \
  --env-vars-file deploy/env.yaml \
  --service-account ahvi-backend-runtime@ahvi-485510.iam.gserviceaccount.com
```

### B. Schedule Daily Execution with Cloud Scheduler

Trigger the job daily (e.g. at 03:00 UTC):

```bash
gcloud scheduler jobs create http ahvi-purge-expired-accounts-daily \
  --location asia-south1 \
  --schedule "0 3 * * *" \
  --time-zone "UTC" \
  --uri "https://asia-south1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/ahvi-485510/jobs/ahvi-purge-expired-accounts:run" \
  --http-method POST \
  --oauth-service-account-email ahvi-backend-runtime@ahvi-485510.iam.gserviceaccount.com
```

### C. Manual Run & Log Inspection

To run a dry-run or execute immediately:

```bash
# Execute job manually in Google Cloud
gcloud run jobs execute ahvi-purge-expired-accounts --region asia-south1 --wait

# Inspect execution logs
gcloud beta run jobs executions logs tail --region asia-south1
```

