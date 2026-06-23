# Catalog Validation

## ROUND 2 — catalog generation independent of RMBG cleanup

**Problem:** prior candidate saved OK but catalog never generated. Log audit confirmed the bug:
```
ahvi.save_selected_v4 ... saved=1
ahvi.catalog.persistence_stripped document_id=893a41e7... stripped=['catalog_status']
(no ahvi.catalog.start / centered / uploaded)
```
Root cause: catalog hook lived in `_try_upload_inline_images`, which early-returns when `masked_url`
already exists (RMBG skipped) → catalog never ran.

**Fix (commit `66e0d4e`):** moved catalog generation to a guaranteed per-item step in `save_selected`
with a byte-source fallback (inline masked b64 → inline raw b64 → fetch resolved image URL). New logs
`ahvi.catalog.skip_flag_off`, `ahvi.catalog.skip_no_bytes`, `ahvi.catalog.start` (with source).
Idempotent, never blocks save. Tests: 28/28.

| | |
|---|---|
| Commit | `66e0d4e` (routers/wardrobe_capture.py + tests/test_catalog_image_service.py) |
| origin/main | `66e0d4e` |
| Candidate revision | `ahvi-backend-00526-zag` — flags ON, 0% traffic, tag `candidate` |
| Template scrub | `00453-5fv` (flags removed) |
| prod | `00445-4c4` @ 100%, flags absent |

**Gate check (round 2):**
| check | result |
|---|---|
| prod flags absent | ✅ `00445` clean |
| candidate flags present | ✅ `00526-zag` |
| no prod traffic moved | ✅ `00445` @ 100% |

**Live validation — PENDING manual upload to candidate `00526-zag`.** Re-upload Red Polka Dot Dress via
the candidate APK, then re-run the §6 log query. Expect now:
`ahvi.catalog.start` → `ahvi.catalog.uploaded` (or `ahvi.catalog.failed`) + `save_selected_v4 saved=1`,
no `Unknown attribute`.

Pre-upload audit (this revision): not yet exercised (the 10:28 entries above are the prior candidate).

---

# Catalog Persistence-Strip — Validation (Round 1)

P0: catalog_* fields must not break wardrobe save when Appwrite schema lacks them.

## 1. Commit ✅
- Hash: **`e0fc4b2`** — `fix(catalog): prevent catalog fields from blocking wardrobe save`
- Staged ONLY (no `git add .`):
  - `services/wardrobe_persistence_service.py`
  - `tests/test_catalog_persistence_strip.py`
- Tests: 4/4 passing.

## 2. Push ✅
- `53a197b..e0fc4b2  main -> main`
- `git log origin/main -1` → **`e0fc4b2`**.

## 3. Candidate deploy ✅
- Revision: **`ahvi-backend-00524-jal`** — flags ON, 0% traffic, tag `candidate`.
- URL: `https://candidate---ahvi-backend-lz3aebcusq-el.a.run.app`
- Prod traffic NOT moved.

## 4. Template scrub ✅
- Scrub revision `00451-jnz` (flags removed from template) → future prod deploys stay dormant.

### Gate verification
| check | result |
|---|---|
| prod traffic | `ahvi-backend-00445-4c4` @ 100% (unmoved) |
| prod revision flags | **absent** (`00445` clean) |
| candidate tag → | `ahvi-backend-00524-jal` |
| candidate revision flags | **present** (`ENABLE_CATALOG_IMAGE_GENERATION`, `ENABLE_CATALOG_NORMALIZATION`) |
| template (latest `00451`) | flags removed |

## 5. Real upload validation — PENDING (manual, on-device)
Uploading **Red Polka Dot Dress** runs through the candidate APK's `save-selected` flow on the phone —
a manual UI action I cannot drive headlessly (no app automation / no JWT). The installed candidate APK
already targets the candidate backend (`.env` → candidate URL), and candidate revision `00524-jal`
carries the persistence-strip fix, so an upload now exercises the fix.

**Run on the phone:** open AHVI (candidate APK) → add wardrobe item → Red Polka Dot Dress → save.

Success criteria:
- `save_selected_v4: saved=1`
- NO `Unknown attribute: "catalog_status"`
- Expect `ahvi.catalog.persistence_stripped` (schema lacks catalog_* → stripped + retried) OR catalog
  fields persisted.

## 6. Log audit
Command:
```
gcloud logging read 'resource.type="cloud_run_revision"
AND resource.labels.service_name="ahvi-backend"
AND (textPayload:"ahvi.catalog" OR textPayload:"persistence_stripped"
OR textPayload:"save_selected_v4" OR textPayload:"catalog_ready"
OR textPayload:"catalog_failed")' --project ahvi-485510 --freshness=20m
--limit=100 --format="value(timestamp,textPayload)"
```
Result (pre-upload, freshness 25m): **no matching entries** — no upload has hit the candidate yet
(candidate is 0% traffic; awaiting the manual phone upload).

## Pass / Fail
| area | verdict |
|---|---|
| Fix + tests (4/4) | ✅ PASS |
| Commit `e0fc4b2` + push | ✅ PASS |
| Candidate deploy (flags on, 0% traffic) | ✅ PASS |
| Template scrub / prod dormant | ✅ PASS |
| Gates (prod clean / candidate flagged / no traffic moved) | ✅ PASS |
| Live Red-Polka-Dot-Dress upload | ⏳ PENDING — manual phone upload, then re-run §6 |

**Not done:** prod catalog flags NOT enabled, prod traffic NOT moved, Appwrite schema NOT modified.

_After you upload on the phone, re-run the §6 command (or ask me) to fill in the live result._
