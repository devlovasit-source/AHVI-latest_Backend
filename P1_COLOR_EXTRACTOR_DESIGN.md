# P1 Color Metadata Track — Image Dominant-Color Extractor (DESIGN ONLY)

No code edited, no DB write, no R2 write, no commit, no deploy. This is a build spec for a **dry-run** script.

## 1. Current state
- **Storage**: `style_assets.colors` is an Appwrite **string array** (`string`, size 64, `array=True` — `create_style_assets_collection.py`). Values are lowercase canonical color words, e.g. `["burgundy","cream"]`.
- **How scoring uses it** (`services/style_reasoning_engine.py::_asset_score`): rewards **matches** between an asset's `colors`/blob and the direction palette/hero color (`+2` per palette match, `+10` exact hero-color via `_extract_simple_colors`). It only *rewards similarity* — there is **no clash/harmony penalty**.
- **Why missing colors break Phase C**: the Shared-Style-Brain visual guard (`colors_clash` / `palette_clashes`) needs each asset's `colors` to detect e.g. burgundy+green. On an asset with `colors=[]`, the clash check is **skipped** (`AHVI_VISUAL_GUARD_COLOR_SKIP_NOCOLOR`) → wrong-but-pretty boards still slip through.
- **Counts** (`data/style_asset_metadata_audit.csv`): **209 / 529 assets missing colors**. Text inference (`scripts/patch_asset_colors.py::_infer_colors`, word-boundary, fixed) confidently fills only **~29** (`data/style_asset_backfill_proposal.csv`, confidence=high). The remaining **~180** carry no color word in name/subcategory/asset_id → **require image extraction**.

## 2. Extractor design — `scripts/extract_style_asset_colors_dryrun.py`
**Input** (one of):
- live `style_assets` (page via `AppwriteProxy.list_documents`, read-only) filtered to `colors==[]`, OR
- `--from-csv data/style_asset_metadata_audit.csv` (rows where `missing_colors==1`), reading `image_url` from the live doc or a joined export.
**Auth/env**: load `.env` like `patch_asset_colors.py` (importer/proxy need `APPWRITE_*`; R2 not required — public `image_url` GET is enough).
**Output**: `data/style_asset_color_image_proposal.csv`
```
asset_id, name, category, image_url, current_colors, suggested_colors,
dominant_hex, secondary_hexes, confidence, method, reason, status
```
- `method`: `image_kmeans` | `image_quantize` | `text_fallback` | `skipped`
- `status`: `ok` | `low_conf` | `failed` | `skipped_nonimage` | `skipped_unsafe`

## 3. Algorithm (deterministic, Pillow-based; no ML dependency required)
1. **Fetch (safe)**: HTTP GET (not HEAD — r2.dev blocks HEAD, returns 403), `User-Agent` set, `timeout=15s`, cap body (`MAX_BYTES=6MB`), verify `Content-Type` starts `image/`; else `status=skipped_nonimage`. Sleep `RATE_LIMIT=0.2s` between fetches.
2. **Decode**: `PIL.Image.open(BytesIO)`. Keep alpha if present (`RGBA`).
3. **Resize**: `thumbnail((128,128))` — speed + denoise.
4. **Background removal (heuristic)**:
   - Drop pixels with `alpha < 16` (transparent PNG cutouts).
   - Sample 4 corners; if corners are near-uniform, treat as background color → drop pixels within `ΔE/RGB distance < 18` of it.
   - Drop near-white (`min(r,g,b) > 240`) and near-black (`max < 16`) **as background** — UNLESS, after removal, <12% of pixels remain → the garment likely **is** white/black → re-include and mark color white/black at **medium** confidence.
5. **Cluster** visible pixels:
   - Primary: `Image.quantize(colors=6, method=MEDIANCUT)` → palette + per-bucket pixel counts (pure Pillow, deterministic).
   - Optional faster/cleaner path if `numpy`+`sklearn` available: KMeans(k=5) on sampled pixels (`method=image_kmeans`). Keep Pillow as the no-dep fallback (`method=image_quantize`).
6. **Rank**: keep clusters ≥ `MIN_SHARE=12%` of garment pixels; take top **1–3**.
7. **Map centroid → canonical name** (RGB→HSV, table in §3a). `dominant_hex` = top cluster hex; `secondary_hexes` = others.
8. **Limit** to 1–3 deduped canonical names, ordered by share.

### 3a. RGB/HSL → canonical color table (allowed set only)
Canonical names: `black, white, grey, navy, blue, brown, beige, cream, green, olive, red, burgundy, maroon, pink, yellow, orange, purple, gold, silver`.
Mapping (HSV, H 0–360, S/V 0–1) — deterministic thresholds:
- `V<0.18` → **black**
- `S<0.12 & V>0.85` → **white**; `S<0.12 & 0.85≥V>0.65` → **silver/grey** (silver if slight cool tint), `S<0.12 & 0.65≥V>0.18` → **grey**
- low-sat warm light (`S 0.12–0.35, V>0.75, H 30–60`) → **cream**; (`S 0.2–0.45, V 0.55–0.8, H 30–55`) → **beige**
- H 20–45 mid: `S>0.5, V 0.3–0.6` → **brown**; metallic warm high-V low-mid-sat near H40–50 → **gold**
- H 0–15 / 345–360: `V>0.5,S>0.5` → **red**; dark (`V 0.25–0.5`) → **maroon**; desaturated-dark wine (`H 330–360, V 0.25–0.5`) → **burgundy**
- H 15–45 high-sat bright → **orange**; H 45–65 → **yellow**
- H 65–160 → **green**; (`H 60–90, V<0.5, S 0.3–0.7`) → **olive**
- H 160–250: bright→**blue**; dark (`V<0.35`)→**navy**
- H 250–290 → **purple**; H 290–345 → **pink**
(Thresholds tuned on the 8 test fixtures §7; keep table in one dict for easy iteration.)

## 4. Safety
- **Dry-run only**: writes one CSV; **no DB write, no R2 write, no image mutation** (decode in memory).
- HTTP: GET with `timeout`, byte cap, content-type gate, `RATE_LIMIT` sleep, retry≤1.
- **Skip unsafe URLs**: non-`https`, non-r2.dev/known-host, non-image content-type → `status=skipped_unsafe/nonimage`, no fetch of arbitrary hosts.
- Every failure → a **row** (`status=failed`, `reason=<exc[:120]>`), never crash the batch (item-level try/except).
- Idempotent: re-runnable; overwrites only the proposal CSV.

## 5. Validation
- Cross-check vs **filename inference** (`patch_asset_colors._infer_colors(name, subcategory, asset_id)`):
  - image color ∈ filename colors → **confidence boost** (agreement).
  - conflict (image says blue, name says red) → cap at **medium**, `reason=text_image_conflict` for human review.
  - filename empty (the 180 case) → rely on image, confidence from clustering quality.
- Vs **existing colors**: target set has `colors=[]`; if any non-empty slips in, only propose additions, never overwrite (flag).
- **Human-review threshold**: only `high` is auto-appliable later (§6).

### Confidence rules
- **high**: one dominant garment cluster ≥ `HIGH_SHARE=55%` after bg removal, clean (not accessory/jewellery/skincare category), agrees-with or no-conflict-with filename.
- **medium**: 2–3 colors, or noisy background, or text/image mild conflict, or white/black-is-garment fallback.
- **low**: category in {accessory, jewellery, grooming/skincare-like} (small/metallic/reflective → unreliable), white-background dominance with no garment pixels, or cluster failure.

## 6. Apply strategy (FUTURE — not this task)
1. Human reviews `data/style_asset_color_image_proposal.csv`.
2. Apply **high-confidence rows only**, first batch, via a dry-run-defaulted apply step (extend `scripts/patch_asset_colors.py` to accept a proposal CSV, or a sibling `apply_color_proposal.py --dry-run/--apply`). Writes `colors` + `updated_at` via `AppwriteProxy.update_document` (the existing safe path).
3. **Medium** → manual review queue.
4. **Low** → skip; revisit after RMBG/cutout improves background removal.
5. Re-run the metadata audit to confirm `missing_colors` drops.

## 7. Tests (`tests/test_color_extractor.py`) — local fixture images, no network
| fixture | expected |
|---|---|
| white shirt on white bg | garment-is-bg fallback → `white`, medium |
| black dress on dark bg | `black`, medium/high |
| multicolor saree | 2–3 colors, medium, `reason=multi_cluster` |
| beige bag | `beige` (or beige+brown), high/medium |
| gold jewellery | low confidence (metallic/reflective, accessory) |
| transparent PNG (cutout) | alpha-drop works → garment color, high |
| broken URL | `status=failed`, row written, no crash |
| non-image URL (html) | `status=skipped_nonimage` |
Plus unit tests on the **RGB→canonical** mapper with synthetic swatches (burgundy `#5b1a2b`→burgundy, navy `#1b2a4a`→navy, olive `#6b6b23`→olive, etc.).

## 8. Deliverable
This doc only (`P1_COLOR_EXTRACTOR_DESIGN.md`). No code, no DB, no commit, no deploy.

### Dependencies / notes
- **Pillow** present (12.1.1). `numpy`/`scikit-learn` **optional** (faster KMeans path); pure-Pillow `quantize` is the dependency-light default.
- Effectiveness ceiling: backgrounds are not always clean → accessory/jewellery stay low-confidence until a cutout/RMBG step exists (ties to P5 RMBG pipeline). Expect ~120–150 of the 180 to land high/medium; rest manual.

---

## Backend engineer — two parallel tasks
1. **Shared Style Brain A+B+C** behind `STYLE_SHARED_BRAIN` — per `P1_SHARED_STYLE_BRAIN_IMPLEMENTATION.md`.
2. **Color extractor dry-run script** — per this doc → produces `data/style_asset_color_image_proposal.csv` for review.

Together they kill the top remaining quality issue — **wrong-but-pretty visual boards**: (1) adds the clash/occasion/gender guard; (2) supplies the color metadata that guard depends on.
```
```
