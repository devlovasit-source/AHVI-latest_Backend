# Catalog Candidate Report — staging smoke (flags ON)

## UPDATE — validation gate calibrated → 7/7 PASS

After the calibration fix (foreground-only color, per-category area thresholds, skin skip for
accessories + tightened skin heuristic), the same 7-type smoke set now passes:

| # | case | category | rotation | occupancy % | clipping | validation | latency |
|---|---|---|---|---:|---|---|---:|
| 1 | hanger top | top | 0 | 40.3 | no | ✅ pass | 281 ms |
| 2 | mirror selfie | top | 0 | 35.9 | no | ✅ pass | 250 ms |
| 3 | flat-lay shirt | top | 0 | 44.7 | no | ✅ pass | 286 ms |
| 4 | dress | dress | 0 | 20.2 | no | ✅ pass | 232 ms |
| 5 | kurta (sideways) | ethnic | −90 | 19.8 | no | ✅ pass | 364 ms |
| 6 | footwear | footwear | 0 | 18.4 | no | ✅ pass | 217 ms |
| 7 | handbag | bag | 0 | 40.5 | no | ✅ pass | 245 ms |

**7/7 pass** (target ≥6/7). Unit suite: 24 passed. Fixes:
- **color** → `_foreground_avg_color` compares garment-foreground vs garment-foreground (ignores
  off-white padding/background + transparent pixels); threshold `_COLOR_DIST_MAX=120`.
- **skin** → skipped for `bag/handbag/accessory/footwear/jewellery`; `_is_skin` tightened
  (`r>g>b`, `g−b<45`, `b>40`) so marigold/gold/mustard garments aren't flagged.
- **area** → per-category min occupancy: top/bottom/dress/ethnic/outerwear = 12%, footwear/bag/
  accessory/jewellery = 8% (was flat 20%). Still rejects empty mask + tiny artifacts.
- **logs** → `ahvi.catalog.validation.{area,skin,color}` (category, threshold, measured, decision).

Note: the cloud `candidate` revision was built before this fix — rebuild it before relying on the
staging URL.

---

## Original run (pre-fix) — 7/7 REJECTED — recorded for history

**Build:** commit `600dbab`, candidate revision tagged `candidate` (no prod traffic).
**Flags:** `ENABLE_CATALOG_IMAGE_GENERATION=true`, `ENABLE_CATALOG_NORMALIZATION=true`. `STYLE_SHARED_BRAIN` NOT enabled.
**Smoke type:** pipeline-level on representative **synthetic** inputs for the 7 requested image types
(no real hanger/selfie assets or live auth available). Same code path as the candidate revision.

## Result: 7/7 REJECTED — validation is the blocker (not the rotate/center stages)

| # | case | category | rotation* | occupancy % | clipping | validation | latency |
|---|---|---|---|---:|---|---|---:|
| 1 | hanger top | top | n/a | 40.3 | no | ❌ color_mismatch | 206 ms |
| 2 | mirror selfie | top | n/a | 35.9 | no | ❌ color_mismatch | 170 ms |
| 3 | flat-lay shirt | top | n/a | 44.7 | no | ❌ color_mismatch | 190 ms |
| 4 | dress | dress | n/a | 20.2 | no | ❌ color_mismatch | 144 ms |
| 5 | kurta (sideways) | ethnic | −90 (built) | 19.8 | no | ❌ visible_area_below_20pct | 83 ms |
| 6 | footwear | footwear | 0 | 18.4 | no | ❌ visible_area_below_20pct | 83 ms |
| 7 | handbag | bag | 0 | 40.5 | no | ❌ skin_region_dominant | 108 ms |

\*rotation field is omitted on a failed result, so it reads n/a; the rotate stage still ran during build
(kurta built at −90 before validation rejected it).

Latency 83–206 ms (pure CPU pipeline; excludes RMBG + R2 upload in the real save flow).
Clipping: none in any case (8% padding holds — center/trim stages are healthy).

## Root causes (all in `services/catalog_image_service.py`)

1. **`color_mismatch` — PRIMARY bug, hits real images too.**
   `validate_catalog_image` → `_avg_color_rgba(canvas)` counts every pixel with `alpha>16`. After
   centering, the canvas background is **opaque off-white (alpha 255)**, so the average is dragged toward
   white. Compared against the tight garment-only `original` average, the distance exceeds the 180
   threshold and rejects. Any garment smaller than the full canvas (i.e. all of them) false-rejects.
   - **Fix:** compute the catalog's dominant color over **non-background pixels only** (distance from
     off-white > 24), or compare garment-region→garment-region. Then re-tune the threshold.

2. **`skin_region_dominant` — false positive on leather/tan.**
   `_is_skin` (RGB rule) flags brown handbag `(120,72,40)` as skin.
   - **Fix:** skip the skin check for `bag`/`footwear`/`accessory`/`jewellery`; and/or require an actual
     face-shaped region, not just skin-tone ratio. Raise the dominance threshold.

3. **`visible_area_below_20pct` — threshold too high for thin/structured silhouettes.**
   Kurta (19.8%) and footwear (18.4%) sit just under 20%. Same gap flagged in the visual review
   (thin garments under-fill the square).
   - **Fix:** lower threshold to ~12–15%, OR measure occupancy within the garment bounding box rather
     than the whole 1600² canvas.

## Verdict

Rotate / trim / center / canvas stages = healthy (correct rotation, no clipping, consistent fill,
fast). **The validation gate is mis-calibrated and currently rejects 100% of valid garments** — so with
flags on, every item would persist `catalogStatus=catalog_validation_failed` and no catalog image would
ever be produced. This is correctly contained: prod flags are OFF, so production is unaffected.

**Recommend:** a small follow-up fix commit to `validate_catalog_image` (3 fixes above), then re-run this
smoke before any prod enablement. No prod rollout performed.

## Artifacts
- Montage (original | masked | catalog/rejected): `catalog_candidate_montage.png`
- Per-case PNGs: `{case}_original.png`, `{case}_masked.png` in the run temp dir.
