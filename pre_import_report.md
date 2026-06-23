# Pre-Import Snapshot — style_assets (production)

Captured before the planned `meghna_female` import. **Import was NOT executed** — a blocking discrepancy was found (see bottom).

## Totals
- total style_assets: **505**
- male: **268**
- female: **229**   ← expected 0 per task premise
- unisex: **8**

## Category distribution (all)
bottom 83 · top 87 · accessory 106 · dresses 60 · ethnic 49 · footwear 55 · outerwear 26 · travel 21 · loungewear 10 · grooming 8

## Occasion coverage (all, top tags)
casual_day 207 · party 60 · casual 59 · date 57 · wedding 50 · festive 50 · startup_office 38 · client_meeting 38 · coffee_date 38 · weekend 35 · occasion 35 · travel 25 · vacation 23 · diwali 15 · smart_casual 14

## Existing FEMALE assets (229) — origin
- asset_id prefix: **`womens_assets_*`** (229/229)
- source: **`manifest_import`** (229/229) → imported via `scripts/import_manifest_assets.py`
- `meghna_female_*` present: **0**
- female categories: dresses 60 · accessory 40 · bottom 40 · top 37 · ethnic 35 · footwear 17 (note: **`dresses`** plural; **no outerwear**)
- sample ids: `womens_assets_tops_white_spagetti_strap_top`, `womens_assets_tops_pink_and_white_sports_bra` (← a sports-bra/innerwear leaked into their import)

## 🛑 BLOCKING DISCREPANCY — import halted
Female assets already exist (229), imported from the **same `womens assets.zip`** via a **different pipeline** (`import_manifest_assets.py`, ids `womens_assets_*`). The `meghna_female` seed (255 rows, ids `meghna_female_*`) targets the **same garments under different asset_ids** → importing would **NOT upsert**; it would create ~255 **duplicate** female docs (second id scheme + `dress`/`dresses` category clash), inflating female to ~484.

Per task SAFETY ("if failures/issues, STOP, do not retry blindly, report"), the import was not run. Decision required on source-of-truth before any write.
