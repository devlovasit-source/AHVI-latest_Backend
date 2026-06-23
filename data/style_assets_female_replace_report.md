# Female Style Asset Replacement — Report

Replaced the legacy `manifest_import` female set (229) with the higher-quality
`meghna_female` seed (255) in production `style_assets`. No code, schema,
R2-delete, or deployment changes.

## Import summary
| step | result |
|---|---|
| backup (pre-delete) | 229 docs → `data/backups/style_assets_womens_manifest_import_backup_20260615T095738Z.json` |
| delete plan | `data/style_assets_replace_delete_plan.json` (229 rows) |
| dry-run delete | count = **229** (matched, guards passed) |
| delete (applied) | **229 deleted, 0 failed** |
| import | **created 255, updated 0, failed 0** |

## Before → After
| metric | before (manifest_import 229) | after (meghna 255) |
|---|---|---|
| total style_assets | 505 | **531** |
| female | 229 | **255** |
| male / unisex | 268 / 8 | 268 / 8 (untouched) |

### Category coverage (female)
| cat | before | after |
|---|---|---|
| dress | 60 (`dresses` plural) | **57 (`dress`)** |
| accessory | 40 | 65 |
| bottom | 40 | 36 |
| top | 37 | 32 |
| ethnic | 35 | 31 |
| footwear | 17 | 18 |
| **outerwear** | **0** | **16** |

### Occasion coverage (female) — the big win
| occasion | before | after |
|---|---|---|
| office | 0 | **54** |
| coffee_date | 0 | **112** |
| brunch | 0 | **126** |
| dinner | 0 | **94** |
| party | 60 | 143 |
| wedding | 35 | 47 |
| haldi | 0 | 3 |
| mehendi | 0 | 15 |
| airport | 0 | **10** |
| travel | 0 | **36** |
| dailywear | 0 | **157** |

The legacy set only tagged party/wedding; everyday/office/travel female styling had **zero** assets. Now all 11 occasions are covered.

## Validation results
- female count: **255** ✓
- duplicate asset_ids: **none** ✓
- missing gender: **0** ✓
- invalid categories: **none** (`dress` singular; no `dresses`) ✓
- missing image_url: **0** ✓
- URL reachability (GET sample, 7 across categories incl. converted): **all 200 / image/jpeg** ✓
- broken image URLs: none found in sample (legacy set had control-char/space keys → broken; meghna keys are clean slugified `.jpg`)

## ⚠ Known defects / failures
1. **2 non-fashion leaked into the imported female set** (against exclude rules):
   - `meghna_female_top_black_and_blue_bralette_top` (bralette = innerwear)
   - `meghna_female_bottom_pantyliner` (non-fashion)
   Root cause: the seed classifier matched garment keywords (`top`, `pant`) **before** the innerwear DROP check. They are LIVE; not broken URLs. **Recommend deleting these 2 docs** (or deactivating). 253 clean.
2. **Import-command gotcha (process, not data):** `python scripts/import_style_assets.py …` failed initially with *"Missing Appwrite backend configuration"* because that script does **not** load `.env`. It must be run with `APPWRITE_*` exported (the import was completed via a `.env`-loading runner). Not a code change — flagged for the runbook.

## Recommendations
- Delete the 2 non-fashion strays (above) → female = 253 clean.
- 16 orphaned R2 objects remain from the earlier upload of the original 271 (the dropped non-fashion). Harmless/unreferenced; optional cleanup (no R2 deletes performed, per instruction).
- haldi (3) + airport (10) remain thin — source-zip limited; future female ethnic/airport assets needed.
- The legacy 229 are fully recoverable from the backup JSON if rollback is ever needed.

## Safety / scope
No deployment. No schema change. No R2 deletes. No unrelated files touched. No `git add`. Backup captured before any delete.
