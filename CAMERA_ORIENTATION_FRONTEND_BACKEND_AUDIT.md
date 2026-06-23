# Camera Image Orientation — Frontend + Backend Audit (AUDIT ONLY)

Symptom: camera photos save/display rotated after wardrobe upload; gallery photos fine.
Example: baseball cap from camera rotated even after catalog normalization.

No code changed.

## 1. End-to-end flow + where orientation is (mis)handled

```
Camera (Android sensor)         Gallery
  | takePicture() JPEG            | image_picker pickMultiImage(maxW/maxH/quality)
  | pixels stored sideways        | image_picker RE-ENCODES -> BAKES orientation
  | EXIF Orientation=6/3/8        | pixels upright, EXIF=1
  v                               v
File.readAsBytes() RAW   <-----  f.readAsBytes()            [lib/wardrobe.dart:2151 / :2192]
  | (EXIF intact)                 | (already upright)
  v
Flutter preview Image.memory  -> respects EXIF -> LOOKS upright (masks the bug)   [Q2]
  v
base64Encode(bytes)                                          [backend_service.dart:11]
  v
POST /api/wardrobe/capture/analyze
  v
_decode_image_base64  ==> Image.open(...).convert("RGB")   NO exif_transpose  [wardrobe_capture.py:276]  <-- PRIMARY BUG
  |  (camera image now = rotated pixels; gallery already upright)
  v
Gemini detect_and_crop(image)  -> bbox on ROTATED pixels   [gemini_multi_garment_detector.py:330]
  v
_crop_to_png_bytes(image, bbox_px) -> crop re-encoded PNG, original EXIF LOST  [:322]
  v
remove_bg_bytes(crop)  -> RMBG on rotated crop (raw passthrough)   [bg_service.py:86]
  v
catalog normalize: _open_rgba -> exif_transpose  (NO-OP: crop has no EXIF) +
  rotation only for top/dress/outerwear/ethnic (cap=accessory -> never rotates)  [catalog_image_service.py]
  v
R2 -> wardrobe UI  => rotated
```

## 2. Root cause: BOTH (backend primary)

- **Backend (primary):** the first decode `_decode_image_base64` does **not** apply
  `ImageOps.exif_transpose`. Every downstream stage (Gemini bbox, crop, RMBG, catalog) runs on the
  raw camera pixels. The crop is re-encoded without EXIF, so the wrong rotation is **baked in** and
  catalog normalization can't recover it.
- **Frontend (contributing):** camera and gallery take **different paths**. Gallery is incidentally
  fixed because `image_picker` with `maxWidth/maxHeight/imageQuality` re-encodes and bakes orientation
  upright; camera sends raw bytes with EXIF orientation. This inconsistency is why "gallery is fine,
  camera is rotated."

## 3. Exact risky files / functions

Frontend (`C:\tmp\AHVI-frontend-clean`):
- `lib/wardrobe.dart:2146 _captureAndDetect` — camera: `takePicture()` → `readAsBytes()` raw, EXIF kept, no bake.
- `lib/wardrobe.dart:2169 _pickGallery` — gallery: `pickMultiImage(maxWidth:1600,maxHeight:1600,imageQuality:82)` → bakes orientation (divergent path).
- `lib/services/backend_service.dart:11 _encodeBytes` / `:698 MultipartFile.fromBytes` — sends bytes verbatim (no normalization).
- (No `flutter_image_compress`; `image_cropper` only in `profile.dart` avatar, not wardrobe.)

Backend:
- `routers/wardrobe_capture.py:~265-278 _decode_image_base64` — **PRIMARY**: `Image.open().convert("RGB")`, no `exif_transpose`. Used by `analyze_capture` (:1242).
- `services/gemini_multi_garment_detector.py:330 detect_and_crop` + `:322 _crop_to_png_bytes` — bbox + crop on un-normalized image; crop drops EXIF.
- `services/bg_service.py:86 remove_bg_bytes` — raw passthrough; inherits rotation.
- `services/catalog_image_service.py _open_rgba/_build_catalog_canvas` — transpose is a no-op on EXIF-less crops; accessory/bag never auto-rotate → can't fix caps.
- Note: `services/image_normalizer.py:22` DOES `exif_transpose`, but it runs in the R2 board-normalize path, not on the analyze/crop path — so it doesn't help capture.

## Answers

Frontend:
1. Preserve EXIF on send? Camera: YES (raw, EXIF intact). Gallery: NO (re-encoded, baked upright, EXIF=1).
2. Display correct only because Flutter respects EXIF? YES — `Image.memory` honors EXIF so the camera preview looks upright, hiding the rotated stored pixels.
3. Does compression strip EXIF without rotating? Gallery re-encode **bakes** (rotates pixels then drops EXIF) → safe. Camera has no compression → EXIF retained. No "strip-without-rotate" case in wardrobe.
4. Camera vs gallery different? YES — separate functions, different EXIF handling. The core inconsistency.
5. base64 from raw camera bytes? YES — `takePicture()` file → `readAsBytes()` unmodified → base64.
6. Frontend crop/resize before backend? Gallery resizes+bakes (1600/q82); camera none; no wardrobe crop.
7. Orientation normalization before upload? NONE for camera; gallery only incidental via image_picker.

Backend:
1. First decode point? `_decode_image_base64` (`wardrobe_capture.py:~265-278`).
2. `exif_transpose` before Gemini? **NO**.
3. OpenCV before transpose? No `cv2` in the capture decode path (PIL only).
4. Gemini bboxes on unnormalized image? **YES** (un-transposed image passed to `detect_and_crop`).
5. Crop uses raw or normalized pixels? **RAW** (un-transposed); crop re-encoded without EXIF → wrong rotation baked.
6. Catalog normalization too late? **YES** — runs after the wrong crop is baked, and for caps (accessory) it never rotates; transpose is a no-op on the EXIF-less crop.

## 4. Minimal patch plan

Backend (PRIMARY — fixes both camera + keeps gallery correct):
- In `_decode_image_base64`, decode with `ImageOps.exif_transpose(Image.open(io.BytesIO(data)))` BEFORE `.convert("RGB")`. One change normalizes the image for Gemini + crop + RMBG + catalog.
- Result: Gemini bbox + crop operate on upright pixels; downstream stages inherit upright; catalog stays as a backup only.

Frontend (consistency / defense-in-depth, optional once backend fixed):
- Normalize orientation right after camera capture: decode + bake orientation (e.g. `image` package `bakeOrientation`, or re-encode) before `readAsBytes()`/base64, so the bytes are physically upright.
- Make camera + gallery use the SAME normalize step so both send baked-upright bytes.

Do NOT rotate in both layers blindly — pick backend transpose as the single source of truth; if frontend also bakes, backend transpose becomes a no-op (EXIF=1) → safe, no double rotation.

## 5. Tests (to add when patching)

Frontend:
- Camera portrait with EXIF Orientation=6 → uploaded bytes are upright.
- Gallery image → unchanged (still upright).
- Compressed image → remains upright.
- Preview matches the backend-received image.

Backend:
- Image with EXIF Orientation=6 → `_decode_image_base64` returns upright pixels (assert size W/H swapped vs raw, or pixel probe).
- Gemini receives normalized (upright) bytes.
- No double rotation when EXIF already =1.
- Catalog normalization does not rotate an already-upright crop (rotation_applied=0).

_Audit only — no code edited, nothing committed or deployed._
