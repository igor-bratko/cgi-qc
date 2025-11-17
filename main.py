# validator/main.py
import io
import os
import re
import itertools
from typing import List, Dict
from fastapi import FastAPI, Request, HTTPException
import boto3
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import imagehash
import cv2
from PIL import Image, ImageOps
import mediapipe as mp

SEG = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

# ---------- Config from env ----------
# S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9002")
S3_REGION = os.getenv("S3_REGION", "eu-central-1")
S3_BUCKET = os.getenv("S3_BUCKET", "test-web-qwe")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "...")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "...")
# S3_PATH_STYLE = os.getenv("S3_PATH_STYLE", "true").lower() == "true"
PORT = int(os.getenv("PORT", "8085"))
REMBG_ENABLED = os.getenv("REMBG_ENABLED", "false").lower() == "true"

# ---------- Policy thresholds ----------
MIN_SIDE = 1000
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# angle diversity
PHASH_DUP_THRESHOLD = 0.10  # normalized 0..1, below means near-duplicate
PHASH_DIFFERENT_THRESHOLD = 0.22  # normalized 0..1, above means clearly different
ORB_SAME_VIEW_SIM = 0.30  # above means similar viewpoint
DIVERSITY_MIN = 0.50  # fraction of diverse pairs
UNIQUE_VIEWS_MIN = 4

# background cleanliness (simple heuristic)
BG_CLEAN_MIN = 0.65

# ---------- S3 client ----------
S3 = boto3.client(
    "s3",
    # endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION,
    config=boto3.session.Config(signature_version="s3v4",
                                s3={"addressing_style": "virtual"})
)


# ---------- Models ----------
class AssetDto(BaseModel):
    assetId: str
    s3Keys: List[str]


# ---------- Helpers ----------
def list_image_keys(bucket: str, prefix: str) -> List[str]:
    out: List[str] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = S3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            ext = os.path.splitext(o["Key"])[1].lower()
            if ext in ALLOWED_EXT:
                out.append(o["Key"])
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(out)


def get_image(bucket: str, key: str) -> Image.Image:
    data = S3.get_object(Bucket=bucket, Key=key)["Body"].read()
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")


def score_bg_clean(img: Image.Image) -> float:
    # RGBA as uint8
    im_rgba = img.convert("RGBA")
    arr = np.asarray(im_rgba, dtype=np.uint8)
    h, w, _ = arr.shape

    # border thickness
    b = max(4, min(16, int(0.02 * min(h, w))))

    # border mask
    mask = np.zeros((h, w), dtype=bool)
    mask[:b, :] = True
    mask[-b:, :] = True
    mask[:, :b] = True
    mask[:, -b:] = True

    # transparency score for cutouts
    alpha = arr[..., 3]
    border_alpha = alpha[mask]
    trans_ratio = float((border_alpha < 8).mean())  # 0..1

    # RGB border pixels
    border_rgb_u8 = arr[..., :3][mask]  # uint8 0..255
    if border_rgb_u8.size == 0:
        return max(trans_ratio, 0.0)

    # HSV in 8-bit space: H 0..179. S,V 0..255
    hsv = cv2.cvtColor(border_rgb_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    S8 = hsv[:, 0, 1].astype(np.float32)
    V8 = hsv[:, 0, 2].astype(np.float32)

    # thresholds tuned for white or very light gray sweeps with soft shadows
    WHITE_S_MAX = 0.18 * 255.0
    WHITE_V_MIN = 0.88 * 255.0
    white_ratio = float(((S8 <= WHITE_S_MAX) & (V8 >= WHITE_V_MIN)).mean())

    # soft uniformity score on normalized RGB
    border_rgb = border_rgb_u8.astype(np.float32) / 255.0
    col_std = float(border_rgb.std(axis=0).mean())  # 0..~0.5
    UNIFORM_REF_STD = 0.06
    uniform = float(1.0 / (1.0 + (col_std / UNIFORM_REF_STD)))

    # combine - cutouts pass via transparency
    return max(trans_ratio, 0.7 * white_ratio + 0.3 * uniform)


# ---------- Angle diversity helpers you asked about ----------
def orb_features(img_rgb: Image.Image):
    """Return keypoints count and descriptors for ORB on grayscale."""
    arr = cv2.cvtColor(np.array(img_rgb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=12, scaleFactor=1.2, nlevels=8)
    kps, des = orb.detectAndCompute(arr, None)
    return (len(kps) if kps else 0), des


def orb_similarity(des1, des2, kp1, kp2) -> float:
    """0..1. fraction of good matches relative to min(kp1, kp2)."""
    if des1 is None or des2 is None or kp1 == 0 or kp2 == 0:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.75 * n.distance]
    denom = max(1, min(kp1, kp2))
    return min(1.0, len(good) / denom)


def angle_diversity_stats(images: List[Image.Image], keys: List[str]) -> Dict:
    phashes = [imagehash.phash(img) for img in images]
    kps_des = [orb_features(img) for img in images]

    def is_different(i: int, j: int) -> bool:
        # normalized pHash distance 0..1
        d_ph = (phashes[i] - phashes[j]) / 64.0
        sim_orb = orb_similarity(kps_des[i][1], kps_des[j][1], kps_des[i][0], kps_des[j][0])
        # different if clearly different by pHash or dissimilar by ORB
        return d_ph >= PHASH_DIFFERENT_THRESHOLD or sim_orb <= ORB_SAME_VIEW_SIM

    # greedy selection up to 5 unique views
    selected_idx: List[int] = []
    for idx in range(len(images)):
        if all(is_different(idx, s) for s in selected_idx):
            selected_idx.append(idx)
        if len(selected_idx) >= 5:
            break

    # diversity score across all pairs
    pairs = list(itertools.combinations(range(len(images)), 2))
    diverse = sum(1 for i, j in pairs if is_different(i, j))
    diversity_score = diverse / max(1, len(pairs))

    # near duplicates check by pHash only
    dup_pairs = []
    for i, j in pairs:
        d_ph = (phashes[i] - phashes[j]) / 64.0
        if d_ph < PHASH_DUP_THRESHOLD:
            dup_pairs.append([keys[i], keys[j]])

    return {
        "unique_views_count": len(selected_idx),
        "selected_keys": [keys[i] for i in selected_idx[:max(4, min(5, len(selected_idx)))]],
        "diversity_score": float(diversity_score),
        "dup_pairs": dup_pairs
    }


# ---------- API ----------
app = FastAPI(title="Image Intake Validator", version="0.1.0")


@app.post("/debug/echo")
async def debug_echo(request: Request):
    raw = await request.body()
    print("RAW HEADERS:", dict(request.headers))
    print("RAW BODY:", raw.decode("utf-8", "ignore"))
    return {"headers": dict(request.headers), "raw": raw.decode("utf-8", "ignore")}


@app.post("/validate")
def validate(req: AssetDto):
    print("Start validation")
    bucket = "test-web-qwe"
    keys = req.s3Keys

    reasons: List[str] = []
    metrics: Dict[str, object] = {}

    if len(keys) < 4:
        reasons.append(f"Need at least 4 images. found={len(keys)}")

    # optional naming convention
    # name_re = re.compile(req.naming_regex) if req.naming_regex else None
    # bad_names = [k for k in keys if name_re and not name_re.search(os.path.basename(k))]
    # if bad_names:
    #     reasons.append(f"Names not matching pattern. count={len(bad_names)}")

    # load images and collect stats
    pil_images: List[Image.Image] = []
    sizes = []
    bg_scores = []
    good_keys: List[str] = []

    for k in keys:
        try:
            img = get_image(bucket, k)
        except Exception:
            reasons.append(f"Unreadable image. key={k}")
            continue
        w, h = img.size
        sizes.append(min(w, h))
        if min(w, h) < MIN_SIDE:
            reasons.append(f"Resolution too small {w}x{h}. key={k}")
        bg_scores.append(score_bg_clean(img))
        pil_images.append(img)
        good_keys.append(k)

    if not pil_images:
        return {"pass": False, "reasons": reasons, "metrics": {}, "selected": []}

    # angle diversity and duplicates
    stats = angle_diversity_stats(pil_images, good_keys)
    if stats["unique_views_count"] < UNIQUE_VIEWS_MIN:
        reasons.append(f"Not enough distinct angles. unique_views={stats['unique_views_count']}")

    if stats["diversity_score"] < DIVERSITY_MIN:
        reasons.append(f"Low diversity. score={stats['diversity_score']:.2f}")

    if stats["dup_pairs"]:
        reasons.append(f"Near-duplicate pairs detected. count={len(stats['dup_pairs'])}")

    # background cleanliness
    bg_median = float(np.median(bg_scores)) if bg_scores else 0.0
    if bg_median < BG_CLEAN_MIN:
        reasons.append(f"Background not clean enough. score={bg_median:.2f}")

    selected = stats["selected_keys"]

    metrics.update({
        "count": len(keys),
        "min_side_px": int(min(sizes) if sizes else 0),
        "angle_diversity": stats["diversity_score"],
        "unique_views_count": stats["unique_views_count"],
        "bg_cleanliness": bg_median,
        "dup_pairs": stats["dup_pairs"],
        "selected_count": len(selected)
    })

    return {
        "pass": len(reasons) == 0 and len(selected) >= 4,
        "reasons": reasons,
        "metrics": metrics,
        "selected": selected
    }


def _read_s3_png(bucket: str, key: str) -> Image.Image:
    obj = S3.get_object(Bucket=bucket, Key=key)
    img = Image.open(io.BytesIO(obj["Body"].read()))
    return ImageOps.exif_transpose(img)


def _derive_mask_key(orig_key: str) -> str:
    # as requested: replace /validated/ with /masks/
    return orig_key.replace("/validated/", "/masks/")


def _derive_out_key(orig_key: str) -> str:
    # write cleaned PNG next to validated. change folder to /cleaned/ and extension to .png
    base, _ext = os.path.splitext(orig_key)
    cleaned_base = base.replace("/validated/", "/cleaned/")
    return f"{cleaned_base}.png"


@app.post("/apply-masks-batch")
def apply_masks_batch(req: AssetDto):
    cleaned = []
    for key in req.s3Keys:
        mask_key = _derive_mask_key(key)
        out_key = _derive_out_key(key)
        try:
            # 1) load original
            orig = _read_s3_png(S3_BUCKET, key).convert("RGBA")
            w, h = orig.size

            # 2) load mask
            m = _read_s3_png(S3_BUCKET, mask_key).convert("L")
            if m.size != (w, h):
                m = m.resize((w, h), Image.BILINEAR)

            # 3) normalize to alpha
            arr = np.asarray(m, dtype=np.uint8)
            alpha = arr if arr.mean() < 128 else (255 - arr)
            alpha = (alpha > 32).astype(np.uint8) * 255
            alpha_img = Image.fromarray(alpha, mode="L")

            # 4) compose transparent PNG
            r, g, b, _ = orig.split()
            cleaned_img = Image.merge("RGBA", (r, g, b, alpha_img))

            # 5) upload
            buf = io.BytesIO()
            cleaned_img.save(buf, format="PNG", optimize=True)
            buf.seek(0)
            print("Upload to: " + out_key)
            S3.put_object(Bucket=S3_BUCKET, Key=out_key, Body=buf.getvalue(), ContentType="image/png")

            cleaned.append(out_key)

        except Exception as e:
            raise HTTPException(status_code=400,
                                detail=f"apply-mask failed for key='{key}' mask='{mask_key}': {e}")

    return {"assetId": req.assetId, "processed": len(cleaned), "cleaned": cleaned}


# Optional local dev entrypoint
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
