# app.py — Streamlit GTSRB Traffic Sign Recognition (Tuned Model + Error Analysis)

import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


# ----------------------------
# Constants / Labels (GTSRB)
# ----------------------------

CLASS_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]
N_CLASSES = 43
assert len(CLASS_NAMES) == N_CLASSES


# ----------------------------
# Paths / Repo structure
# ----------------------------

st.set_page_config(page_title="GTSRB Traffic Sign Demo (Tuned)", layout="wide")

PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "dataset"

# Default model paths (edit as needed)
DEFAULT_MODEL_CANDIDATES = [
    PROJECT_ROOT / "gtsrb_cnn_tuned.keras",
    PROJECT_ROOT / "traffic_classifier.keras",
    PROJECT_ROOT / "traffic_classifier.h5",
]

# Allow override via env var if you want
ENV_MODEL_PATH = os.getenv("MODEL_PATH")
if ENV_MODEL_PATH:
    DEFAULT_MODEL_CANDIDATES.insert(0, Path(ENV_MODEL_PATH))


# ----------------------------
# Utilities
# ----------------------------

def pretty_label(class_id: int) -> str:
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"ClassId {class_id}"

def resolve_model_path(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    for p in DEFAULT_MODEL_CANDIDATES:
        if p.exists():
            return p
    # Fall back to first candidate even if missing (we'll raise later)
    return DEFAULT_MODEL_CANDIDATES[0]

@st.cache_resource
def get_model(model_path_str: str):
    model_path = Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Tip: Put your tuned model file in the project folder or set MODEL_PATH env var."
        )
    m = load_model(str(model_path))
    return m

def infer_img_size(model) -> tuple[int, int]:
    """
    Infer expected input size from model.input_shape.
    Works for (None, H, W, C). If unknown, default to 30x30.
    """
    shp = getattr(model, "input_shape", None)
    if not shp or len(shp) < 4:
        return (30, 30)
    h, w = shp[1], shp[2]
    if h is None or w is None:
        return (30, 30)
    return (int(w), int(h))  # PIL expects (W, H)

def preprocess_image(img: Image.Image, img_size: tuple[int, int]) -> np.ndarray:
    img = img.convert("RGB").resize(img_size)
    x = np.array(img).astype("float32") / 255.0
    return np.expand_dims(x, axis=0)

def predict_one(model, img: Image.Image, img_size: tuple[int, int], top_k=5):
    x = preprocess_image(img, img_size)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    top_idx = np.argsort(probs)[::-1][:top_k]
    top = [(int(i), float(probs[i])) for i in top_idx]
    return pred, probs, top

@st.cache_data
def load_csvs():
    train_csv = DATASET_DIR / "Train.csv"
    test_csv  = DATASET_DIR / "Test.csv"
    meta_csv  = DATASET_DIR / "Meta.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing: {test_csv}")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)
    meta_df  = pd.read_csv(meta_csv) if meta_csv.exists() else None
    return train_df, test_df, meta_df

def safe_open_image(rel_path: str) -> Image.Image:
    p = DATASET_DIR / rel_path
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p)

def apply_roi_crop(img: Image.Image, row: pd.Series) -> Image.Image:
    required = ["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]
    if not all(c in row.index for c in required):
        return img

    x1, y1, x2, y2 = (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"]))
    w, h = img.size
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))

def image_quality_diagnostics(img: Image.Image) -> dict:
    """
    Lightweight “why might it be wrong?” hints:
    - brightness/contrast via numpy
    - optional blur score using OpenCV if installed
    """
    arr = np.array(img.convert("RGB")).astype(np.float32)
    gray = arr.mean(axis=2)

    brightness = float(gray.mean())  # 0..255
    contrast = float(gray.std())

    out = {
        "brightness_mean_0_255": brightness,
        "contrast_std_0_255": contrast,
        "blur_laplacian_var": None,
    }

    # Optional blur metric if cv2 exists
    try:
        import cv2
        g = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        out["blur_laplacian_var"] = float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        pass

    return out


# ----------------------------
# UI
# ----------------------------

st.title("🚦 Traffic Sign Recognition (GTSRB) — Tuned Model Demo")

st.sidebar.header("Model & Controls")

model_path_input = st.sidebar.text_input(
    "Model path (relative to project or absolute)",
    value="gtsrb_cnn_tuned.keras"
)

top_k = st.sidebar.slider("Top-K predictions", 3, 10, 5)
mode = st.sidebar.radio("Input source", ["Upload image", "Pick from dataset (CSV)"])
use_roi = st.sidebar.checkbox("Use ROI crop (recommended for CSV images)", value=True)

show_debug = st.sidebar.checkbox("Show debug diagnostics", value=False)

# Load model
try:
    model_path = resolve_model_path(model_path_input)
    model = get_model(str(model_path))
    IMG_SIZE = infer_img_size(model)  # inferred from model input
except Exception as e:
    st.error(str(e))
    st.stop()

if show_debug:
    st.sidebar.caption(f"Loaded model: {model_path}")
    st.sidebar.caption(f"Expected input size: {IMG_SIZE[0]}×{IMG_SIZE[1]}")

# Load dataset CSVs if needed
train_df = test_df = meta_df = None
if mode == "Pick from dataset (CSV)":
    try:
        train_df, test_df, meta_df = load_csvs()
    except Exception as e:
        st.error(str(e))
        st.stop()

left, right = st.columns([1, 1], gap="large")

selected_img = None
caption = ""
true_class = None
rel_path = None


# ----------------------------
# Left column: pick input
# ----------------------------

if mode == "Upload image":
    with left:
        st.subheader("Upload an image")
        up = st.file_uploader("Upload a traffic sign image", type=["png", "jpg", "jpeg", "webp", "bmp"])
        if up is not None:
            selected_img = Image.open(up)
            caption = f"Uploaded: {up.name}"

else:
    with left:
        st.subheader("Pick an image from your dataset")

        split = st.radio("Choose split", ["Test.csv", "Train.csv"])
        df = test_df if split == "Test.csv" else train_df

        if "Path" not in df.columns:
            st.error(f"{split} must contain a 'Path' column. Found: {list(df.columns)}")
            st.stop()

        if "ClassId" in df.columns:
            all_classes = sorted(df["ClassId"].unique().tolist())
            chosen = st.selectbox("Filter by ClassId (optional)", ["All"] + all_classes)
            df_view = df if chosen == "All" else df[df["ClassId"] == chosen]
        else:
            df_view = df

        st.caption(f"Available rows: {len(df_view):,}")

        if "picked_idx" not in st.session_state:
            st.session_state.picked_idx = 0

        c1, c2 = st.columns([1, 2], gap="small")
        with c1:
            if st.button("🎲 Random sample", use_container_width=True, disabled=(len(df_view) == 0)):
                st.session_state.picked_idx = int(np.random.randint(0, len(df_view)))

        with c2:
            idx = st.number_input(
                "Row index (in filtered view)",
                min_value=0,
                max_value=max(len(df_view) - 1, 0),
                value=int(st.session_state.picked_idx),
                step=1,
            )
            st.session_state.picked_idx = int(idx)

        if len(df_view) > 0:
            row = df_view.iloc[int(st.session_state.picked_idx)]
            rel_path = str(row["Path"])
            img = safe_open_image(rel_path)

            img_used = apply_roi_crop(img, row) if use_roi else img
            selected_img = img_used

            if "ClassId" in row.index:
                true_class = int(row["ClassId"])
                caption = f"{split} → {rel_path} | True: {pretty_label(true_class)}"
            else:
                caption = f"{split} → {rel_path}"

            if use_roi:
                st.caption("Original image (before ROI crop):")
                st.image(img, use_container_width=True)


# ----------------------------
# Right column: prediction + evaluation
# ----------------------------

with right:
    st.subheader("Prediction")

    if selected_img is None:
        st.info("Choose an image to run prediction.")
    else:
        st.image(selected_img, caption=caption, use_container_width=True)

        pred_class, probs, top = predict_one(model, selected_img, IMG_SIZE, top_k=top_k)

        st.markdown(f"### ✅ Predicted: **{pretty_label(pred_class)}**")
        st.caption(f"Confidence (top-1): {float(probs[pred_class]):.3f}")

        if true_class is not None:
            correct = (pred_class == true_class)
            st.markdown(f"**Match with ground truth:** {'✅ Yes' if correct else '❌ No'}")

        top_df = pd.DataFrame(
            [{"ClassId": cid, "Label": pretty_label(cid), "Probability": p} for cid, p in top]
        )
        st.dataframe(top_df, use_container_width=True, hide_index=True)

        chart_df = top_df.set_index("Label")[["Probability"]]
        st.bar_chart(chart_df)

        with st.expander("Why might the model be wrong? (quick diagnostics)"):
            diag = image_quality_diagnostics(selected_img)
            st.write(diag)

            hints = []
            if diag["brightness_mean_0_255"] < 60:
                hints.append("Image looks quite dark → digits/icons can disappear in shadow.")
            if diag["brightness_mean_0_255"] > 200:
                hints.append("Image is very bright → glare/overexposure can wash out details.")
            if diag["contrast_std_0_255"] < 25:
                hints.append("Low contrast → edges and digits may be hard to distinguish.")
            if diag["blur_laplacian_var"] is not None and diag["blur_laplacian_var"] < 80:
                hints.append("Likely blurry (low Laplacian variance) → common cause of confusion between similar signs.")

            if hints:
                st.markdown("**Possible reasons:**")
                for h in hints:
                    st.write(f"- {h}")
            else:
                st.write("No obvious low-quality signal detected. If wrong, it may be a visually similar class.")

        with st.expander("Show full probability distribution (all 43 classes)"):
            full_df = pd.DataFrame({
                "ClassId": list(range(N_CLASSES)),
                "Label": [pretty_label(i) for i in range(N_CLASSES)],
                "Probability": probs
            }).sort_values("Probability", ascending=False)
            st.dataframe(full_df, use_container_width=True, hide_index=True)


# ----------------------------
# Extra: Dataset-wide error analysis (only in CSV mode)
# ----------------------------

if mode == "Pick from dataset (CSV)":
    st.divider()
    st.subheader("📉 Error Analysis (Dataset Mode)")

    st.caption(
        "Run evaluation on a subset of the selected split to find wrong predictions, "
        "most confident wrong predictions, and top confusions."
    )

    split_eval = st.radio("Evaluate split", ["Test.csv", "Train.csv"], horizontal=True)
    df_eval = test_df if split_eval == "Test.csv" else train_df

    if "ClassId" not in df_eval.columns:
        st.info("This CSV does not include ClassId; dataset-wide error analysis needs ground truth labels.")
    else:
        max_n = min(3000, len(df_eval))
        n_eval = st.slider("How many samples to evaluate", 100, max_n, min(800, max_n), step=100)

        do_eval = st.button("Run evaluation", type="primary")

        if do_eval:
            # sample rows for speed
            df_sample = df_eval.sample(n=n_eval, random_state=42).reset_index(drop=True)

            y_true = []
            y_pred = []
            conf_pred = []
            paths = []

            progress = st.progress(0)
            for i in range(len(df_sample)):
                row = df_sample.iloc[i]
                p = str(row["Path"])
                img = safe_open_image(p)
                img_used = apply_roi_crop(img, row) if use_roi else img

                pred, probs, _ = predict_one(model, img_used, IMG_SIZE, top_k=top_k)

                t = int(row["ClassId"])
                y_true.append(t)
                y_pred.append(pred)
                conf_pred.append(float(probs[pred]))
                paths.append(p)

                if (i + 1) % 25 == 0 or (i + 1) == len(df_sample):
                    progress.progress((i + 1) / len(df_sample))

            y_true = np.array(y_true, dtype=int)
            y_pred = np.array(y_pred, dtype=int)
            conf_pred = np.array(conf_pred, dtype=float)

            acc = float((y_true == y_pred).mean())
            st.success(f"Accuracy on evaluated subset: {acc:.4f} ({n_eval} samples)")

            wrong_idx = np.where(y_true != y_pred)[0]
            st.write(f"Misclassified: {len(wrong_idx)} / {len(y_true)} ({len(wrong_idx)/len(y_true):.2%})")

            # Top confusions
            pairs = [(int(y_true[i]), int(y_pred[i])) for i in wrong_idx]
            top_pairs = Counter(pairs).most_common(10)

            if top_pairs:
                st.markdown("**Top 10 confusion pairs (True → Pred):**")
                for (t, p), c in top_pairs:
                    st.write(f"- {pretty_label(t)} → {pretty_label(p)} | count={c}")

            # Show random wrong examples
            st.markdown("### Wrong predictions (examples)")
            n_show = st.slider("How many wrong examples to show", 6, 24, 12, step=3)

            if len(wrong_idx) == 0:
                st.info("No wrong predictions found in the evaluated subset 🎉")
            else:
                cols = st.columns(3)
                pick = np.random.choice(wrong_idx, size=min(n_show, len(wrong_idx)), replace=False)

                for j, wi in enumerate(pick):
                    row = df_sample.iloc[int(wi)]
                    p = str(row["Path"])
                    img = safe_open_image(p)
                    img_used = apply_roi_crop(img, row) if use_roi else img

                    t = int(y_true[wi])
                    pr = int(y_pred[wi])
                    cf = float(conf_pred[wi])

                    with cols[j % 3]:
                        st.image(img_used, use_container_width=True)
                        st.caption(f"{p}")
                        st.write(f"True: **{pretty_label(t)}**")
                        st.write(f"Pred: **{pretty_label(pr)}** ({cf:.2f})")

                # Most confident wrong
                st.markdown("### Most confident wrong predictions")
                top_conf_wrong = wrong_idx[np.argsort(-conf_pred[wrong_idx])]
                pick2 = top_conf_wrong[: min(n_show, len(top_conf_wrong))]

                cols2 = st.columns(3)
                for j, wi in enumerate(pick2):
                    row = df_sample.iloc[int(wi)]
                    p = str(row["Path"])
                    img = safe_open_image(p)
                    img_used = apply_roi_crop(img, row) if use_roi else img

                    t = int(y_true[wi])
                    pr = int(y_pred[wi])
                    cf = float(conf_pred[wi])

                    with cols2[j % 3]:
                        st.image(img_used, use_container_width=True)
                        st.caption(f"{p}")
                        st.write(f"True: **{pretty_label(t)}**")
                        st.write(f"Pred: **{pretty_label(pr)}** ({cf:.2f})")
