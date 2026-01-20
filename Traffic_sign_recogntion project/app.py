import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model


# Page setup

st.set_page_config(page_title="GTSRB Traffic Sign Demo", layout="wide")

# Your exact repo structure
PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "traffic_classifier.h5"  # matches your screenshot
IMG_SIZE = (30, 30)
N_CLASSES = 43


# Caching (fast reload)

@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return load_model(str(MODEL_PATH))

@st.cache_data
def load_csvs():
    train_csv = DATASET_DIR / "Train.csv"
    test_csv = DATASET_DIR / "Test.csv"
    meta_csv = DATASET_DIR / "Meta.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing: {test_csv}")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    meta_df = pd.read_csv(meta_csv) if meta_csv.exists() else None
    return train_df, test_df, meta_df

def class_name_map(meta_df):
    """
    Try to build ClassId -> SignName mapping if Meta.csv contains it.
    Different GTSRB variants use different column names, so we probe.
    """
    mapping = {}
    if meta_df is None:
        return mapping

    # Common patterns
    # If Meta.csv has columns like: ClassId, SignName
    possible_name_cols = ["SignName", "Name", "label", "Label", "Description", "ClassName"]

    if "ClassId" in meta_df.columns:
        name_col = next((c for c in possible_name_cols if c in meta_df.columns), None)
        if name_col:
            mapping = dict(zip(meta_df["ClassId"].astype(int), meta_df[name_col].astype(str)))
    return mapping

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img).astype("float32") / 255.0
    return np.expand_dims(x, axis=0)

def predict(model, img: Image.Image, top_k=5):
    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    top_idx = np.argsort(probs)[::-1][:top_k]
    top = [(int(i), float(probs[i])) for i in top_idx]
    return pred, probs, top

def safe_open_image(rel_path: str) -> Image.Image:
    """
    Your CSV uses paths like 'Test/00000.png' which resolve to dataset/Test/00000.png
    """
    p = DATASET_DIR / rel_path
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p)

def apply_roi_crop(img: Image.Image, row: pd.Series) -> Image.Image:
    """
    ROI columns in your CSV: Roi.X1, Roi.Y1, Roi.X2, Roi.Y2
    Crop to that box to focus on the sign.
    """
    required = ["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]
    if not all(c in row.index for c in required):
        return img

    x1, y1, x2, y2 = (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"]))

    # clamp to image bounds
    w, h = img.size
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return img

    return img.crop((x1, y1, x2, y2))


# App UI

st.title("🚦 Traffic Sign Recognition (GTSRB) — Streamlit Demo")

# Load model + data
try:
    model = get_model()
    train_df, test_df, meta_df = load_csvs()
    id_to_name = class_name_map(meta_df)
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.header("Controls")
mode = st.sidebar.radio("Input source", ["Upload image", "Pick from dataset (CSV)"])
use_roi = st.sidebar.checkbox("Use ROI crop (recommended for CSV images)", value=True)
top_k = st.sidebar.slider("Top-K predictions", 3, 10, 5)

def pretty_label(class_id: int) -> str:
    return id_to_name.get(class_id, f"ClassId {class_id}")

left, right = st.columns([1, 1], gap="large")

selected_img = None
caption = ""
true_class = None


# Mode: Upload

if mode == "Upload image":
    with left:
        st.subheader("Upload an image")
        up = st.file_uploader("Upload a traffic sign image", type=["png", "jpg", "jpeg", "webp", "bmp"])
        if up is not None:
            selected_img = Image.open(up)
            caption = f"Uploaded: {up.name}"

# Mode: Pick from dataset (CSV)

else:
    with left:
        st.subheader("Pick an image from your dataset")

        split = st.radio("Choose split", ["Test.csv", "Train.csv"])
        df = test_df if split == "Test.csv" else train_df

        # basic validation based on your snippet
        if "Path" not in df.columns:
            st.error(f"{split} must contain a 'Path' column. Found: {list(df.columns)}")
            st.stop()

        # Optional: filter by true class
        if "ClassId" in df.columns:
            all_classes = sorted(df["ClassId"].unique().tolist())
            chosen = st.selectbox("Filter by ClassId (optional)", ["All"] + all_classes)
            df_view = df if chosen == "All" else df[df["ClassId"] == chosen]
        else:
            df_view = df

        st.caption(f"Available rows: {len(df_view):,}")

        # -Random sampler state 
        if "picked_idx" not in st.session_state:
            st.session_state.picked_idx = 0

        # Controls row: random button + index input
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

            # Optionally crop ROI
            if use_roi:
                img_used = apply_roi_crop(img, row)
            else:
                img_used = img

            selected_img = img_used

            # True class (if present)
            if "ClassId" in row.index:
                true_class = int(row["ClassId"])
                caption = f"{split} → {rel_path} | True: {pretty_label(true_class)}"
            else:
                caption = f"{split} → {rel_path}"

            # Show original too if ROI is on (so users see what happened)
            if use_roi:
                st.caption("Original image (before ROI crop):")
                st.image(img, use_container_width=True)


# Prediction view

with right:
    st.subheader("Prediction")

    if selected_img is None:
        st.info("Choose an image to run prediction.")
    else:
        st.image(selected_img, caption=caption, use_container_width=True)

        pred_class, probs, top = predict(model, selected_img, top_k=top_k)

        st.markdown(f"### ✅ Predicted: **{pretty_label(pred_class)}**")
        if true_class is not None:
            correct = (pred_class == true_class)
            st.markdown(f"**Match with ground truth:** {'✅ Yes' if correct else '❌ No'}")

        top_df = pd.DataFrame(
            [{"ClassId": cid, "Label": pretty_label(cid), "Probability": p} for cid, p in top]
        )
        st.dataframe(top_df, use_container_width=True, hide_index=True)

        # Bar chart of top-k
        chart_df = top_df.set_index("Label")[["Probability"]]
        st.bar_chart(chart_df)

        with st.expander("Show full probability distribution (all 43 classes)"):
            full_df = pd.DataFrame({
                "ClassId": list(range(N_CLASSES)),
                "Label": [pretty_label(i) for i in range(N_CLASSES)],
                "Probability": probs
            }).sort_values("Probability", ascending=False)
            st.dataframe(full_df, use_container_width=True, hide_index=True)
