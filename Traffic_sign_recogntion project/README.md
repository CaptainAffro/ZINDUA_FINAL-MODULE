# Traffic Sign Recognition (GTSRB) — CNN Classifier + Streamlit Demo

This capstone project trains a **Convolutional Neural Network (CNN)** to classify **43 categories** of German traffic signs using the **GTSRB** dataset, then deploys the trained model in a **Streamlit** web app for interactive predictions.

---

## Project Goals

- Build an end-to-end computer vision classification pipeline:
  - data loading + preprocessing
  - CNN model training with augmentation
  - evaluation on an official test split
  - deployment as a Streamlit demo
- Achieve strong generalization performance on unseen traffic sign images.

---

## Dataset

**Dataset:** GTSRB (German Traffic Sign Recognition Benchmark)  
**Kaggle source:** https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

### Important files
- `Train/` — folders `0` to `42` containing class images
- `Train.csv` — metadata for training images (dataset variant dependent)
- `Test.csv` — metadata for test images including ROI coordinates and paths
- `Test/` — test images referenced by `Test.csv`
- (Optional) `Meta.csv` — sometimes includes label descriptions (depends on dataset variant)

### Test.csv schema (used in this project)
Example columns:
- `Width`, `Height`
- `Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`
- `ClassId`
- `Path` (e.g., `Test/00000.png`)

---

## Notebook Summary

### Preprocessing
- Resize each image to **30×30**
- Convert to RGB arrays
- Normalize pixels to **[0, 1]** (`/255.0`)
- One-hot encode labels with `to_categorical(..., 43)`

### Data split (from notebook outputs)
- Total images loaded: **39,209**
- Training set: **31,367**
- Validation set: **7,842**

### Data augmentation
Used `ImageDataGenerator`:
- `rotation_range=10`
- `zoom_range=0.15`
- `width_shift_range=0.1`
- `height_shift_range=0.1`
- `shear_range=0.15`
- `fill_mode="nearest"`

### Model architecture (Keras Sequential CNN)
- `Conv2D(32, 5×5)` → `Conv2D(32, 5×5)` → `MaxPool2D` → `Dropout(0.25)`
- `Conv2D(64, 3×3)` → `Conv2D(64, 3×3)` → `MaxPool2D` → `Dropout(0.25)`
- `Flatten` → `Dense(256)` → `Dropout(0.5)` → `Dense(43, softmax)`

### Training setup
- Loss: `categorical_crossentropy`
- Optimizer: `adam`
- Metric: `accuracy`
- Epochs: **6**
- Batch size: **32** (via augmented generator)

---

## Results (Test Set)

**Final Test Accuracy:** **96.47%**

Classification report summary:
- Accuracy: **0.96** (12,630 samples)
- Macro avg F1: **0.94**
- Weighted avg F1: **0.96**

The notebook also includes a confusion matrix plot and example predictions.

---

## Saved Model

The notebook saves models in these formats (both appear in the notebook):
- `traffic_classifier.h5`
- `my_model.keras`

Use whichever you prefer in deployment. The Streamlit demo below defaults to `traffic_classifier.h5`.

---

## Streamlit Demo (Deployment)

The Streamlit app supports:
- Predict from an **uploaded image**
- Predict by selecting an image from **Train.csv/Test.csv**
- Optional **ROI cropping** using `Roi.X1..Roi.Y2` to focus on the sign region (recommended for CSV-based images)
- **Random sample** button for quick demo testing
- Top-K probabilities + bar chart + full probability table

---

## Project Structure (recommended)

your_project/
traffic_sign recognition.ipynb
app.py
traffic_classifier.h5
(optional) my_model.keras
dataset/
Train/
0/ ... 42/
Test/
Train.csv
Test.csv
(optional) Meta.csv

## Limitations

- Input resolution is 30×30, which can reduce fine-detail discrimination (e.g., speed limit number differences).
- Some classes are visually similar; confusion can occur between similar sign types.
- The baseline CNN is strong, but transfer learning could improve accuracy and robustness.

## Future Improvements

- Train with higher resolution (e.g., 64×64)
- Use transfer learning (MobileNetV2/EfficientNet)
- Add ClassId → SignName mapping for readable outputs (using Meta.csv if available)
- Add ROI bounding box overlay visualization in the app
- Probability calibration for more reliable confidence values