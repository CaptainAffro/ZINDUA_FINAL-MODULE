# 🚦 Traffic Sign Recognition (GTSRB) — CNN + Error Analysis + Hyperparameter Tuning

## Project Overview
This project builds an end-to-end **traffic sign recognition** system that can classify **43 German traffic sign categories** from images. Beyond training a high-accuracy CNN, the project focuses on being *capstone-ready*: it includes **clear class labeling**, **diagnostic error analysis** (what the model gets wrong and why), and a **hyperparameter tuning workflow** using **KerasTuner (Hyperband)**.  

A lightweight **Streamlit app** is included to demo predictions interactively.

---

## Problem Statement
Given an input image of a traffic sign, predict the correct sign category (one of 43 classes).  

Why this matters:
- Traffic sign recognition is a core component of **ADAS** and **autonomous driving**
- Reliable recognition improves navigation, safety, and compliance
- Real-world deployment requires more than accuracy: we need **interpretability and failure analysis**

---

## Dataset
**GTSRB (German Traffic Sign Recognition Benchmark)** via Kaggle:  
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign  

What’s inside (typical structure in this Kaggle version):
- `Train/` and `Test/` image folders
- `Train.csv` and `Test.csv` metadata including:
  - `Path` (relative path to image)
  - `ClassId` (0–42) — the target label
  - ROI box coordinates: `Roi.X1, Roi.Y1, Roi.X2, Roi.Y2` (useful for cropping to the sign)
- `Meta.csv` (sometimes available): class metadata and/or names depending on dataset variant

---

## Approach Summary (From Baseline → Tuned Model)

### 1) Data Loading
- Load images using paths from CSV
- Build arrays `X` (images) and `y` (labels)
- Normalize pixel values to `[0,1]`

### 2) Preprocessing
- Resize images to a fixed input size (e.g., **30×30**, matching GTSRB standard variants)
- Train/validation split
- One-hot encode labels for multi-class classification

### 3) Data Augmentation
To improve robustness to real-world variation, training uses `ImageDataGenerator` with:
- rotation
- zoom
- horizontal/vertical shifts
- shear transforms
- nearest fill mode

This simulates viewpoint and camera differences common in road scenes.

### 4) Baseline CNN Model
A compact CNN was trained as a strong baseline:

**Architecture (baseline)**
- Conv2D(32, 5×5) → Conv2D(32, 5×5) → MaxPool → Dropout(0.25)
- Conv2D(64, 3×3) → Conv2D(64, 3×3) → MaxPool → Dropout(0.25)
- Flatten → Dense(256) → Dropout(0.5) → Dense(43 softmax)

Loss: `categorical_crossentropy`  
Optimizer: `adam`  
Metric: `accuracy`

### 5) Evaluation + Diagnostics (Baseline)
Beyond accuracy, the notebook includes:
- Classification report (precision/recall/F1 per class)
- Confusion matrix visualization
- **Misclassification visualization**:
  - Random wrong predictions
  - **Most confident wrong predictions**
  - Top confusion pairs (True → Pred)
  - Top-k probability breakdown for individual failures

This moves the project from “trained a model” to “understand the model.”

---

## ✅ Class Labeling (Human-Readable Predictions)
A key improvement is mapping each `ClassId (0–42)` to a readable traffic sign name.

Example:
- `14 → Stop`
- `13 → Yield`
- `1 → Speed limit (30km/h)`

This is used in:
- Prediction displays
- Confusion analysis
- Error visualization titles
- Streamlit UI outputs

---

## 🔍 Error Analysis: What the Model Gets Wrong (and Why)
The notebook explicitly visualizes failure cases and summarizes patterns:

### What we plot
1. **Random misclassifications** (gives a general feel)
2. **Most confident misclassifications** (reveals systematic errors)
3. **Top confusion pairs** (True class most frequently confused with Pred class)
4. **Top-k probabilities** for wrong samples (was the model uncertain or confidently wrong?)

### Common failure patterns (typical for GTSRB)
- **Visually similar signs**  
  e.g., speed limits (30 vs 50), “end of restriction” variants, similar arrow/mandatory signs
- **Small sign footprint** in the image (not enough pixels for digits/icons)
- **Blur and motion blur**
- **Harsh lighting / glare / shadows**
- **Perspective distortion / rotation**
- **Occlusion** (partial sign, poles/trees)

---

## ⚙️ Hyperparameter Tuning (KerasTuner — Hyperband)

### Why tuning?
Even a good CNN can often improve with better choices for:
- convolution filters
- dropout rates
- dense layer size
- learning rate
- early-layer kernel size (handled carefully due to tuner constraints)

### What was tuned
The tuner searched over hyperparameters including:
- `conv1_kernel` (tuned as integer {3,5} then converted to (k,k))
- `conv1_filters`, `conv1_filters_2`
- `conv2_filters`, `conv2_filters_2`
- `drop1`, `drop2`, `drop_dense`
- `dense_units`
- `lr` (log-sampled)

### Important Implementation Detail (Bug Fix)
KerasTuner `Choice()` cannot store tuples like `(3,3)`.  
So the project uses:

```python
k = hp.Choice("conv1_kernel", [3, 5])
kernel_size = (k, k)

### 🏁 Tuned Model Evaluation

After tuning:

- The best hyperparameters are extracted
- The tuned best model is evaluated on the test set
- The same error analysis pipeline is re-run for the tuned model:
  - confusion matrix
  - report
  - wrong predictions grid
  - confident wrong
  - top confusions
  - top-k probability explanations
This ensures improvements are measurable and explainable, not just anecdotal.

### 📌 Findings & Recommendations (Executive Summary)

### Findings

- Tuning improves stability and generalization by selecting better regularization + learning rate
- Remaining errors cluster around:
  - visually similar classes (digit/icon-level distinctions)
  - low-quality inputs (blur, small signs, glare, occlusion)

### Recommendations

- Increase effective sign resolution (e.g., 64×64 inputs or crop-zoom on ROI)
- Add augmentation targeted to failure modes (brightness/contrast jitter, mild blur, perspective)
- Use class weights or focal loss if imbalance/hard classes dominate errors
- Upgrade to a pretrained backbone (MobileNetV2 / EfficientNetB0) for stronger features
- Add interpretability (Top-k + Grad-CAM) for a stronger “why” narrative


### 🌐 Streamlit App (Interactive Demo)

The Streamlit app allows you to:
- Upload an image and predict the traffic sign class
- Or pick a sample from Train.csv / Test.csv
- Optionally crop using ROI coordinates for better focus on the sign
- Display Top-K predictions + probability bar chart
- (Optional extension) run evaluation on a subset and visualize wrong predictions

### Results 

- Baseline Test Accuracy: 96%
- Tuned Test Accuracy: 98.08%

**Most common confusions (Top 3):**

- Beware of ice/snow → Slippery road (count = 25)
- Pedestrians → Right of way at the next intersection (count = 23)
- General caution → Double curve (count = 15)


### Limitations

- Performance can drop on images that differ strongly from training distribution (new camera styles, very low light, heavy motion blur).
- Small signs in large scenes remain challenging without detection/cropping.
- A dedicated detection stage (localizing the sign first) would improve real-world robustness.

### Future Work

- Add a sign detector (e.g., YOLO) + classifier pipeline
- Increase input resolution and/or ROI-based cropping everywhere emphasizes the sign
- Transfer learning with MobileNet/EfficientNet
- Interpretability: Grad-CAM for “what the model looked at”
- Add model monitoring metrics for deployment: confidence thresholds, abstention, and drift checks

### Credits

**Dataset:** GTSRB via Kaggle link above
**Tools:** TensorFlow/Keras, KerasTuner, Streamlit, scikit-learn, matplotlib