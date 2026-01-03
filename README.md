# Human Action Recognition with Image Captioning (HMDB51)
## 📌 Project Overview
This project implements an **end-to-end Human Action Recognition system** using **image frame sequences** from the **HMDB51 dataset**. The system not only recognizes the action being performed but also generates a **natural language caption** describing the action.

The complete pipeline includes:

* Frame-based action recognition using **CNN + LSTM**
* Backend inference using **Flask**
* Frontend interface for uploading frames
* Caption generation for recognized actions

---

## 🧠 Methodology

### 1. Dataset

* **HMDB51 dataset** (frame-level version)
* Each action contains multiple folders of image frames

### 2. Feature Extraction

* Frames resized to **224 × 224**
* Normalized to range **[0, 1]**
* **16 frames per sequence** sampled uniformly
* CNN backbone: **MobileNetV2 (ImageNet pretrained)**

### 3. Model Architecture

```
Input (16 × 224 × 224 × 3)
   ↓
TimeDistributed MobileNetV2
   ↓
Global Average Pooling
   ↓
LSTM (256 units)
   ↓
Dense + Softmax
```

* Temporal modeling handled by **LSTM**
* Classification performed over **14 action classes**

---

### 4. Training Details

* Optimizer: Adam
* Loss: Categorical Crossentropy
* Batch Size: 4
* Epochs: 10
* Train/Test Split: 80/20

Trained model outputs:

* `hmdb_frames_cnn_lstm.h5`
* `hmdb_labels.npy`

---

## 🖥 Backend (Flask API)

### API Endpoint

```
POST /predict
```

### Input

* Multiple image frames uploaded as `frames[]`

### Output (JSON)

```json
{
  "prediction": "clap",
  "confidence": 0.9999,
  "caption": "A person is clapping their hands."
}
```

---

## 🖼 Frontend

* HTML + JavaScript interface
* Allows uploading multiple frames
* Displays:

  * Predicted action
  * Confidence score
  * Generated caption

---

## 📝 Caption Generation

Captioning is implemented using **semantic mapping**:

```python
action_to_caption = {
  "clap": "A person is clapping their hands.",
  "walk": "A person is walking forward.",
  "run": "A person is running quickly."
}
```

This approach ensures:

* Fast inference
* Grammatically correct captions
* Easy extensibility

---

## ▶ How to Run the Project

### 1. Install Dependencies

```
pip install tensorflow flask opencv-python numpy scikit-learn tqdm
```

### 2. Train the Model

```
python train_model.py
```

### 3. Start Backend Server

```
python app.py
```

### 4. Open Frontend

* Navigate to: `http://127.0.0.1:5000`
* Upload image frames
* View prediction and caption

---

## 🎯 Results

* Accurate recognition of human actions
* Real-time inference on frame sequences
* Natural language caption generation

---

## 🎓 Academic Relevance

This project demonstrates:

* Deep learning for video understanding
* Temporal modeling with LSTM
* Practical ML system deployment
* End-to-end integration (ML + Backend + UI)

Suitable for:

* Final Year Project (FYP)
* MS / BS thesis
* Research prototyping

---

## 🔮 Future Enhancements

* Transformer-based video models
* Attention-based image captioning
* Real-time webcam inference
* FastAPI deployment

---

## 📜 License

This project is for academic and research purposes.
