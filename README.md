# 👁️ TrackVision: Real-Time Object Detection & Tracking using YOLOv8 and Deep SORT

A computer vision project built with **Python**, **OpenCV**, **YOLOv8**, and **Deep SORT** for detecting and tracking multiple objects in real time using a webcam or video file.

The system detects objects frame by frame, draws bounding boxes, and assigns unique tracking IDs so objects can be followed continuously while moving.

---

## 📌 Project Overview

This project performs **real-time object detection and multi-object tracking** using a pre-trained **YOLOv8** model and **Deep SORT**.

A webcam or video stream is captured using OpenCV. YOLO detects objects in each frame and predicts:

- object type
- confidence score
- bounding box location

Deep SORT then:

- assigns a unique ID to each object
- tracks movement across frames
- keeps IDs stable when objects move
- handles overlapping objects more accurately

This project demonstrates:

- computer vision with OpenCV
- real-time webcam/video processing
- object detection using YOLOv8
- multi-object tracking with Deep SORT
- working with pre-trained deep learning models
- drawing bounding boxes, labels, and tracking IDs

---

## 🚀 How to Use

1. Run the project.
2. The webcam opens automatically.
3. YOLO detects visible objects.
4. Deep SORT assigns tracking IDs.
5. Bounding boxes and labels are shown in real time.
6. IDs stay consistent while objects move.
7. Press **Q** to exit.

Example:

```text
person ID:1
chair ID:2
cell phone ID:3
```

---

## ✨ Features

- 📷 Real-time webcam input
- 🎥 Optional video file support
- 🧠 YOLOv8 object detection
- 🎯 Deep SORT multi-object tracking
- 📦 Bounding boxes with labels
- 🔢 unique tracking IDs
- ⚡ fast real-time inference

---

## 🛠️ Tech Stack

### Computer Vision

- OpenCV

### Deep Learning

- YOLOv8 (Ultralytics)

### Object Tracking

- Deep SORT

### Programming Language

- Python

### Utilities

- NumPy

---


## ⚙️ Installation & Setup

### 1. Clone repository

```bash
git clone https://github.com/your-username/CodeAlpha_TrackVision.git
```

Move into project:

```bash
cd your-project-folder
```

---

### 2. Create virtual environment

### Windows

```bash
python -m venv cvenv
cvenv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run project

```bash
python main.py
```

The webcam opens automatically.

Press:

```bash
q
```

to quit.

---

## 🎥 Use Video File Instead of Webcam

By default:

```python
cap = cv2.VideoCapture(0)
```

To use a video file:

```python
cap = cv2.VideoCapture("videos/sample.mp4")
```

Then run:

```bash
python main.py
```

---

## 🧠 How It Works

### Step 1 – Capture frame

OpenCV captures live frames:

```text
Webcam
↓
Frame captured
```

---

### Step 2 – YOLOv8 detection

YOLO processes each frame.

Detects objects like:

- person
- chair
- laptop
- phone
- car

Predicts:

- class
- confidence
- coordinates

Example:

```text
person → 98%
chair → 87%
```

---

### Step 3 – COCO label mapping

`coco.txt` converts YOLO class IDs into readable names.

Example:

```text
0 → person
56 → chair
67 → cell phone
```

---

### Step 4 – Deep SORT tracking

Deep SORT:

- predicts object movement
- matches detections to previous tracks
- maintains consistent IDs

Example:

```text
Frame 1:
person ID:1

Frame 2:
same person → ID:1
```

Even if objects overlap.

---

### Step 5 – Display output

OpenCV draws:

- green bounding boxes
- class labels
- tracking IDs

Example:

```text
person ID:1
chair ID:2
```

---


## 🎯 Future Improvements

Possible enhancements:

- people counting
- line crossing detection
- save processed video
- object analytics dashboard
- GPU acceleration
- web dashboard with Streamlit
- alert system for selected objects

---

## 👨‍💻 Author

Built by **Fuvad**
