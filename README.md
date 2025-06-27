# Real-Time Object Detection with YOLOv5 and Faster R-CNN

This project is a Flask-based web application for **real-time object detection** using two models:
- **YOLOv5**
- **Faster R-CNN (PyTorch)**

You can dynamically switch between YOLOv5 and Faster R-CNN during runtime via the web interface.

---

## 🚀 Features
- Real-time webcam object detection.
- Supports switching between YOLOv5 and Faster R-CNN models without restarting the server.
- Flask web interface to select the model.
- Live video streaming using Flask routes.

---

## 📂 Project Structure
```text
├── app.py               # Flask app entry point
├── rcnn_detect.py       # Faster R-CNN model logic
├── yolo_detect.py       # YOLOv5 model logic
├── yolov5s.pt           # YOLOv5 model weights (optional to upload)
├── templates/
│   └── index.html       # Frontend interface
├── static/              # CSS, JS, and other static files
└── README.md
