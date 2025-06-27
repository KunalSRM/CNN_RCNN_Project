import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load pre-trained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
cap = cv2.VideoCapture(0)

# COCO class names (hardcoded for clarity)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect_frame():
    ret, frame = cap.read()
    if not ret:
        return None

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = F.to_tensor(img)

    with torch.no_grad():
        preds = model([img])[0]

    boxes = preds['boxes'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()
    classes = preds['labels'].cpu().numpy()

    for box, score, cls in zip(boxes, scores, classes):
        if score > 0.5:
            x1, y1, x2, y2 = box
            label = COCO_INSTANCE_CATEGORY_NAMES[cls]
            text = f'{label}: {score:.2f}'

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()
