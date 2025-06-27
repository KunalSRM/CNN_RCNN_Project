import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('yolov5', 'yolov5s', source='local')
model.eval()
cap = cv2.VideoCapture(0)

def detect_frame():
    ret, frame = cap.read()
    if not ret:
        return None

    # Inference
    results = model(frame)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    for idx in range(len(labels)):
        row = cords[idx]
        if row[4] >= 0.5:
            x1, y1, x2, y2 = int(row[0]*frame.shape[1]), int(row[1]*frame.shape[0]), int(row[2]*frame.shape[1]), int(row[3]*frame.shape[0])
            label = model.names[int(labels[idx])]
            confidence = row[4].item()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()
