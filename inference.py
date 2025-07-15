import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os
import torch


MODEL_PATH = './models/fasterrcnn_hands2.pth'
NUM_CLASSES = 5
CONF_THRESH = 0.5  
DEVICE = torch.device('cpu')  #



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=5)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)
model.eval()

tracker = DeepSort(max_age=30, max_cosine_distance=0.5, nn_budget=100)

frame_files = sorted([frame for frame in os.listdir(IMG_DIR) if frame.endswith('.jpg')])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_tensor = F.to_tensor(frame).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)[0]

    detections = []
    for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
        if score < CONF_THRESH:
            continue
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], float(score), str(label.item())))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()