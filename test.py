import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
from tqdm import tqdm
from dataset import handsssDataset, collate_fn  # your existing dataset

# ---- Config ----
MODEL_PATH = './models/fasterrcnn_hands.pth'
DATA_PATH = '/Users/arya/Downloads/hands_dataset/test'
ANN_PATH = '/Users/arya/Downloads/hands_dataset/test/_annotations.coco.json'
NUM_CLASSES = 4  # including background as class 0 if you used that setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PRED_JSON = './predictions.json'

# ---- Load model ----
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---- Load dataset ----
from torchvision import transforms as T
transform = T.Compose([T.ToTensor()])
dataset = handsssDataset(DATA_PATH, ANN_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

# ---- Run inference and collect predictions in COCO format ----
results = []
print("Running inference on test set...")
for images, targets in tqdm(dataloader):
    images = [img.to(DEVICE) for img in images]
    with torch.no_grad():
        outputs = model(images)

    for target, output in zip(targets, outputs):
        image_id = int(target["image_id"].item())
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
            results.append({
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": bbox,
                "score": float(score)
            })


# ---- Save predictions to JSON ----
with open(SAVE_PRED_JSON, 'w') as f:
    json.dump(results, f)

print(f"Saved predictions to {SAVE_PRED_JSON}")

# ---- COCO Evaluation ----
coco_gt = COCO(ANN_PATH)
coco_dt = coco_gt.loadRes(SAVE_PRED_JSON)

coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
