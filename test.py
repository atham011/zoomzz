import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
from tqdm import tqdm
from dataset import handsssDataset, collate_fn
import os
from torchvision import transforms as T


MODEL_PATH = './models/fasterrcnn_hands.pth'
DATA_PATH = '/Users/arya/Downloads/hands_dataset/test'
ANN_PATH = '/Users/arya/Downloads/hands_dataset/test/_annotations.coco.json'
NUM_CLASSES = 5
DEVICE = torch.device('cpu')
SAVE_PRED_JSON = './predictions.json'
SAVE_GT = './groundtruth.json'



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=5)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)
model.eval()

transform = T.Compose([T.ToTensor()])
data = handsssDataset(DATA_PATH, ANN_PATH, transform=transform)
dataloader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn)


# Generate predictions
results = []
for image_id, (images, targets) in enumerate(tqdm(dataloader)):
    images = [img.to(DEVICE) for img in images]
    outputs = model(images)

    for i in range(len(images)):
        boxes = outputs[i]['boxes'].detach().cpu().numpy()
        scores = outputs[i]['scores'].detach().cpu().numpy()
        labels = outputs[i]['labels'].detach().cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < 0.3:
                continue
            x1, y1, x2, y2 = box
            results.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            })

# Save predictions to JSON
with open(SAVE_PRED_JSON, "w") as f:
    json.dump(results, f)

# Build ground truth in COCO format
def build_coco_gt(dataset, save_path):
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    ann_id = 1
    for image_id in range(len(dataset)):
        image, target = dataset[image_id]
        coco_gt["images"].append({
            "id": image_id,
            "width": image.shape[2],
            "height": image.shape[1],
            "file_name": f"{image_id}.jpg"
        })

        for j in range(len(target["boxes"])):
            box = target["boxes"][j]
            x1, y1, x2, y2 = box.tolist()
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(target["labels"][j].item()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0
            })
            ann_id += 1

    # Use label2index from dataset or fallback to default if missing
    if hasattr(dataset, 'label2index'):
        categories = [{"id": idx, "name": name} for name, idx in dataset.label2index.items()]
    else:
        categories = [{"id": i, "name": f"class_{i}"} for i in range(1, NUM_CLASSES)]
    
    coco_gt["categories"] = categories

    with open(save_path, "w") as f:
        json.dump(coco_gt, f)

# Save GT file
build_coco_gt(dataset, SAVE_GT)

# Run COCO Evaluation
coco_gt = COCO(SAVE_GT)
coco_dt = coco_gt.loadRes(SAVE_PRED_JSON)

coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()