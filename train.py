import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import os
import sys
from tqdm import tqdm
import torchvision.transforms as T
from deep_sort_realtime.deepsort_tracker import DeepSort


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import handsssDataset, collate_fn



def collate_fn(batch):
    return tuple(zip(*batch))


def train(data_path, ann_path):
    print("Loading model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    print("Loaded model...")

    num_classes = 5
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    device = torch.device('cpu') 

    print("Initializing Dataset...")
    transform = T.Compose([T.ToTensor()])
    dataset = handsssDataset(data_path, ann_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    print("Initialized Dataset...")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 1

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for images, targets in progress_bar:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            lr_scheduler.step()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Time: {time.time() - start:.2f}s")

    print("Training complete")
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/fasterrcnn_hands2.pth")


if __name__ == "__main__":
    train(
        data_path='/Users/arya/Downloads/hands_dataset/train',
        ann_path= '/Users/arya/Downloads/hands_dataset/train/_annotations.coco.json'
    )