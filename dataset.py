import os
import json
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import ToTensor


# a dataloader essentially just takes your data and puts it into a batch
class handsssDataset(torch.utils.data.Dataset):
    # this is the constructor, takes where the data is stored and then loads it into the handsssDataset object
    def __init__(self, data_path, ann_path, transform=None):
        self.data_path = data_path
        self.ann_path = ann_path
        self.transform = transform

        with open(self.ann_path, 'r') as f:
            self.ann_data = json.load(f)
        
        self.images = {img['id']: img for img in self.ann_data['images']}
        self.annotations = {}
        for ann in self.ann_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.image_ids = list(self.images.keys())
        #self.image_ids = self.image_ids[:50]
    # this is a helper function that returns the length of the data


    def __len__(self):
        return len(self.image_ids)

    # this is a helper function that returns the item at the index idx
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_path = os.path.join(self.data_path, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        image_width, image_height = image.size

        if self.transform is not None:
            image = self.transform(image)

        anns = self.annotations.get(image_id, [])

        boxes = []
        labels = []
        for anni in anns:
            x, y, w, h = anni['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(anni['category_id'])

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(image_id)
        }

        return image, target
    # this is a helper function that returns the data
    def get_data(self):
        return self.data
    
def collate_fn(batch):
    return tuple(zip(*batch))
