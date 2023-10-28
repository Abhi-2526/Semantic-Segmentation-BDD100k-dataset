import cv2
import numpy as np 
import glob
from torch.utils.data import Dataset, DataLoader
import torch

class DrivableDataset(Dataset):

    def __init__(self, images):

        self.images = images

    def __getitem__(self, item):

        img_path = self.images[item]
        orig_image = cv2.imread(img_path)
        resize_image = cv2.resize(orig_image, (512, 512))
        image = np.float32(resize_image) / 127.5 - 1
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2,0,1)

        label_path = self.images[item].replace('images/100k', 'labels/drivable/masks').replace('.jpg', '.png')
        label = cv2.imread(label_path)[:, :, 0]
        label = cv2.resize(label, (512, 512))
        mask = np.zeros_like(resize_image)
        mask[:, :, 0] = (np.array(label) == 0).astype(np.uint8)
        mask[:, :, 1] = (np.array(label) == 1).astype(np.uint8)
        mask[:, :, 2] = (np.array(label) == 2).astype(np.uint8)

        mask_tensor = torch.from_numpy(mask).long()
        label_tensor = torch.from_numpy(label).long()

        return image_tensor, label_tensor
    
    def __len__(self):
        return len(self.images)





