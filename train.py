import os
import glob
import json

import cv2
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from preprocess import DrivableDataset
from models import Unet
from metric import compute_miou

device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "bdd100k/images/100k/"
train_images = glob.glob(data_dir + "train/*.jpg")
val_images = glob.glob(data_dir + "val/*.jpg")

trainset = DrivableDataset(train_images)
valset = DrivableDataset(val_images)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)

model = Unet()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

num_epochs = 10
train_losses, val_losses = [], []
min_val_loss = np.Inf
for epoch in range(num_epochs):
  # training loop
    model.train()
    running_loss = 0.0
    train_iou = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iou = compute_miou(outputs, labels)
        train_iou += iou

    train_losses.append(running_loss/len(trainloader))

    # validation loop
    model.eval()
    val_running_loss = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            iou = compute_miou(outputs, labels)
            val_iou += iou

    val_losses.append(val_running_loss/len(valloader))

    if val_running_loss/len(valloader) < min_val_loss:
        min_val_loss = val_running_loss/len(valloader)
        checkpoint = {"model": model.state_dict(), "optimizer":optimizer.state_dict(), "epoch":epoch}
        torch.save("best_model.pth", checkpoint)
    
    log_dict = {'train_iou': train_iou, 'val_iou': val_iou, 'epoch':epoch, 'train_loss':running_loss/len(trainloader), 'val_loss':val_running_loss/len(valloader)}
    with open("log.json") as f:
        json.dump(f, log_dict)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(trainloader)}, Val Loss: {val_running_loss/len(valloader)}")
    print(f"Epoch {epoch+1}/{num_epochs}, Train IoU: {train_iou/len(trainloader)}, Val IoU: {val_iou/len(valloader)}")

