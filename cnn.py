import torch
import torch.nn as nn
import torch.functional as F 
import numpy as np 
import os
import sys
from utils import get_imagefolder_dataset, get_loader
from torchvision import transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import argparse
import gc
import time


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.n_classes = num_classes
        self.conv1 = nn.Conv2d(3, 10, 5, 1, 1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, 5, 1, 1)
        self.batch_norm2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 10, 7, 1)
        self.batch_norm3 = nn.BatchNorm2d(10)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(2250, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 6)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, X):
        output = self.conv1(X)
        output = self.batch_norm1(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.conv2(output)
        output = self.batch_norm2(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.conv3(output)
        output = self.batch_norm3(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = output.view(-1, 2250)
        output = self.linear1(output)
        output = self.dropout1(output)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.dropout2(output)
        output = self.linear3(output)
        return output


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise ValueError("Cuda not available")

    Args = argparse.ArgumentParser()

    Args.add_argument("--epochs", "-e", required=True, help="Number of Epochs")
    Args.add_argument("--batch-size", "-bs", required=True, help="Batch Size")
    Args.add_argument("--learning-rate", "-l", required=True, help="Learning Rate")
    Args.add_argument("--model-type", "-mt", required=True, help="Model Type. Enter 'resnet' or 'custom'")
    Args.add_argument("--experiment-name", "-n", required=True, help="Experiment Name")
    Args.add_argument("--weight-decay", "-w", required=True, help="Weight Decay")

    arg_dict = Args.parse_args()


    NUM_EPOCHS = int(arg_dict.epochs)
    lr =  float(arg_dict.learning_rate)
    BATCH_SIZE = int(arg_dict.batch_size)
    model_type = arg_dict.model_type
    experiment_name = arg_dict.experiment_name
    weight_decay = float(arg_dict.weight_decay)

    print("Parameters parsed ...")

    writer = SummaryWriter(log_dir = "tensor_board_cnn/"+experiment_name)

    transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])


    train_dataset = get_imagefolder_dataset("data/seg_train/", transform = transforms)
    train_loader = get_loader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = get_imagefolder_dataset("data/seg_test/", transform = transforms)
    test_loader = get_loader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    n_categories = 6

    if model_type == "resnet":
        print("Loading pretrained ResNet18 ...")
        model = models.resnet18(pretrained=True).to(device)
        model.fc = nn.Linear(512, 6).to(device)
    elif model_type == "custom":
        model = CustomCNN(n_categories).to(device)
    else:
        raise TypeError("Mode type should be resnet or custom")

    print("Model Loaded!")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)

    test_loss_record = []
    test_acc_record = []
    time_log = []
    for j in range(NUM_EPOCHS):
        start = time.time()
        model = model.train()
        epoch_loss = 0
        for i, (X,y) in enumerate(train_loader):
            optimizer.zero_grad()

            X = X.to(device)
            y = y.to(device)

            output = model(X)
            loss = loss_fn(output, y)
            epoch_loss += loss.item()*BATCH_SIZE

            loss.backward()
            optimizer.step()

        gc.collect()
        train_total = (i+1)*BATCH_SIZE
        print("Train total: ", train_total)
        epoch_loss = epoch_loss/train_total
        print("Epoch {0} train loss: {1}".format(j+1, epoch_loss))
        writer.add_scalar("Train Loss", epoch_loss, j+1)
        model = model.eval()
        acc = 0
        total = 0
        test_loss = 0
        loss = 0
        for i, (X,y) in enumerate(test_loader):
            optimizer.zero_grad()

            X = X.to(device)
            y = y.to(device)

            output = model(X)
            loss = loss_fn(output, y).item()
            test_loss += loss

            label = torch.argmax(output, 1)

            acc += torch.sum(label == y).item()
            total += len(y)

        test_loss = test_loss*BATCH_SIZE/total
        acc = acc/total
        test_loss_record.append(test_loss)
        test_acc_record.append(acc)
        writer.add_scalar("Test Accuracy", acc, j+1)
        writer.add_scalar("Test Loss", test_loss, j+1)
        print("Test total: ", total)
        print("Epoch {0} test loss: {1}".format(j+1, test_loss))
        print("Epoch {0} test accuracy: {1}".format(j+1, acc))
        end = time.time()
        time_log.append(end-start)

    average_time = np.mean(time_log)
        
    experiment_details = {
        "model": model_type,
        "lr": lr,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS
    }

    metrics = {
        "final_train_loss": epoch_loss,
        "final_test_loss": test_loss,
        "test_accuracy": acc,
        "best_test_loss": min(test_loss_record),
        "best_test_accuracy": max(test_acc_record),
        "average_epoch_time": average_time
    }

    writer.add_hparams(experiment_details, metrics)
    writer.flush()