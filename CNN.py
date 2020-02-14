"""
Author: Georgios Voulgaris
Date: 22/01/2020
Description: Create a simple CNN to train the classifier. Start from small and expand in order to achieve
Optimal performance. Initially there will be 2 classes to train the classifier (water and barren land)
Once a good performance is achieved a total of 10 classes will be added.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from matplotlib import style


REBUILD_DATA = True  # set to true once, then back to false unless I want to change something in my training data.


class Barren_landVSWater:
    IMG_SIZE = 50
    WATER = "Data/Water"
    BARREN_LAND = "Data/Barren_Land"
    TESTING = "Data/Testing"
    LABELS = {WATER: 0, BARREN_LAND: 1}
    training_data = []

    watercount = 0
    barren_landcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f or "jpeg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append(
                            [np.array(img), np.eye(2)[self.LABELS[label]]]
                        )

                        if label == self.WATER:
                            self.watercount += 1
                        elif label == self.BARREN_LAND:
                            self.barren_landcount += 1

                    except Exception as e:
                        pass
                        # print(label, f, str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print('Water:', barren_landvwater.watercount)
        print('Barren_land:', barren_landvwater.barren_landcount)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


net = Net().to(device)

if REBUILD_DATA:
    barren_landvwater = Barren_landVSWater()
    barren_landvwater.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)
print(f"len training data: {len(training_data)}")

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
print(f"X: {X}")
X = X / 255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X) * VAL_PCT)
print(f"validation size: {val_size}")

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(f"len(train x): {len(train_X)}")
print(f"len test_x: {len(test_X)}")


def fwd_pass(X, y, train=False):  #  train=False preventsmodifing weights when we do our validation data
    outputs = net(X)
    acc = (outputs.argmax(dim=1) == y.argmax(dim=1)).float().mean()
    # matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    # acc = matches.count(True) / len(matches)
    loss = loss_function(outputs, y)

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return acc, loss

MODEL_NAME = f"model-{int(time.time())}" # gives a dynamic model name, to

def train(net):
    BATCH_SIZE = 100
    EPOCHS = 100

    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i: i + BATCH_SIZE].view(-1, 1, 50, 50)
                batch_y = train_y[i: i + BATCH_SIZE]

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                # Version 4: Use a graph for monitoring
                f.write(
                    f"{MODEL_NAME},{round(time.time(), 3)},in_sample,{round(float(acc), 2)},{round(float(loss), 4)}\n")
                """
                Version 3:
                Instead of dispalying like version 2, it saves it into a log file.
                f.write(f"{MODEL_NAME}, {int(time.time())}, in_sample, {round(float(loss), 4)}\n")
                """

                """
                Version 2:
                prints this all out every step.
                print(f"In - sample acc: {round(float(acc), 2)} Loss: {round(float(loss), 2)}")
                """

                """
                Version 1:
                Has too many things repeated so it is better to define a fwd_pass function that will be called instead
                of repeating all the bellow.
                
                net.zero_grad()
                V1 outputs = net(batch_X)

                matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, batch_y)]
                in_sample_acc = matches.count(True) / len(matches)

                loss = loss_function(outputs, batch_y)
                loss.backward()
                optimizer.step()
            print(loss)
            print("In-sample acc:", round(in_sample_acc, 2))
            """


train(net)


style.use("ggplot")

model_name = "model-1579796306"


def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, sample_type, acc, loss = c.split(",")

            times.append(timestamp)
            accuracies.append(acc)
            losses.append(loss)

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    ax1.plot(times, accuracies, label='in_samp_acc')
    ax1.legend(loc=2)
    ax2.plot(times, losses, label='in_sample_loss')
    ax2.legend(loc=2)
    plt.show()


create_acc_loss_graph(model_name)


def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]

            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy:", round(correct / total, 3))
