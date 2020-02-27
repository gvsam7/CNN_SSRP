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
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, Grayscale, Resize
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import itertools
import time
import matplotlib.pyplot as plt
from matplotlib import style
import wandb
wandb.init(project="cnn_ssrp")


def _infer_conv_size(w, k, s, p, d):
    """Infers the next size after convolution.

    Args:

        w: Input size.

        k: Kernel size.

        s: Stride.

        p: Padding.

        d: Dilation.

    Returns:

        int: Output size.

    """

    x = (w - k - (k - 1) * (d - 1) + 2 * p) // s + 1

    return x


class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        def _add_layer(in_channels, out_channels, kernel_size, _input_size, stride=1, padding=0, dilation=1):
            in_height, in_width = _input_size[1:]
            conv = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding, dilation=dilation)
            out_h = _infer_conv_size(in_height, kernel_size, stride, padding, dilation)
            out_w = _infer_conv_size(in_width, kernel_size, stride, padding, dilation)

            relu = nn.ReLU(inplace=True)
            mp = nn.MaxPool2d(2, 2)
            out_h //= 2
            out_w //= 2

            conv_block = nn.Sequential(conv, relu, mp)

            return conv_block, (out_channels, out_h, out_w)

        self.conv1, input_size = _add_layer(input_size[0], 32, 5, input_size)
        self.conv2, input_size = _add_layer(input_size[0], 64, 5, input_size)
        self.conv3, input_size = _add_layer(input_size[0], 128, 5, input_size)

        input_size_flattened = np.product(input_size)
        self.fc1 = nn.Linear(input_size_flattened, 512)
        self.fc2 = nn.Linear(512, 9)

        # Define sigmoid activation and softmax output
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

    def convs(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.softmax(x)
        # x = F.softmax(x)
        return x


def step(x, y, net, optimizer, loss_function, train):

    with torch.set_grad_enabled(train):
        outputs = net(x)
        # acc = (outputs.sigmoid().round() == y).float().mean() * 100
        # matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
        # then I want to count how many trues and how many false
        # acc = matches.count(True) / len(matches)  # this will give a percentage of accuracy
        # acc = (outputs.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        acc = outputs.argmax(dim=1).eq(y).sum().item()
        # print(f"Outputs: {outputs}, y: {y}")
        loss = loss_function(outputs, y)

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return acc, loss


@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        x, y = batch
        preds = model(x)
        all_preds = torch.cat((all_preds, preds), dim=0)
    return all_preds


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    EPOCHS = 100
    TRAIN_BATCH_SIZE = 100
    # VAL_BATCH_SIZE = 100
    TEST_BATCH_SIZE = 100
    PRED_BATCH_SIZE = 100

    transforms = Compose([Resize((50, 50)), ToTensor()])
    dataset = ImageFolder("Data_WetSeason", transform=transforms)
    testset = ImageFolder("Test_WetSeason", transform=transforms)
    INPUT_SIZE = dataset[0][0].shape
    """
    train_val_len = int(0.9 * len(dataset))
    test_len = int(len(dataset) - train_val_len)
    train_len = int(0.8 * 0.9 * len(dataset))
    val_len = int(len(dataset) - test_len - train_len)
    """
    """train_len = int(0.7 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = int(len(dataset) - train_len - val_len)
    # train, test = random_split(dataset, lengths=(train_len, test_len))
    train, validation, test = random_split(dataset, lengths=(train_len, val_len, test_len))
    train_loader = DataLoader(train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(validation, batch_size=VAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test, batch_size=TEST_BATCH_SIZE, shuffle=False)
    # prediction_loader = DataLoader(dataset, batch_size=PRED_BATCH_SIZE)
    """
    train_len = int(0.8 * len(dataset))
    test_len = int(len(dataset) - train_len)
    train, test = random_split(dataset, lengths=(train_len, test_len))
    train_loader = DataLoader(train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, batch_size=TEST_BATCH_SIZE, shuffle=False)
    prediction_loader = DataLoader(testset, batch_size=PRED_BATCH_SIZE)

    net = Net(INPUT_SIZE).to(device)
    import pdb; pdb.set_trace()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    # with open("CNN_model.log", "a") as f:
    for epoch in range(EPOCHS):

        net.train()
        sum_acc = 0
        for x, y in train_loader:
            x = x.to(device)
            save_image(x, "x.png")
            y = y.to(device) # .float()
            acc, loss = step(x, y, net=net, optimizer=optimizer, loss_function=loss_function, train=True)
            sum_acc += acc
        train_avg_acc = sum_acc / len(train_loader)
        print(f"Training accuracy: {train_avg_acc:.2f}")

        net.eval()
        sum_acc = 0
        for x, y in test_loader:
            x = x.to(device)
            save_image(x, "x2.png")
            y = y.to(device)  # .float()
            val_acc, val_loss = step(x, y, net=net, optimizer=optimizer, loss_function=loss_function, train=True)
            sum_acc += val_acc
        test_avg_acc = sum_acc / len(test_loader)

        print(f"Validation accuracy: {test_avg_acc:.2f}")
        train_steps = len(train_loader) * (epoch + 1)
        wandb.log({"Train Accuracy": train_avg_acc, "Validation Accuracy": test_avg_acc}, step=train_steps)

    # train_preds = get_all_preds(net, test_loader)
    train_preds = get_all_preds(net, prediction_loader)
    cm = confusion_matrix(testset.targets, train_preds.argmax(dim=1))
    names = ('Apartment Housing', 'Barren Land', 'Brick Kilns', 'Forest', 'Informal/Small Housing',
             'Large Industry', 'Non-irrigated Agriculture', 'Small Industry',
             'Water (river/lake)')
    # wandb.log(cm)
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cm, names)
    plt.show()

    precision, recall, f1_score, support = score(testset.targets, train_preds.argmax(dim=1))
    print('precision: {}'.format(precision, average="None"))
    print('recall: {}'.format(recall, average="None"))
    print('f1_score: {}'.format(f1_score, average="None"))
    print('support: {}'.format(support, average="None"))


"""MODEL_NAME = f"model-{int(time.time())}"
    f.write(f"{MODEL_NAME}, "
            f"epoch: {epoch}, "
            f"time:{round(time.time(), 3)}, "
            f"train accuracy: {round(float(acc), 2)}, "
            f"train loss: {round(float(loss), 4)}, "
            f"validation accuracy: {round(float(val_acc), 2)}, "
            f"validation loss: {round(float(val_loss), 4)}\n")
            """

"""style.use("ggplot")
model_name = "model-1579796306"


def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    ax1.plot(times, accuracies, label='in_sample_acc')
    ax1.plot(times, val_accs, label="out_sample_acc")
    ax1.legend(loc=2)

    ax2.plot(times, losses, label='in_sample_loss')
    ax2.plot(times, val_losses, label="out_sample_loss")
    ax2.legend(loc=2)

    plt.show()
    


create_acc_loss_graph(model_name)
"""
if __name__ == "__main__":
    main()
