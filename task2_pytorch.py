import torch
from torch import nn,optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
from torch.autograd import Variable

BATCH_SIZE = 128
NUM_EPOCHS = 10

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

class SimpleNet(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(D_in, H1)
        self.layer2 = nn.Linear(H1, H2)
        self.layer3 = nn.Linear(H2, D_out)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model = SimpleNet(28 * 28, 300, 100, 10)

# TODO:define loss function and optimiter
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# train and evaluate
for epoch in range(NUM_EPOCHS):
    for images, labels in tqdm(train_loader):
        images = images.view(images.size(0), -1)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        else:
            images = Variable(images)
            labels = Variable(labels)
        out = model(images)
        loss = criterion(out, labels)

        optimizer.zero_grad()#clear all the gradients
        loss.backward()
        optimizer.step()#perform an optimization step

model.eval()
train_accuracy = 0
test_accuracy = 0
#train accuracy
for images, labels in tqdm(train_loader):
    images = images.view(images.size(0), -1)
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
    out = model(images)
    loss = criterion(out, labels)
    _,pred = torch.max(out, 1)
    num_correct = (pred == labels).sum()
    train_accuracy += num_correct.item()
train_accuracy /= len(train_dataset)
#test accuracy
for images, labels in tqdm(test_loader):
    images = images.view(images.size(0), -1)
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
    out = model(images)
    loss = criterion(out, labels)
    _,pred = torch.max(out, 1)
    num_correct = (pred == labels).sum()
    test_accuracy += num_correct.item()
test_accuracy /= len(test_dataset)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))