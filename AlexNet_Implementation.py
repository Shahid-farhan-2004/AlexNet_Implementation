import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_data=datasets.CIFAR10(root="./data",train=True,transform=transform,download=True)
test_data=datasets.CIFAR10(root="./data",train=False,transform=transform,download=True)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=1000)

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,10)
        )
    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,1)
        return self.classifier(x)

model=AlexNet()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0005)

model.train()
for epoch in range(20):
    for images,labels in train_loader:
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss is {loss.item():.4f}")

model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        outputs=model(images)
        _,predictions=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predictions==labels).sum().item()
    print(f"correctnes is {((correct/total)*100):.2f}")