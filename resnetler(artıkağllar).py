#resnet sınıflandırma -> cıfar10
#transfer learning ile ör: optimize ve hızlı anlanda
#custom resnet build ile ör:büyük veri setlerinde
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models #önceden eğitilmiş modelleri içeriye aktar 
from tqdm import tqdm
#veri yükleme
device=torch.device('cuda' if torch.cuda.is_available() else'cpu')
#veri yükleme işlemi
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_dataset=torchvision.datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)
test_dataset=torchvision.datasets.CIFAR10(root='./data',train=False,transform=transform,download=True)
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=False) 
#residual blokların oluşturulması
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(ResidualBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.downsample=downsample
    def forward(self,x):
        identity=x#kendi kedine bağlanacak giriş verisi 
        if self.downsample is not None:
            identity=self.downsample(x)
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out+=identity#skip connection
        out=self.relu(out)
        return out
#renset oluşturma(custom)
class CustomResnet(nn.Module):
    """convd2->bn->relu->maxpool->(resnet blocks)layer1->layer2->layer3->layer4->avgpool->(flatten)fc"""
    def __init__(self):
        super(CustomResnet, self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._make_layer(64,64,2)
        self.layer2=self._make_layer(64,128,2,stride=2)
        self.layer3=self._make_layer(128,256,2,stride=2)
        self.layer4=self._make_layer(256,512,2,stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512,10)#cifar10 için 10 sınıf

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample=None
        if stride!=1 or in_channels!=out_channels:
            downsample=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers=[ResidualBlock(in_channels,out_channels,stride,downsample)]
        for _ in range(1,blocks):
            layers.append(ResidualBlock(out_channels,out_channels))
        return nn.Sequential(*layers)    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x
model=CustomResnet()  
#resnet ile transfer learning ve custom resnet ile training
use_transfer_learning=True
if use_transfer_learning:
    model=CustomResnet().to(device)
else:
    model=models.resnet18(pretrained=True)#hazır resnet modeli ile fine tuning
    num_ftrs=model.fc.in_features#tam bağlı katmandaki giriş boyutu
    model.fc=nn.Sequential(#kendi sınıflandırıcı blok
    nn.Linear(num_ftrs,10),
    nn.ReLU(),
    nn.Linear(256,10)
    )
    model=model.to(device)
#kayıp fonksiyonu ve optimizer tanımlama
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
#model eğitimi
num_epochs=5
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss=0.0
    for inputs,labels in train_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*inputs.size(0)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataset):.4f}') 
#model değerlendirme ve testi
model.eval()
correct=0
total=0
with torch.no_grad():
    for inputs,labels in test_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        outputs=model(inputs)
        _,preds=torch.max(outputs,1)
        total += labels.size(0)
        correct += (preds==labels).sum().item()
print(f'Accuracy of the model on the test images: {100 * correct / total} %')
