#mobilnet ile transfer öğrenme
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
#veri yükleme ve data augmention
#cihaz ayarı
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#veri dönüşümü 
#klasik dönüşümler:tensör dönüşümü ve normalizasyon 
#mobilnet e uygun giriş boyutunun ayarlanması 
#data augmention 
transform_train=transforms.Compose([
    transforms.Resize((224,224)),#net mobilet boyutu
    transforms.RandomHorizontalFlip(),#görüntüyü yatay olarak çevirerek veri artırma
    transforms.RandomRotation(10),#görüntüyü rastgele 10 derece döndürme
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2),#renk varyasyonları
    transforms.ToTensor(),#görüntüleri tensöre çevirme
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))#pixel değerlerini normalleştirme
])
transform_test=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])
#oxford flowers veri seti yükleme
train_dataset=datasets.CIFAR10(root='./data',train=True,transform=transform_train,download=True)
test_dataset=datasets.CIFAR10(root='./data',train=False,transform=transform_test,download=True)
#rastgele 5 örnek seçimi
indices=torch.randint(len(train_dataset),size=(5,))#indeksleri rastgele seç
samples=[train_dataset[i] for i in indices]#örnekler
#görselleştirme 
fig,axes=plt.subplots(1,5,figsize=(15,5))
for i,(image,label) in enumerate(samples):
    image=image.numpy().transpose((1,2,0))#tensörü görüntü formatına çevir
    image=(image*0.5)+0.5#normalizasyonu tersine çevir
    axes[i].imshow(image)
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')
train_loader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)
#transfer learning tanımlama ve fine tuning(train)
#mobilnetv2 modelini yükleme
model=models.mobilenet_v2(pretrained=True)#önceden eğitilmiş ağırlıklar
#sınıflandırcı katmanını değiştirme
num_ftrs=model.classifier[1].in_features
model.classifier[1]=nn.Linear(num_ftrs,102)#son katmanı oxford 102 için değiştiriyoruz
model=model.to(device)# modeli belirtilen cihaza taşıyoruz
#kayıp fonksiyonu ve optimizer tanımlama
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.0001)
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)#öğrenme oranı zamanla azalır
#model eğitimi
epochs=3 
for epoch in tqdm(range(epochs)):#her epoch için ilerleme çubuğu gösteriyoruz
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
    scheduler.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dataset):.4f}')
#modeli kaydetme 
torch.save(model.state_dict(),'mobilenet_transfer_learning.pth')
#test ve değerlendirme
model.eval()
all_preds=[]
all_labels=[]
with torch.no_grad():
    for inputs,labels in test_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        outputs=model(inputs)
        _,preds=torch.max(outputs,1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
#confusion matrix ve sınıflandırma raporu
cm=confusion_matrix(all_labels,all_preds)
plt.figure(figsize=(10,8))  
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
print(classification_report(all_labels,all_preds))
#transfer learning ile mobilnet modeli kullanarak CIFAR-10 veri seti üzerinde sınıflandırma yapıldı.        

