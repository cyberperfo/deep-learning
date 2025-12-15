#veri skıştırma otomatik kodlayıcı (autoencoder) ağı
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
#veri seti yükleme ve ön işleme
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset=datasets.FashionMNIST(root='./data',train=True,transform=transform,download=True)
test_dataset=datasets.FashionMNIST(root='./data',train=False,transform=transform,download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=128,shuffle=False)
#autoencoder modeli tanımlama
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        #encoder
        self.encoder=nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
        )
        #decoder
        self.decoder=nn.Sequential(
            nn.Linear(16,64),
            nn.ReLU(),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Unflatten(1,(1,28,28))
        )
    def forward(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded
 #early stopping için yardımcı fonksiyon
class EarlyStopping:
    def __init__(self,patience=5,min_delta=0.001):
        self.patience=patience#kaç epoch boyunca iyileşme olmazsa durducağımızı belirten parametre
        self.min_delta=min_delta#kayıptaki minimum iyileşme miktarı
        self.counter=0#sabit kalan epoch sayısı
        self.best_loss=None#en iyi kayıp değeri
    def __call__(self,loss):
        if self.best_loss is None or loss<self.best_loss-self.min_delta:#gelişme var 
            self.best_loss=loss
        else:
            self.counter+=1
        if self.counter>=self.patience:
            return True
        return False
#model eğitimi    
#hyperparametreler
epochs=50
learning_rate=0.001
model=Autoencoder()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
early_stopping=EarlyStopping(patience=7,min_delta=0.0005)
#eğitim döngüsü
for epoch in range(epochs):
    model.train()
    train_loss=0
    for data,_ in train_loader:
        optimizer.zero_grad()
        prediction=model(data)
        loss=criterion(prediction,data)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item
    avg_loss=total_loss/len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}')
    if early_stopping(avg_loss):
        print("Early stopping triggered")
        break
#model testi
from scipy.ndimage import gaussian_filter
def compute_ssim(img1,img2,sigma=1.5):
    C1=(0.01*255)**2
    C2=(0.03*255)**2
    mu1=gaussian_filter(img1,sigma)
    mu2=gaussian_filter(img2,sigma)
    mu1_sq=mu1**2
    mu2_sq=mu2**2
    mu1_mu2=mu1*mu2
    sigma1_sq=gaussian_filter(img1**2,sigma)-mu1_sq#varyans hesabı
    sigma2_sq=gaussian_filter(img2**2,sigma)-mu2_sq
    sigma12=gaussian_filter(img1*img2,sigma)-mu1_mu2#kovaryans hesabı
    #ssim haritası hesaplama
    ssim_map=((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return ssim_map.mean()
def evaluate(model,test_loader,n_images=10):
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs,_=batch
            outputs=model(inputs)#modelin çıktılarını üretiyoruz
            break
    inputs = inputs.numpy()
    outputs = outputs.numpy()
    fig, axes = plt.subplots(2, n_images, figsize=(15, 3))
    ssim_scores=[]
    for i in range(n_images):
        img1=np.squeeze(inputs[i])#orijinal görüntü sıkıştır
        img2=np.squeeze(outputs[i])#yeniden oluşturulan görüntü sıkıştır
        ssim_score=compute_ssim(img1,img2)#ssim skorunu hesapla yani benzerlik ölçütü
        ssim_scores.append(ssim_score)#skorları listeye ekle
        axes[0,i].imshow(img1,cmap='gray')
        axes[0,i].axis('off')
        axes[1,i].imshow(img2,cmap='gray')
        axes[1,i].axis('off')
    axes[0,0].set_tittle("Original")
    axes[1,0].set_tittle("Decoded image")
    plt.show()
    avg_ssim=np.mean(ssim_scores)#ortalama ssim skorunu hesapla
    print(f'Average SSIM over {n_images} images: {avg_ssim:.4f}')
evaluate(model,test_loader,n_images=10)    