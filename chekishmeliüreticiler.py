#problem tanımı :çekişmeli üretici üretme
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
#veri seti hazırlama:MNİST
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=64
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
#mnist veri seti yükleme
train_dataset=datasets.MNIST(root='./data',train=True,transform=transform,download=True)
#veri yükleyici oluşturma batchler halinde veriyi yükleme
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
#discirminator hazırlama
image_size=28*28  # MNIST images are 28x28 pixels
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model=nn.Sequential(#sequential ardışık sentezleme
            nn.Linear(image_size,1024),#input: image size,1024 :nöron sayısı yani layerin outputu
            nn.LeakyReLU(0.2),
            nn.Linear(1024,512),#512 boyuta indirgeme
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),#256 boyuta indirgeme
            nn.LeakyReLU(0.2),  
            nn.Linear(256,1),#256 dan 1 e indirgeme gerçekmi sahte kararını ver
            nn.Sigmoid()#gerçeklik değerini yüzde ile ölçüyor 0-1 arasında değer
        )
    def forward(self,img):
        return self.model(img.view(-1,image_size))#gorüntüyü düzleştirerek modele ver 
#generator hazırlama
class Generator(nn.Module):
    def __init__(self,latent_dim):
        super(Generator, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(latent_dim,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,image_size),
            nn.Tanh()#-1 ile 1 arasında çıktı verir
        )
    def forward(self,z):
        return self.model(z).view(-1,1,28,28)#28x28 boyutunda görüntü oluştur ,reshape ayarı -1 ile 1 arb kanalı veya farklı ayarlar    
#model eğitme
#hyperparametreler
learning_rate=0.02
z_dim=100 #rastgele gurultu vektör boyutu (noise görüntüsü)
epochs=20# eğitim dongu sayısı 
#model başlatma 
# #model başlatma:discriminator ve generator tanımlama
generator=Generator(z_dim).to(device)
discriminator=Discriminator().to(device)
#kayıp fonksiyonu ve optimizatör tanımlama
criterion = nn.BCELoss()#binary cross entoropy 
g_optimizer=optim.Adam(generator.parameters(),lr=learning_rate,betas=(0.5,0.999))
d_optimizer=optim.Adam(discriminator.parameters(),lr=learning_rate,betas=(0.5,0.999))
#eğitim döngüsünün başlatılması 
for epoch in range(epochs):
    for i, (real_images,_) in enumerate(train_loader):
        real_images=real_images.to(device)
        batch_size= real_images.size(0)
        real_labels=torch.ones(batch_size,1).to(device)#gerçek görüntüleri 1 olucak şekilde etiketle
        fake_labels=torch.zeros(batch_size,1).to(device)#fake görüntüleri 0 olucak şekilde etiketle  
        #discrimatör eğitimi
        #discriminator egitimi
        z = torch.randn(batch_size, z_dim).to(device) # rastgele gurultu uret
        fake_imgs = generator(z) # generator ile sahte goruntu olustur
        real_loss = criterion(discriminator(real_images), real_labels) # gercek goruntu kaybi
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels) # sahtgoruntulerin kaybi
        d_loss=real_loss + fake_loss # toplam discriminator kaybi
        d_optimizer.zero_grad() # gradyanlari sifirla
        d_loss.backward() # geriye yayilim
        d_optimizer.step() # parametreleri guncelle
        #generator egitilmesi
        g_loss = criterion(discriminator(fake_imgs), real_labels) # generator kaybi 
        g_optimizer.zero_grad() # gradyanlari sifirla
        g_loss.backward() # geriye yayilim
        g_optimizer.step() # parametreleri guncelle
    print(f"Epoch {epoch + 1}/{epochs}, d_Loss: {d_loss.item():.3f}, gloss: {g_loss.item():.3f}")
#model testi
with torch.no_grad():#gradyan hesaplamalarını devre dışı bırak
    z = torch.randn(16, z_dim).to(device) # 16 rastgele gurultu uret
    sample_imgs = generated_images = generator(z).cpu() # generator ile sahte goruntu olustur ve cpu ya tası
    grid= np.transpose(torchvision.utils.make_grid(sample_imgs, nrow=4, padding=2, normalize=True),(1,2,0))#goruntuleri ızgara sekline getir
    plt.imshow(grid) # goruntuleri goster
    plt.show()