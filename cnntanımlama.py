#Problem tanımı :Cıfar10 veri seti üzerinde CNN kullanarak görüntü sınıflandırması yapma
#Gerekli kütüphaneler
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#veri seti yükleme
def get_data_loaders(batch_size=64): #her itarasyonda işlenecek veri miktar
    transforms_pipeline = transforms.Compose([transforms.ToTensor(),#goruntuyu tensore çevirme 0 251 arası değerlerni 0 1 arasına çevirme
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#normalizasyon(RGB ayarından dolayı 3 ölçekte işlem)
    #train ve test setini oluştur ve cifar10 veri setini indir
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_pipeline)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_pipeline)
    #pytorch veri yükleyicisi oluşturma
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
train_loader, test_loader = get_data_loaders()
#veri görselleşitrme
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy() #tensorden numpy array e çevir
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #3 kanal için renkleri doğru şekilde gösterme 
    plt.show()
def get_sample_images(train_loader):
    dataiter = iter(train_loader)#veri gruplarına tekrar tekrar erişmek için iterator oluştur
    images, labels =next(dataiter) #ilk batch den görüntü ve etiketleri al
    return images, labels
def visualize(n):
    train_loader, test_loader = get_data_loaders()
    images ,labels = get_sample_images(train_loader)
    plt.figure()     
    for i in range(n):#n tane görüntü için alt grafik oluştur
        plt.subplot(1, n, i+1)
        imshow(images[i])
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.show()
visualize(5)
#model tanımlama
class CNN(nn.Module): #CNN sınıfı
    def __init__(self): #CNN inşası için gerekli bileşenleri tanımlama 
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) #ilk konvolüsyon katmanı 3 giriş kanalı(RGB) 32 çıkış kanalı
        self.relu = nn.ReLU() #aktivasyon fonksiyonu
        self.pool = nn.MaxPool2d(kernel_size=2,stride= 2)#2x2 pooling katmanı
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #ikinci konvolüsyon katmanı 32 giriş kanalı 64 çıkış kanalı
        self.dropot = nn.Dropout(0.25) #dropout katmanı
        self.fc1 = nn.Linear(64 * 8 * 8, 512) #tam bağlantılı katman
        self.fc2 = nn.Linear(512, 10) #çıkış katmanı 10 sınıf için
        #image 3x32x32->conv1(32)-> relu(32)->pool(16)->conv2(16)->relu(16)->pool(8)->image=8x8
    def forward(self, x): #ileri besleme fonksiyonu
        """
        image 3x32x32->conv1(32)-> relu(32)->pool(16)->conv2(16)->relu(16)->pool(8)->image=8x8
        flatten ->fc1->relu->dropout->fc2->output
        """
        x = self.pool(self.relu(self.conv1(x))) #ilk konvolüsyon + aktivasyon + pooling
        x = self.pool(self.relu(self.conv2(x))) #ikinci konvolüsyon + aktivasyon + pooling
        x =self.dropot(self.relu(self.fc1(x))) #fully conccetd layer(dropout9
        x = x.view(-1, 64 * 8 * 8) #tensörü düzleştir
        x = self.relu(self.fc1(x)) #tam bağlantılı katman + aktivasyon
        x = self.fc2(x) #çıkış katmanı
        return x

device = torch.device('cpu')
model = CNN().to(device) #modeli cpu ya taşı
#kayıp fonksiyonu ve optimizer tanımlama
define_loss_and_optimizer= lambda model: (nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001,momentum=0.9)) #kayıp fonksiyonu ve optimizer tanımlama    
#train
def train_model(model, train_loader,criterion,optimizer,num_epochs=10):
    model.train() #modeli eğitim moduna al
    train_losses=[] #her bir epoch için kayıp değerlerini tutmak için liste
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) #veriyi cihaza taşı
            optimizer.zero_grad() #gradyanları sıfırla
            prediction = model(images) #modelden çıktı al forward pro(prediction)
            loss = criterion(prediction, labels) #kayıp hesapla
            loss.backward() #geri yayılım
            optimizer.step() #optimize et
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    return train_losses
train_loader, test_loader = get_data_loaders()
model = CNN().to(device)
criterion, optimizer = define_loss_and_optimizer(model)
train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=10)

#loss graph
plt.figure()
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Training Loss', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()  
#test
def test_model(model, test_loader,dataset_type):
    model.eval() #modeli değerlendirme moduna al
    correct = 0
    total = 0
    with torch.no_grad(): #gradyan hesaplamalarını devre dışı bırak
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) #veriyi cihaza taşı
            predictions = model(images) #modelden çıktı al
            _, predicted = torch.max(predictions.data, 1) #en yüksek olasılıklı sınıfı al
            total += labels.size(0) #toplam örnek sayısını güncelle
            correct += (predicted == labels).sum().item() #doğru tahmin sayısını güncelle    
    print(f'Accuracy of the model on the {dataset_type} images: {100 * correct / total} %')
test_model(model, test_loader,dataset_type='test')#test accuary değerini yazdır
test_model(model, train_loader,dataset_type='train')#train accuracy değerini yazdır
#main
if __name__ == "__main__":
    #veri seti yükleme 
    train_loader, test_loader = get_data_loaders()
    #veri görselleştirme
    visualize(5)
    #model oluşturma    
    model = CNN().to(device)
    #kayıp fonksiyonu ve optimizer tanımlama
    criterion, optimizer = define_loss_and_optimizer(model)
    #train
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)
    #test
    test_model(model, test_loader,dataset_type='test')
    test_model(model, train_loader,dataset_type='train')            