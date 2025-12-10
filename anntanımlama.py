#problem tanımlama mlist veri seti ile rakam sınıflandırma ANN
#library
import torch 
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#veri seti yükleme
def get_data_loaders(batch_size=64): #her itarasyonda işlenecek veri miktar
    transforms_pipeline = transforms.Compose([transforms.ToTensor(),#goruntuyu tensore çevirme 0 251 arası değerlerni 0 1 arasına çevirme
                                      transforms.Normalize((0.5,), (0.5,))])#normalizasyon
    #train set
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms_pipeline)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms_pipeline)
    #pytorch veri yükleyicisi oluşturma
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
train_loader, test_loader = get_data_loaders()

#veri görselleşitrme
def visualize_samples(loader, n):
    images, labels = next(iter(loader)) #ilk batch den görüntü ve etiketleri al
    fig ,axes = plt.subplots(1, n, figsize=(n*2,2))#n fazla görüntü için alt grafik oluştur
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray') #görüntüyü gri tonlamalı olarak göster
        axes[i].set_title(f'Label: {labels[i].item()}') #etiketi başlık olarak ayarla
        axes[i].axis('off') #eksenleri kapat
    plt.show()
visualize_samples(train_loader, 5)

#model tanımlama 
class Neuralnetwork(nn.Module): #yapay sinir ağı sınıfı
    def __init__(self): #nn inşası için gerekli bileşenleri tanımlama 
        super(Neuralnetwork, self).__init__()
        self.flatten = nn.Flatten() # elimizde bulunan görüntüleri vektörleştirme
        self.fc1 = nn.Linear(28*28, 128) #giriş katmanı 
        self.relu = nn.ReLU() #aktivasyon fonksiyonu
        self.fc2 = nn.Linear(128, 64) #ikinci katmanı input 128 output 64 sınıf için
        self.fc3 = nn.Linear(64, 10) #çıkış katmanı 10 sınıf için
    def forward(self, x): #ileri besleme fonksiyonu (az önce tanımlamadık burda oluşturcaz)
        x = self.flatten(x) #görüntüyü vektörleştir
        x = self.fc1(x) #ilk katman
        x = self.relu(x) #aktivasyon
        x = self.fc2(x) #ikinci katman
        x = self.relu(x) #aktivasyon
        x = self.fc3(x) #çıkış katmanı
        return x


# model oluşturma
model = Neuralnetwork().to('cpu') #modeli cpu ya taşı
#kayıp fonksiyonu ve optimizer tanımlama
define_loss_and_optimizer= lambda model: (nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.01)) #kayıp fonksiyonu ve optimizer tanımlama
#train 
def train_model(model, train_loader, num_epochs=5):
    criterion, optimizer = define_loss_and_optimizer(model) #kayıp fonksiyonu ve optimizer al
    model.train() #modeli eğitim moduna al
    train_losses=[] #her bir epoch için kayıp değerlerini tutmak için liste
    for epoch in range(num_epochs):#belirtilen epoch sayısı kadar eğitim yap
        running_loss = 0.0 #kayıp değerini tutmak için değişken
        for images, labels in train_loader: #tüm veri seti üzerinde iterasyon yap
            images, labels = images.to('cpu'), labels.to('cpu') #veriyi cpu ya taşı
            optimizer.zero_grad() #gradyanları sıfırla
            predictions = model(images) #modelden çıktı al
            loss = criterion(predictions, labels) #kayıp hesapla
            loss.backward() #geri yayılım
            optimizer.step() #optimize et
            running_loss += loss.item() #kayıp değerini topla
        avg_loss = running_loss/len(train_loader)
        train_losses.append(avg_loss)
    return train_losses
#loss graph 
train_losses = train_model(model, train_loader, num_epochs=1)
plt.figure()
plt.plot(range(1,len(train_losses)+1),train_losses,marker="o",linestyle="-",label="train loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.show()
#test
def test_model(model, test_loader):
    model.eval() #modeli değerlendirme moduna al
    correct = 0 
    total = 0
    with torch.no_grad(): #gradyan hesaplamalarını devre dışı bırak
        for images, labels in test_loader: #test veri seti üzerinde iterasyon yap
            images, labels = images.to('cpu'), labels.to('cpu') #veriyi cpu ya taşı
            predictions = model(images) #modelden çıktı al
            _, predicted = torch.max(predictions.data, 1) #en yüksek olasılıklı sınıfı al
            total += labels.size(0) #toplam örnek sayısını güncelle
            correct += (predicted == labels).sum().item() #doğru tahmin sayısını güncelle
    print("Accuracy of the model on the test images: {} %".format(100 * correct / total))
test_model(model, test_loader)        
#main
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()#veri yükleyicileri al
    visualize_samples(train_loader, 5)
    model = Neuralnetwork().to('cpu')
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, num_epochs=5)
    test_model(model, test_loader)