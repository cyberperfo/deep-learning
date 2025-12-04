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

#model tanımlama 


# model oluşturma

#train 

#test

