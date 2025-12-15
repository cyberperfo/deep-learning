#radyal temelli fonksiyon (RBF) ağı
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#veri setinin içiriye aktarılaması
df=pd.read_csv("iris.data",header=None) #örnek veri seti csv dosyasından yükleniyor
X=df.iloc[:, :-1].values #ilk 4 sütünu x içine atar
y=df.factorize(df.iloc[:, -1])#hedef etiketi alıp y değişkeni için sayısal değerlere dönüştürür
#veriyi standartize etme 
scaler=StandardScaler()
X=scaler.fit_transform(X)
#train ve test verisi ayırma
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
 #tensöre çevirme işlemi 
def to_tensor(X,y):
    return torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.long)
X_train,y_train=to_tensor(X_train,y_train)
X_test,y_test=to_tensor(X_test,y_test)
#rbfn modeli ve rbf_kernel,in tanımlanması
def rbf_kernel(X,centers,beta):
    return torch.exp(-beta*torch.cdist(X,centers)**2)
class RBFN(nn.Module):
    def __init__(self,input_size,num_centers,output_size,beta=2.0):
        super(RBFN,self).__init__()
        self.centers=nn.Parameter(torch.randn(num_centers,input_size))#rbf merkezlerini rastgele başlat 
        self.beta=nn.Parameter(torch.ones(1)*beta) #beta paremtresi rbf nin genişliğini kontrol eder
        self.linear=nn.Linear(num_centers,output_size)#outputu tam bağlantılı katmana gönderir
    def forward(self,x):
        #rbf çekirdek fonksiyonu hesapla 
        phi = rbf_kernel(x,self.centers,self.beta)
        out = self.linear(phi)
        return out  
model=RBFN(input_size=4,num_centers=10,output_size=3)        
# model training
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
num_epochs=100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs=model(X_train)
    loss=criterion(outputs,y_train)
    loss.backward()
    optimizer.step()#ağırlıkları güncelle
    if (epoch+1)%10==0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# model testi ve değerlendirme
with torch.no_grad():
    y_pred=model(X_test)#test verisi ile tahmin et 
    accuary=(y_pred.argmax(dim=1)==y_test).float().mean()#doğruluk hesapla
    print(f'Accuracy: {accuary.item()*100:.2f}%')

