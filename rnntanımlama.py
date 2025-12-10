#rnn da zaman setleri kullanılır
#veri seti seçme
#librry import etme
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#veri oluşturma ve görselleştirme
def generate_data(seq_length=50, num_samples=1000):
    """example :3lu paket 
    sequence : [2,3,4] giriş dizileri saklmak için 
    target [5]hedef değerleri saklamak için"""
    x = np.linspace(0, 100, seq_length)
    y=np.sin(x)
    sequence = []
    targets = []
    for i in range(len(x)-seq_length):
        sequence.append(y[i:i+seq_length])
        targets.append(y[i+seq_length])
    plt.figure(figsize=(10,5))
    plt.plot(x, y,label='sin(t)',color='blue',linewidth=2)
    plt.title('Sine Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()    

    return np.array(sequence), np.array(targets)   
sequence , targets = generate_data()        
#model oluşturma
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        RNN->linear(output)"""
        super(RNNModel, self).__init__()
        #giriş boyutu input_size
        #gizli katman boyutu hidden_size
        #num layers :rnn katman sayısı
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)#fully conected layer :output

    def forward(self, x):
        out, _ = self.rnn(x)# rnnye girdiyi ver çıktıyı al 
        out = self.fc(out[:, -1, :])#son zaman adımındaki çıktıyı al ve fully connected layera bağla
        return out
model=RNNModel(input_size=1, hidden_size=50, num_layers=1, output_size=1)    
#model eğitme
#hyperparametreler
seq_length=50
input_size=1
hidden_size=16
output_size=1
num_layers=1
epochs=20
batch_size=32
learning_rate=0.001
#veri hazırlama
X,y=generate_data(seq_length)
X=torch.tensor(X,dtype=torch.float32).unsqueeze(-1)#pytorch tensörüne çevirme ve sonuna 1 boyut ekleme 
y=torch.tensor(y,dtype=torch.float32).unsqueeze(-1)#pytorch tensörüne çevirme ve sonuna 1 boyut ekleme
dataset =torch.utils.data.TensorDataset(X,y)#pytorch veri seti oluşturma
dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)#veri yükleyici oluşturma
#modeli tanımlama
model=RNNModel(input_size,hidden_size,num_layers,output_size)
criterion=nn.MSELoss()#kayıp fonksiyonu
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)#optimizasyon algoritması
for i in range(epochs):
    for batch_x,batch_y in dataloader:
        optimizer.zero_grad()
        pred_y=model(batch_x)#batch_x=inputs ve pred_y tahmini değer #batch_y ise targets 
        loss=criterion(pred_y,batch_y)#kayıp hesaplama
        loss.backward()
        optimizer.step()
    print(f'Epoch {i+1}/{epochs}, Loss: {loss.item():.4f}') 
#model test etme ve değerlendirme
#test verisi oluşturma
X_test=np.linspace(100,110,seq_length).redshape(-1,1) #reshape (-1,1) zorunlu ilk test verisi
y_test=np.sin(X_test)
X_test2=np.linspace(110,120,seq_length).reshape(-1,1)#ikinci test verisi
y_test2=np.sin(X_test2)
X_test=torch.tensor(X_test,dtype=torch.float32).unsqueeze(-1)#test verisini tensöre çevirme
X_test2=torch.tensor(X_test2,dtype=torch.float32).unsqueeze(-1)#ikinci test verisini tensöre çevirme
#önce tensöre çevirdik yoksa pytorch nezlinde veri kabul edilemez
model.eval()#değerlendirme moduna geçme 
prediction1 =model(X_test).detach().numpy()#ilk test verisi ile tahmin yapma
prediction2 =model(X_test2).detach().numpy()#ikinci test verisi ile tahmin yapma
#sonuçları görselleştirme
plt.figure()
plt.plot(np.linespace(0,100,len(y)),y,marker='o',label="Training Data")#eğitim verisi
plt.plot(X_test.numpy().flatten(),marker='o',label="Test 1")
plt.plot(X_test2.numpy().flatten(),marker='o',label="Test 2")
plt.plot(np.arange(seq_length,seq_length+1),prediction1.flatten(),"ro",label="Prediction 1")
plt.plot(np.arange(seq_length,seq_length+1),prediction2.flatten(),"go",label="Prediction 2")
plt.legend()
plt.show()
