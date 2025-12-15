#sınıflandırma projesi positive ve negative commetnlerden oluşan bir veri seti
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
from collections import Counter
#veri tanımlanması ve ön işleme
positive_comments=["I love this product! It works great and exceeded my expectations."]
negative_comments=["This is the worst purchase I've ever made. Completely disappointed."]
def preprocess_text(text):
    text=text.lower()#küçük harfe çevir
    text=text.translate(str.maketrans("","",string.punctuation))#noktalama işaretlerini kaldır
    return text
#veri seti oluşturma
data=positive_comments+negative_comments
labels=[1]*len(positive_comments)+[0]*len(negative_comments)#1:pozitif,0:negatif 
data=[preprocess_text(comment) for comment in data]#veri ön işleme    
#vocab oluşturma(kelime dağarcığı)
all_words=" ".join(data).split()#tüm kelimeleri birleştir ve böl
world_counts=Counter(all_words)#kelime frekansını hesapla
vocab={word:i for i,word in enumerate(world_counts)}#kelimeleri indekslere dönüştür
vocab["<PAD>"]=0#padding için özel token ekle
#veri setini tensöre dönüştürme
max_len=10#maksimum yorum uzunluğu
def encode_comment(comment,vocab,max_len):
    tokens=comment.split()#yorumları kelimelere böl
    indices=[vocab.get(token,0) for token in tokens]#kelimeleri indekslere dönüştür
    indices=indices[:max_len]#maksimum uzunluğa kes
    indices+=[vocab["<PAD>"]]*(max_len-len(indices))#eksikse padding ekle
    return torch.tensor(indices)
X=torch.stack([encode_comment(comment,vocab,max_len) for comment in data])#tüm yorumları tensöre dönüştür
y=torch.tensor(labels)#etiketleri tensöre dönüştür
#train ve test verisi ayırma
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#transformer modelinin oluşturulması
class TransformerClass(nn.Module):
    def __init__(self,vocab_size,embed_size,num_heads,hidden_dim,num_layers):
        super(TransformerClass,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.positional_encoding=nn.Parameter(torch.zeros(1,max_len,embed_size))#pozisyonel kodlama
        self.transformer=nn.Transformer(d_model=embed_size,nhead=num_heads,num_encoder_layers=num_layers,dim_feedforward=hidden_dim)
        self.fc=nn.Linear(embed_size,num_classes)
        self.out=nn.Linear(hidden_dim,2)#2 sınıf için çıkış katmanı
    def forward(self,x):
        embeded=self.embedding(x)+self.positional_encoding#gömme ve pozisyonel kodlama ekleme
        output=self.transformer(embeded,embeded)#transformer katmanı
        output=output.view(output.size(0),-1)#tensörü düzleştir
        output=self.fc(output)#tam bağlantılı katman   
        output=self.sigmoid(output)#çıkış katmanı
        return output
#train
vocab_size=len(vocab)
embedding_dim=32
num_heads=2
hidden_dim=64
num_layers=2
num_classes=1
model=TransformerClass(vocab_size,embedding_dim,num_heads,hidden_dim,num_layers)
criterion=nn.BCELoss()#binary cross entropy loss
optimizer=optim.Adam(model.parameters(),lr=0.001)
num_epochs=20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs=model(X_train)
    loss=criterion(outputs.squeeze(),y_train.float())
    loss.backward()
    optimizer.step()
    if (epoch+1)%5==0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#test
model.eval()
with torch.no_grad():
    y_pred=model(X_test.long()).squeeze()
    y_pred=(y_pred>=0.5).float()
    y_pred_training= model(X_train.long()).squeeze()
    y_pred_training=(y_pred_training>=0.5).float()
accuracy=accuracy_score(y_test,y_pred)
print(f'Test Accuracy: {accuracy*100:.2f}%')
accuracy_train=accuracy_score(y_train,y_pred_training)
print(f'Train Accuracy: {accuracy_train*100:.2f}%')    