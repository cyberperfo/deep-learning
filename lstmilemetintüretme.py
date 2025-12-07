#lstm ile metin türetme
#kütüphane
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from itertools import product
#veri seti yükleme ve önişleme preprocessing
# urun yorumlari
text = """Bu ürün beklentimi fazlasıyla karşıladı.
Malzeme kalitesi gerçekten çok iyi.
Kargo hızlı ve sorunsuz bir şekilde elime ulaştı.
Fiyatına göre performansı harika.
Kesinlikle tavsiye ederim ve öneririm!"""
#veri önişleme
#noktamalam işaretlerinden kurtul ,küçük harflerin dönüşümü kelimeleri böl 
words = text.replace('.', '').replace(',', '').replace('!', '').lower().split()
#kelime frekansını hesapla
word_counts = Counter(words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)#kelime frekansını büyükten küçüğe sırala
word_to_ix={word: i for i, word in enumerate(vocab)}#kelimeleri indekslere dönüştür 
ix_to_word={i: word for word, i in word_to_ix.items()}#indeksleri kelimelere dönüştür
#eğitim verisi hazırlama
data=[(words[i], words[i+1]) for i in range(len(words)-1)]#her kelime ve bir sonraki kelime çifti
#lstm modeli oluşturma
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        """
        input ->embedding ->lstm ->fc->output
        """
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x, hidden)
        out = self.fc(lstm_out.view(1,-1))  # Sadece son zaman adımını kullan
        return out,
model=LSTMModel(vocab_size=len(vocab), embedding_dim=8, hidden_dim=32)
#hiperparamtre tuning 
def prepare_sequence(seq, word_to_ix):
    return torch.tensor([word_to_ix[w] for w in seq], dtype=torch.long)#liste şeklinde kelime indekslerini tensöre çevir
#hyparametreler tuning kombinasyonları belirle 
embedding_sizes=[8,16]#denenecek embedding boyutları
hidden_sizes=[16,32]#denenecek gizli katman boyutları
learning_rates=[0.01,0.005]#denenecek öğrenme oranları
best_loss=float('inf')#en düşük kayıp değeri hesaplamak için kullanılacak değişken (sonsuz değer şuan)
best_params= {}#en iyi parametreleri saklamak için boş sözlük
print("Hiperparametre Tuning Başlıyor...")
for emb_size, hidden_size, lr in product(embedding_sizes, hidden_sizes, learning_rates):#grid search product:kombinasyon oluşturma fonskiyonu
    print(f'Deniyor: Embedding Dim={emb_size}, Hidden Dim={hidden_size}, Learning Rate={lr}')
    #model tanımla
    model=LSTMModel(vocab_size=len(vocab), embedding_dim=emb_size, hidden_dim=hidden_size)
    loss_function =nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)
    epochs=50
    total_loss=0
    #model eğitme
    for epoch in range(epochs):
        hidden = None
        epoch_loss=0#epoch kaybı başlangıçta sıfır
        for word, next_word in data:#input=word ve target=next word kelimeler
            model.zero_grad()
            input_seq=prepare_sequence([word],word_to_ix)#girdiyi tensöre çevir
            target_seq=prepare_sequence([next_word],word_to_ix)#hedefi tensöre çevir
            prediction=model(input_seq,hidden)#output
            loss=loss_function(prediction,target_seq)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(data):.4f}')
        total_loss = epoch_loss    
    #en iyi modeli kaydet 
    if total_loss<best_loss:
        best_loss=total_loss
        best_params={'embedding_dim':emb_size,'hidden_dim':hidden_size,'learning_rate':lr}
print("En İyi Hiperparametreler:", best_params)        