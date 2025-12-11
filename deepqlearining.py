#deeplearning cartpole
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from IPython import display
#cartpole ortamını oluşturma
env=gym.make('CartPole-v1',render_mode='human')
device="cpu"
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
#replay memory oluşturma
class ReplayMemory:
    def __init__(self,capacity):#maximum size of the memory haline boşaltma 
        self.memory=deque([],maxlen=capacity)
    def push(self,*args):#save a transition
        self.memory.append(Transition(*args))
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):#memory uzunluğu
        return len(self.memory)
#dql modeli oluşturma
class DQN(nn.Module):
    def __init__(self,state_size,action_size):
        super(DQN,self).__init__()
        self.fc1=nn.Linear(state_size,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,action_size)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
#hiperparametrelerin ve yardımcı fonksiyonların tanımlanması
batch_size=128
gamma=0.999#uzun vadeli düşünme için indirim faktörü
eps_start=0.9#ilk başta zamanla rastgele davranma eğilimi
eps_end=0.05#en sonda öğrenerek hareket etme eğilimi
eps_decay=200#oranın düşmesi hızı
tau=0.005#target network güncelleme oranı
lr=1e-4#ağırlık güncelleme için öğrenme hızı
n_actions=env.action_space.n#ajanın alabileceği action sayısı
state,info=env.reset()#her yeni durumda ortamı sıfırla
n_observations=len(state)#gözlem sayısı
policy_net=DQN(n_observations,n_actions).to(device)#her adımda eğitilen ağ
target_net=DQN(n_observations,n_actions).to(device)#diğer ağ daha az sıklıkla güncelenen kopy ağ ardışık karelerin birbirine benzmesini engeller 
target_net.load_state_dict(policy_net.state_dict())#hedef ağ ile politik ağın ağırlıklarını eşitleyerek kopyla
optimizer=optim.AdamW(policy_net.parameters(),lr=lr)
memoru=ReplayMemory(10000)
steps_done=0#ajanın yaptığı toplam adım sayısı
def select_action(state):
    global steps_done
    sample=random.random()
    eps_threshold=eps_end+(eps_start-eps_end)*math.exp(-1.*steps_done/eps_decay)#ajan öğrendikçe rastgele hareket etme olasılığı azalır en iyi öğrenme artar
    #eps_threshold :rastgele keşifmi yoksa öğrenmeye dayalı action mı seçileceğini belirler
    steps_done+=1#toplam adım sayısını artır
    #eğer sample eps_threshold dan büyükse ajan neural network ile action seçer ,değilse rastgele action seçilir
    if sample>eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]],device=device,dtype=torch.long)
episode_durations=[]#her bir eğitimin ne kadar sürdüğünü ve kaç adım aldığını tutar
def plot_durations(show_result=False):
    plt.figure(1)#grafik alaını başlatır
    durations_t=torch.tensor(episode_durations,dtype=torch.float)#matematiksel uygunluk için tensöre çevir
    if show_result:#eğitim tamamlandıysa başlık ayarla yoksa grafiği temizleyip devam et eğitime
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    #eksen etiketi ayarlama     
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t)>=100:#en az 100 episode tamamlandıysa 100 episode ortalamasını çiz
        means=durations_t.unfold(0,100,1).mean(1).view(-1)
        means=torch.cat((torch.zeros(99),means))#ilk 99 bölüm için ortalama hesaplanmadığından ilk 99 bölümü sil 
        plt.plot(means.numpy())#ortalama çizgiyi çiz
    plt.pause(0.001)#grafiği güncelle
    display.clear_output(wait=True)#görüntüleme
    display.display(plt.gcf())
def optimize_model():
    if len(memoru)<batch_size:#yeterli veri yoksa eğitim yapma 
        return
    transitions=memoru.sample(batch_size)#hafızadan rastgele örneklem al
    batch=Transition(*zip(*transitions))#hafızadan gelen veri listesini batch formatına dönüştür
    non_final_mask=torch.tensor(tuple(map(lambda s:s is not None,batch.next_state)),device=device,dtype=torch.bool)#hangi adımlarla oyunun devam ettiğini hangisinde etmediğini belirler
    non_final_next_states=torch.cat([s for s in batch.next_state if s is not None])#sadece oyunun devam ettiği adımları birleştirerek tensör yapar
    #tensör birleştirme
    state_batch=torch.cat(batch.state)
    action_batch=torch.cat(batch.action)
    reward_batch=torch.cat(batch.reward)
    #tensör birleştirme 
    state_action_values=policy_net(state_batch).gather(1,action_batch)#ajan o anki duruma göre seçtiği actionların Q değerlerini al
    next_state_values=torch.zeros(batch_size,device=device)#gelecek tüm değerleri sıfır olarak başlatma
    with torch.no_grad():
        next_state_values[non_final_mask]=target_net(non_final_next_states).max(1)[0]#oyunun devam ettiği adımlar için hedef ağdan gelecek en iyi Q değerlerini al
    expected_state_action_values=(next_state_values*gamma)+reward_batch#beklenen Q değerlerini hesapla ballman denklemi
    criterion=nn.SmoothL1Loss()
    loss=criterion(state_action_values,expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)#kararlılığı sağlamak için gradyanları kırp
    optimizer.step()
#model eğitimi ve sonuçların değerlendirmesi
num_episodes=500
for i_episode in range(num_episodes):
    state,info=env.reset()
    state=torch.tensor([state],device=device,dtype=torch.float32)
    for t in count():
        action=select_action(state)
        next_state,reward,terminated,truncated,info=env.step(action.item())
        reward=torch.tensor([reward],device=device,dtype=torch.float32)
        done=terminated or truncated
        if terminated:
            next_state=None        
        else:
            next_state=torch.tensor([next_state],device=device,dtype=torch.float32,device=device).unsqueeze(0)
        memoru.push(state,action,next_state,reward)#geçişi replay memory e ekle
        state=next_state
        optimize_model()#modeli optimize et
        target_net_state_dict=target_net.state_dict()
        policy_net_state_dict=policy_net.state_dict()
    #hedef ağı yumuşak güncelleme
    for key in policy_net_state_dict:
        target_net_state_dict[key]=tau*policy_net_state_dict[key]*tau+(1-tau)*target_net_state_dict[key]*(1-tau)
        if done:
            episode_durations.append(t+1)
            plot_durations()
            break   
print('Complete')       
plot_durations(show_result=True)
plt.ioff()
plt.show()