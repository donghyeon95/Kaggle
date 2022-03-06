#!/usr/bin/env python
# coding: utf-8

# # N-Beat 

# !! 질문 사항 !!
# - N-beat stack 구현이 Trend, Trend, Trend...., seasonality, seasonality,seasonality....., noise, noise, ... 순이던데
# - Trend, seasonality, noise, Trend, seasonality, noise, .... 순서이면 안되나요?
# 
# - 예상 이유 -> seasonality, noise를 계산 하면서 Trend를 잃어버리기 때문에??
# - 

# ### 관련 정리  
# https://joungheekim.github.io/2020/09/09/paper-review/

# - paper 
# https://arxiv.org/abs/1905.10437

# https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/ar.html

# https://github.com/philipperemy/n-beats

# In[1]:


from datetime import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

wandb.login()


# In[2]:


train_df = pd.read_csv("/USER/traffic/data/train.csv")
valid_df = pd.read_csv("/USER/traffic/data/validate.csv")
test_df = pd.read_csv("/USER/traffic/data/test.csv")


# In[12]:


# train_df['datetime'] = pd.to_datetime(train_df['날짜'].astype('str')) + train_df['시간'].astype('timedelta64[h]')`


# In[3]:


train_df['10'].values


# In[4]:


loads = ["10", "100", "101", "120", "121", "140", "150", "160", "200", "201", "251", "270", "300", "301", "351", "352", "370", "400", "450", "500", "550", "600", "650", "652", "1000",
"1020", "1040", "1100", "1200", "1510", "2510", "3000", "4510", "5510", "6000"]


# ### Data Loader

# In[5]:


class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, data, backcast_length, forecast_length, scaler=None):
        self.data = data.values
        self.backcast_length = backcast_length #어떤 주기로 예측을 수행할 지 
        self.forecast_length = forecast_length #얼마나 에측할 것인지?        

    def __len__(self): 
        # 전체 중에서 윈도우를 밀고 가면서 생성한다고 보면 된다. 빼주는 이유는 마지막 윈도우의 길이를 확보해주기 위함이다. 
        return len(self.data) - self.backcast_length - self.forecast_length + 1

    def __getitem__(self, idx):
        x_end = idx + self.backcast_length
        y_end = x_end + self.forecast_length

        x = self.data[idx:x_end]
        y = self.data[x_end:y_end]
        
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
        return x, y 


# In[6]:


batch_size = 4
num_workers = 2
pin_memory = True

backcast_length = 24 * 7 # 1주일 data를 보고 예측할 것이다. 
forecast_length = 24 * 7 # 1주일치를 예측할 것이다. 
theta_dim=(2, 8, 3)
n_trend=3
n_seasonality=3
n_residual=3
hidden_dim=256 #논문에서는 256인데 학습을 빠르게 하기위해 64로 설정한 듯. 
learning_rate = 1e-4
num_epoch = 100

criterion = nn.MSELoss()


# In[38]:


config = {
    'batch_size': batch_size,
    'backcast_length' : backcast_length,
    'forecast_length' : forecast_length,
    'theta_dim': theta_dim,
    'n_trend' : n_trend,
    "n_seasonality": n_seasonality,
    "n_residual": n_residual,
    "hidden_dim" : hidden_dim,
    "learning_rate" : learning_rate,
    "num_epoch": num_epoch,
    "criterion" : criterion,
}

wandb.init(config=config, project="TRAFFIC", entity="team6", name="N-beat-02")


# In[22]:


train_loaders = []
val_loaders = []
test_loaders = []

for load in loads: 
    train_dataset = TimeseriesDataset(train_df[load], backcast_length, forecast_length) #현재 데이터의 평균과 분산으로 정규화
    val_dataset = TimeseriesDataset(valid_df[load], backcast_length, forecast_length) #치팅을 하면 안되기에 train의 평균과 분산으로 
    test_dataset = TimeseriesDataset(test_df[load], backcast_length, forecast_length) #치팅을 하면 안되기에 train의 평균과 분산으로

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, # train data의 순서를 외우지 못하게 하기위해 shffle을 해준다. 
                                                   num_workers=num_workers, pin_memory=pin_memory)
    train_loaders.append(train_dataloader)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                                   num_workers=num_workers, pin_memory=pin_memory)
    val_loaders.append(val_dataloader)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                                   num_workers=num_workers, pin_memory=pin_memory)
    test_loaders.append(test_loaders)
    
    


# ### N-beat Model

# In[25]:


class NBeatsNet(nn.Module):
    def __init__(self,
                 forecast_length=168, #미래 24개의 데이터를 보겠다. 
                 backcast_length=168, #과거 96개의 데이터를 보겠다.
                 theta_dim=(2, 8, 3), # 2차식으로 trend를 보겠다. 
                 n_trend=3,
                 n_seasonality=3,
                 n_residual=3,
                 hidden_dim=256,
                 feature_dim=1):
        super(NBeatsNet, self).__init__()        
            
        self.trend_stack = []
        self.seasonality_stack = []
        self.residual_stack = [] #bs =[] 이건했는데 fs=[] 이거 안해줌
        # We can simply call pairplot on our DataFrame for an automatic visual analysis 

        for i in range(n_trend):
            self.trend_stack.append(Block(backcast_length, forecast_length, hidden_dim, theta_dim[0], mode='trend', feature_dim=feature_dim))    

        for i in range(n_seasonality):
            self.seasonality_stack.append(Block(backcast_length, forecast_length, hidden_dim, theta_dim[1], mode='seasonality', feature_dim=feature_dim))
        
        for i in range(n_residual):
            self.residual_stack.append(Block(backcast_length, forecast_length, hidden_dim, theta_dim[2], mode='residual', feature_dim=feature_dim))
        # We can simply call pairplot on our DataFrame for an automatic visual analysis 

        
        self.trend_stack = nn.ModuleList(self.trend_stack)
        self.seasonality_stack = nn.ModuleList(self.seasonality_stack)
        self.residual_stack = nn.ModuleList(self.residual_stack)

    def forward(self, backcast):# We can simply call pairplot on our DataFrame for an automatic visual analysis 
# We can simply call pairplot on our DataFrame for an automatic visual analysis 

        backcast_stack = []
        forecast_stack = []

        for layer in self.trend_stack:
            b, f = layer(backcast)
            backcast_stack.append(b)
            forecast_stack.append(f)
            backcast = backcast - b

        for layer in self.seasonality_stack:
            b, f = layer(backcast)
            backcast_stack.append(b)
            forecast_stack.append(f)
            backcast = backcast - b


        for layer in self.residual_stack:
            b, f = layer(backcast)
            backcast_stack.append(b)
            forecast_stack.append(f)
            backcast = backcast - b

        backcast = torch.stack(backcast_stack, 0)
        forecast = torch.stack(forecast_stack, 0)
        return backcast, forecast


# ours
class Block(nn.Module):
    def __init__(self, backcast_length, forecast_length, hidden_dim, theta_dim, mode, feature_dim):
        # mode: season, trend, residual 
        # MLP : FC stack
        # 
        super(Block, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.theta_dim = theta_dim
        self.mode = mode

        self.MLP = nn.Sequential(
                nn.Linear(backcast_length * feature_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) # univariate??multivariate??

        if mode == 'residual':
            # residual  
            self.theta_b = nn.Linear(hidden_dim, theta_dim * feature_dim)             
            self.theta_f = nn.Linear(hidden_dim, theta_dim * feature_dim)
            self.linear_b = nn.Linear(theta_dim, backcast_length)
            self.linear_f = nn.Linear(theta_dim, forecast_length)
            #레지듀얼은 과거와 미래의 경향성이 너무 없기 때문에 따로 계산을 한다. 

        else:
            # trend, seasonality# We can simply call pairplot on our DataFrame for an automatic visual analysis 
            self.theta = nn.Linear(hidden_dim, theta_dim*feature_dim)
            #수업과는 달리 backcast랑 forecast를 같은 것을 쓰기 위해.


    def forward(self, x):
        shape = x.shape
        # x: (batch size, backcast length, feature_dim)
        #24로 나눈 것은 24 step을 가면 1주기가 되도록....콜코
        t_b = ((torch.arange(start=0, end=self.backcast_length, device=x.device, dtype=torch.float) - self.backcast_length) / 24 ) # (backcast_len,)
        t_f = ((torch.arange(start=0, end=self.forecast_length, device=x.device, dtype=torch.float)) / 24 ) # (forecast_len,)

        # x -> h
        x = self.MLP(x.reshape(x.shape[0], -1)) # (batch, backcast length * feature_dim) -> (batch, theta_dim)
# We can simply call pairplot on our DataFrame for an automatic visual analysis 

        # h -> theta -> backcast, forecast
        if self.mode == 'residual':# We can simply call pairplot on our DataFrame for an automatic visual analysis 

            # residual block
            theta_b = self.theta_b(x).reshape(shape[0], shape[2], self.theta_dim)
            theta_f = self.theta_f(x).reshape(shape[0], shape[2], self.theta_dim)
            b = self.linear_b(theta_b).permute(0,2,1) # (batch, backcast length, feature_dim)
            f = self.linear_f(theta_f).permute(0,2,1) # (batch, forecast length, feature_dim)
        else:
            theta_b = self.theta(x).reshape(shape[0], self.theta_dim, shape[2])
            theta_f = theta_b

            if self.mode == 'trend':
                b = self.get_trend(theta_b, t_b)
                f = self.get_trend(theta_f, t_f)
                
            elif self.mode == 'seasonality':
                b = self.get_seasonality(theta_b, t_b)
                f = self.get_seasonality(theta_f, t_f)
        return b, f# We can simply call pairplot on our DataFrame for an automatic visual analysis 


    def get_trend(self, theta, t):
        # theta dim = 0 -> 수평선 (y=a)
        # theta dim = 1 -> 직선 (y=bx)
        # theta dim = 2 -> 이차곡선 (y=cx^2)
        # ...
        T = torch.stack([t ** i for i in range(theta.shape[1])])  # (theta_dim ,sequence length)
        out = torch.einsum('btf,ts->bsf', theta, T)
        return out


    def get_seasonality(self, theta, t):
        # import pdb; pdb.set_trace() 
        s1 = torch.stack([torch.cos(2 * np.pi * (i+1) * t) for i in range(self.theta_dim//2)]).float()  # H/2-1
        s2 = torch.stack([torch.sin(2 * np.pi * (i+1) * t) for i in range(self.theta_dim//2)]).float()
        S = torch.cat([s1, s2])# We can simply call pairplot on our DataFrame for an automatic visual analysis 

        out = torch.einsum('btf,ts->bsf', theta, S)
        return out


# In[26]:


model = NBeatsNet(forecast_length, backcast_length, theta_dim, n_trend, n_seasonality, n_residual, hidden_dim)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ### Train

# In[ ]:





# In[ ]:


import time
train_losses = []
val_losses = []
best_epoch = 0
start_time = time.time()
best_model_state_dict = None
best_val_loss = 10000
models = []
for k in range(35):
    model = NBeatsNet(forecast_length, backcast_length, theta_dim, n_trend, n_seasonality, n_residual, hidden_dim)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epoch):
        train_loss_mean = 0
        val_loss_mean = 0

        # train
        model.train()
        for i, (inputs, labels) in enumerate(train_loaders[k]):


            inputs = inputs.cuda()
            labels = labels.cuda()


            # forecast = model(inputs)

            backcast, forecast = model(inputs.unsqueeze(2))
            # import pdb; pdb.set_trace()
            loss = criterion(backcast.sum(0), inputs.unsqueeze(2)) + criterion(forecast.sum(0), labels.unsqueeze(2))
            #summation을 해주는 부분이다. 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_mean += loss.item()

            if i % 100 == 0:# We can simply call pairplot on our DataFrame for an automatic visual analysis 

                print('load [{}] epoch [{}/{}] iter [{:03d}/{:03d}] loss [{:.4f}] elapsed time [{:.2f}min]'.format(k,epoch, num_epoch, i, len(train_dataloader), loss.item(), (time.time()-start_time)/60))

        train_loss_mean = train_loss_mean/len(train_dataloader)
        wandb.log({
            f'train_loss_mean_{k}' : train_loss_mean
        })

        train_losses.append(train_loss_mean)

        # validation
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loaders[k]):
                inputs = inputs.cuda()
                labels = labels.cuda()

                backcast, forecast = model(inputs.unsqueeze(2))
                loss = criterion(backcast.sum(0), inputs.unsqueeze(2)) + criterion(forecast.sum(0), labels.unsqueeze(2))
                val_loss_mean += loss.item()

            val_loss_mean = val_loss_mean/len(val_dataloader)
            wandb.log({
                f'val_loss_mean_{k}': val_loss_mean,
            })

            print('load [{}] epoch [{}/{}] train loss [{:.4f}] validation loss [{:.4f}] elapsed time [{:.2f} min]\n'.format(k, epoch, num_epoch, train_loss_mean, val_loss_mean, (time.time()-start_time)/60))

            if val_loss_mean < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss_mean
                best_model_state_dict = model.state_dict()
        val_losses.append(val_loss_mean)
    models.append(model)


# In[ ]:





# In[ ]:





# ### Predict

# In[36]:


predictions = []

with torch.no_grad():
    for i in range(35):
        for i, (inputs, labels) in enumerate(test_dataloader[i]):
            inputs = inputs.cuda()

            backcast, forecast = model(inputs.unsqueeze(2))
            predictions.append(forecast.sum(0).data.cuda())
    predictions.append(torch.cat(predictions, dim=0))


# print(predictions)
test_real = test_df.iloc[backcast_length:]  # 입력으로만 사용된 데이터는 제외하였습니다.
# plot_prediction_after_N_days(test_real, train_dataset.scaler.unscale(predictions), N=3)


# ### Submmit

# In[ ]:




