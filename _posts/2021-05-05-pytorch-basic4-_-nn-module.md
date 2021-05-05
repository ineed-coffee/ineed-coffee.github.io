---
title: pytorch basic4 _ nn module
author: INEED COFFEE
date: 2021-05-05 14:00:00 +0800
categories: [Pytorch101]
tags: [colab]
toc: true
comments: true
typora-root-url: ../
---
**In [None]:**

{% highlight python %}
from google.colab import drive
drive.mount('/content/drive')
{% endhighlight %}

    Mounted at /content/drive
    
 
### 연결된 그래픽 카드와 CUDA 버전 확인하기 

**In [None]:**

{% highlight python %}
!nvidia-smi
{% endhighlight %}

    Sun Apr 11 07:45:10 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.67       Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   68C    P8    11W /  70W |      0MiB / 15109MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    

**In [None]:**

{% highlight python %}
!nvcc -V
{% endhighlight %}

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2020 NVIDIA Corporation
    Built on Wed_Jul_22_19:09:09_PDT_2020
    Cuda compilation tools, release 11.0, V11.0.221
    Build cuda_11.0_bu.TC445_37.28845127_0
    
 
### torch, torchvision, torchtext, 버전 확인 

**In [None]:**

{% highlight python %}
import torch
import torchvision
import torchtext

print(f'torch version: {torch.__version__}')
print(f'torchvision version: {torchvision.__version__}')
print(f'torchtext version: {torchtext.__version__}')

{% endhighlight %}

    torch version: 1.8.1+cu101
    torchvision version: 0.9.1+cu101
    torchtext version: 0.9.1
    
 
# Basic 4. nn module

__modeling을 위하여 Tensorflow의 keras 프레임워크와 같이 pytorch에는 nn module이라는 high-level
API 존재__ 

**In [45]:**

{% highlight python %}
# basic 3. 에서 작성했던 선형 회귀 모델을 nn module로 작성해보기 (+ nn.Functional)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 학습 device 설정
torch.manual_seed=1120 # 시드 고정(CPU)
if torch.cuda.is_available():
  device_='cuda'
  torch.cuda.manual_seed_all(1120) # 시드 고정(GPU)
else:
  device_='cpu'
device = torch.device(device_)

# Set Conditions & Hyper-parameters
dtype_=torch.float
R,C = 500,3 # 500개 데이터 샘플 , 각 데이터는 3-dim
epochs=10000
learning_rate=1e-3

# Generate data, initialize weights & bias
x = torch.randn((R,C),device=device, dtype=dtype_)
y=torch.tensor([(2*sum(i)+1).item() for i in x],device=device, dtype=dtype_).view(-1,1)  # 목표식 y=2*x+1

# Generate model
torch.random.manual_seed(1120) # 파라미터 초기화 시드 고정은 별도로 해줘야함. 위의 시드 고정은 다른 무작위 수 생성에만 영향을 줌.
model=nn.Linear(C,1).to(device_)
#model.to_device()
print('초기화된 가중치',list(model.parameters()))
print("="*65)

# Set Optimizer
optimizer=optim.SGD(model.parameters(),lr=learning_rate)

# Train
for epoch in range(epochs):
  
  y_hat=model(x)        # Forward-pass
  cost=F.mse_loss(y,y_hat)

  optimizer.zero_grad() # reset gradient to avoid accumulation
  cost.backward()       # compute gradient of each parameter
  optimizer.step()      # update each parameter

  if not epoch%1000:
    print(f'{epoch+1}/{epochs} : Cost={cost}')

print("="*65)
print(f'목표치: w1=(2,2,2) , b1=1')
print(f'학습결과: w1={[v.item() for v in model.weight[0]]} , b1={model.bias[0].item()}')

{% endhighlight %}

    초기화된 가중치 [Parameter containing:
    tensor([[ 0.0367,  0.0082, -0.4146]], device='cuda:0', requires_grad=True), Parameter containing:
    tensor([0.2584], device='cuda:0', requires_grad=True)]
    =================================================================
    1/10000 : Cost=15.626565933227539
    1001/10000 : Cost=0.19387540221214294
    2001/10000 : Cost=0.0025998216588050127
    3001/10000 : Cost=3.8721154851373285e-05
    4001/10000 : Cost=6.508944920824433e-07
    5001/10000 : Cost=1.2670970050976393e-08
    6001/10000 : Cost=1.9142774032587795e-09
    7001/10000 : Cost=1.9142774032587795e-09
    8001/10000 : Cost=1.9142774032587795e-09
    9001/10000 : Cost=1.9142774032587795e-09
    =================================================================
    목표치: w1=(2,2,2) , b1=1
    학습결과: w1=[2.0, 1.9999690055847168, 1.9999727010726929] , b1=0.9999865889549255
    
