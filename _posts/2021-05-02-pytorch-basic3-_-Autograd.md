---
title: pytorch basic3 _ Autograd
author: INEED COFFEE
date: 2021-05-02 14:00:00 +0800
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

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    
 
### 연결된 그래픽 카드와 CUDA 버전 확인하기 

**In [None]:**

{% highlight python %}
!nvidia-smi
{% endhighlight %}

    Mon Apr 19 14:37:53 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.67       Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   38C    P0    32W / 250W |    899MiB / 16280MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
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
    
 
# Basic 3. Autograd

__Torch의 기본 자료형인 Tensor의 경우 연산이 이루어질때 그 기록 또한 누적되어 자동미분이 가능한 구조로 계산된다.__ 

**In [None]:**

{% highlight python %}
# Tensor 자료형으로 사칙연산 해보기 (과정이 grad_fn 에 저장된다)
x = torch.tensor(1.5, requires_grad=True)
y = torch.tensor(3.5, requires_grad=True)
z=y**2+x

print('x:',x)
print('y:',y)
print('z:',z)
print()

# 자동미분 계산
z.backward()

# 각 변수에 할당될 기울기(Gradient)
print(f"x에 계산된 기울기: {x.grad}")
print(f"y에 계산된 기울기: {y.grad}")
{% endhighlight %}

    x: tensor(1.5000, requires_grad=True)
    y: tensor(3.5000, requires_grad=True)
    z: tensor(13.7500, grad_fn=<AddBackward0>)
    
    x에 계산된 기울기: 1.0
    y에 계산된 기울기: 7.0
    
 
__이때, Tensor의 .grad 속성은 처음 requires\_grad=True 로 설정한 Leaf Tensor에 대해서만 가능하다. 연산
중간의 텐서에서는 참조 불가__ 

**In [None]:**

{% highlight python %}
x = torch.tensor(1.5, requires_grad=True) # Leaf Tensor
y=x*2+3
z=y**2

print('x:',x)
print('y:',y)
print('z:',z)
print()

# 자동미분 계산
z.backward()

# 각 변수에 할당될 기울기(Gradient)
print(f"x에 계산된 기울기: {x.grad}")
print(f"y에 계산된 기울기: {y.grad}") # Not a Leaf Tensor => Error!
{% endhighlight %}

    x: tensor(1.5000, requires_grad=True)
    y: tensor(6., grad_fn=<AddBackward0>)
    z: tensor(36., grad_fn=<PowBackward0>)
    
    x에 계산된 기울기: 24.0
    y에 계산된 기울기: None
    

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:15: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the gradient for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations.
      from ipykernel import kernelapp as app
    
 
### 이러한 Autograd 특성으로부터 선형회귀 모델을 생성해보자 

**In [None]:**

{% highlight python %}
import torch
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

w1 = torch.zeros((C,1),device=device, dtype=dtype_,requires_grad=True) # 학습할 가중치와 편향
b1 = torch.zeros(1,device=device, dtype=dtype_,requires_grad=True)

# Set Optimizer
optimizer=optim.SGD([w1,b1],lr=learning_rate)

# Train
for epoch in range(epochs):
  
  y_hat=x.matmul(w1)+b1
  cost=torch.mean((y-y_hat)**2) # Forward-pass

  optimizer.zero_grad() # reset gradient to avoid accumulation
  cost.backward()       # compute gradient of each parameter
  optimizer.step()      # update each parameter

  if not epoch%1000:
    print(f'{epoch+1}/{epochs} : Cost={cost}')

print("="*65)
print(f'목표치: w1=(2,2,2) , b1=1')
print(f'학습결과: w1={[v.item() for v in w1]} , b1={b1.item()}')
{% endhighlight %}

    1/10000 : Cost=14.351768493652344
    1001/10000 : Cost=0.17560093104839325
    2001/10000 : Cost=0.002341908635571599
    3001/10000 : Cost=3.5410659620538354e-05
    4001/10000 : Cost=6.201415203577199e-07
    5001/10000 : Cost=1.2345105382394195e-08
    6001/10000 : Cost=1.9163828302026786e-09
    7001/10000 : Cost=1.9163828302026786e-09
    8001/10000 : Cost=1.9163828302026786e-09
    9001/10000 : Cost=1.9163828302026786e-09
    =================================================================
    목표치: w1=(2,2,2) , b1=1
    학습결과: w1=[2.0, 1.9999690055847168, 1.999973177909851] , b1=0.9999855756759644
    
 
### 어느정도 학습이 됐다면 테스트도 해보자 

**In [None]:**

{% highlight python %}
test_in=torch.tensor([[1.,2.,4.]],device=device, dtype=dtype_) # 예상 정답은 2*(1+2+4)+1 = 15

with torch.no_grad(): # gradient를 추적하지 않음을 의미
  y_hat=test_in.matmul(w1)+b1
  print(f"예상 정답: {[(2*sum(i)+1).item() for i in test_in]}")
  print(f"회귀식 예측 값: {y_hat}")
{% endhighlight %}

    예상 정답: [15.0]
    회귀식 예측 값: tensor([[14.9998]], device='cuda:0')
    
