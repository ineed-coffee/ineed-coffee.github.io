---
title: pytorch basic5 _ modeling with 3 API
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

    Thu Apr 22 07:05:12 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.67       Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   34C    P8    27W / 149W |      0MiB / 11441MiB |      0%      Default |
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
    
 
### Basic 5. 3-ways to build model from nn.Module
- Sequential API (easy, high-level)
- Functional API (general way)
- Subclassing API (pytorch standard) 

**In [1]:**

{% highlight python %}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
{% endhighlight %}
 
1. __Sequential API 방식. 간단한 모델을 설계하기에 최적__ 

**In [None]:**

{% highlight python %}
# In: 20-dim -> hidden1: 100-unit -> hidden2: 100-unit -> Out: 10-dim 구조의 2계층 다중 분류 신경망 설계하기 

model1 = nn.Sequential(
    nn.Linear(20,30),
    nn.ReLU(),
    nn.Linear(30,10),
    nn.Softmax()
)

print("구성확인(default layer name): ")
print(list(model1.modules()))

# 각 layer에 이름을 부여하고자 한다면 다음과 같이 작성할 수 있다.
model2 = nn.Sequential()
model2.add_module("hidden1",nn.Linear(20,30))
model2.add_module("activation1",nn.ReLU())
model2.add_module("hidden2",nn.Linear(30,10))
model2.add_module("activation2",nn.Softmax())

print()
print("="*65)
print()

print("구성확인(defined layer name): ")
print(list(model2.modules()))
print("접근 또한 가능, model2.hidden1")
print(model2.hidden1)
{% endhighlight %}

    구성확인(default layer name): 
    [Sequential(
      (0): Linear(in_features=20, out_features=30, bias=True)
      (1): ReLU()
      (2): Linear(in_features=30, out_features=10, bias=True)
      (3): Softmax(dim=None)
    ), Linear(in_features=20, out_features=30, bias=True), ReLU(), Linear(in_features=30, out_features=10, bias=True), Softmax(dim=None)]
    
    =================================================================
    
    구성확인(defined layer name): 
    [Sequential(
      (hidden1): Linear(in_features=20, out_features=30, bias=True)
      (activation1): ReLU()
      (hidden2): Linear(in_features=30, out_features=10, bias=True)
      (activation2): Softmax(dim=None)
    ), Linear(in_features=20, out_features=30, bias=True), ReLU(), Linear(in_features=30, out_features=10, bias=True), Softmax(dim=None)]
    접근 또한 가능, model2.hidden1
    Linear(in_features=20, out_features=30, bias=True)
    
 
2. __Functional API 방식. Sequential 방식으로는 설계가 까다로운 경우 활용. 가장 일반적인 방법__
- keras에서 이런 functional api 방식을 지원한다. Pytorch에서는 다음 방법인 subclassing api를 많이 씀 

**In [None]:**

{% highlight python %}
from keras.layers import Input, Dense, concatenate
from keras.models import Model
 
# 두 종류의 입력이 있는 모델을 가정
In_a = Input(shape=(28,))
In_b = Input(shape=(64,))
 
# In_a 에 대한 모듈 정의
module1 = Dense(16, activation="relu")(In_a)
module1 = Dense(8, activation="relu")(module1)
module1 = Model(inputs=In_a, outputs=module1)
 
# In_b 에 대한 모듈 정의
module2 = Dense(64, activation="relu")(In_b)
module2 = Dense(32, activation="relu")(module2)
module2 = Dense(8, activation="relu")(module2)
module2 = Model(inputs=In_b, outputs=module2)
 
# 두 모듈의 출력을 연결하는 층 생성(concatenate)
concat_layer = concatenate([module1.output, module2.output])
 
# 최종 단 정의
top_layer = Dense(2, activation="relu")(concat_layer)
final = Dense(1, activation="linear")(top_layer)
 
# 최종 모델 정의
model = Model(inputs=[module1.input, module2.input], outputs=final)

# 모델 아키텍쳐 확인
print(model.summary())
{% endhighlight %}

    Model: "model_4"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_4 (InputLayer)            [(None, 64)]         0                                            
    __________________________________________________________________________________________________
    input_3 (InputLayer)            [(None, 28)]         0                                            
    __________________________________________________________________________________________________
    dense_9 (Dense)                 (None, 64)           4160        input_4[0][0]                    
    __________________________________________________________________________________________________
    dense_7 (Dense)                 (None, 16)           464         input_3[0][0]                    
    __________________________________________________________________________________________________
    dense_10 (Dense)                (None, 32)           2080        dense_9[0][0]                    
    __________________________________________________________________________________________________
    dense_8 (Dense)                 (None, 8)            136         dense_7[0][0]                    
    __________________________________________________________________________________________________
    dense_11 (Dense)                (None, 8)            264         dense_10[0][0]                   
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 16)           0           dense_8[0][0]                    
                                                                     dense_11[0][0]                   
    __________________________________________________________________________________________________
    dense_12 (Dense)                (None, 2)            34          concatenate_1[0][0]              
    __________________________________________________________________________________________________
    dense_13 (Dense)                (None, 1)            3           dense_12[0][0]                   
    ==================================================================================================
    Total params: 7,141
    Trainable params: 7,141
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    
 
3. __Subclassing API 방식. Pytorch 에서는 가장 standard 방식이며 객체형 프로그래밍 방식으로 구현한다.__
 

**In [21]:**

{% highlight python %}
# 직렬로 설계가 가능한 모듈형태의 layer의 경우 sequential 방식으로 작성하며 이들을 연결 시 Subclassing API 방식으로 작서하며 forward 메소드에 순전파 과정을 작성해주면 된다.
class my_CNN(nn.Module):

  def __init__(self,model_in,model_out):
    super(my_CNN,self).__init__()

    self.conv_idx=1
    self.conv1=self._make_conv_module(model_in,32,3,1,1)
    self.conv2=self._make_conv_module(32,64,3,1,1)
    self.fc1  =nn.Linear(7*7*64,model_out)
    nn.init.xavier_uniform_(self.fc1.weight)
    self.classifier=nn.Softmax(dim=1)

  def _make_conv_module(self,ch_in,ch_out,filter_size,stride,pad):
    conv=nn.Sequential()
    conv.add_module(f"conv_{self.conv_idx}",nn.Conv2d(ch_in,ch_out,kernel_size=filter_size,stride=stride,padding=pad))
    conv.add_module(f"relu_{self.conv_idx}",nn.ReLU())
    conv.add_module(f"maxpool_{self.conv_idx}",nn.MaxPool2d(2))
    return conv

  def forward(self,x):

    yhat=self.conv1(x)
    yhat=self.conv2(yhat)
    yhat=yhat.view(yhat.shape[0],-1)
    yhat=self.fc1(yhat)
    yhat=self.classifier(yhat)
    return yhat

model=my_CNN(3,10)
print(model)
{% endhighlight %}

    my_CNN(
      (conv1): Sequential(
        (conv_1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu_1): ReLU()
        (maxpool_1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv2): Sequential(
        (conv_1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu_1): ReLU()
        (maxpool_1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (fc1): Linear(in_features=3136, out_features=10, bias=True)
      (classifier): Softmax(dim=1)
    )
    
