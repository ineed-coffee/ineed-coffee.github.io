---
title: pytorch basic1 _ Tensor
author: INEED COFFEE
date: 2021-04-25 14:00:00 +0800
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

    Wed Apr  7 11:23:37 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.67       Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   69C    P8    34W / 149W |      0MiB / 11441MiB |      0%      Default |
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
    
 
# Basic 1. Tensor
Tensor vs. Ndarray 

**In [None]:**

{% highlight python %}
import torch
import numpy as np
{% endhighlight %}

**In [None]:**

{% highlight python %}
# make random 2D-array data
import random as r
pick=lambda :r.randint(5,10)
R,C=pick(),pick()
base = [[r.randint(0,100) for __ in range(C)] for _ in range(R)]
                                          
print('random base data')
for row in base:
  print(*row)
{% endhighlight %}

    random base data
    46 13 8 58 43 78 52 97 29 10
    15 51 70 14 3 16 88 29 71 58
    86 63 96 83 2 80 56 18 63 70
    48 88 52 84 79 70 72 50 48 97
    75 67 59 41 89 18 67 85 46 12
    39 31 16 33 91 0 14 46 41 27
    65 18 73 77 63 69 42 42 90 49
    90 49 8 19 47 60 77 64 29 93
    

**In [None]:**

{% highlight python %}
as_ndarray = np.array(base)
as_tensor  = torch.tensor(base,dtype=torch.int)

print("casted to ndarray:")
print(as_ndarray)
print()
print("="*50)
print()
print("casted to tensor:")
print(as_tensor)
{% endhighlight %}

    casted to ndarray:
    [[46 13  8 58 43 78 52 97 29 10]
     [15 51 70 14  3 16 88 29 71 58]
     [86 63 96 83  2 80 56 18 63 70]
     [48 88 52 84 79 70 72 50 48 97]
     [75 67 59 41 89 18 67 85 46 12]
     [39 31 16 33 91  0 14 46 41 27]
     [65 18 73 77 63 69 42 42 90 49]
     [90 49  8 19 47 60 77 64 29 93]]
    
    ==================================================
    
    casted to tensor:
    tensor([[46, 13,  8, 58, 43, 78, 52, 97, 29, 10],
            [15, 51, 70, 14,  3, 16, 88, 29, 71, 58],
            [86, 63, 96, 83,  2, 80, 56, 18, 63, 70],
            [48, 88, 52, 84, 79, 70, 72, 50, 48, 97],
            [75, 67, 59, 41, 89, 18, 67, 85, 46, 12],
            [39, 31, 16, 33, 91,  0, 14, 46, 41, 27],
            [65, 18, 73, 77, 63, 69, 42, 42, 90, 49],
            [90, 49,  8, 19, 47, 60, 77, 64, 29, 93]], dtype=torch.int32)
    
 
### Dimension, Shape
> np.ndim == torch.dim()
> np.shape == torch.shape
 

**In [None]:**

{% highlight python %}
print(f'dimension of as_ndarray, {as_ndarray.ndim}')
print(f'dimension of as_tensor, {as_tensor.dim()}')
print()
print(f'shape of as_ndarray, {as_ndarray.shape}')
print(f'shape of as_tensor, {as_tensor.shape}') # or .size()
{% endhighlight %}

    dimension of as_ndarray, 2
    dimension of as_tensor, 2
    
    shape of as_ndarray, (8, 10)
    shape of as_tensor, torch.Size([8, 10])
    
 
### Broadcasting ndarray == Broadcasting Tensor 

**In [None]:**

{% highlight python %}
a=[[1,2,],
   [3,4]]
b=[5,6]

na,nb=np.array(a),np.array(b)
nc=na+nb
print("Broadcasting in ndarray")
print(nc)
print()
print("="*50)
ta,tb=torch.tensor(a),torch.tensor(b)
tc=ta+tb
print("Broadcasting in tensor")
print(tc)
{% endhighlight %}

    Broadcasting in ndarray
    [[ 6  8]
     [ 8 10]]
    
    ==================================================
    Broadcasting in tensor
    tensor([[ 6,  8],
            [ 8, 10]])
    
 
### Numpy `axis` parameter == Torch `dim` paratemer 

**In [None]:**

{% highlight python %}
a=[[1.,2.,3.],
   [4.,5.,6.]]

ta=torch.tensor(a)

print("dim=1 적용")
print(ta.mean(dim=1))
print()
print("dim=0 적용")
print(ta.sum(dim=0))
print()
print("dim=-1 적용 (마지막 차원이 삭제되는 효과라 생각하면 된다.)")
print(ta.max(dim=-1)) # tensor.max() 는 (max_value,arg max) 의 두 정보를 동시에 반환한다.
print(f'max values: {ta.max(dim=-1)[0]}')
print(f'max indexes: {ta.max(dim=-1)[1]}')
{% endhighlight %}

    dim=1 적용
    tensor([2., 5.])
    
    dim=0 적용
    tensor([5., 7., 9.])
    
    dim=-1 적용 (마지막 차원이 삭제되는 효과라 생각하면 된다.)
    torch.return_types.max(
    values=tensor([3., 6.]),
    indices=tensor([2, 2]))
    max values: tensor([3., 6.])
    max indexes: tensor([2, 2])
    
