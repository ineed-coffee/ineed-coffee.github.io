---
title: pytorch basic2 _ Tensor
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

**In [28]:**

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
    
 
# Basic 2. Tensor
Tensor vs. Ndarray 

**In [29]:**

{% highlight python %}
import torch
import numpy as np
{% endhighlight %}

**In [30]:**

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
    60 66 29 88 12 84 1 54 45
    63 31 25 40 61 40 65 27 68
    58 33 31 62 89 46 28 49 67
    48 39 25 57 62 81 7 98 4
    68 15 51 47 64 1 19 11 14
    44 41 51 72 44 73 59 79 54
    82 5 50 71 60 25 88 1 14
    79 74 71 78 78 40 71 44 47
    2 84 37 28 33 8 56 54 80
    18 31 2 69 59 66 34 51 14
    

**In [31]:**

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
    [[60 66 29 88 12 84  1 54 45]
     [63 31 25 40 61 40 65 27 68]
     [58 33 31 62 89 46 28 49 67]
     [48 39 25 57 62 81  7 98  4]
     [68 15 51 47 64  1 19 11 14]
     [44 41 51 72 44 73 59 79 54]
     [82  5 50 71 60 25 88  1 14]
     [79 74 71 78 78 40 71 44 47]
     [ 2 84 37 28 33  8 56 54 80]
     [18 31  2 69 59 66 34 51 14]]
    
    ==================================================
    
    casted to tensor:
    tensor([[60, 66, 29, 88, 12, 84,  1, 54, 45],
            [63, 31, 25, 40, 61, 40, 65, 27, 68],
            [58, 33, 31, 62, 89, 46, 28, 49, 67],
            [48, 39, 25, 57, 62, 81,  7, 98,  4],
            [68, 15, 51, 47, 64,  1, 19, 11, 14],
            [44, 41, 51, 72, 44, 73, 59, 79, 54],
            [82,  5, 50, 71, 60, 25, 88,  1, 14],
            [79, 74, 71, 78, 78, 40, 71, 44, 47],
            [ 2, 84, 37, 28, 33,  8, 56, 54, 80],
            [18, 31,  2, 69, 59, 66, 34, 51, 14]], dtype=torch.int32)
    
 
### Dimension, Shape
> np.reshape  == torch.view
 

**In [32]:**

{% highlight python %}
print(f'shape of as_tensor, {as_tensor.shape}') # or .size()
for row in as_tensor:
  print(row)
print()
reshaped_tensor=as_tensor.view(3,2,-1)
print(f'shape of reshaped_tensor, {reshaped_tensor.shape}')
for row in reshaped_tensor:
  print(row)
{% endhighlight %}

    shape of as_tensor, torch.Size([10, 9])
    tensor([60, 66, 29, 88, 12, 84,  1, 54, 45], dtype=torch.int32)
    tensor([63, 31, 25, 40, 61, 40, 65, 27, 68], dtype=torch.int32)
    tensor([58, 33, 31, 62, 89, 46, 28, 49, 67], dtype=torch.int32)
    tensor([48, 39, 25, 57, 62, 81,  7, 98,  4], dtype=torch.int32)
    tensor([68, 15, 51, 47, 64,  1, 19, 11, 14], dtype=torch.int32)
    tensor([44, 41, 51, 72, 44, 73, 59, 79, 54], dtype=torch.int32)
    tensor([82,  5, 50, 71, 60, 25, 88,  1, 14], dtype=torch.int32)
    tensor([79, 74, 71, 78, 78, 40, 71, 44, 47], dtype=torch.int32)
    tensor([ 2, 84, 37, 28, 33,  8, 56, 54, 80], dtype=torch.int32)
    tensor([18, 31,  2, 69, 59, 66, 34, 51, 14], dtype=torch.int32)
    
    shape of reshaped_tensor, torch.Size([3, 2, 15])
    tensor([[60, 66, 29, 88, 12, 84,  1, 54, 45, 63, 31, 25, 40, 61, 40],
            [65, 27, 68, 58, 33, 31, 62, 89, 46, 28, 49, 67, 48, 39, 25]],
           dtype=torch.int32)
    tensor([[57, 62, 81,  7, 98,  4, 68, 15, 51, 47, 64,  1, 19, 11, 14],
            [44, 41, 51, 72, 44, 73, 59, 79, 54, 82,  5, 50, 71, 60, 25]],
           dtype=torch.int32)
    tensor([[88,  1, 14, 79, 74, 71, 78, 78, 40, 71, 44, 47,  2, 84, 37],
            [28, 33,  8, 56, 54, 80, 18, 31,  2, 69, 59, 66, 34, 51, 14]],
           dtype=torch.int32)
    
 
> np.new_axis == torch.unsqeeze 

**In [33]:**

{% highlight python %}
print(f'shape of as_tensor, {as_tensor.shape}')
for row in as_tensor:
  print(row)
print()
unsqeezed_tensor=as_tensor.unsqueeze(1)
print(f'shape of reshaped_tensor, {unsqeezed_tensor.shape}')
for row in unsqeezed_tensor:
  print(row)
{% endhighlight %}

    shape of as_tensor, torch.Size([10, 9])
    tensor([60, 66, 29, 88, 12, 84,  1, 54, 45], dtype=torch.int32)
    tensor([63, 31, 25, 40, 61, 40, 65, 27, 68], dtype=torch.int32)
    tensor([58, 33, 31, 62, 89, 46, 28, 49, 67], dtype=torch.int32)
    tensor([48, 39, 25, 57, 62, 81,  7, 98,  4], dtype=torch.int32)
    tensor([68, 15, 51, 47, 64,  1, 19, 11, 14], dtype=torch.int32)
    tensor([44, 41, 51, 72, 44, 73, 59, 79, 54], dtype=torch.int32)
    tensor([82,  5, 50, 71, 60, 25, 88,  1, 14], dtype=torch.int32)
    tensor([79, 74, 71, 78, 78, 40, 71, 44, 47], dtype=torch.int32)
    tensor([ 2, 84, 37, 28, 33,  8, 56, 54, 80], dtype=torch.int32)
    tensor([18, 31,  2, 69, 59, 66, 34, 51, 14], dtype=torch.int32)
    
    shape of reshaped_tensor, torch.Size([10, 1, 9])
    tensor([[60, 66, 29, 88, 12, 84,  1, 54, 45]], dtype=torch.int32)
    tensor([[63, 31, 25, 40, 61, 40, 65, 27, 68]], dtype=torch.int32)
    tensor([[58, 33, 31, 62, 89, 46, 28, 49, 67]], dtype=torch.int32)
    tensor([[48, 39, 25, 57, 62, 81,  7, 98,  4]], dtype=torch.int32)
    tensor([[68, 15, 51, 47, 64,  1, 19, 11, 14]], dtype=torch.int32)
    tensor([[44, 41, 51, 72, 44, 73, 59, 79, 54]], dtype=torch.int32)
    tensor([[82,  5, 50, 71, 60, 25, 88,  1, 14]], dtype=torch.int32)
    tensor([[79, 74, 71, 78, 78, 40, 71, 44, 47]], dtype=torch.int32)
    tensor([[ 2, 84, 37, 28, 33,  8, 56, 54, 80]], dtype=torch.int32)
    tensor([[18, 31,  2, 69, 59, 66, 34, 51, 14]], dtype=torch.int32)
    
 
### Concatenation , Stacking 
 
> np.concat([] , axis= ) == torch.cat([] , dim= ) 

**In [34]:**

{% highlight python %}
x = torch.Tensor([[1, 2], [3, 4]])
y = torch.Tensor([[5, 6], [7, 8]])

dim0_concat = torch.cat([x,y],dim=0) # 행이 늘어난다.
dim1_concat = torch.cat([x,y],dim=1) # 열이 늘어난다.

print("concat with dim=0")
for row in dim0_concat:
  print(row)
print()
print("concat with dim=1")
for row in dim1_concat:
  print(row)
{% endhighlight %}

    concat with dim=0
    tensor([1., 2.])
    tensor([3., 4.])
    tensor([5., 6.])
    tensor([7., 8.])
    
    concat with dim=1
    tensor([1., 2., 5., 6.])
    tensor([3., 4., 7., 8.])
    
 
> torch.stack() : unsqeeze 작업을 자동으로 수행 가능. 

**In [None]:**

{% highlight python %}
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
k=torch.stack([x,y,z],axis=0)
print(k)
print(k.shape)
{% endhighlight %}

    tensor([[1., 4.],
            [2., 5.],
            [3., 6.]])
    torch.Size([3, 2])
    
 
### ones_like , zeros_like
> 행렬 shape을 참고하여 0이나 1로 초기화 시키는 함수, numpy 와 비슷하다. 

**In [None]:**

{% highlight python %}
sample = torch.Tensor([[0, 1, 2], [2, 1, 0]])

print(torch.ones_like(sample))
print()
print(torch.zeros_like(sample))
{% endhighlight %}

    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    
 
### In-place Operation (중요! ★)
> C의 ++ 나, Python의 += 연산 같은 덮어쓰는 연산을 수행하며 그 `기록 정보`를 유지하고 있는것이 특징이다.
(이후 인공신경망 생성시에 역전파의 디버깅이 가능한 구조이다.) 

**In [None]:**

{% highlight python %}
sample = torch.Tensor([[1, 2], [3, 4]])

print(sample.mul(2))
print(sample) # sample 자료는 변하지 않음
print()
# 어떤 연산이든 뒤에 언더바(_)를 추가하면 In-place operation이 된다.
print(sample.mul_(2))
print(sample)
{% endhighlight %}

    tensor([[2., 4.],
            [6., 8.]])
    tensor([[1., 2.],
            [3., 4.]])
    
    tensor([[2., 4.],
            [6., 8.]])
    tensor([[2., 4.],
            [6., 8.]])
    
