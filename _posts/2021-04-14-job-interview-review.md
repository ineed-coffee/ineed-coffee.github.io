---
title: AI,ML,DL 관련 직접 받았던 면접 질문과 셀프 검토
author: INEED COFFEE
date: 2021-04-14 14:00:00 +0800
categories: [Reviews]
tags: [deep learning,review,job interview,machine learning,ai]
toc: true
comments: true
typora-root-url: ../
---

​	

### 면접에서 직접 받았던 질문과 내가 답한 문장을 평가하고 수정하는 목적의 리뷰입니다.

​	

__`2020.04.14 : 스타트업`__ 

__Q__ : CNN, RNN에 대하여 설명해주세요.

__A__ : DNN 모델의 파생 심화 모델로 CNN 모델은 이미지와 같은 공간정보를 유지하며 특성을 추출해야 하는 데이터의 성격에 맞게 convolution 이라는 연산으로 Local receptive field 개념을 구현한 모델. RNN 모델은 데이터와 데이터 사이 시차의 연관성이 존재하여 추출하고자 하는 특성이 이전 특성으로부터 받는 영향을 고려할 수 있도록 Cell-state, forget-gate 와 같은 장치를 통하여 구현한 모델.

__self-feedback__ : 3/5 . CNN 개념보다 RNN 모델 설명을 훨씬 두루뭉실하게 설명한것 같다. 한마디로 표현할 수 있는 핵심 단어로만 구성해놓은 답변을 준비해야 할듯함.

​	

__Q__ : Cross-validation이란? 이것을 왜 하는지? 꼭 해야하는지? Test-set은 validation-set으로 사용하면 안되나?

__A__ : 

- 검증용 데이터셋을 고정 데이터로 활용하면 편향의 위험이 존재하여 K-fold 와 같은 방식으로 서로 다른 부분데이터셋에 의하여 검증이 이루어지도록 하기 위함. 
- 모델의 학습 단계에서 모델이 원활히 학습이 이루어지고 있는지를 확인하는데 있어 학습 오류만을 통해 확인할 수 없기 때문에 검증 오류의 추이로부터 커브 포인트가 발생하는지 관찰해야한다.
- 학습 데이터가 부족한 경우, 검증 데이터를 별도로 설정하지 않은채 train-test 로만 구성하여 진행하는 경우도 존재하는걸로 알고 있다고 답변.
- 가중치 갱신에는 관여를 하지 않더라도 검증용 데이터와 테스트 데이터를 별도로 나누는 이유는 모델에 처음보는 입력을 넣기 위해서로 알고 있다고 답변

__self-feedback__ : 2/5 . cross-validation 에 대한 정의와 기본 개념보다 방법에 대해서만 얘기할 수 있었다. 경우에 따라 데이터셋이 큰 경우 테스트 데이터를 별도로 두지 않는 경우도 있다는 것을 면접관님께 처음 들음. 

​	

__Q__ : 딥러닝의 장점과 단점을 말해주세요.

__A__ : 

- ML 모델과 비교하였을때 비선형적으로 엮인 데이터로부터 특성 추출에 강하다.
- 모델의 복잡도가 ML에 비해 월등히 높다(하드웨어 스펙 또한 요구된다) , 기울기 소실&증폭 문제가 완벽히 해결되지 않았다. 

__self-feedback__ : 0.5/5 정말 꽝인 답변이었다. 그동안 생각해보지 못한 부분이라 답변 자체도 10초 정도 머뭇거렸다. 지인에게도 생각을 물어봤는데 아마 딥러닝의 단점은 'ML 모델에 비하여 결과해석이 어렵다' 가 좋은 답변이 되었을것 같다.

​	

__Q__ : CNN 모델과 단순 feed-forward 모델을 비교한다면, 어느쪽이 파라미터가 많을까요?

__A__ : kernel은 fully-connecting 방식과 다르게 각 연산에 쓰이는 가중치가 공유되는 형식의 연산이라 Conv. layer 쪽이 훨씬 파라미터가 적다고 답변.

__self-feedback__ : 3.5/5 개념 자체는 알고 있었으나 '공유된다' 나 '무조건 적다' 와 같은 표현이 모호하게 해석될 수 있어 면접관님께서도 정확한 추가 질문을 주셨음. (ex. 공유된다는게 kernel 끼리를 말씀하신 건가요?)

​	

__Q__ : 차원의 저주에 대해 말씀해주세요.

__A__ : PCA 기법에서 맵핑하고자 하는 차원 자체가 방대한 경우, 불필요한 정보까지 축약되어 신뢰할 수 없는 저차원 정보가 되는것이라 답변

__self-feedback__ : 0/5 아예 틀린 답변을 함. PCA 쪽에서도 차원의 저주라는 표현을 쓰는 상황이 있기는 한듯하나 일반적인 차원의 저주는 모델 학습에 있어 학습 데이터에 비해 입력 차원의 수가 큰 경우 일정 차원을 기점으로 학습 능력이 급격히 감소하는 현상을 차원의 저주라고 한다고 면접관님께서 바로 잡아주심.

​	

__Q__ : Over-fitting 의 대응 기법들을 설명해주세요.

__A__ : 더 많은 데이터를 확보하는 것이 가장 이상적, 그 외에는 L1,L2 정칙화를 통한 weight-decay, model-capacity를 줄일 수 있는 bottleneck 기법, 해당 epoch 동안 일정 비율의 가중치를 고정시키는 drop-out, batch-normalization 등을 답변

__self-feedback__ : 3.5/5 나머지는 아는 선에서 더듬지 않고 설명하였으나 batch-normalization의 경우 사실 이름만 알고 정확한 개념을 몰랐던 상태라 `batch-normalization이 CNN모델에서만 쓰이나요?` 라는 면접관님의 추가질문에 답변할 수 없었음.

​		

__그 외 기타 배운것들__ 

- 내가 했던 프로젝트들을 설명하는 중간에 'XXX은 생성 모델이었나요?' 라는 질문을 주셨는데 이게 GAN 모델 말씀하시는줄 모르고 train-from-scratch 를 말씀하신걸로 알아들었다. '생성 모델' 이라는 말을 알고 있자.
- 기술 면접관님께서 프로젝트에 대해 평을 남겨주셨는데,  '어떤 것을 했는지' , '어떻게 했는지' 같은 내용은 알 수 있었으나 그 과정에서 `ai engineer의 관점에서` , `데이터 분석자의 관점에서` 시도한 것들이 드러나지 않는다고 말씀해주셨다. (ex. xx모델을 ~~이렇게 변형하여 사용했는데 어떤 문제점 때문에 그런 시도를 했다. xx과 같은 변수는 xx의 문제점이 있어 ~~이런식으로 피쳐 엔지니어링을 진행했다.) __뭘? 했는지 뿐만 아니라 왜? 에 대한 답변이 잘 드러나게끔 프로젝트 설명을 하거나 최소한 질문했을때 술술 답변할 수 있어야겠다고 느낌__ 
-   면접이 처음이라 질문에 대한 답변만 생각하고 막상 내가 어떤 질문을 할 것인가를 전혀 준비를 못해갔다. 임원 면접에서 면접관님들께서 먼저 근무 환경이나 추구하는 근무 방식, 연봉 등에 대하여 말씀해주셨고 지인들에게도 물어보니 __퇴직금 포함 여부, 포괄 임금제, 식대 지원, 스톡 옵션 제도, 청년내일채움공제__ 와 같은 내용은 면접 자리에서 꼭 물어봐야한다는 것을 알았다.