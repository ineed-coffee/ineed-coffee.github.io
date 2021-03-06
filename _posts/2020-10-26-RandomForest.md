---
title: RandomForest 개념 정리 및 활용 (sklearn)
author: INEED COFFEE
date: 2020-10-26 14:00:00 +0800
categories: [ML/DL,Machine Learning]
tags: [random forest,sklearn,python]
toc: true
comments: true
typora-root-url: ../
---
# :mag: Index

- [랜덤 포레스트(Random Forest)는 무엇일까?](#idx1) 
- [랜덤 포레스트의 핵심 : bagging (bootstrap + aggregate)](#idx2) 
- [랜덤 포레스트의 중요 매개변수](#idx3)
- [sklearn 라이브러리를 활용한 랜덤 포레스트](#idx4) 
- [Reference](#idx5)



---



### :radio_button: 랜덤 포레스트(Random Forest)는 무엇일까? <a id="idx1"></a>



이전에 [의사결정 나무 (Decision Tree)](https://github.com/ineed-coffee/TIL/blob/master/machine%20learning/DecisionTree%20%EA%B0%9C%EB%85%90%20%EC%A0%95%EB%A6%AC.md) 에 대한 개념을 정리하였는데 이름에서도 알 수 있듯이 __무작위 숲 (Random Forest)__ 는 의사결정 나무를 활용한 모델의 한계점 및 취약점을 극복하기 위한 __개선&확장 알고리즘__ 이다. 



의사결정나무는 그 학습 과정에서도 알 수 있듯이 트리의 높이에 따라 과적합 학습으로 빠질 수 있는 정도가 차이가 크고, 그에 따라 특이값에 상당히 민감한 모델이 되기 쉽다. 트리의 높이를 제한하거나 만들어진 트리에서 오류를 최소로 하는 가지를 몇개 잘라내는 방식으로 일반화 성능을 높일 수 있지만 데이터 자체가 잘 문서화되어 있어야 그나마 효과를 볼 수 있어 다양한 데이터에 활용하기 어렵다.



> 그럼 무작위 숲은 이러한 의사결정나무의 한계점을 어떻게 극복하는것일까?



우리가 쇼핑몰에서 어떤 옷을 사려하는데 상품평이 3개만 달려있고 3개 모두 긍정적인 아이템과 , 상품평이 7천개 이상 달려있고 긍정/부정의 리뷰가 고루 있는데 긍정적인 리뷰가 조금 더 많은 아이템이 있다고 하면 우린 어떤 아이템을 좋은 상품이라 생각할까?



__당연히 후자!__ 각기 다른 취향과 체형을 가진 사람들로부터 평가 받은 상품에 대한 리뷰를 훨씬 신뢰할 수 있다고 생각할것이다. 바로 이러한 점이 랜덤 포레스트에서 취하고자 하는 방식이다. 표면적인 이해를 돕자면, 랜덤 포레스트를 통해 모델을 구성하는 방식은 학습을 위해 만들어 놓은 데이터셋에서 조금씩 조금씩 데이터 일부를 뽑아 의사결정 나무들을 만들고, 각 의사결정 나무들에게 같은 예측 데이터셋을 입력으로 넣어주어 어떤 결과를 예측하는지 종합하여 판단한다. 분류 문제의 경우 더 많은 의사결정 나무들이 선택한 카테고리를 , 회귀 문제의 경우 각 의사결정 나무들이 예측한 값의 평균을 최종적으로 모델의 출력으로 결정한다.  



__※ 무작위 숲의 이해를 돕기 위한 그림__ 

![무작위 숲](https://t1.daumcdn.net/cfile/tistory/99555D335E218AE131)

출처 [https://kuduz.tistory.com/1202](https://kuduz.tistory.com/1202)



---

### :radio_button: 랜덤 포레스트의 핵심 : bagging (bootstrap + aggregate) <a id="idx2"></a>



> 어떻게 생각해보면 당연한 말 같다. 모델 하나만의 결과는 신뢰도가 낮으니 모델을 여러개 만들어서 출력을 종합해보자! 
>
> 근데 그럼 트리만 여러개 만들면 끝인걸까?



각각의 트리가 비슷한 모양을 가진다면, 조금더 정확히는 숲을 이루는 나무들이 서로 상관화(correlated)되어 있다면 트리가 아무리 많아도 성능 개선을 기대할 수 없다. 따라서 단순히 `숲` 이 아니라 `무작위 숲` 이 되기 위해서는 트리마다 생김새가 다르도록 __랜덤성__ 을 가진 나무들로 구성되어야 한다.



표면적으로만 이해하기에는 __'그냥 트리 여러개 만들어서 평균이나 빈번값 활용하는 그거'__  정도로 이해할 수 있겠지만 사실 랜덤 포레스트의 핵심 키워드는 바로 __`bagging`__ :star: ​이다.



bagging 을 설명하기 앞서 랜덤 포레스트 알고리즘과 같이 기존 알고리즘들을 엮어 활용하는 것을 __앙상블(ensemble) 기법__ 이라고 하는데 bagging 은 이러한 앙상블 기법 중 하나이다. 카테고리 제목에서도 알 수 있듯이 bagging 이라는 단어는 bootstrapping + aggregating 을 합쳐놓은 용어이다. 두 단어는 간략히는 각각 다음과 같은 의미를 갖는 용어인데,

```
Bootstrapping : 전체 집합에서 무작위 복원추출을 통해 여러 부분집합을 만드는 행위
Aggregating : '집계한다' 는 광범위한 용어로 평균이나 최빈값 등을 도출하는 행위
```



두 용어 중에서 더 큰 의미를 가지는 단어는 bootstrapping 이다. 숲을 이룰 각 나무를 학습하는데 쓰일 데이터셋으로 학습데이터 전체를 사용하지 않고 그중 일부만 활용하는 것인데 그 일부를 무작위로 복원추출하여 생성하겠다는 것이다. 만약 100개의 학습 데이터가 있다고할때 그 중 20개의 데이터만 뽑아 하나의 나무를 구성하고 다시 20개의 데이터를 100개에 포함시킨 뒤 또 랜덤하게 20개를 뽑아 다음 나무를 생성하며 각 나무는 서로 다른 모양의 가지와 높이를 구성하게 된다. 



이 두 용어를 합친 bagging 기법은 다음과 같이 정리할 수 있다.

__`"머신 러닝에서 앙상블 기법 중 하나인 bagging 은 boostrapping을 통해 train_set 에서 무작위 복원 추출을 통해 기저 모델을 생성하고 , 이러한 기저 모델을 어려개 생성하여 이들의 출력 결과를 aggregating 하여 학습하는 알고리즘이다."`__ 



__※ 사실 bagging 이외에도 랜덤포레스트의 각 나무는 임의 노드 최적화라는 방법을 통해 더욱 일반화 성능을 얻고 있는데 이 정리에서는 bagging 에 대해서만 다루었다.__ 



---


### :radio_button: 랜덤 포레스트의 중요 매개변수 <a id="idx3"></a>

- __숲의 크기__ :star:
  - 숲을 구성할 나무의 수를 뜻하는 매개변수로 나무의 수가 증가할수록 일반화 성능이 우수하지만 , 훈련과 검증의 시간은 오래걸린다.

- __각 트리의 높이__ :star: ​
  - 숲을 구성하는 각 나무들의 최대 깊이를 설정하는 매개변수로 과소적합&과대적합과 밀접한 관계를 가지므로 적절한 높이를 통해 학습하는 것이 중요하다.
- __임의성의 정도와 종류__ 
  - 전체 데이터에서 부분 데이터를 추출하는 과정에서 어떤 분포 함수를 적용하는지에 따라 랜덤성에 차이가 있다. 깊은 설명은 생략
- __각 트리의 특징 벡터 선정__ 
  - 참고내용에서 언급한 부분, 실제 각 트리는 부분 데이터 중에서도 모든 특성을 다 학습에 활용하지 않고 각 노드에서 일부 특성만을 고려하여 분기하는데 이때 활용되는 훈련 목적 함수의 종류에 따라 일반화 성능의 차이가 있다. 깊은 설명 생략



---

### :radio_button: sklearn 라이브러리를 활용한 랜덤 포레스트 <a id="idx4"></a> 

:pencil2: __무작위 숲 객체 생성__ 

```python
from sklearn.ensemble import RandomForestClassifier
from slkearn.ensemble import RandomForestRegressor

RF_C = RandomForestClassifier()
RF_R = RandomForestRegressor()
```

> __추가 옵션__ : 

(다른 나머지 옵션은 의사결정 나무와 동일)

- `n_estimators` : 숲을 구성할 나무의 수를 지정하는 옵션
  - default = 10 이며 , 너무 작으면 일반 의사결정 나무와 성능 차이가 없음
  - 나무의 수가 많을수록 노이즈에 강한 일반화 모델을 얻을 수 있지만 학습에 오랜 시간이 걸림

  

  


:pencil2: __무작위 숲 객체 학습__ 

```python
model_c = RF_C.fit(xdata,ydata,sample_weight)
model_r = RF_R.fit(xdata,ydata,sample_weight)
```

> 파라미터 세부내용

- `xdata` : 학습데이터의 입력값
- `ydata` : 학습데이터의 출력값
- `sample_weight` : optional 파라미터로 각 설명변수의 가중치 배열을 전달 가능. 분기 조사 과정에서 적용된다.

  

  

:pencil2: __무작위 숲 객체 예측/분류__ 

```python
predict_C = model_C.predict(xtest)
predict_R = model_r.predict(xtest)
```

> 참고

예측 분류/값 에 대해 __배열__ 형태로 반환받는다.

  

  

 :pencil2: __무작위 숲 객체 평가__ 

```python
score_c = model_c.score(xtest,ytest,sample_weight)
score_r = model_r.score(xtest,ytest,sample_weight)
```

> 각 문제별 평가기준

- `분류(classification)`
  - 단일 레이블의 경우 ACC (정확도) 를 기준으로 0~1 사이 값으로 평가
  - 다중 레이블의 경우 각 레이블의 ACC 값의 평균으로 평가한다.

※ 다중 레이블은 ydata 의 컬럼이 2 이상인 데이터이다.

- `회귀(Regression)`
  - __R2__ 라고 많이 표현하는 결정계수를 평가 지표를 활용한다. 모든 데이터를 평균치로 예측하는 단순 모델(Zero-R)과 비교하였을 때 얼마나 개선된 성능을 보이는지를 구하는 결정계수의 값 또한 ACC 와 마찬가지로 0~1 사이의 값을 가진다.

  

  



---


### :radio_button: Reference <a id="idx5"></a>

- [https://ko.wikipedia.org/wiki/%EB%9E%9C%EB%8D%A4_%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8#%ED%9B%88%EB%A0%A8_%EB%AA%A9%EC%A0%81_%ED%95%A8%EC%88%98](https://ko.wikipedia.org/wiki/%EB%9E%9C%EB%8D%A4_%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8#%ED%9B%88%EB%A0%A8_%EB%AA%A9%EC%A0%81_%ED%95%A8%EC%88%98)
- http://hleecaster.com/ml-random-forest-concept/
- https://lsjsj92.tistory.com/542



