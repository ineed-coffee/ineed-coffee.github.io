---
title: Types of data and how to analyze
author: INEED COFFEE
date: 2020-10-21 14:00:00 +0800
categories: [Probability & Statistics]
tags: [probability,statistics,pearson,spearman,phi]
toc: true
comments: true
typora-root-url: ../
---
# :mag: Index

- [데이터의 종류와 예시](#idx1) 
- [데이터 유형에 따른 상관 분석 방법](#idx2) 
  - [연속형 데이터 - 연속형 데이터](#idx2_1)
  - [이진 데이터 - 연속형 데이터](#idx2_2)
  - [이진 데이터 - 이진 데이터](#idx2_3)
- [참고자료](#idx3)



---



### :radio_button: 데이터의 종류와 예시 <a id="idx1"></a>

세상에는 많은 유형의 데이터가 존재하고 때에 따라 의미는 같지만 형태가 변할 수도 있다. 이런 데이터를 분석하기 위해서는 데이터의 성격에 따라 구분하고 각 특성에 대하여 알고 있어야 하는데 데이터는 크게 다음과 같이 분류할 수 있다.

​	

- __양적 데이터 (Quantitative data)__ 
  - 연속형 (Continuous) - 키 , 나이 , 수입과 같은 연산 가능한 실수형 데이터
  - 이산형 (Discrete) - 책의 페이지 , 동전의 개수와 같은 연산 가능한 정수형 데이터 ( 혹은 단위형 데이터)
- __질적 데이터 (Qualitative data)__ 
  - 명목형 (norminal) - 혈액형 , 전공 구분과 같이 연산이 불가하고 순위의 개념이 없는 데이터
  - 순서형 (ordinal) - 옷의 사이즈 구분 , 반 등수와 같은 순서 구분이 있는 연산 불가 데이터
  
  

__※ 때에 따라서는 데이터의 형태가 바뀌는 경우도 있을 수 있다.__ 

ex ) 순서형 데이터를 인코딩하여 이산형 데이터로 다루거나 ,

ex ) 연속형 데이터를 구간 mapping 을 통해 이산형 데이터로 다루는 경우, 등

​	

---


### :radio_button: 데이터 유형에 따른 상관 분석 방법 <a id="idx2"></a>

​	

데이터간 선형도 혹은 단조 정도를 알아보기위해 상관 분석을 진행하는데 , 데이터의 유형에 따라 이를 분석하는 방법이 다른다.

​	

1. __연속형 데이터 - 연속형 데이터__ <a id="idx2_1"></a>

연속적인 데이터로 이루어진 두 데이터를 상관 분석하는 방법은 2가지 유형으로 나뉜다.

  	

`데이터가 정규성을 띄는 경우 or 띈다고 가정할 수 있을때`  : 

- 평균, 표준편차와 같은 모수(parameter)를 활용할 수 있으므로 __피어슨 상관계수 (Pearson Correlation Coefficient)__ 를 통해 상관계수를 구한다.

  

`데이터의 수가 적거나 정규성 가정을 할 수 없는 경우` : 

- 각 데이터를 크기에 따라 나열하여 순위를 매긴 후 해당 순위 데이터 사이의 피어슨 상관계수를 통해 분석하는 __스피어만 상관계수 (Spearman Rank Correlation Coefficient)__ 를 이용한다.
- 순위 데이터를 통해 계산하므로 질적 데이터 중 __순서형 데이터__ 또한 가변수화하여 적용할 수 있다.
- 스피어만 상관계수와 거의 비슷한 방법으로 __켄달의 타우 (Kenddall's Tau)__ 방식도 있다.

  

※ 위의 두 방법 구분을 흔히  __`모수적 방법(Parametic method)`__ , __`비모수적 방법(Non-Parametic method)`__  으로 구분하는데 , 

이에 대한 보다 자세한 내용은 [모수적 VS 비모수적 접근](https://github.com/ineed-coffee/TIL/blob/master/statistics/Parametric%20VS.%20Non-Parametric%20method.md) 에 정리하였다.

 	

---

​	

2. __이진 데이터 - 연속형 데이터__ <a id="idx2_2"></a>

독립 변수의 데이터가 네/아니오 와 같은 이진 형태로 되어 있고 종속 변수의 형태가 연속형 데이터일때 , 둘의 상관분석을 위해 다음 2가지 방법을 활용한다.

`양류 상관계수 (point-biserial correlation coefficient)` :

- 명목형 이진데이터로 이루어진 독립변수와 양적 종속변수 사이의 상관정도를 알아보기 위해 사용.
- 피어슨의 단순적률상관계수와 동일한 결과를 가짐

  

`양분 상관계수 (biserial correlation coefficient)` :

- 양류  상관계수와 거의 동일하지만 굳이 구분되어 있는 이유는 기존의 독립 데이터가 양적 데이터인 상황에서 연구자의 임의 기준에 의해 이분화 된 데이터를 통해 양적 종속 변수와 상관계수를 구하는 경우를 뜻하기 때문이다.

  

※ 두 경우 모두 피어슨 단순적률 상관계수의 변형된 형태로 볼 수 있다.

​	

---

​	

3. __이진 데이터 - 이진 데이터__ <a id='idx2_3'></a>

독립 변수와 종속 변수 모두 네/아니오 형식의 명목형 이진 데이터로 이루어져있을때 둘의 상관관계를 알아볼 때는 다음의 방법을 이용한다.

​	

`파이 계수 (Phi Coefficient)` :

- 빈번히 사용되는 지표는 아니지만 , `L` , `C` , `Lambda` 의 지표와 더불어 두 이진 데이터 사이 상관 정도를 나타내는 지표이다.
- 두 변수에 대해 다음과 같은 테이블을 구성할 때 , 아래와 같은 식으로 파이 계수를 구한다.

|    _    | X=0  | X=1  |
| :-----: | :--: | :--: |
| __Y=1__ |  a   |  b   |
| __Y=0__ |  c   |  d   |

​	

$$ Phi = \frac {ad-bc}{ \sqrt {(a+b)(b+c)(c+d)(d+a)}} $$

​	

---

### :radio_button: 참고자료 <a id="idx3"></a>

- [https://mansoostat.tistory.com/115](https://mansoostat.tistory.com/115) 
- [https://m.blog.naver.com/PostView.nhn?blogId=artquery&logNo=44945345&proxyReferer=https:%2F%2Fwww.google.com%2F](https://m.blog.naver.com/PostView.nhn?blogId=artquery&logNo=44945345&proxyReferer=https:%2F%2Fwww.google.com%2F)  







