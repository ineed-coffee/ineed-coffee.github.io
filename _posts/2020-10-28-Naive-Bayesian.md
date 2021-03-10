---
title: Naive Bayesian 개념 정리 및 활용 (sklearn & R)
author: INEED COFFEE
date: 2020-10-28 14:00:00 +0800
categories: [ML/DL,Machine Learning]
tags: [naive-bayesian,sklearn,r,python]
toc: true
comments: true
---
# :mag: Index

- [나이브 베이즈(Naive Bayes) 알고리즘이란?](#idx1) 

- [조건부 확률에서 베이즈 공식까지](#idx2) 

- [나이브 베이즈 알고리즘을 머신러닝에 어떻게 적용하는걸까?](#idx3)

- [나이브 베이즈 모델의 종류](#idx4) 

- [나이브 베이즈 알고리즘의 장단점](#idx5)

- [R을 활용한 나이브 베이즈 모델링](#idx6)

- [sklearn 패키지를 활용한 나이브 베이즈 모델링](#idx7)

- [참고자료](#idx8)

  

---

### :radio_button: 나이브 베이즈(Naive Bayes) 알고리즘이란? <a id="idx1"></a>

​	

확률 기반 머신러닝 알고리즘의 s대표격인 나이브 베이즈 알고리즘은 데이터의 각 특성이 서로 독립할 것이라는 `나이브(Naive)` 한 가정을 전제로 하고 새로운 입력이 들어왔을때 데이터를 어느 레이블로 분류하는 것이 확률적으로 가장 높은가? 라는 철학으로 동작하는 알고리즘이다. 

​	

>  `조건부 확률`  ,`우도` , `사전확률` , `결합 확률` , `독립` 등의 생소한 단어들이 등장할텐데 천천히 나이브 베이즈 알고리즘이 어떤건지 자세히 알아보도록 하자.

​	

---


### :radio_button: 조건부 확률에서 베이즈 공식까지 <a id="idx2"></a>



> __조건부 확률__ 

어떤 사건이 일어났다는 가정했을 때 다른 어떤 사건이 일어날 확률을 조건부 확률이라 한다.

예를 들어 _내가 버스 정류장까지 5분만에 뛰어갈 수 있다고 할때 버스를 놓치지 않을 확률은 얼마일까?_ 와 같은 형식이 조건부 확률이다.

​	

두 사건 A , B 가 있다고 할때 각 사건이 일어날 확률은 `P(A)` , `P(B)` 와 같이 표현하고 

__'__ 사건 A 가 일어났을때 B가 일어날 확률 __'__  과 같은 __조건부 확률__ 은 `P(B|A)` 와 같이 표현한다.

또 , __'__ A ,B 가 동시에 일어날 확률 __'__ 과 같은 문제는 __결합 확률__ 이라 하는데 `P(A∩B)` 혹은 `P(A,B)` 와 같이 표현한다.

​	

조건부 확률의 계산은 위 표현식들의 조합을 통해 이루어진다.
	
![eq1](http://latex2png.com/pngs/07ca123db3b2f8938fc1e23b23a59b46.png)

말로 풀어보자면 ,

`A가 일어났다고 가정할때 B가 일어날 확률` = `A , B 가 동시에 일어날 확률` ÷ `A가 일어날 확률`  로 계산되는 것이다.

​	

그런데 , 결합 확률이라고 했던 `P(A∩B)` 는 교환법칙의 성질을 만족하는 연산으로 `P(A∩B)` = `P(B∩A)` 이다.

말로하면 당연하긴 하지만 (내일 비가온다(A) 와 내가 지각한다(B) 가 동시에 일어날 확률과 내일 내가 지각한다(B) 와 비가온다(A) 가 동시에 일어날 확률은 같다.)  ,  이는 베이즈 공식의 기초가 되는 개념이 된다.

​	

 앞의 조건부 확률 계산식으로부터 결합확률 `P(B∩A)` 는 다음과 같이 다시 쓸 수 있는데 , 

​	![eq2](http://latex2png.com/pngs/30b647f653394dafd803f3501b8932c7.png) 

​	

`P(B∩A)` = `P(A∩B)` 라고 했으므로 , 다음과 같은 식이 만들어진다.

​	


![eq3](http://latex2png.com/pngs/f4932910004bb23e7ac13a53808d3ccc.png)

![eq4](http://latex2png.com/pngs/8df60e63c6507429f811eaaca10cc2f3.png)

![eq5](http://latex2png.com/pngs/41e8904688dbaf94742f2c506f145cf7.png)

![eq6](http://latex2png.com/pngs/3d7564b2e288e1e91d8e703d31c526d0.png)

![eq7](http://latex2png.com/pngs/42618b976784d87909a85a9037f4dc1e.png)

​	

이때 , 최종적으로 정리된 5번 식이  바로 __베이즈 공식__ 이다.

​	

베이즈 공식은 식 자체로는 두 조건부 확률간의 관계정도를 나타내는 식으로 보일 수 있지만 실제로 이 공식은 우리가 알지 못하는 미래의 확률을 이미 확인된 확률을 통해 추정할 수 있다는 의미를 갖는 중요한 공식이다.

​	

미래 확률을 추정한다는 말을 예시로 풀어보자면 , 

`A 사건` : 무릎이 시리다.

`B 사건` : 비가온다.

라고 두 사건 A , B 을 설정해보자. 

​	

이때 지금 내 무릎이 아프다면 , 곧이어 비가 올 확률은 어떻게 계산할 수 있을까? 

확률식으로 표현했을때 `P(B|A)` 로 표현되는 이러한 미래의 확률이 우리가 궁금한 __사후 확률__ 인 것이다.

​	

이러한 사후 확률은 위에서 확인했던 베이즈 공식에 따라  `[ P(A|B)·P(B) ] ÷ P(A)`  로 계산할 수 있는데 

각 확률을 예시에 맞춰 적용하면 다음과 같다.

​	

`P(A|B)` : 비가 왔을때 , 내가 무릎이 시렸던 비율 ( 경험 확률 )

`P(B)` : 전체 날짜 중 비가 왔던 비율 ( 경험 확률 )

`P(A) ` : 전체 날짜 중 내 무릎이 시렸던 비율 ( 경험 확률 )

​	

계속 부가 설명으로 __경험__ 확률이라는 말을 썼는데 , 우리가 예측하는 확률이 아닌 이미 관찰된 데이터를 통해 계산된 확률값임을 구분하기 위해 사용하였다.

​	

또 , 베이즈 이론에서는 위의 각 경험 확률에 대한 용어가 존재하는데 다음과 같다.

`P(A|B)` : 우도 , Likelihood

`P(B)` : 사전 확률 , Prior

`P(A)` : 주변 우도 , Evidence or Marginal likelihood

`P(B|A)` : (우리가 실제로 관심 있는 확률) 사후 확률 , Posterior :star: 

​	

위 용어로 베이즈 이론을 종합하여 다시 작성하면 , 우리가 실제로 알고 싶은 __사후확률__ 을 계산하는 식은 다음과 같이 나타낼 수 있다.

​	

![eq8](http://latex2png.com/pngs/5bf8c19a135b0811e238644356fc9389.png)
	

---


### :radio_button: 나이브 베이즈 알고리즘을 머신러닝에 어떻게 적용하는걸까? <a id="idx3"></a>

그렇다면 이와 같은 베이즈 이론이 머신러닝에 어떻게 적용되는 것일까?



나이브 베이즈 알고리즘은 확률 기반의 __데이터 분류__  머신러닝 알고리즘이라 하였는데 , 

앞서 예시를 들며 사용하였던 사건 A , B 를 머신러닝과 관련된 용어로 바꾸면 조금 더 이해가 쉽다.

​	

하나의 독립변수와 하나의 (범주형) 종속변수로 이루어진 데이터에 대해 다음과 같이 사건을 설정해보자. 

`A 사건` : 어떤 데이터의  독립변수 값이 3이다. 

`B 사건` : 어떤 데이터의 종속 변수가  7이다. (7번 레이블이다)

​	

이런 상황에서 새로운 입력이 들어왔는데 독립 변수의 값이 3일때 , 7번 레이블로 분류될 확률은 어떻게 될까?

​	

![eq9](http://latex2png.com/pngs/de5ee5958726ceef0eb9eaecf04aae89.png)

​	


위와 같은 식으로 표현되며 , 이를 우리가 궁금한 __사후 확률__ 이라고 하는것이다.

​	

그럼 베이즈 공식에서 확인했듯 , 사후 확률은 우도,주변우도,사전 확률을 통해 계산할 수 있는데 

​	

![eq10](http://latex2png.com/pngs/2c8fa23bde3d5c2038c14b56c64ab435.png)

​	

우도( `P(input=3|label=7)` ) , 사전확률( `P(label=7)` ) , 주변우도 ( `input=3` ) 모두 우리가 이미 학습용으로 가지고 있는 데이터에서 계산할 수 있는 확률값이다.

​	

베이지안 기반 분류 모델에서는 이 같은 사후확률을 모두 계산하여 비교해보고 가장 확률이 높은쪽으로 데이터를 분류하는것이다.

​	

> __독립변수가 여러개인 경우에는 사후확률을 어떻게 계산할까?__ 



일반적인 상황이라면 독립변수가 2개 이상일때는 식의 복잡도가 꽤 올라간다. 앞서 언급한 예시에서 새로운 사건 __C__ 가 추가된 상황을 상상해보자.

​	

`A 사건` : 어떤 데이터의  독립변수1 의 값이 3이다. 

`B 사건` : 어떤 데이터의 종속 변수가  7이다. (7번 레이블이다)

`c 사건` : 어떤 데이터의 독립변수2 의 값이 5이다.

​	

이런 상황에서 사후확률 `P( label=7 | input1=3 , input2=5 )` 은

`P( input1=3 , input2=5 | label=7 ) · P( label=7 )` ÷ `P(input1=3 , input2=5)` 로 계산하게된다.

​	

하나의 사건을 더 추가하여 확인해보면 알겠지만 , 독립변수의 수가 증가할수록 우도와 주변우도의 계산이 복잡하고 많은 연산을 요구하는 확률값으로 예상할 수 있다.

​	

__그렇기 때문에 나이브 베이지안 모델에서는 모든 독립사건을 서로 독립적인 사건이라 가정한다.__ 

​	

독립적인 사건이란 , 서로의 발생에 영향이 없는 관계를 의미하며 독립적인 관계인 사건들의 결합확률은 각 사건의 단일발생 확률들의 곱으로 계산할 수 있다.

​	

![eq11](http://latex2png.com/pngs/3c0b4a46385dda7704ee14a04f928e5c.png)

​	

이러한 가정이 있으면 위의 사후 확률 계산은 상당히 간단해진다.

​	

![eq12](http://latex2png.com/pngs/498b03bf3dd8dd97ed555792e8d63a88.png)

​	

여기서 생각해보면 종속변수가 7일 사후확률이 아니라 다른 분류값일 사후확률의 경우에도 분모에 위치한 주변우도의 값은 변하지 않는다. 새로 들어온 독립변수1 , 독립변수2 의 값은 변하지 않기 때문이다. 따라서 식을 등식이 아닌 비례 관계로 나타내면 다음과 같다.

​		

![eq13](http://latex2png.com/pngs/dac3978996b03e766c971f2382b865bf.png)

​	

즉 , 각 레이블별로 사후확률을 계산함에 있어 주변 우도는 굳이 계산할 필요 없고 우도와 사전확률의 곱만 서로 비교하면 어느 분류로 처리하는것이 확률적으로 가장 맞을지 알 수 있는 것이다. 또 , 우도와 사전확률의 확률값 또한 매 사후확률 계산시 마다 지속적으로 계산할 필요 없이 초기에 1번씩만 계산해놓으면 더 이상 불필요한 계산을 할 필요가 없어진다.



이에 나이브 베이지안 모델은 학습 데이터로부터 모델을 생성하는 과정에서 __우도 표__ 를 만들어 저장하고 데이터가 새로 들어봐 분류 작업을 처리해야할때 , 각 레이블에 대하여 사후확률을 구하는데 우도 표에서 각 우도를 참조하여 사후확률을 계산한다.

​	

> 멋있다!

​	


---


### :radio_button: 나이브 베이즈 모델의 종류 <a id="idx4"></a>



:rocket: __가우시안 모델​__ 

- 독립변수가 연속형 데이터일때 정규 분포를 가정하고 사후 확률을 계산하여 비교한다.

  

:rocket: __베르누이 모델__ 

- 독립변수가 이진 데이터일때 사용하는 모델로 나이브 가우시안 모델의 대표적 활용 예시인 스팸 메일 분류가 베르누이 모델을 활용한다.

  

:rocket: __다항 분포 모델__ 

- 독립변수가 이산형 데이터일때 사용하는 모델로 보통 출현 횟수와 관련된 데이터를 다루는 모델이다.



---

### :radio_button: 나이브 베이즈 알고리즘의 장단점 <a id="idx5"></a>



이러한 확률기반 베이즈 알고리즘의 장·단점을 간략히 정리해보자면 다음과 같다.

​	

> __Pros__ 

- 모든 독립변수가 독립적인 관계일 것이라는 나이브한 대전제에도 불구하고 실전에서 꽤 높은 성능을 보이며 문서 분류 문제의 경우 특히나 강한 면모를 보인다.

  

>__Cons__ 

- 비교적 많은 경우에 성능이 괜찮을뿐이지 독립의 가정을 하기 어려운 데이터들이나 다른 유형의 분류 문제에서는 적용하기 어렵다.
- 단어 빈도와 같이 이산형 독립변수에 대해서는 이전에 관측되지 않았던 데이터라 우도가 0이 되어버리는 경우가 발생할 위험이 있다. 이에 __스무딩__ 이라는 방법을 활용하여 정확하지 않은 zero-likelihood 문제를 해결할 수 있다. 



※ __스무딩(smoothing)__  : 1과 같은 최소의 데이터를 분자에 적용하여 0이 아닌 작은 우도값이 되도록 하는 기법 

​	

---


### :radio_button: R을 활용한 나이브 베이즈 모델링 <a id="idx6"></a>

__R을 활용한 나이브 베이지안 기반 스팸 메일 분류기 실습__ 

```R
# 스팸 메일 분류기를 통해 예측하고 결과 혼동행렬을 출력해보기

library(readxl)
library(tm)
library(SnowballC)
library(e1071)
library(gmodels)

sms_train <- read.csv('C:/임시/RData/sms_spam_ansi.txt')
sms_test <- read.csv('C:/임시/RData/햄스팸테스트.txt',header=F)

# 병합 후 전처리 -> 끝나면 다시 분할
split_point <- dim(sms_train)[1]
names(sms_test) <- c('type','text') # 칼럼명 통일 (병합에 필요)
data <- rbind(sms_train,sms_test) # 병합

table(data$type) # 4818 hams 751 spams

# 전처리 -------------------------------------------

sms_corpus <- VCorpus(VectorSource(data$text)) # 코퍼스 생성
sms_corpus_preprocessed <- tm_map(sms_corpus,removeNumbers) # 숫자 제거
sms_corpus_preprocessed <- tm_map(sms_corpus_preprocessed,content_transformer(tolower)) # 소문자 통일
sms_corpus_preprocessed <- tm_map(sms_corpus_preprocessed,removePunctuation) # 특수문자 제거
sms_corpus_preprocessed <- tm_map(sms_corpus_preprocessed,removeWords,stopwords()) # 불용어 제거

sms_corpus_preprocessed <- tm_map(sms_corpus_preprocessed,stemDocument) # 어근 변환
sms_corpus_preprocessed <- tm_map(sms_corpus_preprocessed,stripWhitespace) # 공백 제거

# 문서 단어 행렬 생성 ----------------------------------

sms_dtm <- DocumentTermMatrix(sms_corpus_preprocessed)
sms_dtm #terms :6909 #Non-/sparse entries: 43348/38416166    #Sparsity: 100% 

# 희소성 리미팅
sms_dtm <-removeSparseTerms(sms_dtm,0.999)
sms_dtm #terms :1200 #Non-/sparse entries: 34159/6648641     #Sparsity: 99%

# 빈도 수치 범주형 변환
sms_dtm <- apply(sms_dtm, MARGIN = 2, function(x)ifelse(x>0,'Y','N'))


# 학습,예측 데이터 분할

xtrain <- sms_dtm[1:split_point,]
ytrain <- factor(data$type[1:split_point])

xtest <- sms_dtm[(split_point+1):5569,]
ytest <- factor(data$type[(split_point+1):5569])

# 베이시안 분류기 생성&예측
sms_Classifier <- naiveBayes(xtrain,ytrain,laplace = 1)
ypred <- predict(sms_Classifier,xtest)

# 성능 출력
CrossTable(ypred,ytest,
           dnn=c('predicted','actual'))

# 정확도 100%
```

​	

```
             | actual 
   predicted |       ham |      spam | Row Total | 
-------------|-----------|-----------|-----------|
         ham |         6 |         0 |         6 | 
             |     1.600 |     2.400 |           | 
             |     1.000 |     0.000 |     0.600 | 
             |     1.000 |     0.000 |           | 
             |     0.600 |     0.000 |           | 
-------------|-----------|-----------|-----------|
        spam |         0 |         4 |         4 | 
             |     2.400 |     3.600 |           | 
             |     0.000 |     1.000 |     0.400 | 
             |     0.000 |     1.000 |           | 
             |     0.000 |     0.400 |           | 
-------------|-----------|-----------|-----------|
Column Total |         6 |         4 |        10 | 
             |     0.600 |     0.400 |           | 
-------------|-----------|-----------|-----------|
```

​	

---


### :radio_button: sklearn 패키지를 활용한 나이브 베이즈 모델링 <a id="idx7"></a>





---


### :radio_button: 참고자료 <a id="idx8"></a>

- [나의 첫 머신러닝/딥러닝](https://wikibook.co.kr/mymlrev/) Chapter 4.5