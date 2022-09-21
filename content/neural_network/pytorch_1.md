---
title: "PyTorch Deep Learning - Tensor"
description: "파이토치 노드를 구성하는 Tensor 소개"
date: "2022-06-11 00:00:00"
slug: "pytorch_1"
image: "neural_network/images/pytorch.jpeg"
tags: [pytorch, 파이토치, 뉴럴넷, pytorch 사용법, 신경망, 머신러닝, 텐서, pytorch tensor]
categories: [Pytorch, Neural Network]
---
## 소개

`tensor`란 `numpy`와 유사하게 다차원 행렬을 다룰수있는 PyTorch 패키치의 자료구조다. 신경망 개론 수업에서 `numpy` 패키지를 활용해 node와 weight, bias 등을 구현하고는 하는데 같은 개념의 연산을 **GPU 등 적합한 하드웨어 자원을 통해 수행하고자 할때** `tensor`를 이용하게 된다. Tensorflow 패키지 또한 동일한 개념과 이름을 가진 `tf.Tensor`를 사용한다.

## Tensor 생성

값이 비어있는 tensor를 생성하기 위해서는 `torch.empty()` 메소드를 다음과 같이 호출한다.

```
 import torch

 x1 = torch.empty(1) # scalar 생성
 x2 = torch.empty(3) # 1d vector 생성
 x3 = torch.empty(2, 3) # 2d matrix 생성
 x4 = torch.empty(2, 2, 3) # 3d matrix 생성
```

유사하게 0과 1 사이의 랜덤한 값이 부여된 tensor를 사용하기 위해서는 `torch.rand()` 함수를, 0값의 경우 `torch.zeros()` 함수를, 1값의 경우 `torch.ones()` 함수를 차원값과 함께 호출한다 (`numpy`와 유사하게 구성).

```
 x1 = torch.rand(2, 2, 3)
```

## 데이터 타입

별도로 데이터 타입을 지정하지 않은 경우 위 저장된 변수의 데이터 타입은 `torch.float32`로 자동 지정된다.

```
 x1 = torch.ones(2, 2, 3)
 print(x1.dtype) # output: torch.float32
```

다른 데이터 타입을 사용하고자 할 경우 `tensor` 생성시 `dtype` 매개변수로 다음과 같이 지정이 가능하다.

```
 x1 = torch.ones(2, 2, 3, dtype=torch.int)
 print(x1.dtype) # output: torch.int

 x2 = torch.ones(2, 2, 3, dtype=torch.double)
 print(x2.dtype) # output: torch.double

 x3 = torch.ones(2, 2, 3, dtype=torch.float16)
 print(x3.dtype) # output: torch.float16
```

## 행렬 구조 확인
생성된 `tensor`의 구조는 `size` 함수를 통해 확인이 가능하다.

```
 x1 = torch.ones(2, 2, dtype=torch.int)
 print(x1.size) # output: torch.Size([2, 2])
```

## 불러오기 기능

### Python List
`numpy` 패키지의 `np.array` 함수와 동일하게 행렬구조를 가진 파이썬 `list` 로부터 `tensor` 생성을 지원한다. `torch.tensor()` 함수의 매개변수로 `list` 를 넣어주는 일반적인 구조다.

```
 x1 = torch.tensor([2.5, 0.1])
```

### Numpy Array
자연스럽게 `numpy.array` 를 활용한 `tensor` 생성 또한 `torch.from_numpy()` 함수를 통해 다음과 같이 지원한다.

```
 import numpy as np

 x1_np = np.array([2,5, 0.1])
 x1 = torch.from_numpy(x1_np)
```

반대로 `tensor` 에서 `numpy` 로의 변환은 다음과 같이 수행한다.

```
 x1 = torch.tensor([2.5, 0.1])
 x1_np = x1.numpy()
```

여기서 유의할 부분은 **`tensor` 의 메모리 위치가 GPU 가 아닌 CPU 일 경우, x1의 변형은 x1_np 에 그대로 반영**되게 된다는 점이다. 이는 위의 두개 예시 (`tensor` -> `numpy`, `numpy` -> `tensor`)에 공통적으로 적용된다.

```
 x1 = torch.tensor([2.5, 0.1])
 x1_np = x1.numpy()

 x1.add_(1)
 print(x1_np) # output: [3.5, 1.1]
```

CUDA 지원 하드웨어 가용이 가능한 경우, 다음 두가지 방식을 통해 `tensor` 저장 위치를 GPU로 설정할 수 있다.

```
 if torch.cuda.is_available():
    device = torch.device("cuda")
    
    x1 = torch.tensor([2.5, 0.1], device=device) # 1. 생성 시 GPU 메모리 가용

    x2 = torch.tensor([2.5, 0.1])
    x2 = x2.to(device) # 2. 생성 후 GPU 메모리 가용

    x3 = x1 + x2
    x3 = x3.to("cpu") # CPU 메모리 가용
```

## 행렬 연산
### 일반적인 연산
덧셈, 곱셈과 같은 기본적인 행렬 연산 방식또한 `numpy`와 크게 다르지 않다. `+`, `*` 등의 수학 기호, 또는 `torch.add()`, `torch.mul()` 등의 함수를 호출해 연산을 수행할 수 있다.

`numpy`와 동일하게 내적 연산을 위해서는 `torch.mul()` 이 아닌 다른 함수를 호출한다. 이와 관련된 내용은 이후 글에서 언급할 예정.

```
 x = torch.rand(2, 2)
 y = torch.rand(2, 2)

 # 덧셈
 z1 = x + y
 z2 = torch.add(x, y)

 # 뺄셈
 z3 = x - y
 z4 = torch.sub(x, y)

 # 곱셈, element-wise
 z5 = x * y
 z6 = torch.mul(x, y)

 # 나눗셈, element-wise
 z7 = x / y
 z8 = torch.div(x, y)
```

### 바꿔치기 연산 (In-Place Operation)
`torch` 는 `.add_`, `.sub_` 등 '_' 접미사가 붙은 바꿔치기 연산 함수를 제공한다. 바꿔치기 라는 단어에서 유추 가능하듯 이는 **타겟 변수의 값을 바꾸는 효과**를 가지게 된다.

```
 x = torch.rand(2, 2)
 y = torch.rand(2, 2)

 y.add_(x) # y 변수의 값이 y + x 의 output으로 변경
```

## 슬라이싱
슬라이싱의 경우 또한 `numpy` 패키지와 동일한 방법을 고수한다. 2차원 행렬구조의 경우 `x[i, j]` 와 같은 포맷으로 `i` 번째 로우, `j` 번째 컬럼을 리턴하며, `x[i1:i2, j1:j2]` 와 같이 범위 설정이 가능하다.

유의가 필요한 부분은 1개의 값이 리턴될때 `tensor` 오브젝트가 아닌 기입된 실제 값을 보고싶다면 `item()` 함수를 별도로 호출해야 하며, 해당 함수는 `tensor` 에 1개 값만 들어있을때 사용 가능하다는 점이다.

```
 x = torch.rand(5, 2)

 print(x[:, 0]) # 1번 컬럼 슬라이스
 print(x[0, :]) # 1번 로우 슬라이스

 print(x[1, 1]) # 2번 로우, 2번 컬럼 슬라이스 (tensor 형태 유지)
 print(x[1, 1]).item() # 2번 로우, 2번 값
```

## 행렬 구조 변경 (Reshaping)
`np.reshape`이 아닌 `view` 함수를 이용하게 된다. 매개변수로 들어가는 **차원의 element 수은 항상 input `tensor`의 element 수와 같아야 하며** (예. (4, 4) -> (2, 8)), 마지막 숫자가 유추 가능한 경우 -1 으로 매개변수를 대체할 수 있다 (하단 예시 참조).

```
 x = torch.rand(4, 4)

 y1 = x.view(16) # x.view(-1)와 동일
 y2 = x.view(2, 8) # x.view(-1, 8)와 동일
```