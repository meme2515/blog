---
title: "Transformer 네트워크 개념 소개"
description: "짧은 개념 소개 및 Self-Attention, Multi-Head Attention, 네트워크 구조 상세 설명"
date: "2022-09-18 00:00:00"
slug: "transformer"
image: "neural_network/images/transformer_1.bmp"
tags: [transformer, neural net, 트랜스포머 네트워크, 트랜스포머]
categories: [transformer, 트랜스포머, neural network]
---
## TL;DR

- 기존 RNN 기반의 모델의 느린 학습 속도 문제를 해결하기 위해 2017년 구글이 주도한 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 논문에서 제시한 NLP 신경망 구조이다.
- LSTM, GRU 등의 NLP 도메인의 선발주자들이 해결하고자 한, 문장 속 단어간의 관계성을 Attention 이라는 개념을 통해 해결했다. 이는 기존 RNN 모델과 달리 병렬 처리가 가능한 구조를 가지고 있다.
- 논문은 통역을 적용 영역으로 다루었으며, 따라서 주어진 문장을 해석하는 Encoder, 새로운 문장을 생성하는 Decoder 로 나뉘어진 구조를 가지고 있다. 여기서 Encoder 의 구조를 차용한 것이 BERT, Decoder 의 구조를 차용한 것이 GPT 모델이다.
- 본 글을 작성하는 시점에 가장 활발하게 사용되고 있는 NLP 신경망 구조이다. 

## 등장 배경

| ![alt text](neural_network/images/transformer_3.bmp) |
|:--:|
| Fig 1. RNN 모델 구조 예시 |

본 모델 구조가 발표되기 이전의 주류 자연어 신경망 구조는 기본적으로 단어, 또는 글자를 순차적으로 처리하는 구조를 가지고 있다. Input 단어가 신경망을 활성화해 다음 단어를 예측하는 방식이다. 언어는 단어의 순차적인 조합을 통해 구성되고, 또한 Input 과 Output 문장의 길이가 항상 다르기 때문에 이는 가장 자연스러운 방식으로 여겨졌는데, 이러한 구조는 학습이 느리고, 간격이 먼 단어 간의 관계를 해석하지 못한다는 단점을 가지고 있다. 이 중 후자의 문제를 해결하기 위해 연구자들은 Memory Unit 이라는 개념을 고안해 일종의 단어 기억 장치를 만들어낸다. 이 Memory Unit 은 간격이 먼 단어에 대한 정보를 저장해 문장의 맥락을 보다 정확하게 해석하는 것에 의의를 두고 있으며, 널리 알려진 GRU, LSTM 과 같은 셀 구조가 이에 해당한다.

| ![alt text](neural_network/images/transformer_2.bmp) |
|:--:|
| Fig 2. 베이스 RNN, LSTM 및 GRU 셀 구조 예시 |

하지만 셀의 구조가 복잡해질수록 순차적 학습은 연산 부담이 크다는 문제가 악화된다. 한 개 셀에서 처리할 정보가 그만큼 늘어나니 이는 당연한 결과인데, 자연어 처리 분야가 발전하며 연구자들은 보다 방대한 데이터를 학습시키고자 했고, 이를 위해 병렬 처리가 가능한 NLP 모델인 트랜스포머를 2017년 발표하게된다.

## Attention

트랜스포머 구조의 혁신적인 점은 마치 비전 분야의 convolution 개념과 같이 언어에 대한 병렬처리를 가능하게 했다는 점이다. 이와 같은 처리 방식은 "attention" 이라고 불리고 (맥락 이해를 위해 문장의 각 단어에 대한 "집중도" 를 연산한다는 의미), 세부적으로는 Self-Attention 과 Multi-Head Attention 으로 그 구조를 나눌 수 있다. Self-Attention 은 문장의 각 단어에 대해 attention 점수, 즉 가중치를 매기는 과정을 가르키며, Multi-Head Attention 은 이러한 점수 부여 과정을 여러번 수행하는 것이라고 짧게 설명할 수 있다.

아래 설명에서는 관련 [Coursera 강의](https://www.coursera.org/learn/nlp-sequence-models) 에서 등장하는 번역 문제를 예시로 다루고있다. Jane visite l'Afrique en septembre 라는 불어 문장을 Janes visits Africa in September 라는 영어 문장으로 번역하는 예시이다.

### Self-Attention

| ![alt text](neural_network/images/transformer_4.bmp) |
|:--:|
| Fig 3. qKV 매트릭스 연산 과정 예시 |

(1) Jane (2) visite (3) l'Afrique (4) en (5) septembre 와 같은 형태로 토큰화된 각 단어는 q, K, V 라는 세가지 특성을 부여받는다. 여기서 qKV 특성이란 데이터베이스의 query, key, value 에 해당하는 개념인데, 직관적인 비유를 들자면 (3) l'Afrique 에서 어떠한 일이 일어났는가? 를 l'Afrique 의 $q^3$ 특성이라고 가정했을때 $q^3 \cdot k^1$ 은 이 질문에 대한 답변으로 (1) Jane 이라는 단어의 적합도를 나타내게 된다 ($q^3 \cdot k^2$ 는 (2) visite 의 적합도, $q^3 \cdot k^3$ 는 (3) l'Afrique 의 적합도와 같은 식).

이렇게 연산된 적합도는 softmax 함수를 통해 정규화되며, 정규화된 각 질문의 적합도에 value 값을 곱해줌으로서 질문에 적합한 정보를 추출하게 되는 원리이다. 수식으로 표현하면 다음과 같다.

$$
A(q, K, V) = \Sigma_i \frac{\text{exp}(q \cdot k^i)}{\Sigma_j \text{exp}(q \cdot k^j)} v^i
$$

Fig 3. 에서 확인할 수 있듯이 인풋은 임베딩된 단어 벡터이며, 도출되는 qKV 특성 또한 각각의 벡터이다. 따라서 마치 convolution layer 와 같이 행렬 곱셈을 위한 커널 학습이 가능하며 (learned matrix), 병렬처리가 가능해지는 것이다. 또한 도출된 $A(q, K, V)$ 도 벡터의 형태를 유지하게 되며, 각각의 단어에 대해 Attention Vector 를 추출한다 ((1) Jane -> $A^1$, (2) visite -> $A^2$, etc.).

### Multi-Head Attention

| ![alt text](neural_network/images/transformer_6.bmp) |
|:--:|
| Fig 4. Multi-Head Attention 예시 |

Self-Attention 의 개념을 이해했다면, Multi-Head Attention 은 이러한 Self-Attention 을 여러번 수행하는 과정이라고 설명할 수 있다. 한번의 Self-Attention 과정은 "Head" 로 표현되며, 8-Head Attention 이란 Self-Attention 이 8번 수행된 결과가 되는 식이다. 여기서 각각의 Head 는 qKV 특성에 서로 다른 가중치를 적용하여 구분되며, 개념적으로는 (1) 무엇을 했는가? (2) 언제 했는가? 와 같이 질문의 내용이 변형되는 과정이다.

이렇게 구해진 n개의 Attention Vector 를 이어붙인 정보를 통해 Multi-Head Attention 아웃풋 행렬을 도출하며, 실제 학습시에는 for-loop 이 아닌 병렬처리를 수행한다. 질문의 내용이 다양해지기 때문에 도출된 결과는 단일 Attention Vector 보다 더 깊이있는 정보를 가지게된다.

## 모델 구조

| ![alt text](neural_network/images/transformer_1.bmp) |
|:--:|
| Fig 5. Transformer 네트워크 구조 |

트랜스포머 네트워크는 machine translation, 즉 언어 간 해석 문제를 염두하고 만들어졌기 때문에 한개의 언어를 해석하는 encoder 블록, 다른 언어를 생성하는 decoder 블록으로 구분할 수 있다. 

### Encoder

Fig 5. 의 좌측 도식화에 해당하는 부분이다. 우선 임베딩된 인풋 문장 Jane visite l'Afrique en septembre (Input Encoding) 에서 Q, K, V 특성을 추출한 후, 이에 Multi-Head Attention 을 적용해 문장에 대한 해석 정보가 담긴 매트릭스를 생성한다 (여기서 Q, K, V 특성은 세갈래의 화살표로 표기되어있다). 이후 일반적인 신경망 구조를 통해 해당 매트릭스에서 중요한 정보를 선별하며, 여기까지의 과정을 N 번 반복한다. Multi-Head Attention 과 Feed Forward 레이어에서 생성되는 아웃풋에는 여타 신경망의 [Batch Normalization](https://meme2515.github.io/neural_network/batchnorm/) 과 유사한 Add & Normalization 이 적용된다. 

### Decoder

Fig 5. 의 우측 도식화에 해당하는 부분이다. 인풋엔 생성하고자 하는 문장의 단어들이 순차적으로 입력되며, 최초엔 이러한 문장 정보가 없기 때문에 start-of-sentence 라는 의미의 SOS 토큰 등을 활용하게 된다. 이후 임베딩된 단어 인풋에서 Q, K, V 특성을 추출 후, 이에 Multi-Head Attention 을 적용한다. Encoder 와 다른 점은 이 Multi-Head Attention 의 아웃풋이 해석 정보를 가진 매트릭스가 아닌 Q, 즉 인풋 단어에 대한 복수의 질문 정보를 담은 매트릭스라는 점이다. 

앞서 설명한 Encoder 는 궁극적으로 이에 대응하는 K, V 매트릭스를 생성하게 된다 (양 블록을 있는 두개의 화살표로 표현). 따라서 Decoder 의 질문에 대응하는 답변을 줄 수 있게 설계된 것이다. 이러한 두 갈래의 Q, K, V 특성은 다시 Multi-Head Attention 레이어에 적용되며, 이를 통해서 아웃풋된 두 언어의 상관성 정보를 담은 매트릭스는 Feed Forward 레이어를 통해 중요한 정보만을 남기게 된다. 여기까지의 과정 또한 N 번 반복 후, Softmax Activation 을 통해 여러 단어 중 가장 알맞은 단어를 선택하게 된다. 

Decoder 블록 또한 각 레이어 별로 Add & Normalization 이 적용된다.

### Positional Encoding

트랜스포머 구조를 처음 접할때 가장 강조되는 부분이 CNN 과의 유사성이다. 병렬 처리를 통한 연산 속도의 개선은 이미 설명했지만, 언어 영역에서 이러한 유사성이 가지는 단점은 없을까?

| ![alt text](neural_network/images/transformer_5.bmp) |
|:--:|
| Fig 6. Translation Invariance |

비전 분야에서 CNN 이 각광받는 이유 중 하나는 Convolution Layer 내의 커널을 이미지의 여러 영역에 동일하게 적용하기 때문이다. 이로 인해 고양이를 분류하도록 학습된 커널은 고양이가 이미지의 좌하단, 우상단, 중앙 등 어떠한 영역에 있던 문제없이 고양이의 특성을 추출해 그 존재 유무를 추측할 수 있게된다. 입력 위치가 변해도 출력은 변하지 않는다는 의미이며, 학술적으로 이는 Translation Invariance 라 칭한다 (관심이 있다면 [해당 Medium 글](https://seoilgun.medium.com/cnn%EC%9D%98-stationarity%EC%99%80-locality-610166700979)에서 보다 상세한 내용을 확인할 수 있다). 

트랜스포머의 qKV 매트릭스 추출 과정 또한 이와 유사하다. CNN 의 커널과 유사하게 동일한 learned matrix 를 각 단어 벡터에 곱하게 되며, 이로 인해 단어의 위치와 무관하게 qKV 특성을 추출할 수 있게 되는 것이다. 이는 l'Afrique 라는 단어가 문장의 어느 위치에서 등장하던 같은 질문을 던지고, 다른 질문에 대한 동일한 답을 준다는 점을 의미한다. 하지만 언어에서 단어의 위치는 중요한 정보를 담고 있다. 예시로 같은 단어일지라도 문맥과 위치에 따라 주어가 될 수도, 목적어가 될 수도 있기 때문이다.

이렇듯 유실된 단어의 위치 정보를 활용하기 위해 논문의 저자는 position encoder 라는 개념을 소개한다. 우선 (1) Jane 과 같은 각 토큰은 길이 4의 벡터에 임베드 된다고 가정하자. Position 인코딩은 이와 동일한 길이의 벡터에 해당 토큰의 위치 정보 (이 경우 1) 를 표현한 후, 이를 (1) Jane 을 해석한 기존 임베드에 더해줌으로서 의미, 맥락과 더불어 위치 정보 또한 벡터에 추가하게 된다.

| ![alt text](neural_network/images/transformer_8.png) |
|:--:|
| Fig 7. Positional Encoding 벡터 덧셈 예시 |

그렇다면 토큰의 위치 정보는 어떻게 벡터로 표현될 수 있을까? 사실 각 위치에 대한 벡터의 값이 일정하고, 구분될 수 있다면 생성 과정 자체는 크게 중요하지 않다. 다만 감안할 부분은 (1) Position encoding 값이 기존 임베딩 값을 지나치게 변형하면 안된다는 점 (2) 연산 과정이 복잡해 학습 시간을 지연시키면 안된다는 점 등을 들 수 있다. 논문은 sine, cosine 함수를 활용해 다음과 같은 position encoding 생성 함수를 제안한다.

$$
\text{PE}_{\text{pos}, 2i} = \text{sin}(\frac{\text{pos}}{10000^{\frac{2i}{d}}})
$$

$$
\text{PE}_{\text{pos}, 2i+1} = \text{cos}(\frac{\text{pos}}{10000^{\frac{2i}{d}}})
$$

여기서 d 는 타깃 벡터의 길의 (예시의 경우 d=4), i 는 타깃 벡터에 존재하는 모든 인덱스 (i=[1, 2, 3, 4]), pos 는 단어의 위치 (Jane 은 첫 토큰임으로 pos=1) 를 나타낸다. 함수가 두개인 이유는 홀수 인덱스의 경우 하단의 cosine 함수를, 짝수 인덱스의 경우 상단의 sine 함수를 활용하기 위함이다. 이러한 각 변수와 함수간의 관계는 다음과 같이 도식화할 수 있다. 

| ![alt text](neural_network/images/transformer_7.png) |
|:--:|
| Fig 8. Positional Encoding 함수 아웃풋 예시 |

즉, 한정된 레인지에서 추출된 position encoding 값을 통해 기존 임베딩 정보를 지나치게 왜곡하지 않는 선에서 위치 정보를 추가하는 것이다.

### Masking

Decoder 블록의 첫 Multi-Head Attention 레이어는 Masking 이라는 매커니즘을 통해 학습된다. 간단히 말해, 이미 완성된 영문장 Jane visits Africa in September 을 마스킹 처리하여 "Jane visits Africa __ __" 라는 인풋이 "Jane visits Africa in __" 이라는 아웃풋을 생성하도록 유도하고, 학습하는 것이다. 관심이 있다면 [해당 Medium 글](https://medium.com/analytics-vidhya/masking-in-transformers-self-attention-mechanism-bad3c9ec235c) 에서 더욱 자세한 내용을 확인할 수 있다.