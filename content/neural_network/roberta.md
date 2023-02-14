---
title: "(논문 리뷰) RoBERTa: A Robustly Optimized BERT Pretraining Approach"
date: "2023-02-13 00:00:00"
slug: "roberta"
image: "neural_network/images/roberta_1.gif"
tags: [RoBERTa, 로버타, 버트, BERT, 뉴럴넷, 논문리뷰]
categories: [RoBERTa, BERT, 뉴럴넷]
---

## Abstract

- 언어 모델 간의 직접적인 비교는 (1) 연산 자원 (2) 서로 다른 데이터 (3) 하이퍼파라미터 민감도 문제로 많은 어려움이 있다.
- 본 논문에서는 BERT 논문의 선행 학습 과정을 재현하며, 특히 하이퍼파라미터와 데이터 규모에 따른 성능 차이를 탐구.
- 처음 공개된 버전의 BERT 는 매우 undertrain 되어 있었으며, 적합한 학습 과정을 통해 이후 등장한 모든 언어 모델을 상회하는 성능을 기록 (GLUE, RACE, SQuAD).

## Introduction

- 자기지도 학습 방법인 **ELMo, GPT, BERT, XLM, XLNet** 등은 많은 성능 발전을 이루어냈지만, 이러한 모델들의 어떠한 요소가 성능에 직접적인 영향을 끼쳤는지 판별하기 어려운 측면이 존재.
- 상기된 문제들로 인해 이러한 모델 간 직접적인 비교는 난이도가 높은 편.
- 본 논문에서는 하이퍼파리미터와 데이터 규모에 의한 BERT 성능의 변화를 면밀히 관찰하여 개선된 학습 방법을 제시. 구체적인 방법은 다음과 같다.
    - *더 많은 데이터로, 더 큰 배치를 구성하여, 더 긴 기간 동안 학습을 진행*
    - *NSP (Next Sentence Prediction) 과제 제거*
    - *학습 시 더욱 긴 문장 활용*
    - *정적인 (static) 마스킹 패턴을 동적으로 (dynamic) 변경*
- 연구진은 또한 **CC-News** 라는 새로운 데이터셋을 활용해 여타 모델의 데이터셋과 유사한 규모를 구축함.

## Experimental Setup

- 기존 BERT 와 동일한 하이퍼파라미터 세팅을 가져가나, peak lr 과 warmup steps 는 별도 튜닝을 거침. Adam Optimizer 의 epsilon, beta 2 값 또한 일부 변경하였다.
- 특히 학습 성능은 Adam epsilon 값에 크게 민감하게 반응.
- 기존 방식과 다르게 학습 과정에서 최대 sequence length 를 변경시키지 않았으며, 512 로 고정.
- **BookCorpus + Wikipedia 데이터셋 (약 16 GB) 에 CC-News (76 GB), OpenWebText (38 GB), Stories (31 GB) 등의 데이터셋을 추가하였다.**

## Training Procedure Analysis

- 고정된 BERT 모델을 기반으로, 연구진은 다음과 같은 실험을 진행.

### Static vs. Dynamic Masking

- BERT 는 최초 데이터 전처리 과정에서 마스킹 위치를 선정한 후, 적용된 마스크를 학습 과정에서 정적으로 활용한다.
- 기존 방식에서는 epoch 간 동일한 마스크 활용을 방지하기 위해, 각 문장에 대한 마스킹을 10 번씩 개별적으로 진행. 40 epoch 간 학습을 진행했기 때문에 모델이 완벽히 동일한 마스킹에 노출된 횟수는 총 4회 이다.
- 연구진은 이러한 방식을 sequence 가 모델에 학습될 때마다 마스킹을 진행하는 동적 방식과 비교하였으며, 이는 데이터셋이 커질수록 성능을 결정하는 핵심 요소가 될 수 있다.

| Masking                 | SQuAD 2.0 | MNLI-m | SST-2 |
|-------------------------|-----------|--------|-------|
| reference               | 76.3      | 84.3   | 92.8  |
| RoBERTa Implementation: |           |        |       |
| static                  | 78.3      | 84.3   | 92.5  |
| dynamic                 | 78.7      | 84.0   | 92.9  |

- 테이블에 따르면, 동적 마스킹은 정적 마스킹 방식에 비해 아주 약간의 성능 개선을 제공.
- (개인적인 생각이나, 알려진 바에 의해 그렇게 유효한 차이인지는 잘 모르겠다. 마치 AlexNet 의 Local Response Normalization 을 보는 것 같음)

### Model Input Format and Next Sentence Prediction

- 기존 BERT 의 인풋은 두 개의 문장으로 구성되어 있으며, 50% 의 확률로 실제 연속적으로 등장하는 문장이거나, 나머지 50% 의 확률로 서로 다른 문서에서 추출된 문장.
- 모델은 이러한 두 문장이 실제로 연속적으로 등장하는 문장인지 여부를 학습하게 되며, 이를 NSP 과제라 지칭함 (NSP 손실 함수 활용).
- 기존 연구에서는 이러한 NSP 학습 과제를 배제할 경우, 성능이 대폭 하락하는 결과를 확인하였지만, 최근 연구들은 NSP 학습의 필요성에 의문을 제기함.
    - ***Segment Pair + NSP :** 기존 BERT 의 인풋 포맷과 동일한 형태. 인풋 토큰은 두 개의 Segment Pair 를 기반으로 하며, 하나의 Segment 는 다수의 Sentence 를 포함할 수 있다. 여기서 Segment 의 최대 길이는 512 로 한정 되어 있음.*
    - ***Sentence Pair + NSP :** 인풋은 두 개의 Sentence 를 기반으로 함. 두 문장의 합이 최대 길이인 512 에 한참 못 미치기 때문에 연구진은 배치 사이즈를 키우는 방식을 선택.*
    - ***Full Sentences :** NSP Loss 가 배제되며, 512 사이즈에 맞게 한 개, 또는 복수의 문서에서 텍스트를 추출한다. 하나의 문서에 끝에 도달한 경우 (SEP) 토큰을 삽입한 후, 이후 문서로 넘어가게 됨.*
    - ***Doc Sentences :** Full Setences 방식과 동일하나, 하나의 문서에서만 텍스트를 추출한다. 512 사이즈에 미치지 못하는 경우 동적으로 배치 사이즈를 조정함.*

| Model                             | SQuAD 1.1/2.0 | MNLI-m | SST-2 | RACE |
|-----------------------------------|---------------|--------|-------|------|
| Reimplementation with NSP loss    |               |        |       |      |
| Segment Pair                      | 90.4/78.7     | 84.0   | 92.9  | 64.2 |
| Sentence Pair                     | 88.7/76.2     | 82.9   | 92.1  | 63.0 |
| Reimplementation without NSP loss |               |        |       |      |
| Full Sentences                    | 90.4/79.1     | 84.7   | 92.5  | 64.8 |
| Doc Sentences                     | 90.6/79.7     | 84.7   | 92.7  | 65.6 |
| Reference                         |               |        |       |      |
| Bert Base                         | 88.5/76.3     | 84.3   | 92.8  | 64.3 |
| XLNet Base (K = 7)                | _/81.3        | 85.8   | 92.7  | 66.1 |
| XLNet Base (K = 6)                | _/81.0        | 85.6   | 93.4  | 66.7 |

- Sentence Pair 는 Segment Pair 에 비해 Downstream Task (전이 학습 영역) 에서 낮은 성능을 보였으며, 이는 모델이 거리가 먼 단어 간 dependency 를 학습하지 못해서 일 것이라 추측.
- NSP loss 를 제거한 후 성능이 소폭 상승하였으며, 이는 기존 방법이 Segment Pair 방식을 그대로 유지했기 때문이라 추측할 수 있다.
- Doc Sentences 가 Full Sentences 에 비해 나은 성능을 보였지만, 변동적인 배치 사이즈로 인해 RoBERTa 모델은 Full Sentences 방식을 사용 (다른 모델과 비교 조건을 통일하기 위함).

### Training with Large Batches

- Learning Rate 만 적합하게 조정된다면, mini-batch 사이즈를 키우는 것은 학습 속도와 최종 과제 수행 능력에 긍정적인 영향을 끼친다.
- 기존 BERT 모델은 256 Batch Size/1M Steps 세팅 값으로 학습을 진행. 이는 연산 비용 차원에서 (1) 2K Batch Size/125K Steps, (2) 8K Batch Size/31K Steps 와 동일하다.

| batch size | steps | learning rate | perplexity | MNLI-m | SST-2 |
|------------|-------|---------------|------------|--------|-------|
| 256        | 1M    | 1e-4          | 3.99       | 84.7   | 92.7  |
| 2K         | 125K  | 7e-4          | 3.68       | 85.2   | 92.9  |
| 8K         | 31K   | 1e-3          | 3.77       | 84.6   | 92.8  |

- 연구진은 배치 사이즈를 키울 경우, 모델의 perplexity (cross-entropy 기반 loss metric) 가 감소한다는 점을 발견. 또한 end-task 의 정확도 또한 향상된다는 점을 발견한다.
- 배치 사이즈가 큰 경우 병렬 처리가 쉬워진다는 장점 또한 있다. 이후 연구진은 RoBERTa 학습에 8K Batch Size 를 적용함.

### Text Encoding

- **Byte-Pair Encoding (BPE)** 란 캐릭터와 단어 레벨 representation 의 중간에 있는 인코딩 방식. 실제 단어가 아닌 단어의 부분들을 기반으로 인코딩을 수행한다.
- 보통의 BPE 단어 사전은 10K~100K 정도의 규모를 가지는데, 이 중 대부분은 유니코드 캐릭터에 의한 것이며, 이를 바이트 기반으로 변경하여 unknown token 을 필요로 하지 않는 50K 규모의 단어 사전을 구축할 수 있다.
- 기존 BERT 논문은 캐릭터 레벨의 30K 규모 BPE 사전을 활용하였으나, 연구진은 이를 바이트 기반의 50K BPE 사전으로 변경. 때문에 기존 방식에 적용된 데이터 전처리를 필요로 하지 않는다.
- 이로 인해 BERT Base 는 파라미터 수가 약 15M, BERT Large 는 약 20M 개 상승.
- 성능 평가 면에서는 바이트 기반 BPE 가 약간 낮은 성능을 보이나, 연구진은 바이트 기반 BPE 의 장점이 단점을 상쇄한다고 판단, 이를 RoBERTa 에 적용한다.

## RoBERTa

- 동적 마스킹, Full Setences w/o NSP loss, 증가한 mini-batch 사이즈, 바이트 레벨 BPE 등을 적용한 BERT 모델을 연구진은 Robustly optimized BERT approach (RoBERTa) 라 명명.
- 또한 RoBERTa 는 (1) 데이터 크기와 성질 (2) 학습의 정도가 모델 성능에 끼치는 영향을 탐구한다.
- XLNet 의 경우 기존 BERT 모델에 비해 10배 많은 데이터를 활용해 학습되었으며, 배치 사이즈의 경우 8배가 컸던 반면 학습 step 은 1/2 정도의 규모였음으로 BERT 에 비해 약 4배 정도의 시퀀스에 노출된 것.
- 보다 직접적인 비교를 위해 연구진은 규모가 유사한 데이터를 활용해 다음 세개의 모델을 테스트했다.

| Model                    | data  | batch size | steps | SQuAD     | MNLI-m | SST-2 |
|--------------------------|-------|------------|-------|-----------|--------|-------|
| RoBERTa                  |       |            |       |           |        |       |
|   with BOOKS + WIKI      | 16GB  | 8K         | 100K  | 93.6/87.3 | 89.0   | 95.3  |
|   + additional data      | 160GB | 8K         | 100K  | 94.0/87.7 | 89.3   | 95.6  |
|   + pretrain longer      | 160GB | 8K         | 300K  | 94.4/88.7 | 90.0   | 96.1  |
|   + pretrain even longer | 160GB | 8K         | 500K  | 94.6/89.4 | 90.2   | 96.4  |
| BERT Large               |       |            |       |           |        |       |
|   with BOOKS + WIKI      | 13GB  | 256        | 1M    | 90.9/81.8 | 86.6   | 93.7  |
| XLNet Large              |       |            |       |           |        |       |
|   with BOOKS + WIKI      | 13GB  | 256        | 1M    | 94.0/87.8 | 88.4   | 94.4  |
|   + additional data      | 126GB | 2K         | 500K  | 94.5/88.8 | 89.8   | 95.6  |

- 이외에도 RoBERTa 모델은 GLUE, SQuAD, RACE 등의 벤치마크 과제에서 state-of-the-art 성능을 기록. 주로 XLNet 과 비교되었는데, 성능 차이가 아주 큰 편은 아님.
- 특히 GLUE 벤치마크의 경우, train set 을 활용한 환경에서 9개 과제 모두 가장 높은 성능을 기록했지만 test set 기반의 리더보드에서는 4개 과제에서만 가장 높은 성능을 기록했다. 하지만 리더보드의 대부분 모델과 다르게 RoBERTa 는 복수과제를 활용한 fine-tuning 을 진행하지 않음.
- SQuAD 또한 비슷한 양상을 보이는데, 리더보드에서 XLNet + SG-Net Verifier 에 비해 약간 낮은 성능을 기록했다 (외부 데이터를 활용하지 않았기 때문이라고 하는데, 뭔가 자꾸 사족이 붙는 느낌).