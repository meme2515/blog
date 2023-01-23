---
title: "Full Stack Deep Learning 2022 부트캠프 - Week 6"
description: "Continual Learning"
date: "2022-12-06 00:00:00"
slug: "fsdl_6"
image: "mlops/images/fsdl_6_title.png"
tags: [FSDL, Full Stack Deep Learning, MLOps, 부트캠프, fsdl 2022, fsdl 2022 후기, fsdl 후기, full stack deep learning 2022, 풀스택딥러닝]
categories: [FSDL, MLOps, 부트캠프]
draft: "true"
---
- [YouTube](https://www.youtube.com/watch?list=PL1T8fO7ArWleMMI8KPJ_5D5XSlovTW_Ur&v=nra0Tt3a-Oc), [Lecture Notes](https://fullstackdeeplearning.com/course/2022/lecture-6-continual-learning/), [Slides](https://drive.google.com/file/d/10fDYIEELIeT3Nju001GTAxM_YYUDFMpB/view)

## Lecture 내용 요약

- [FSDL 2022 Course Overview](https://meme2515.github.io/mlops/fsdl/)
- [Lecture 1 - When to Use ML and Course Vision](http://meme2515.github.io/mlops/fsdl_1/)
- [Lecture 2 - Development Infrastureture & Tooling](http://meme2515.github.io/mlops/fsdl_2/)
- [Lecture 3 - Troubleshooting & Testing](http://meme2515.github.io/mlops/fsdl_3/)
- [Lecture 4 - Data Management](http://meme2515.github.io/mlops/fsdl_4/)
- [Lecture 5 - Deployment](http://meme2515.github.io/mlops/fsdl_5/)
- [Lecture 6 - Continual Learning](http://meme2515.github.io/mlops/fsdl_6/)
- [Lecture 7 - Foundation Models](http://meme2515.github.io/mlops/fsdl_7/)
- [Lecture 8 - ML Teams and Project Management](http://meme2515.github.io/mlops/fsdl_8/)
- [Lecture 9 - Ethics](http://meme2515.github.io/mlops/fsdl_9/)

## Overview

- 실전의 ML 제품은 항상 변동하는 통계 분포를 다루기 때문에 academic setting 에서 다루는 방식과는 달리 **continual learning, 즉 지속적 학습이 동반되어야 한다**.

- **Data Flywheel** : 유저가 유입될수록 더 많은 데이터가 쌓이며, 이러한 데이터를 활용해 개선된 ML 모델을 제공할 수 있도록 하는 과정을 일컫는다. 개선된 모델은 다시 유저들이 더 나은 제품 경험을 할 수 있도록 도우며, 이로 인해 또 다시더 많은 유저들이 유입되는 것.

- Andrej Karpathy 는 이상적인 Data Flywheel 체계를 [Operation Vacation](https://www.youtube.com/watch?v=hx7BXih7zx8) 이라 설명한다. 즉, 파이프라인이 구축된 후에는 Data Scientist 가 크게 관여할 부분이 없다.

![alt text](mlops/images/fsdl_6_1.png)

- 모델을 개발하기 위해서는 우선 데이터에 대한 수집/정제/레이블 처리가 이루어져야 한다. 이러한 데이터를 기반으로 모델을 학습하며, 학습된 모델을 평가해 평가 결과에 따른 모델 개선 과정이 수반된다. 이러한 과정을 거쳐야만 배포 가능한 minimum viable model 이 완성된다고 볼 수 있는 것. 

![alt text](mlops/images/fsdl_6_2.png)

- 문제는 모델 배포 후 발생한다. 배포 시 성능에 대한 명확한 측정 방법이 없기 때문에 모델러는 일정 케이스에 대해 모델이 정상적으로 작동하는지에 대한 테스트를 진행한 후, 큰 문제가 발생하지 않는다면 다른 작업을 시작하게 된다.

![alt text](mlops/images/fsdl_6_3.png)

- 하지만 대부분의 문제는 ML Engineer 에 의해 발견되지 않는다. 비즈니스 유저, 혹은 제품 담당자가 유저의 컴플레인을 수리한 후, 이에 대한 조사 차원의 작업이 시작된다. 이 단계에서 이미 회사는 문제를 처리해야 하는 입장에 놓이기에 비용이 발생하게 된다.

![alt text](mlops/images/fsdl_6_4.png)

- 결국 

## How to Think About Continual Learning

## Periodic Retraining

## Iterating On Your Retraining Strategy

## Retraining Strategy

## The Continual Improvement Workflow