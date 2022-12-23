---
title: "Full Stack Deep Learning 2022 부트캠프 - Week 8"
description: "ML Teams and Project Management"
date: "2022-12-08 00:00:00"
slug: "fsdl_8"
image: "mlops/images/fsdl_8_title.png"
tags: [FSDL, Full Stack Deep Learning, MLOps, 부트캠프, fsdl 2022, fsdl 2022 후기, fsdl 후기, full stack deep learning 2022, 풀스택딥러닝]
categories: [FSDL, MLOps, 부트캠프]
---

- [YouTube](https://www.youtube.com/watch?v=a54xH6nT4Sw&list=PL1T8fO7ArWleMMI8KPJ_5D5XSlovTW_Ur), [Lecture Notes](https://fullstackdeeplearning.com/course/2022/lecture-8-teams-and-pm/), [Slides](https://drive.google.com/file/d/1o2x8ywivp555__AEbLLI28BsiQmHobOh/view)

## Lecture 내용 요약

- [Lecture 1 - When to Use ML and Course Vision](http://meme2515.github.io/mlops/fsdl_1/)
- [Lecture 2 - Development Infrastureture & Tooling](http://meme2515.github.io/mlops/fsdl_2/)
- [Lecture 3 - Troubleshooting & Testing](http://meme2515.github.io/mlops/fsdl_3/)
- [Lecture 4 - Data Management](http://meme2515.github.io/mlops/fsdl_4/)
- [Lecture 5 - Deployment](http://meme2515.github.io/mlops/fsdl_5/)
- [Lecture 6 - Continual Learning](http://meme2515.github.io/mlops/fsdl_6/)
- [Lecture 7 - Foundation Models](http://meme2515.github.io/mlops/fsdl_7/)
- [Lecture 8 - ML Teams and Project Management](http://meme2515.github.io/mlops/fsdl_8/)
- [Lecture 9 - Ethics](http://meme2515.github.io/mlops/fsdl_9/)

## Why is this hard?

제품을 만드는 과정은 다음과 같은 이유로 어렵고 험난하다.

- 좋은 인력을 채용해야 한다.
- 채용된 인력을 관리하고, 성장시켜야 한다.
- 팀의 결과물들을 관리하고, 방향성을 맞춰야 한다.
- 장기간 제품에 영향을 끼칠 기술적인 요소들을 적절히 선택해야 한다.
- 리더십의 기대치를 관리해야 한다.
- 필요 조건을 정의하고, 관련 인력에게 커뮤니케이션 해야 한다.

머신러닝은 이와 같은 과정을 더욱 어렵게 만든다.

- ML 관련 인력은 아직 시장에 많지 않고, 채용 비용이 비싼 편이다.
- ML 팀에는 상대적으로 다양한 롤 (role) 이 존재한다.
- 대부분 프로젝트의 타임라인이 불명확하고, 실패 확률이 높다.
- 분야가 빠르게 발전하고 있으며, ML 제품 관리는 어렵고 아직 체계가 확립되지 않았다.
- 리더십은 대게 ML 기술을 깊게 이해하지 못한다.
- 비전문 인력이 ML 제품의 실패 원인을 파악하기 어렵다.

8주차 강의는 다음과 같은 내용을 담고있다.

- ML 분야의 롤 (role) 과 롤 별 필요 전문성
- ML 엔지니어 채용 방식 (그리고 취업 방식)
- ML 팀의 구성 방식과 전체 조직과 협업하는 법
- ML 팀과 ML 제품을 운영하는 법
- ML 제품 기획 시 고려 요소

## Roles

![alt text](mlops/images/fsdl_8_1.png)

**대표적인 롤**

- **ML Product Manager** : ML Product Manager 는 ML 팀, 비즈니스 영역, 제품 유저, 데이터 오너 간 협업하여 도큐먼트를 작성하고, 제품의 뼈대를 세우고, 계획을 수립하고, 그 중 작업의 우선순위를 정해 ML 프로젝트를 진행하는 역할을 맡는다.

- **MLOps/ML Platform Engineer** : 모델 배포 과정을 보다 쉽고 스케일링이 가능하도록 인프라를 설계하는 역할을 맡는다. 이후 AWS, GCP, Kafka, 혹은 다른 ML 툴을 활용해 배포된 제품의 인프라를 관리하는 역할을 수행.

- **ML Engineer** : 모델을 학습하고 배포하는 역할. TensorFlow, Docker 등의 툴을 활용해 예측 시스템을 실제 데이터에 적용한다.

- **ML Researcher** : 예측 모델을 학습하는 역할을 맡지만, 주로 최신 모델을 실험적인 환경에서 사용해보거나 이외 제품 적용이 시급하지 않은 문제를 다룬다. TensorFlow, PyTorch 등의 라이브러리를 노트북 환경에서 다루며, 실험 결과를 공유하는 역할을 맡는다.

- **Data Scientist** : 위에 설명된 모든 역할을 포괄하는 단어. 조직에 따라 비즈니스 문제에 대한 해결을 구하는 분석가 역할을 수행할 수도 있으며, SQL, Excel, Pandas, Sklearn 등의 다양한 툴을 다룬다.

**필요한 스킬**

이러한 롤들을 수행하기 위해선 어떤 스킬셋이 필요할까? 하단 차트는 이러한 롤들이 필요로 하는 스킬셋을 도식화 한다 - *수평축은 ML 전문성을, 동그라미 크기는 커뮤니케이션과 기술 문서 작성에 대한 스킬을 뜻함*.

![alt text](mlops/images/fsdl_8_2.png)

- **MLOps** 란 기본적으로 소프트웨어 엔지리어링 롤이며, 기존의 소프트웨어 엔지니어링 파이프라인에 대한 이해가 필요하다.

- **ML Engineer** 는 ML 과 소프트웨어 개발 기술에 대한 지식을 모두 요구한다. 이러한 요구조건은 시장에 흔하지 않으며, 상당한 self-study 를 거친 엔지니어, 혹은 소프트웨어 엔지니어로 근무하는 과학/엔지니어링 분야 박사 학위자가 적합.

- **ML Researcher** 는 컴퓨터 공학, 통계학 등의 석/박사 학위를 소지하고 있는 ML 전문가가 적합하다.

- **ML Product Manager** 는 기존의 Product Manager 의 역할과 크게 다르지 않지만, ML 제품 개발 과정과 관련 지식에 능통해야 한다.

- **Data Scientist** 란 학사 학위자 부터 박사 학위자 까지 다양한 배경을 가질 수 있다.

- 버클리 EECS 박사 과정을 밟고 있는 [Shreya Shankar 가 게시한 글](https://www.shreya-shankar.com/phd-year-one/)에 따르면, **ML 엔지니어는 Task ML 엔지니어, Platform ML 엔지니어**로 세분화해 구분할 수 있다.
    - **Task ML Engineer** 는 구체적인 ML 파이프라인을 관리하는 역할을 맡는다. 이러한 ML 파이프라인이 정상적으로 작동하는지, 주기적인 업데이트가 이루어지는지 등을 확인하며, 대체로 업무량이 많은 편.
    - **Platform ML Engineer** 는 다른 ML Engineer 들이 수행하는 반복적인 작업들을 자동화 하는 업무를 맡는다.

## Hiring

**AI 역량 갭**

- FSDL 이 처음 시작된 2018 년의 경우 채용시장에서 AI 기술을 이해하는 인력을 찾기는 어려운 일이었다. 따라서 기업 내 AI 활용의 가장 큰 걸림돌은 인력 확보 문제였다.

- 2022년 현재, 이러한 채용시장 내 AI 역량에 대한 수요/공급 간 불균형은 여전히 존재하지만, 4년간 이루어진 관련 인력들의 커리어 전환, 이미 ML 수업을 수강한 학부생들의 시장 유입으로 어느 정도 해소된 면이 있다.

- 하지만 아직 시장에는 ML 이 어떻게 실패하고, ML 제품을 성공적으로 배포하는 방법을 아는 인력이 부족하다. 특히 **제품 배포 경험**을 가진 인력에 대한 품귀 현상이 존재.

**채용 소스**

- 이렇듯 작은 인력 풀과 급성장하는 수요로 인해 ML 직군 채용은 어려운 편이다. MLOps, Data Engineer, Product Manager 와 같은 롤은 많은 ML 지식을 요구하지 않기 때문에, 본 섹션에서는 코어 ML 직군에 대한 채용 방법을 설명한다.

![alt text](mlops/images/fsdl_8_3.png)

- 위와 같은 완벽한, 그리고 비현실적인 JD 를 통한 채용은 잘못된 방식이다.
    - 이보다는 소프트 엔지니어링 스킬셋을 갖춘 후보 중 ML 분야에 대한 관심이 있고, 배우고자 하는 인력을 추리는 편이 낫다. 기본적인 개발 역량이 있다면 ML 은 충분히 학습 가능한 영역이다.
    - 주니어 레벨의 채용 또한 고려해 볼 수 있다. 최근 졸업생등은 ML 지식을 상당 수준 가지고 있는 편.
    - 필요한 시킬셋이 무엇인지 가능한 자세히 기술하는 편이 좋다. DevOps 부터 알고리즘 개발까지 모든 ML 개발 과정에 능통한 인력을 찾기란 불가능하다.

- ML Researcher 를 채용하기 위해 강사진은 다음과 같은 팁을 제시한다.
    - 논문의 양보다는 질을 검토할 것. 아이디어의 독창성, 수행 방식 또한 면밀히 검증.
    - 트렌디한 문제보다 본질적인 문제에 집중하는 연구자를 우선 채용할 것.
    - 학계 밖에서의 경험은 비즈니스 환경 적응에 도움이 되기 때문에 이 또한 중요하다.
    - 박사 학위가 없거나, 유사 분야인 물리학, 통계학 등을 공부한 인력 또한 진중하게 검토할 것.

- 좋은 지원자를 찾기 위해서는 다음과 같은 경로를 시도할 것.
    - LinkedIn, 리크루터, 캠퍼스 방문 등 기존 채용 경로 검토.
    - ArXiv, 유명 컨퍼런스 등을 모니터링하고, 마음에 드는 논문의 1 저자 플래그.
    - 좋아하는 논문을 누군가 수준있게 구현한 경우 플래그.
    - NeurIPS, ICML, ICLR 등 ML 컨퍼런스 참석.

![alt text](mlops/images/fsdl_8_4.png)

- 리크루팅을 진행하면서 지원자들이 회사에 바라는 바를 파악하고, 이에 맞춰 회사를 포지셔닝 하는 과정이 필요하다. ML 전문가들은 흥미로운 데이터를 기반으로 영향력있는 일을 하고 싶어하기 때문에 배움과 영향력을 지향하는 문화를 만들고, 이를 통해 좋은 인력이 지원할 동기를 만들어 주어야 한다.

- 좋은 지원자를 모으기 위해선 채용을 진행중인 팀이 어떻게 우수하고, 미션이 어떻게 의미있는지에 대한 적극적이고 구체적인 설명이 곁들여져야 한다.

**인터뷰**

- 지원자를 인터뷰 할 때는, **지원자의 강점은 재확인하고, 약점은 최소 기준점을 충족하는지 확인**하자. ML Researcher 의 경우 새로운 ML 프로젝트에 대해 창의적으로 생각할 수 있는지 검증이 필요하지만, 코드 퀄리티의 경우 최소한의 요건만 충족하면 된다. 

- ML 인터뷰는 기존 소프트웨어 엔지니어링 인터뷰에 비해 덜 성숙한 분야이다. Chip Huyen 의 [**Introduction to ML Inteviews Book**](https://huyenchip.com/ml-interviews-book/) 과 같은 레퍼런스를 참조.

## Organizations

- ML 팀의 구성이란 아직 정답이 존재하지 않는 영역이다. 하지만 조직 내 ML 활용 특성과 성숙도에 따라 존재하는 best practice 는 다음과 같이 정리할 수 있다. 

**타입 1 - 초기단계 혹은 Ad-Hoc 성 ML**

- 사실상 조직 내 ML 활용이 없으며, 필요시 단발성 프로젝트가 진행되는 경우. 인하우스 ML 전문성은 매우 낮은 편이다.
- 중소규모의 비즈니스이거나, 교육, 물류 등 상대적으로 IT 중요도가 낮은 분야일 것.
- ML 적용으로 인한 단기적 이점이 상당히 적은 편.
- ML 프로젝트에 대한 지원이 적으며, 좋은 인력을 채용하고 유지하는 것에 상당한 어려움이 있다.

**타입 2 - ML R&D**

**타입 3 - 비즈니스 & 제품 팀 내 적극적인 활용**

**타입 4 - 독립적인 ML 기능**

**타입 5 - ML 우선주의**

## Managing

## Design