---
title: "A/B Testing 개요"
description: "기본 개념, 왜곡 요소 및 Multi-armed Bandit 기반 자동화"
date: "2023-07-22 00:00:00"
slug: "abtest"
image: "statistics/images/abtest_1.jpeg"
tags: ["Statistics", "AB Testing", "AB 테스팅"]
categories: ["Statistics", "AB Testing", "통계학"]
---

## Introduction

### A/B Testing 이란?

| ![alt text](statistics/images/abtest_2.png) |
|:--:|
| Fig 1. [SplitMetrics.com](https://brunch.co.kr/@digitalnative/19) |

- AB Test 란 **디지털 환경에서 실사용자를 대조군 (Control Group) 과 실험군 (Experimental Group) 으로 나누어 특정한 UI 나 알고리즘의 효과를 비교하는 방법론**이다.
- 과거 오프라인에서 신제품을 출시할 때 경쟁성을 파악하기 위해 수행하던 일부 점포를 대상으로 한 실험 또는 소비자 조사를, 변화한 온라인 환경에서 더 높은 검증력으로 수행할 수 있도록 발전한 형태이다.

### 활용 예시

| ![alt text](statistics/images/abtest_4.png) |
|:--:|
| Fig 2. [제페토 AB Test 사례](https://www.youtube.com/watch?v=2UW-vfHN0O4) |

- 네이버 제페토 대화방에 기본적으로 공지 예시를 노출하면, 공지 기능 활용율이 올라갈 것이다 라는 가설을 세움.
- 공지 예시가 노출된 대화방 그룹 A, 노출되지 않은 대화방 그룹 B 에 대해 일정 기간 테스트를 진행.
- 공지 예시가 노출된 대화방 그룹 A 가 그룹 B 에 비해 공지 생성수가 37.9%p 높게 확인됨에 따라, 모든 사용자에게 노출 결정.

### A/B Testing 툴

| ![alt text](statistics/images/abtest_3.png) |
|:--:|
| Fig 3. [Codavel - Optimizely Tutorial](https://blog.codavel.com/a/b-testing-tools-how-to-integrate-optimizely) |

- 규모가 있는 회사의 경우 자체적으로 AB 테스팅 플랫폼을 보유하는 것이 일반적이다.
- 빠르고 쉽게 AB 테스팅을 수행할 수 있는 대표적인 툴로는 [Optimizely](https://www.optimizely.com/) 와 [Google Optimize360](https://marketingplatform.google.com/about/optimize/) 제품이 있다. 테스트 수행 뿐만 아니라 실제 결과값을 분석하고, 대시보드로 표현까지 해주게 된다.
- [Optimizely 개발자 문서](https://docs.developers.optimizely.com/full-stack-experimentation/docs/example-usage)에서 예시로 삼은 Python 환경의 AB 테스팅 코드는 다음과 같다 :

{{< highlight python >}}
from optimizely import optimizely

# Instantiate an Optimizely client
optimizely_client = optimizely.Optimizely(datafile)

# Evaluate a feature flag and variable
enabled = optimizely_client.is_feature_enabled('price_filter', user_id)
min_price = optimizely_client.get_feature_variable_integer('price_filter', 'min_price', user_id)

# Activate an A/B test
variation = optimizely_client.activate('app_redesign', user_id)
if variation == 'control':
    pass
    # Execute code for variation A
elif variation == 'treatment':
    pass
    # Execute code for variation B
else:
    pass
    # Execute code for users who don't qualify for the experiment

# Track an event
optimizely_client.track('purchased', user_id)
{{< /highlight >}}

## 수행 프로세스

### 평가 기준 선정

- 실제 AB 테스팅을 진행하기 전, **결과 해석을 위한 평가 기준을 산정하는 작업이 우선적으로 필요**하다. 경우에 따라 **KPI (Key Performance Indicator), OEC (Overall Evaluation Criterion)** 으로 불리기도 한다.
- AB 테스팅을 위한 좋은 평가 기준은 다음과 같이 정의할 수 있다 :
    - 실험 기간 동안 단기적으로 측정 가능하며, 장기적으로는 전략적인 목표를 추진할 수 있을 것
    - 시기적절하게 중요한 변화를 감지할 수 있을 정도의 민감도를 갖출 것 *(샘플 사이즈, 분산 등 실험 환경 영향 고려)*
    - 비즈니스 특성과 상황의 충분한 이해를 바탕으로 할 것 *(연간 갱신 서비스에 대한 실험을 몇 달간 수행하는 경우 효과를 확인할 수 없음)*
    - 트레이드 오프에 대한 충분한 고려를 바탕으로 할 것 *(매출 vs 사용자 수, 체류시간 vs 이탈율 등)*

### 실험 변수 및 통제 변수 선정

- **실험 변수 (Experimental Variable)** : 실험 과정에서 실험군과 대조군을 상대로 변경할 조건. 상기된 제페토의 예시에서 공지 기능 관련 메시지 노출을 예로 들 수 있다.
- **통제 변수 (Control Variable)** : 실험군과 대조군에서 동등한 조건을 지녀야 하는 변수. 예시로 평균 나이, 성비 등을 들 수 있으며, 관련 경험이 부족할수록 종속 변수에 영향을 미치는 요소를 두루 살펴 샘플링 에러가 발생하지 않도록 하여야 한다.

### 가설 서술

- AB 테스팅을 위한 가설은 참/거짓 판별이 가능한 수준의 단순한 문장으로 서술하여야 한다.
- **귀무가설**이란 변형군 간 차이가 없는 경우를, **대립가설**은 연구를 통해 입증되기를 주장하는 가설을 의미.
- 예시로 체크아웃 페이지에 쿠폰 코드 필드를 더할 시, 사용자 당 매출이 증가하는지 여부를 판별하기 위해 다음과 같은 가설 수립이 가능하다 :

**귀무가설 (H0)** : *"체크아웃 페이지에 쿠폰 코드 필드를 더할 시, 사용자 당 매출은 증가하지 않을 것이다."*

**대립가설 (H1)** : *"체크아웃 페이지에 쿠폰 코드 필드를 더할 시, 사용자 당 매출은 증가할 것이다."*

### 샘플링 및 대조군 & 실험군 생성

- 많은 경우 AB 테스팅 수행에는 비용이 발생하며, 목적을 감안해 **유의미한 통계치를 산출할 수 있는 샘플 사이즈를 특정할 수 있다면 불필요하게 모든 데이터를 수집하는 것은 권장되지 않음**.
    - 검증 가설, 검정력 등 실험 설계 조건 입력시 필요한 샘플 수를 계산해주는 사이트 - [Evan's Awesome A/B Tools](https://www.evanmiller.org/ab-testing/sample-size.html)
- 특히 유저 반응이 불확실한 영역의 경우 작은 비율의 사용자를 대상으로 실험을 진행하는 것이 안전함 *(유저수가 많을 수록 작은 차이로 매출에 매우 부정적인 영향을 끼칠 수 있다)*.
- 샘플링을 수행한 후, 대조군과 실험군을 생성하는 방식은 크게 아래와 같이 세가지로 나눌 수 있다 :

**노출 분산 방식**
- AB Test 가 진행되는 페이지가 렌더링 될 시, 단순 확률을 기반으로 그룹 간 다른 환경 노출
- 통계적 유의성이 높지만, UI/UX 테스팅 시 사용자에게 혼란을 줄 수 있는 위험이 존재
- 따라서 유저에게 직접적으로 노출되지 않는 알고리즘 테스팅이 가장 적합하다

**사용자 분산 방식**

- 쿠키로 발급한 UID 등으로 사용자를 A/B 그룹으로 분할하여, 고정적으로 다른 환경을 노출시킴
- UI/UX 테스팅 시 혼란을 줄 수 있는 위험도가 낮지만, 일반적이지 않은 행동 패턴을 보이는 특정 유저에 의해 결과값이 왜곡될 수 있음

**시간 분할 방식**

- 초~분 단위로 시간대를 세밀하게 분할하여 A/B 안 노출
- 보안/설계 문제로 분산 방식 활용이 어려운 경우 쉽게 구현이 가능함

### 집행 기간 결정

- 샘플링 사이즈와 유사하게 검증 과제에 비해 과도한 집행 기간 설정은 불필요한 비용을 발생시킬 수 있음.
- 서비스 상황에 따라 테스크 초기 효과가 유달리 크거나 작은 경향을 보일 수 있으며, 이를 **초두 효과** 라고 부른다. 이러한 경우 테스트가 안정화되기 전 종료된다면 결과값에 대한 신뢰성이 매우 저하될 수 있음으로 이를 확인하는 과정이 필요.
- 주말 또는 공휴일 등 사용자 행동에 영향을 끼치는 외부 기간 요인을 감안할 것.

## 결과의 유의미성

- 모든 실험과 동일하게 AB 테스팅을 수행한 후에는 결과에 대한 기본적인 신뢰성 검증이 수반된다.
- 대표적인 방식으로는 AA 테스팅과, p-value 분석을 들 수 있다.

### A/A Testing

- 실제 AB 테스팅을 수행하기 전, **동일한 환경 A 에 노출된 두 집단 간 결과값 차이를 확인**하는 과정.
- 실험군 선정 과정에서 왜곡이 발생했거나, 샘플 사이즈가 지나치게 작은 경우 같은 환경임에도 결과값에 차이가 발생할 수 있다.
- 이러한 경우 문제 원인을 파악하고, 동일 결과가 산출되는 것을 확인한 후 실제 AB 테스팅을 진행하는 것이 필수적.

### p-value

- 통계 분석에서 가장 널리 활용되는 유의성 검증 방식.
- 기본적으로 두 집간 단 차이가 없다고 가정했을 때, 산출된 결과값보다 극단적인 값이 도출될 확률을 계산한다.
- 샘플수에 큰 영향을 받으며, 차이가 발생했으나 p-value 가 지나치게 클 경우 실험 기간을 늘려야 함.
- 대부분의 AB 테스팅 툴이 p-value 산출 기능을 제공.

### 왜곡 요소

- 테스팅 대상 트래픽이 충분히 크고, p-value 가 낮게 나타나는 상황에서도 결과가 왜곡될 수 있는 여러 외부 요인이 존재한다. 
    - 일반적이지 않은 행동 패턴을 가진 유저에 의해 결과값이 변질되는 경우이다.
    - 또 다른 예시로 Seasonality를 들 수 있다. 성수기와 비성수기 간 성능이 좋은 추천 알고리즘이 다를 수 있으며, 또한 시기 마다 방문자 성향에 차이가 발생할 수 있다.
    - 내부적인 사유로 의도적으로 AB 테스팅 환경을 조작하는 경우가 발생할 수 있다. 

- 이외에 실험 설계적인 관점에서 발생할 수 있는 대표적인 실수는 다음과 같다 :
    - 실험 내 너무 많은 시나리오를 설계해 충분한 샘플 사이즈가 확보되지 않는 경우
    - 실험 도중 실험 환경을 변경하는 경우
    - 단순히 통계적 유의미성을 기반으로 실험을 중단하는 경우

## MAB (Multi-armed Bandit)

| ![alt text](statistics/images/abtest_5.jpeg) |
|:--:|
| Fig 4. [A/B Testing vs. MAB Testing](https://www.linkedin.com/pulse/basics-multi-armed-bandit-its-application-testing-mengyao-zhang/) |

- AB 테스팅의 가장 큰 단점 중 하나는 **실행 수행 기간 동안 저성과 그룹에서 발생할 수 있는 매출 손실**이다.
- 예를 들어 1만명을 대상으로 한 AB 테스팅 과정에서 페이지 A 가 페이지 B 에 비해 인당 매출 액수가 평균 1,000 원 정도 높았다고 가정했을 때, 약 5백만원의 기회 비용이 발생한다.
- 이러한 문제를 해결하기 위해 **MAB 알고리즘**을 활용할 수 있다. 기본적인 개념은 다음과 같이 설명 가능 :
    - 여러 variation 에 대한 결과를 자동으로 측정하고, 높은 결과값을 보이는 그룹에게 더 많은 트래픽을 분할해 궁극적으로 가장 높은 성과를 보이는 variation 을 찾게됨.
    - Seasonality 등의 효과로 낮은 성과를 보이던 그룹에게 다시 트래픽을 부여하는 등 상황에 따라 유동적으로 반응할 수 있음.

## A/B Testing 은 모든 문제를 해결하는가?

- 애플은 AB 테스트를 수행하지 않는 기업으로 유명하다. 그 이유는 AB 테스팅이 제품의 파편적인 성능을 판별하는 것에는 도움을 주지만, 통합된 하나의 서비스를 만드는데에는 기여하지 못하기 때문.
- AB 테스트를 중요하게 생각하는 기업은 대중의 선호에 좌우되지만, 혁신적인 무언가를 만드는데에는 장애 요소로 작용할 수 있다. 
- [데이블 담당자 분의 블로그](https://brunch.co.kr/@digitalnative/20) 에 따르면, 사용자는 결국 익숙한 것을 선호하는 패턴을 보인다. 예를 들어 국내의 경우 네이버/카카오 등의 기업이 자주 활용하는 UI 패턴이 결국 가장 높은 효율을 보임.
- 하지만 모두가 혁신가는 아니다. 현재 AB 테스팅은 서비스 세부 효율을 개선하기 위한 매우 유용한 툴이며, 가장 확실한 방법임.

## Sources

1. Digital Native Magazine - AB Test 기본부터 심화까지 [[1]](https://brunch.co.kr/@digitalnative/19)[[2]](https://brunch.co.kr/@digitalnative/20)[[3]](https://brunch.co.kr/@digitalnative/17)
2. Rightbrain Lab - 신뢰도 높은 온라인 통제 실험 A/B테스트 [[1]](https://brunch.co.kr/@rightbrain/206)[[2]](https://brunch.co.kr/@rightbrain/221)
3. [Hello Darwin - Everything You Need to Know About A/B Testing](https://hellodarwin.com/blog/about-ab-testing)
4. [Convert - A/B Testing Statistics](https://www.convert.com/blog/a-b-testing/decode-master-ab-testing-statistics/)
5. [Codavel - Optimizely Tutorial](https://blog.codavel.com/a/b-testing-tools-how-to-integrate-optimizely)
6. [Conversion - Top 3 Mistakes That Make Your A/B Test Results Invalid](https://conversion.com/blog/3-mistakes-invalidate-ab-test-results/)