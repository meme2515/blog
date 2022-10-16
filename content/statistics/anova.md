---
title: "ANOVA (분산 분석) 란 무엇인가?"
description: "자유도, MSE, MSB, F값, 사후분석 까지의 One-Way ANOVA Testing 과정 설명"
date: "2022-10-09 00:00:00"
slug: "anova"
image: "statistics/images/anova_1.png"
tags: ["anova", "분산 분석", "아노바", "anova table"]
categories: ["Statistics", "Parametric Testing", "통계학"]
draft: "true"
---

## 소개

ANOVA (Analysis of Variance, 분산 분석) 란 **여러 집단의 평균치 차이가 통계적으로 의미있다고 볼 수 있는지에 대한 검증 방법론**이다. 예를 들어 A, B, C 반에 속한 학생의 평균 키를 비교해본다고 가정하자. 이 경우 모든 그룹의 키가 동일한지, 아니면 최소한 한 개 그룹의 평균 키가 유의미하게 다른지 판별하기 위해 ANOVA 검증을 수행하게된다.

단순히 A 와 B, 두 개 그룹 간 평균 키를 비교한다면 이에 상응하는 검증 방법인 **[t-test](https://www.youtube.com/watch?v=mQXj456SWco)** 수행이 가능하겠지만, 세 개 이상의 그룹에서 t-test 를 반복 수행할 시 발생할 수 있는 **[다중검정문제](https://syj9700.tistory.com/6)** 를 방지하기 위해 우선적으로 ANOVA 를 통해 통계적 유의미성을 판단하는 것이다. ANOVA 를 통해 유의미성이 검증되었다면, 이후 그룹 간 차이를 검증하기 위한 개별적인 t-test 수행이 따를 수 있으며 *(이외에도 Tukey test 등 많은 방법론이 존재한다)* 이와 같은 과정을 **[사후 검정 (Post-Hoc Analysis)](https://m.blog.naver.com/statsol/221472155248)** 이라 칭한다.

## t-test 의 확장 개념

ANOVA 를 설명하기에 앞서 t-test 의 수식을 살펴보자. 

$$
t = \frac{\bar{X_1} - \bar{X_2}}{}
$$

## 레퍼런스

1. [BioinformaticsAndMe - 분산분석(ANOVA)](https://bioinformaticsandme.tistory.com/198)
2. 공돌이의 수학정리노트
    - [ANOVA 가볍게 설명해드립니다](https://www.youtube.com/watch?v=SfbcHZm4xyM)
    - [F-value와 ANOVA의 의미](https://www.youtube.com/watch?v=VZ6WPnI82Z8)
3. Khan Academy - ANOVA Series
    - [ANOVA 1: Calculating SST](https://www.youtube.com/watch?v=EFdlFoHI_0I)
    - [ANOVA 2: Calculating SSW and SSB](https://www.youtube.com/watch?v=j9ZPMlVHJVs)
    - [ANOVA 3: Hypothesis Test with F-Statistic](https://www.youtube.com/watch?v=Xg8_iSkJpAE)
4. [Bozeman Science - Student's t-test](https://www.youtube.com/watch?v=pTmLQvMM-1M)