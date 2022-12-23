---
title: "(번역) Michelangelo : Uber 의 머신러닝 플랫폼"
description: "Uber Blog 에 소개된 플랫폼 상세 내용 소개"
date: "2022-12-23 00:00:00"
slug: "fsdl"
image: "temp/images/michelangelo_2.jpg"
tags: [Uber, Michelangelo, 우버, 우버 머신러닝, 머신러닝 플랫폼 예시, 미켈란젤로, 우버 테크 블로그]
categories: [Uber, 번역, Tech Blog]
---

## Introduction

- 본 글은 2017 년 Uber Blog 에 개제된 [Meet Michelangelo: Uber's Machine Learning Platform](https://www.uber.com/en-KR/blog/michelangelo-machine-learning-platform/) 의 내용을 번역한 것이다. 문단 형태의 글을 bullet-point 로 축약하였고, 약간의 의역이 있다.

![alt text](temp/images/michelangelo_3.webp)

- Michelangelo (미켈란젤로) 는 Uber 사내 활용을 위한 ML-as-a-Service 플랫폼이며, ML 모델 구축 및 배포 과정을 전직원이 보다 쉽게 접근할 수 있도록 돕는것을 그 목적으로 한다. 

- 데이터 관리, 학습, 평가, 배포, 예측, 모니터링 까지 모든 end-to-end 기능을 제공하며, 전통적인 ML 모델은 물론 시계열 예측과 딥러닝 기능까지 제공하고 있다.

- 글이 써진 2017년을 기준으로 시스템은 약 1년간 활용되었고, Uber 의 여러 데이터 센터에 설치되고, 실제 모델 배포에 활용되는 등 이미 머신러닝을 수행하기 위한 기본 시스템으로 자리매김했다.

- 본 글은 이러한 미켈란젤로 시스템을 소개하고, 유즈케이스 및 기본적인 작업 과정을 순차적으로 설명한다.

## Motivation behind Michelangelo

## Use Case : UberEATS

![alt text](temp/images/michelangelo_4.png)

## System Architecture

## Machine Learning Workflow

## Manage Data

![alt text](temp/images/michelangelo_5.png)

### Offline

### Online

### Shared Feature Store

## Train Models

![alt text](temp/images/michelangelo_6.png)

## Evaluate Models

## Model Accuracy Report

![alt text](temp/images/michelangelo_7.png)
![alt text](temp/images/michelangelo_8.png)

### Decision Tree Visualization

![alt text](temp/images/michelangelo_9.png)

### Feature Report

![alt text](temp/images/michelangelo_10.png)

## Deploy Models

![alt text](temp/images/michelangelo_11.png)

## Make Predictions

![alt text](temp/images/michelangelo_12.png)

## Referencing Models

## Scale and Latency

## Monitor Predictions

![alt text](temp/images/michelangelo_13.png)

## Management Plane, API, and Web UI