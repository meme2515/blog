---
title: "[Project] 프레스 금형 Optimization 프로젝트 개요"
description: "프로젝트 내용과 활용 기술에 대한 간략한 소개"
date: "2023-01-23 00:00:00"
slug: "layout_optimization"
image: "projects/images/layout_title.jpg"
tags: []
categories: []
draft: True
---

## Introduction

자동차를 양산할때는 정말이지 엄청난 인력과 장비가 동원됩니다. 이 중 가장 큰 면적을 차지하는 부분은 우리가 직접 눈으로 볼 수 있는 차체인데요, 아래 그림을 통해 짐작하실 수 있겠지만 차체는 주로 나뉘어진 형상을 마치 프라모델 처럼 이어 붙이는 형태로 제작하게 됩니다.

| ![alt text](projects/images/layout_1.jpg) |
|:--:|
| Fig 1. 자동차의 차체 |

이러한 차체 형상을 만드는 방법은 간단하지만, 복잡한데요, 기본적인 개념은 원하는 형상을 본뜬 음각/양각의 틀로 평평한 강판을 찍어 누르는 과정을 거친다고 생각하면 됩니다. 이를 "프레스 금형" 이라 부르며, 제조 단계에서 원하는 모양으로 철제를 가공하기 위한 많은 방법 중 하나입니다. 

| ![alt text](projects/images/layout_2.jpg) |
|:--:|
| Fig 2. 프레스금형 제품 예시 |

어떻게 보면 굉장히 단순한 공정으로 생각될 수 있습니다. 이미 디자인이 완료된 차체 형상에 알맞은 금형을 만든 후, 강판을 찍어내기만 하면 되는 문제니까요. 하지만 자동차의 제조 완성도란 사실 인명과 직결된 문제이고, 강판이라는 것이 다루기 쉬운 자제가 아니기 때문에 많은 고려사항이 발생하게 됩니다.

이런 고려사항 중 하나가 최종 형상 바깥에 남은 강판 영역을 어떻게 처리할까 하는 문제입니다. 강판은 보통 직사각형 형상을 띄게 되는데, 이렇게 남은 영역을 어떻게 구부리고, 접는가에 따라 최종 형상의 완성도에 영향을 미치기 때문입니다. 

하단 그림의 "제품" 은 최종 차체의 형상을, "성형완료" 는 프레스금형에 의해 눌린 강판을 나타냅니다. 즉, 여러 고려사항을 감안해 눌린 강판에서, 결국 불필요한 부분을 잘라냄으로 최종적인 제품이 완성된다고 생각할 수 있습니다.

| ![alt text](projects/images/layout_3.png) |
|:--:|
| Fig 3. 실제 형상과 프레스금형 예시 |
