---
title: "Conda 환경 공유 방법"
description: "YAML 파일 기반 환경 저장 및 구축 노트"
date: "2022-06-23 00:00:00"
slug: "conda_1"
image: machine_learning/images/conda.png
tags: [콘다 YAML, 콘다 환경 공유, conda 환경, 파이썬, 주피터]
categories: [Python, Conda, Anaconda, Environment Sharing]
---
## 배경

[콘다](https://docs.conda.io/en/latest/#)는 윈도우, 맥OS, 리눅스에서 동작하는 패키지 관리 시스템이며, 데이터 분석 환경에서 주로 사용되지만 파이썬, R, 루비, 자바 등 다양한 언어를 지원한다. 본 글에서는 짧게 콘다 환경 생성과 세팅, 저장, 그리고 다른 컴퓨터에서 저장된 환경을 불러오는 법을 살펴보고자 한다.

## 환경 생성 및 세팅, 저장
### 생성 및 패키지 설치

Conda 환경은 다음과 같이 생성할 수 있다. 

```
 conda create --name [환경이름] python=3.10
```

생성된 모든 conda 환경은 다음 커맨드로 확인할 수 있다. `*` 표시는 현재 환경을 나타낸다.

```
 conda env list
 >>> conda environments:
     base                       *
     environment1
     environment2
     ...
```

환경을 바꾸기 위해서는 `activate` 커맨드를 사용한다.

```
 conda activate environment1
 conda env list
 >>> conda environments:
     base
     environment1               *
     environment2
     ...
```

현재 환경에 설치된 패키지는 다음과 같이 확인할 수 있다.

```
 conda list
 >>> numba             0.48.0              py37h47e9c7a_0
     numpy             1.18.1              py37h93ca92e_0
     openssl           1.1.1d                  he774522_4
     pandas            1.0.1               py37h47e9c7a_0
     ...
```

패키지를 설치하기 위해서는 주로 `pip install`, 혹은 `conda install` 커맨드를 사용하게 된다. [pip](https://pypi.org/project/pip/)은 파이썬 전용 패키지인 반면, conda는 기타 언어의 패키지 관리를 지원한다는 차이점을 가지고있다. 다음 예시는 pip 패키지 매니저를 활용했다.

```
 pip install cython
 conda list
 >>> cython            0.29.15             py37ha925a31_0
     numba             0.48.0              py37h47e9c7a_0
     numpy             1.18.1              py37h93ca92e_0
     openssl           1.1.1d                  he774522_4
     pandas            1.0.1               py37h47e9c7a_0
     ...
```

### YAML 파일 저장

[YAML](https://www.redhat.com/en/topics/automation/what-is-yaml) 포맷으로 환경 설정을 저장하기 위해서는 다음 커맨드를 활용한다. YAML 파일명은 굳이 환경 이름과 매칭되지 않아도 괜찮다.

```
 conda env export > environment1.yaml
```

이후 해당 커맨드를 실행한 경로에 environment1.yaml 이라는 파일이 생성되게 된다. 해당 파일을 열어보면 다음과 같이 설치된 패키지가 나열되어 있는것을 확인할 수 있다.

```
 name: environment1
 channels:
   - conda_forge
   - defaults
 dependencies:
   - cython=0.29.15=py37ha925a31_0
   - numba=0.48.0=py37h47e9c7a_0
   ...
```

## YAML 파일을 활용한 환경 생성

다른 컴퓨터에서 저장된 conda 환경과 동일한 환경을 생성하고자 할때, 커맨드창에서 YAML 파일 경로로 이동 후 다음을 실행시키면 된다.

```
 conda env create --file environment1.yaml
```