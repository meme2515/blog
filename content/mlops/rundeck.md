---
title: "배치 관리 소프트웨어 런데크 (Rundeck)"
description: "배치 스케줄링 및 관리 소프트웨어 런데크 소개"
date: "2022-06-13 00:00:00"
slug: "rundeck"
image: "mlops/images/rundeck-wordmark.svg"
tags: [배치관리, 배치 스케줄링, 태스크 스케줄링, 태스크 관리, 원격 pc 커맨드, MLOps, DevOps, SRE]
categories: [MLOps, Batch Job, Batch Scheduling, Task Management, SRE]
---
## 프로그램 사용 배경

업무 중 머신러닝 학습 데이터 생성을 위해 6대 로컬 PC에서 다소 리소스 인텐시브한 작업을 반복적으로, 장기간 진행할 니즈가 생겼다. 최초에는 6대 각각의 로컬 환경에서 Windows 공식 배치관리 툴인 [Task Scheduler](https://docs.microsoft.com/en-us/windows/win32/taskschd/task-scheduler-start-page) 에 관련 .bat 파일을 등록할 요량이었으나 다음과 같은 이유로 별도 배치 관리 툴을 찾아보게 되었다.

1. 6대 PC에서 개별적인 로컬 스케줄러를 관리한다는 것은 물리적인 모니터링을 필요로하기에 데이터 생성 기간동안 지나치게 많은 시간을 뺏길 것 같았다. **빠른 대응이 가능한 중앙화된 모니터링 체계**가 필요했다.
2. 프로세스는 경우에 따른 작업 시간이 달라 일정기간 지속 시 재시작 가능한 **룰 기반 배치 관리**가 필요했다.
3. 데이터 생성 도중 프로세스에 변동이 있을 가능성이 있었기때문에 **프로세스 일괄 수정이 가능**한 툴이 필요했다.

최초 머리에 떠오른 솔루션은 [Apache Airflow](https://airflow.apache.org/) 였으나 그닥 익숙한 솔루션도 되지 못했고, 모니터링 환경이 Windows 10 이었기때문에 환경 세팅에 어려움이 있었다. 그렇게 구글링을 계속하며 [Ansible](https://www.ansible.com/)과 같은 SSH 기반 솔루션을 생각했으나 보안상 이유로 다시 세팅에 어려움이 있었고... 적합한 오픈소스 솔루션인 [Rundeck](https://www.pagerduty.com/integrations/rundeck-runbook-automation/)를 발견했다.

## Rundeck 소개

| ![alt text](mlops/images/rundeck_example.png) |
|:--:|
| Fig 1. Rundeck 파이프라인 예 - 유저가 생성한 Job 들을 Node 별로 할당 및 실행, 에러 발생 등 유사시 알림 설정 |

미국의 클라우드 소프트웨어 업체인 [PagerDuty](https://www.pagerduty.com) 사에서 개발한 작업 관리 소프트웨어이며, Physical, VM, Container, Serverless 환경에서 스크립트, API 호출 등의 작업을 스케줄링 및 관리 할 수 있다. 유학 중 룸메이트가 취업했다고 좋아하던 회사인데 좋은 프로그램을 만들고있었다.

많은 유즈 케이스들이 있는데, 가장 대중적인 예시는 [SRE (사이트 신뢰성 엔지니어링)](https://sre.google/) 영역이다. Google 엔지니어 [Ben Treynor Sloss](https://www.linkedin.com/in/benjamin-treynor-sloss-207120/)가 창안한 개념인데, DevOps가 개발과 운영영역 간 사일로를 줄이는 철학적 접근이라고 한다면, SRE란 operation 영역의 문제들을 엔지니어링 관점에서 해결하는 방법론이라고 정의할 수 있다. 조직의 SRE팀이 계정 및 권한 관리, 인프라 리소스 관리 등의 운영 관점의 문제들을 자동화를 통해 해결하고나면, Dev팀은 소프트웨어 개발에, Ops팀은 제품 안정화에 더욱 집중할 수 있다는 식이다 (나도 현재는 이정도로만 이해하고 있고, 관심이 있다면 [1번](https://www.dynatrace.com/news/blog/what-is-site-reliability-engineering/), [2번](https://www.youtube.com/watch?v=uTEL8Ff1Zvk) 링크에서 더욱 상세한 내용을 확인할 수 있다).

Rundeck 솔루션은 이러한 SRE 관점의 운영 절차를 표준화할 수 있는 플랫폼을 제공하며, 이러한 절차들은 조직 내에서 안전하게 공유되게 된다. 나의 경우는 아직 관련 지식이 부족하며, 당장 필요한 영역은 workload automation 으로 한정되어있기 때문에 깊은 내용은 추후에 더 알아보기로 하자.

핵심적으로 짚고 넘어가야 할 개념은 다음과 같다.

### Projects

[Rundeck Documentation - Projects](https://docs.rundeck.com/docs/manual/projects/)

Rundeck 내 작업 환경의 개념이다. 한개 Rundeck 서버에 여러개의 Project를 관리할 수 있으며, 프로젝트의 개념은 사용자가 정의하기 나름이다. 팀, 인프라, 어플리케이션, 환경 등 사용 목적에 맞게 Project를 구분하게 된다.

### Jobs

[Rundeck Documentation - Jobs](https://docs.rundeck.com/docs/manual/04-jobs.html)

실행하고자 하는 프로세스의 묶음이다. [윈도우 batch 파일](https://en.wikipedia.org/wiki/Batch_file), [Airflow의 DAG](https://airflow.apache.org/docs/apache-airflow/stable/concepts/dags.html) 개념과 유사하다.

| ![alt text](mlops/images/airflow_example.png) |
|:--:|
| Fig 2. Airflow의 DAG 개념 예 - branch_b를 에러 케이스라고 보면 될 듯 하다 |

Rundeck 내에서는 Job 단위로 스케줄러 설정이 가능하고, 개별적인 히스토리가 남게된다. 한개 Job을 생성할 때 input option을 설정하거나, 에러 핸들링 룰을 생성하는 등 부수적인 옵션이 주어지게 된다.

### Steps

CLI 커맨드, 스크립트 실행, 다른 Job 호출 등 하나의 Job을 구성하는 개별적인 태스크를 지칭하는 용어이다. 또한 개별 Step 내에서 다양한 플러그인 활용이 가능하다.

### Nodes

[Rundeck Documentation - Nodes](https://docs.rundeck.com/docs/manual/05-nodes.html)

Job이 실행되는 대상이다 (Physical, VM, Container, API, Database 등). 나의 경우에는 6대로 분할된 로컬 PC에 해당한다. 각각의 Node는 태그와 속성값을 지니게된다.

Rundeck의 [공식 소개 영상](https://www.youtube.com/watch?v=QSY_qw9Buic)을 확인하면 Projects -> Jobs -> Steps -> Nodes 순으로 계층구조 개념을 띄고있다. 

## 설치 방법 💻

1. [윈도우 설치 Doc](https://docs.rundeck.com/docs/administration/install/windows.html#folder-structure)
2. [Ubuntu 설치 Doc](https://docs.rundeck.com/docs/administration/install/linux-deb.html#installing-rundeck)
3. [CentOS 설치 Doc](https://docs.rundeck.com/docs/administration/install/linux-rpm.html)

## 내가 사용한 방법
### WinRM

네트워크를 통해 원격으로 터미널을 제어하는 방법은 SSH (Secure Shell) 커맨드가 가장 익숙했고, Windows 10 부터는 OpenSSH라는 연관 툴을 기본으로 제공한다는 [공식 가이드](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse)를 확인해 세팅을 시도했다. 하지만 세팅에 필요한 PowerShell이 보안상의 이유로 제한되어있어 진행이 어려웠다. 

이런 저런 대안을 찾아보다 Rundeck에서 제공하는 [pywinrim](https://github.com/diyan/pywinrm) 이라는 플러그인을 통해 Windows Node 설정이 가능하다는 [공식 가이드](https://docs.rundeck.com/docs/learning/howto/configuring-windows-nodes.html)를 확인했다. WinRM (Windows Remote Management)은 SSH의 Windows 네이티브 버전 정도로 이해가 되는데, 실제 프로토콜 방식은 굉장히 다르다고한다 ([연관 글](https://www.reddit.com/r/sysadmin/comments/nadfbs/winrm_vs_openssh/)). 

pywinrm은 이런 WinRM 연결을 파이썬 환경에서 구현 가능하도록 하는 패키지인데, Rundeck내에서 해당 패키지를 활용한 노드 생성 기능을 구현한 듯 했다. 하지만 세팅이 생각보다 간단하지는 않았고, 나는 파이썬 스크립팅을 선호했기에 해당 패키지를 별도로 사용해 Rundeck에서는 .py 파일만 실행하는 접근법을 택했다.

하단은 pywinrm 패키지 사용 예시이다.

```
 import winrm
 
 s = winrm.Session('windows-host.example.com', auth=('username', 'password'))
 r = s.run_cmd('ipconfig', ['/all'])
 >>> r.status_code
 0
 >>> r.std_out
 Windows IP Configuration
 
    Host Name . . . . . . . . . . . . : WINDOWS-HOST
    Primary Dns Suffix  . . . . . . . :
    Node Type . . . . . . . . . . . . : Hybrid
    IP Routing Enabled. . . . . . . . : No
    WINS Proxy Enabled. . . . . . . . : No
 ...
```

### FileZilla, PSCP

학습 데이터 생성에 필요한 초기 데이터를 6대 PC에 분할하는 작업을 위해 메인 PC에 세팅한 [FileZilla](https://filezilla-project.org/) 서버를 활용했다. 세팅 난이도도 높지 않고, 단순한 파일공유 (FTP) 프로그램으로 생각하면 될 듯 하다.

일련의 과정을 통해 생성된 학습 데이터는 각각 6대 PC로 부터 실제 학습을 수행할 리눅스 서버에 [PSCP](https://documentation.help/PuTTY/pscp.html) 커맨드를 통해 전송했다. 윈도우 환경에서 리눅스 환경으로 파일을 전송하기 위해 주로 사용되는 명령어라고 한다.

## 결론

6대 PC에 스케줄링된 batch job의 성공 여부를 하나의 환경에서 모니터링 가능한 체계를 구축했다. 또한 일정시간 이상 batch job 지속 시 이를 취소하는 룰을 손쉽게 세팅할 수 있었고, 핵심 코드 또한 중심이 되는 서버 PC에서 수정이 가능하도록 했다. 언급한 3가지 요건을 어느정도 충족한 결과였다.

MLOps와 어느정도 연관성이 있는지는 사실 잘 모르겠다. 리소스 인텐시브한 데이터 생성 과정에서 유지/보수가 가능한 체계를 구축했다는데 의미가 있을수는 있으나 구축하게 될 모델과 직접적인 연관성이 있는건 아니고, Rundeck 이라는 프로그램 또한 분야에서 자주 활용되는 툴은 아닌 것 같다는 인상을 받았다. 

다만 데이터 생성 과정을 여러대의 PC에 분산하고, 이를 모니터링 할 수 있는 체계는 생각보다 유용했고, 다시 사용할 일이 있지않을까 하는 생각이 들었다. 향후에는 조금 더 언급량이 많은 Ansible이나 Airflow같은 툴을 리눅스 기반의 환경에서 사용해보고 싶다.