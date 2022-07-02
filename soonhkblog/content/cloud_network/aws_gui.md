---
title: "AWS EC2 인스턴스 생성 및 XRDP GUI 설정"
description: "EC2 기본 인스턴스 생성 및 XRDP/Putty 설치, RDP 접속 과정 상세"
date: "2022-07-02 00:00:00"
slug: "aws_gui"
image: cloud_network/aws_gui_2.jpg
tags: [EC2, XRDP, AWS GUI, AWS 그래픽 인터페이스, Putty, RDP]
categories: [XRDP, AWS, EC2, GUI]
---
## 소개

본 글에서는 [AWS](https://aws.amazon.com) 에서 가장 기본적인 EC2 인스턴스 생성 및 [우분투](https://ubuntu.com/) 설치, [XRDP](http://xrdp.org/) 를 활용한 GUI를 설정까지의 과정을 정리하고자 한다. 로컬 환경은 Windows 11 이며, 2022년 7월 2일을 기준으로 작성되었다.

## 왜 리눅스를 쓸까?

분석가와 데이터 사이언티스트에게 클라우드 장점은 직관적으로 다가온다. 우선 떠오르는 몇가지를 다음과 같이 정리해보았다.

1. 몇 대의 PC를 사용하듯 수요에 따라 자유롭게 인스턴스 생성 및 **성능/용량 조절이 가능하다**.
2. 하드웨어 관리에 대한 걱정없이 **24시간 사용이 가능**하다.
3. Database, Analytics, ML 등 영역마다 특화된 제품들이 출시되어 있어 별도 **설계없이 쉽게 사용**할 수 있다.

하지만 이와 달리 클라우드 EC2 환경에서 리눅스를 사용이 권장되는 이유는 컴퓨터 공학과가 아니라면 다소 공감하기 어려울 수 있다 (당장 내가 그랬었다). 당장 로컬환경에서 사용되는 운영체제가 Windows, 혹은 MacOS 인데, 사적이건 공적이건 클라우드 환경에서 익숙하지 않은 리눅스를 사용하기 위해서는 그만한 학습시간이 필요하기 때문에 정당화가 필요하다.

우선 EC2 환경은 다음과 같이 Windows, MacOS 를 모두 지원하고 있다.

| ![alt text](cloud_network/aws_gui_3_2.png) |
|:--:|
| Fig 1. EC2 환경 설정 페이지 캡처 |

그럼에도 불구하고 서버, 이 경우 클라우드 환경에서 리눅스 운영체제가 선호되는 이유는 다음과 같이 정리할 수 있다. 

1. 리눅스는 **무료이며, 오픈소스이다**. 이는 개발처가 분명한 Windows나 MacOS에 비해 유지보수 비용이 낮음을 의미한다.
2. 리눅스는 기타 운영체제에 비해 **우수한 안정성/보안성**이 검증되었다. 
3. 리눅스는 커맨드 라인 기반 운영체제이기 때문에 기타 운영체제에 비해 **가볍고 빠르다**.

교과서적인 내용을 벗어나, 당장 [Docker](https://www.docker.com/), [Ansible](https://www.ansible.com/), [Airflow](https://airflow.apache.org/) 과 같은 개발/데이터 분야에서 사용되는 툴들은 기본적으로 윈도우를 지원하지 않는다. 이는 위와 같은 이유로 이미 서버/개발 환경에서 리눅스 사용이 기준으로 자리잡았기 때문이며, 따라서 커뮤니티의 트렌드를 쫒다보면 자연스럽게 리눅스를 사용하고 있는 자신을 발견하게된다.

## EC2

온프레미스 서버를 대체하는 개념으로 AWS의 가장 핵심이 되는 서비스이다. 아마존으로부터 한 대의 컴퓨터를 임대하는 개념으로 생각할 수 있다. 다음을 통해 EC2의 인스턴스를 생성하는 기본적인 방법을 살펴보자.

### 인스턴스 생성

EC2 콘솔에 접속해 다음과 같은 Launch Instance 옵션을 클릭한다. 

![alt text](cloud_network/aws_gui_3_3.png)

서버에 적절할 이름을 지어주고, 운영체제 (AWS는 이를 AMI, Amazon Machine Image, 라 부르며 적절한 소프트웨어가 기본으로 탑재되어있다) 로 Ubuntu 를 선택한다.

![alt text](cloud_network/aws_gui_3_4.png)

인스턴스의 성능과 보안체계를 설정하는 부분이다. 무료 티어가 적용되는 t2.micro 를 선택하고, key pair 를 생성해 로컬 폴더에 다운로드하면 된다. 다운로드 시 파일 포맷은 .pem 으로 지정한다.

![alt text](cloud_network/aws_gui_3_5.png)

이후 Launch Instance 버튼을 클릭하면 EC2 설정이 끝난 것이다. Instances 페이지에서 다음과 같이 생성한 인스턴스가 동작하는 것을 확인할 수 있다.

![alt text](cloud_network/aws_gui_3_6.png)

### SSH 연결

인스턴스 생성이 끝났다면 [SSH (Secure Shell)](https://en.wikipedia.org/wiki/Secure_Shell) 프로토콜을 이용해 인스턴스의 커맨드 라인에 접속할 수 있다. Instances 페이지에서 생성된 인스턴스 선택 후, 상단의 Connect 버튼을 누른다.

![alt text](cloud_network/aws_gui_3_7.png)

SSH Client 탭의 정보를 기반으로 [Git Bash](https://git-scm.com/downloads), 혹은 Windows Powershell 에서 다음 커맨드를 입력한다. 커맨드 입력시에는 생성한 key pair 의 .pem 파일과 동일한 디렉토리에 있어야한다.

```
 ssh -i "{keypair.pem}" ubuntu@{ec2_public_DNS}
```

![alt text](cloud_network/aws_gui_3_8.png)

이로서 EC2 인스턴스의 커맨드 라인 사용을 위한 세팅이 끝났다!

## XDRP GUI

XDRP란 윈도우의 원격 접속 프토로콜인 [RDP (Remote Desktop Protocol)](https://support.microsoft.com/en-us/windows/how-to-use-remote-desktop-5fe128d5-8fb1-7a23-3b8a-41e636865e8c) 를 통해 원격 리눅스 서버에 접속을 가능하게 해주는 오픈소스 소프트웨어이다. 커맨드 라인 접속 프로토콜인 SSH와 달리 우분투의 그래픽 인터페이스 사용이 가능하다. 인스턴스에 대한 SSH 접속을 마쳤다면 다음 커맨드를 순차적으로 실행해 Ubuntu 에 XDRP 를 설치 및 설정하자.

### 우분투 XDRP 설치

```
 sudo apt update && sudo apt upgrade
 sudo sed -i 's/^PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config
 sudo /etc/init.d/ssh restart
```

과정에서 커맨드 라인이 Y/n 값을 요구한다면 Y 를 입력 후 엔터를 누르면 된다. 다음과 같은 화면이 나오면 OK를 선택하자.

![alt text](cloud_network/aws_gui_3_9.png)

```
 sudo passwd ubuntu
```

우분투에 대한 비밀번호 설정이다. 설정하고 싶은 비밀번호를 2회 입력하면 된다.

![alt text](cloud_network/aws_gui_3_10.png)

```
 sudo apt install xrdp xfce4 xfce4-goodies tightvncserver
 echo xfce4-session> /home/ubuntu/.xsession
 sudo cp /home/ubuntu/.xsession /etc/skel
 sudo sed -i '0,/-1/s//ask-1/' /etc/xrdp/xrdp.ini
 sudo service xrdp restart
 sudo reboot
```

인스턴스가 재시작 될때까지 기다린 후, 다시 SSH 로 접속하자.

### Putty

[Putty](https://www.putty.org/) 란 원격 접속을 위한 소프트웨어이다 ([세부 설명 및 사용법 기초](https://dololak.tistory.com/24)). 단순 SSH 접속에 비해 더 세부적인 설정이 가능하며, XDRP 사용을 위해 설치가 필요하다. 다음 커맨드를 통해 Ubuntu 인스턴스에 Putty 를 설치하자.

```
 sudo apt-get install putty
```

이후 AWS 인스턴스 페이지에서 Public IP 와 Private IP 를 확인해야 한다.

![alt text](cloud_network/aws_gui_3_11.png)

Windows 로컬 환경으로 돌아와, https://www.putty.org/ 에서 클라이언트를 다운로드 및 실행하면 다음과 같은 화면을 확인할 수 있다. Host Name 에 확인한 Public IP 를 입력하자.

![alt text](cloud_network/aws_gui_3_12.png)

Putty 클라이언트의 SSH -> Tunnels 설정으로 들어가 Destination 란에 Private IP 와 3389 포트를 입력한다. 이후 Source Port를 9000 으로 지정 후 Add 버튼을 클릭.

![alt text](cloud_network/aws_gui_3_17.png)

SSH -> Auth 설정으로 들어가 저장한 .pem 파일을 찾아준다.

![alt text](cloud_network/aws_gui_3_14.png)

클라이언트의 Open 버튼을 클릭한 후, 확인 버튼을 누르면 로그인창이 뜨게 된다. id는 ubuntu, 비밀번호는 이전에 설정했던 비밀번호를 입력하자.

![alt text](cloud_network/aws_gui_3_15.png)

Putty 접속이 완료됐다. GUI 실행 시에는 이처럼 Putty 접속이 항상 선행되어야 한다.

### Remote Desktop

GUI 접속 단계이다. Windows 에 기본 설치되어있는 RDP 클라이언트 실행 후, 127.0.0.1:8888 에 접속하자.

![alt text](cloud_network/aws_gui_3_18.png)

다음 화면에서 username 은 ubuntu, password 는 지정한 비밀번호를 입력해준다.

![alt text](cloud_network/aws_gui_3_19.png)

GUI 접속 화면을 확인할 수 있다. 이제 필요한 환경설정을 진행해주면 된다.

![alt text](cloud_network/aws_gui_3_20.png)