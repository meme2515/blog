---
title: "웹 어플리케이션 배포 - WSGI 와 NGINX 서버란"
description: "Web Server, WAS 의 개념과 각 서버의 필요성 설명"
date: "2023-03-26 00:00:00"
slug: "wsgi"
image: computer_science/images/wsgi_1.png
tags: [WSGI, NGINX, gunicorn, flask, 웹앱, 웹앱 개발]
categories: [WSGI, NGINX, Flask]
---
## Introduction

이번 글에서는 웹 개발에서 자주 등장하는 단어인 wsgi, nginx, werkzeug 등을 설명한다.

## Web Server vs. WAS

먼저 Web Server 와 Web Application Server (WAS) 의 개념을 살펴볼 필요가 있다. 

현대적인 웹페이지엔 **정적 요소** (Static Pages - image, html, css, javascript) 와 **동적 요소** (Dynamic Pages - python, database) 가 혼재되어 있다. 보통 이러한 정적인 요소를 클라이언트 사이드에서 처리되는 프론트엔드, 동적인 요소를 서버 사이드에서 처리되는 백엔드로 지칭한다.

하지만 클라이언트 사이드에서 처리되는 html, css 또한 결국 웹 서버에서 관련된 파일을 내려받는 과정이 필요한데, 이러한 정적인 요소를 담당하는 서버를 **Web Server**, 그리고 동적인 요소를 담당하는 서버를 **WAS** 로 설정하는 것이다.

| ![alt text](computer_science/images/wsgi_6.png) |
|:--:|
| Fig 1. 웹서버와 WAS 의 구성 |

WSGI 는 이와 같은 구조에서 동적인 요소를 담당하는 WAS 를 구축하는데 활용된다. 반면 NGINX 는 Web Server 의 한 종류로서, 웹사이트의 정적인 요소를 담당하고 필요 시 WSGI 서버와 통신하여 동적인 결과값을 클라이언트에서 서빙하는 역할 또한 수행하는 것.

## WSGI

로컬 환경에서 서버를 구동해본 경험이 있다면 다음과 같은 메시지에 익숙할 것이다. 

{{< highlight zsh >}}
WARNING: This is a development server. Do not use it in a production deployment.
Use a production WSGI server instead

* Restarting with stat
* Debugger is active!
* Debugger PIN: 123-456-789
* Running on http://127.0.0.1:5000/
{{< /highlight >}}

이와 같은 메시지가 뜨는 이유는, Flask 나 Django 같은 백엔드 프레임워크에서 기본적으로 구동되는 서버는 실제 배포를 염두하고 제작된 서버 구조가 아니기 때문이다. 이와 같은 서버는 Werkzeug (벨저크) 패키지에서 제공하는 개발 서버인데, HTTP 기능이 제한되어 있고, 효율성, 안정성, 보안성에서 많은 취약점이 존재한다 *(하지만 속도가 빠르고, 개발에 용이한 기능들을 제공)*. 따라서 배포 환경에 알맞은 WSGI HTTP 서버를 활용할 필요가 발생한다.

| ![alt text](computer_science/images/wsgi_5.png) |
|:--:|
| Fig 2. WSGI 서버 활용 예시 |

WSGI (Web Server Gateway Interface) 란 유저로 부터 전송받은 HTTP Request 를 Django 와 같은 백엔드 프레임워크가 이해할 수 있는 Python 오브젝트로 변환하는 역할을 수행한다. 앞서 언급한 벨저크 서버 또한 사실 이러한 기능을 수행하고 있지만, 개발 환경에 맞는 WSGI 서버란 다수의 워커를 구성해 분산 처리가 가능하고, 개발 서버의 취약점을 보완한 버전이라고 생각할 수 있다.

프레임워크와 WSGI 서버를 구분하는 또 다른 장점은 백엔드 프레임워크가 클라이언트 연결을 유지하는 작업을 수행할 필요 없이, 주어진 Request 에 대한 Response 값을 반환하는 작업에만 집중할 수 있다는 점이다. Django 와 같은 백엔드 프레임워크는 애초에 이러한 목적으로 설계된 패키지이기 때문에, 가능한 모든 상황에서 WSGI 래퍼를 활용해야 한다.

### WSGI Standard

| ![alt text](computer_science/images/wsgi_4.jpeg) |
|:--:|
| Fig 3. Gunicorn 로고 |

WSGI 스탠다드란 모든 WSGI 서버와 프레임워크가 서로 호환할 수 있도록 약속된 통신 프로토콜이다. 대표적인 WSGI 서버인 Gunicorn, uWSGI 등은 이러한 스탠다드를 따르기 때문에 WSGI 프레임워크인 Django, Flask, Pyramid, Falcon 등과 모두 호환이 가능하다.

## NGINX

언급했듯 NGINX 는 웹사이트의 정적인 요소를 담당하는 Web Server 이다. 클라이언트가 웹서버에 html, css 등의 파일을 요청하면, 이를 빠르게 서빙하여 WAS 가 동작할 필요가 없게끔 만들어준다.

### Reverse Proxy

NGINX 와 WSGI 서버는 Reverse Proxy 방식을 통해 연결된다. 여기서 Reverse Proxy 란 사용자가 흔히 활용하는 Proxy 서버와 반대되는 개념인데, 클라이언트가 서버로 보내는 HTTP 정보가 우회될때 이를 **Forward Proxy**, 서버가 클라이언트에게 보내는 HTTP 정보가 우회될때 이를 **Reverse Proxy** 라고 구분지어 부르는 것. 

### Web Server 가 반드시 구분되어야 할까?

결론만 놓고 보면 Django, WSGI 서버만 있어도 웹사이트는 정상적으로 작동한다. 현대적인 WSGI 서버는 정적인 요소까지 모두 서빙하는 기능을 제공하기 때문이다. 

하지만 대부분의 경우 별도 웹서버를 구축하는 것이 권장되는 가장 큰 이유는 다음과 같다.

- 서버 부하 방지
    - WAS 는 데이터베이스 조회와 다양한 연산 처리로 이미 많은 연산 자원을 활용하는 상태이다. 때문에 단순한 정적 컨텐츠는 Web Server 에서 빠르게 전달하는 것이 유저의 체감 속도도 높이고, 서버 부하 또한 줄일 수 있다.
- 여러 대의 WAS 를 활용한 로드 밸런싱
    - 많은 트래픽이 발생할 경우 WAS 의 수를 늘려, 이를 하나의 Web Server 와 연동하는 것이 가능하다. 

| ![alt text](computer_science/images/wsgi_3.png) |
|:--:|
| Fig 4. NGINX 서버 활용 예시 |

정리하자면 Django, Flask 등의 백엔드 서버 배포 시 동적인 요소와 정적인 요소를 구분하기 위해 NGINX 서버를 구축하고, NGINX 서버와 Django 에서 개발된 백엔드와의 소통을 위해 WSGI 서버가 구축된다. 경우에 따라 일부 요소들이 생략되는 경우가 발생할 수 있지만 *(Jekyll, Hugo 와 같은 Static Site 는 구조상 백엔드를 필요로하지 않는다. 또한 트래픽이 작다면 웹서버 없이 프로토타이핑이 가능하다)*, 일반적인 형태의 웹사이트는 보통 이러한 구조를 따라 구축되는 것이 강하게 권장된다고 볼 수 있다.

## Reference

- https://wayhome25.github.io/django/2018/03/03/django-deploy-02-nginx-wsgi/
- https://www.youtube.com/watch?v=66xlIunxWYQ
- https://binux.tistory.com/32