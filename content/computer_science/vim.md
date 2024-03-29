---
title: "Vim 에디터 소개와 기본적인 명령어 정리"
description: "CLI 에디터 사용 이유, 기본 개념 소개 및 단축키 Cheatsheet"
date: "2022-09-05 00:00:00"
slug: "vim"
image: computer_science/images/vim.bmp
tags: [Vim, 리눅스, 유닉스, Vim 에디터, 커맨드 라인]
categories: [리눅스, Vim]
---

## MIT에서 공개한 Vim 관련 강의

유명한 [The Missing Semester of Your CS Education](https://missing.csail.mit.edu/) 의 Vim 관련 강의 비디오 링크.

[![image](computer_science/images/vim_2.png)](https://www.youtube.com/watch?v=a6Q8Na575qc)

## 등장 배경과 특징

1976년 배포된 유닉스 운영체제의 텍스트 에디터 프로그램이었던 [vi](https://en.wikipedia.org/wiki/Vi) 의 개선판으로, 그 이름 또한 Vi IMproved 의 약자이다. 최초 버전은 1991년 배포되었으며, 현재까지 VS Code, Sublime 등과 함께 개발자들이 가장 선호하는 텍스트 에디터 중 하나이다. 아직까지도 리눅스, 맥OS 등의 주요 유닉스 기반 운영체제에 기본으로 탑재되어 있다.

Vim을 처음 접한 사용자는 마우스 사용 없이 문서 수정이 이루어지는 환경이 당황스럽게 받아들여질 수 밖에 없다. 하지만 이는 일정 수준 사용에 익숙해지면 마우스를 조작하는 시간을 절약할 수 있다는 의미이기도 하다 (물론 나는 익숙하지 않다). 당연하지만 Vim 의 문서 수정 방식은 VSCode 같은 기존 텍스트 에디터를 더 편하게 사용하기 위해 고안된 것이 아니라, 주로 검은 화면에 키보드로 입력한 명령어만으로 컴퓨터를 조작하던 시절 문서를 수정하기 위한 가장 현실적인 방안으로 고안된 것이다.

Vim 은 개발자들이 문서를 작성하기보다 수정하는 일에 더 많은 시간을 보낸다는 점에 집중한다. 따라서 기능의 많은 부분들이 방대한 양의 텍스트를 효율적으로 다룰수 있도록 설계되어있다. 또한 맥OS, 리눅스, 윈도우 환경에서 모두 쉽고 빠르게 사용할 수 있기 때문에 범용성이 높다는 특징 또한 존재한다 (사실 개발자용 텍스트 에디터는 모두 이렇기에 특징이라고 하기는 어렵다).

## 장단점 정리 및 기본 개념 소개

### 기타 환경 대비 장단점

장점

- 일정 수준 이상 익숙해진다면 마우스+키보드 조합에 비해 월등한 효율을 낼 수 있다.
- .vimrc 파일 수정을 통해 생각할 수 있는 거의 모든 customization 이 가능하다. 또한 다른 유저가 공개한 설정을 사용함으로서 바로 효율적인 세팅을 이용할 수 있다.
- SSH 터미널 세션에서 별도 GUI 로딩이 필요하지 않다. GUI 세팅이 어렵다면 사실상 유일하게 사용할 수 있는 텍스트 에디터이다 (이게 크다).

단점

- 단축키, 각종 모드 등 배워야 할 것이 많다. 굳이 배우지 않아도 예외적인 사례 몇가지를 제외하면 사실 개발에 큰 지장을 주지 않는다.
- 폰트 구분, 이미지 렌더링, UI 개발 등 여러 현대적인 그래픽 기반 기능들을 태생적으로 제공하지 못한다.

뭣모르고 하는 소리일 수 있겠지만, 개인적으로는 VSCode 와 같은 주류 IDE 를 메인으로 사용하고, SSH 접속과 같은 상황에 Vim 을 서브 에디터로 사용하는 것이 바람직하다고 생각한다. 굳이 모든 상황에 CLI 환경 에디터를 고집함으로서 생기는 득보다는 실이 크지 않을까 조심스럽게 적어본다.

### 데이터 사이언티스트는 Vim을 공부해야할까?

나는 그렇다고 생각한다, 다만 너무 깊이 들어갈 필요는 없을듯하다. 대학교, 부트캠프 등에서 노트북 기반 환경에 익숙해진 초심자는 커맨드 라인에서 문서 편집이 오히려 불편하고, 귀찮은 경험이 될 수 있다. 하지만 클라우드 등 서버 컴퓨팅을 활용 시 발생할 수 있는 여러 문제들 (사내 보안, 자원 절약, 한정된 시간 등) 로 인해 Vim 은 유일한 문서 편집 방법이 될 가능성이 많다.

또한 단순 실험에서 벗어나 스크립트를 작성하고, 모델을 저장/로드하는 과정에서 터미널 사용은 필수적이다. 터미널에서 파일을 브라우징하며 짤막한 코드 수정이 필요할때 Vim 은 실제로 많은 시간을 단축시켜준다.

### Vim 모드

Vim 은 모드 기반의 에디터이다. 여타 텍스트 에디터가 파일을 열게되면 바로 편집, 읽기 기능을 제공하는 것과 다르게 Vim은 주로 문서 탐색 기능을 제공하는 Normal 모드, 편집 기능을 제공하는 Insert 모드, 명령어 입력을 지원하는 Command 모드, 하이라이팅 기능을 제공하는 Visual 모드로 구분할 수 있다.

#### Normal 모드

개발자가 가장 많은 시간을 할애하는 모드이며, `vim filename` 커맨드로 문서를 열면 기본적으로 Normal 모드에서 편집을 시작하게 된다. Undo, redo, find, replace 등 직접적인 텍스트 입력 및 하이라이팅을 제외한 거의 모든 기능을 제공하며, 기타 모드에서 `esc` 키를 클릭하면 다시 Normal 모드로 돌아올 수 있다.

#### Insert 모드

직접적인 텍스트 입력을 지원하며, Normal 모드에서 `i` 키를 클릭해 전환한다. 일반적인 텍스트 에디터와 동일한 상태라고 생각하면 되지만, Vim 의 강점인 단축키가 대부분 지원되지 않는다.

#### Command 모드

프로그래밍 언어처럼 명령어를 입력할 수 있는 모드이며, Normal 모드에서 `:` 키를 입력해 전환할 수 있다. 예를 들자면 `:q` 입력 후 엔터를 누르면 문서를 닫는 식이다.

#### Visual 모드

하이라이팅 기능을 제공하며, Normal 모드에서 `v` 키를 입력해 전환한다. 주로 코드의 특정 부분을 선택해 복사 및 잘라내기 기능을 수행할때 활용한다.

### 기본적인 커맨드 정리

다음 커맨드들은 별도 표기가 없다면 모두 Normal 모드에서만 지원된다.

#### 저장 및 파일 닫기

`:w filename` 현재 문서를 filename 문서에 저장한다. 문서 이름을 지정하지 않으면 현재 문서에 저장한다.

`:q` Vim 을 종료한다. 문서 편집이 이루어졌다면 저장이 필요하다는 문구가 뜨게된다.

`:q!` 문서 저장없이 Vim 을 종료한다.

`wq` 문서를 저장하고 Vim 을 종료한다.

#### 파일 탐색

화살표키, 또는 j, k, h, l 키로 커서를 이동할 수 있다. 권장되는 탐색 방법은 후자인데, 단축키 조합이 빈번한 Vim 에서 손가락의 움직임을 최소화할 수 있기 때문이다.

`j` 아래 `k` 위 `h` 왼쪽 `l` 오른쪽 이동에 해당한다.

`w` 다음 단어로 이동. `B` 이전 단어로 이동.

`b` 단어의 처음으로 이동. `e` 단어의 마지막으로 이동.

`0` 현재 라인의 처음으로 이동. `$` 현재 라인의 마지막으로 이동.

`:123` 123 번째 줄로 이동.

`ctrl-f` 한페이지 위로. `ctrl-b` 한페이지 아래로.

`ctrl-u` 반페이지 위로. `ctrl-d` 반페이지 아래로.

`gg` 파일의 첫 라인으로 이동. `G` 파일의 마지막 라인으로 이동.

#### 찾기 기능

`/foo` 파일에서 foo 를 검색한다. 엔터를 누르면 검색 결과간 이동이 가능하다.

`n` 다음 검색 결과로 이동.

`N` 이전 검색 결과로 이동.

#### 텍스트 편집

`dd` 혹은 `:d` 현재 라인 잘라내기.

`yy` 혹은 `:y` 현재 라인 복사하기.

`p` 잘라내거나 복사한 내용 붙여넣기.

#### Undo/Redo

`u` 마지막 액션 취소.

`U` 현재 라인에 모든 수정 내용 취소.

`ctrl-u` redo.

#### 텍스트 하이라이트

`v` 캐릭터 레벨에서 하이라이트.

`V` 라인 레벨에서 하이라이트 (위/아래 이동만 가능).

`ctrl-v` 행렬 레벨에서 하이라이트.

텍스트 하이라이트 후, 잘라내기 (`d`), 복사 (`y`), 붙여넣기 (`p`) 등의 기능을 사용할 수 있다. Visual 모드에서 Normal 모드로 돌아가기 위해서는 `esc` 키를 누르면 된다.

## 더 나아가기

[The Missing Semester](https://www.youtube.com/watch?v=a6Q8Na575qc)와 [Stanford CS107](https://web.stanford.edu/class/cs107/resources/vim.html)에서 짧지만 양질의 Vim 강의를 제공한다. 두 강의 모두 기본적인 소개는 물론 고급 사용법의 공부법 또한 제공하니 관심이 있다면 참고하면 좋을듯 하다.

## 참고 링크

1. https://en.wikipedia.org/wiki/Vim_(text_editor)
2. https://en.wikipedia.org/wiki/Vi
3. https://medium.com/@fay_jai/what-is-vim-and-why-use-vim-54c67ce3c18e
4. https://www.youtube.com/watch?v=a6Q8Na575qc
5. https://web.stanford.edu/class/cs107/resources/vim.html