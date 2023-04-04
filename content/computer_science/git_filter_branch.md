---
title: "용량이 큰 파일을 실수로 커밋한 경우 대처 방안"
description: "Git Filter Branch 기능 소개 및 실사용 예시"
date: "2023-04-04 00:00:00"
slug: "git_filter_branch"
image: computer_science/images/git_title.svg
tags: [Git, GitHub]
categories: [Git, GitHub, GitLab]
draft: "true"
---

## TL;DR

{{< highlight zsh >}}
git filter-branch --index-filter "git rm -rf --cached --ignore-unmatch path_to_file" HEAD
{{< /highlight >}}

## Introduction

Git 은 본질적으로 코드 버전 관리 (VCS) 를 위해 설계된 툴이기 때문에 용량이 큰 데이터 파일이 커밋된 경우, 여러가지 문제점을 초래할 수 있다. 우선 깃허브는 100MB 가 넘는 용량의 파일이 커밋된 경우 repository 로 push 하는 것 조차 허용하지 않는데, 따라서 데이터 버저닝을 수행하고자 하는 경우 이를 위해 설계된 [Git LFS](https://git-lfs.com/), [DVC](https://dvc.org/) 등의 별도 툴을 Git 과 같이 활용해주어야 한다. 

본 글은 만일 의도치 않게 git 에 용량이 큰 데이터, 혹은 파일을 커밋한 경우, 전체 히스토리를 리셋할 필요 없이 특정 파일만 커밋 히스토리에서 삭제하는 방법을 짧게 소개한다. 

## git filter-branch

우선 filter-branch 기능은 커밋 히스토리를 전반적으로 수정하기 때문에 동작하기 이전 상태로 복구가 불가능하고, 협업시 많은 문제점을 초래할 수 있기 때문에 활용이 권장되지 않는다. 하지만 그럼에도 불구하고 실수로 높은 용량의 파일을 커밋하였다면, 상기된 명령어를 통해 해당 파일과 관련된 히스토리를 삭제할 수 있다. 

여기서 filter-branch 커맨드는 특정한 branch 의 모든 커밋에 필터를 적용한 후, 

## References

1. https://stackoverflow.com/questions/8083282/how-do-i-remove-a-big-file-wrongly-committed-in-git
2. https://git-scm.com/docs/git-filter-branch
3. https://manishearth.github.io/blog/2017/03/05/understanding-git-filter-branch/