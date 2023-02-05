---
title: "Full Stack Deep Learning 2022 부트캠프 - Week 7"
description: "Foundation Models"
date: "2022-12-07 00:00:00"
slug: "fsdl_7"
image: "mlops/images/fsdl_7_title.png"
tags: [FSDL, Full Stack Deep Learning, MLOps, 부트캠프, fsdl 2022, fsdl 2022 후기, fsdl 후기, full stack deep learning 2022, 풀스택딥러닝]
categories: [FSDL, MLOps, 부트캠프]
---
- [YouTube](https://www.youtube.com/watch?v=Rm11UeGwGgk&list=PL1T8fO7ArWleMMI8KPJ_5D5XSlovTW_Ur), [Lecture Notes](https://fullstackdeeplearning.com/course/2022/lecture-7-foundation-models/), [Slides](https://drive.google.com/file/d/17ZAj6izyYhV-SXA_UKNWjYo0adbL2E8n/view)

## Lecture 내용 요약

- [FSDL 2022 Course Overview](https://meme2515.github.io/mlops/fsdl/)
- [Lecture 1 - When to Use ML and Course Vision](http://meme2515.github.io/mlops/fsdl_1/)
- [Lecture 2 - Development Infrastureture & Tooling](http://meme2515.github.io/mlops/fsdl_2/)
- [Lecture 3 - Troubleshooting & Testing](http://meme2515.github.io/mlops/fsdl_3/)
- [Lecture 4 - Data Management](http://meme2515.github.io/mlops/fsdl_4/)
- [Lecture 5 - Deployment](http://meme2515.github.io/mlops/fsdl_5/)
- [Lecture 6 - Continual Learning](http://meme2515.github.io/mlops/fsdl_6/)
- [Lecture 7 - Foundation Models](http://meme2515.github.io/mlops/fsdl_7/)
- [Lecture 8 - ML Teams and Project Management](http://meme2515.github.io/mlops/fsdl_8/)
- [Lecture 9 - Ethics](http://meme2515.github.io/mlops/fsdl_9/)

## Overview

- 기초 모델 (foundation model) 이란 방대한 양의 데이터를 학습한 큰 규모의 모델을 뜻하며, 다양한 task 에 응용될 수 있다.
- 본 주차에서는 Fine-Tuning, Transformers, LLM, Prompt Engineering, CLIP 등의 주제를 다룰 예정.

![alt text](mlops/images/fsdl_7_1.png)

## Fine-Tuning

- 전통적인 ML 은 규모가 큰 모델에 방대한 양의 데이터를 학습하지만, 보유한 데이터가 제한적일 때에는 **전이 학습**을 통해 성능을 향상 시킬 수 있다.
- 전이 학습 (transfer training) 의 기본적인 방법론은 pre-train 된 동일한 모델에 레이어를 덧붙인 후, 일부 가중치를 학습하는 것.
- 비전 분야에서는 2014년 부터 적용되어왔던 방식이며, Model Zoo 상에서 사전 학습이 이루어진 AlexNet, ResNet 과 같은 모델을 쉽게 찾을 수 있다.
- NLP 분야의 경우, 사전 학습은 word embedding 영역에 한정되어왔다.

![alt text](mlops/images/fsdl_7_2.png)

- **[Word2Vec](https://jalammar.github.io/illustrated-word2vec/)** 은 2013년도 자주 함께 등장하는 단어 간 cosine similarity 를 최소화하는 모델을 학습 시켰다. 결과적으로 얻은 벡터를 활용해 연산 작업을 하는 등 다양한 적용 예시를 보임.
- 단순한 단어의 의미 뿐 아니라 맥락에 대한 이해 또한 embedding 작업에 중요한 역할을 수행한다. 2018 년도 등장한 **ELMO, ULMFit** 등이 이에 속하며, 이들 모델은 사전 학습이 완료된 LSTM 모델을 공개했다.
- 하지만 최근 Model Zoo 를 살피면 LSTM 모델은 자취를 감춘 모양새이다. 최근 더욱 활발히 연구되고 있는 모델 구조는 트랜스포머 이기 때문.

## Transformers

- 트랜스포머는 2017년 유명한 논문 “Attention Is All You Need” 에서 처음 등장했으며, 기존 NLP 와는 확연히 다른 구조로 번역 등의 분야에서 독보적인 성능을 보였다.
- 블로그 주인이 별도로 작성한 트랜스포머 소개글이 이미 있으니 대신 참고하면 좋을 듯 하다 - [Transformer 네트워크 개념 소개](https://meme2515.github.io/neural_network/transformer/)

## Large Language Models

### Models

- GPT 와 GPT-2 는 각각 2018 년과 2019 년에 공개되었다. 기존 트랜스포머 구조에서 Decoder 만을 활용해 구축된 모델이며, Masked Self-Attention 을 차용했다.

![alt text](mlops/images/fsdl_7_3.png)

- 이러한 모델들은 약 800 만 개의 웹페이지로 학습되었으며, 가장 큰 모델은 약 15억 개의 파라미터를 가지고 있다.
- GPT-2 가 학습된 방식은 웹 데이터를 기반으로 다음 단어를 예측하는 작업이다. 이러한 작업은 파라미터 수가 증가할 수록 더욱 자연스러워 진다는 특징을 가지고 있다.

![alt text](mlops/images/fsdl_7_4.png)

- BERT 는 유사한 시점에 공개되었다. 트랜스포머 구조의 인코더만을 활용해 만들어졌으며, Attention Masking 을 적용하지 않았다.
- 약 1억 천만 개의 파라미터를 가지고 있으며, 학습 과정에서 BERT 는 random 한 단어를 마스킹 처리한 후, 해당 단어가 어떤 단어인지 예측하는 task 를 수행한다.

![alt text](mlops/images/fsdl_7_5.png)

- **[T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)** (Text-to-Text Transformer) 은 2020 년 공개된 모델이다. 인풋과 아웃풋 모두 텍스트이기 떄문에 모델이 어떤 작업을 수행해야 하는지 특정하는 것이 가능.
- T5 는 Encoder-Decoder 구조를 가지고 있고, 위키피디아 보다 약 100배 큰 C4 데이터를 기반으로 학습되었다.
- **[GPT-3](https://openai.com/blog/gpt-3-apps/)** 는 2020년에 공개된 모델이며 기존 GPT/GPT-2 보다 약 100배의 크기를 가지고 있다. 엄청난 규모로 인해 few-shot learning, zero-shot learning 등의 작업에 이전에 달성하지 못한 정확도를 기록했다.
- 하단 그래프에서 보이듯, 데이터 수가 증가할수록 모델의 성능이 linear 하게 증가하는 것을 확인할 수 있다. 또한, 파라미터가 많을 수록 모델 성능이 증가한다.

![alt text](mlops/images/fsdl_7_6.png)

- OpenAI 는 또한 2022 년 초 **[Instruct-GPT](https://openai.com/blog/instruction-following/)** 를 공개했다. GPT-3 의 아웃풋에 대해 사람이 순위를 매기도록 하여, 파인튜닝을 진행하는 방식.
- Instruct-GPT 는 특정한 지시를 따르는데 개선된 성능을 보이며 (하기 이미지 참고), API 상 ‘text-davinci-002’ 라는 이름으로 공개되어 있는 상태이다.

![alt text](mlops/images/fsdl_7_7.png)

- DeepMind 는 2021년 **[RETRO (Retrieval-Enhanced Transformers)](https://arxiv.org/pdf/2112.04426.pdf)** 라는 모델을 공개했다. 단순히 파라미터에 정보를 담는 것이 아니라, 별도 데이터베이스에 이에 대한 정보를 저장하는 방식.
- BERT 를 활용해 여러 문장을 인코딩 후, 약 1조 개의 토큰을 가진 데이터베이스에 저장하게 된다. 예측 시 매칭되는 문장을 조회해 정보를 추출하게 되며, 항상 최신 정보를 조회할 수 있다는 장점을 가진다.

![alt text](mlops/images/fsdl_7_8.png)

- DeepMind 는 또한 2022 년 **[Chinchilla](https://gpt3demo.com/apps/chinchilla-deepmind)** 라는 모델을 공개했으며, 언어 모델의 크기와 성능에 대한 상관 관계를 관찰했다는 점에 의의가 있다.
- DeepMind 팀은 약 7천만 ~ 160억 개의 파라미터를 가진 [400 개의 언어 모델](https://arxiv.org/pdf/2203.15556.pdf)을, 50억 에서 5000억 개 까지의 토큰을 활용해 학습했으며, 최적의 파라미터와 데이터 셋 사이즈를 유추하는 수식을 통해 대부분의 언어 모델이 undertrain 되었다는 결론을 내렸다 (더 큰 데이터 셋이 필요).

![alt text](mlops/images/fsdl_7_9.png)

- 이를 증명하기 위해 DeepMind 팀은 약 2800 억 개의 파라미터를 가진 **[Gopher](https://gpt3demo.com/apps/deepmind-gopher)** 라는 모델을 3000 개의 토큰으로 학습하였으며, Gopher 의 학습 결과를 Chinchilla (약 700 억 개의 파라미터, 1조 4천 개의) 의 것과 비교해 Chinchilla 의 성능이 더욱 우수하다는 점을 도출.
- 즉 파라미터 수가 적더라도 데이터가 많다면 모델은 더 우수한 성능을 내는 것이 가능하다는 결론을 내렸다.

### Vendors

- OpenAI 는 다음과 같은 모델 사이즈 옵션을 제공 : Ada3 (3억 5천만 파라미터), Babbage (13억 파라미터), Curie (67억 파라미터), Davinci (1750억 파라미터). [각 옵션은 서로 다른 가격 정책과 기능을 제공한다](https://openai.com/api/pricing/).
- GPT-3 의 눈에 띄는 결과들은 대부분 Davinci 를 기반으로 하며, 추가 비용을 들인다면 모델의 fine-tuning 또한 진행할 수 있다.
- OpenAI 의 대안은 다음과 같다:
    - [Cohere AI](https://cohere.ai/) 는 유사한 모델을 유사한 가격으로 제공
    - [AI21](https://www.ai21.com/) 는 보다 큰 규모의 모델을 제공
    - 오픈소스 언어 모델 또한 존재하며 대표적으로 Eleuther GPT-NeoX (200억 파라미터), Facebook OPT-175B (1750억 파라미터), BLOOM from BigScience (1760억 파라미터) 가 있다.

## Prompt Engineering

- GPT-3 와 유사한 언어 모델들은 외계 기술에 가깝다 - 정확히 어떻게 작동되는지가 명확하지 않으며, 이에 대한 인사이트를 위해서는 모델과 상호작용을 해보는 수 밖에는 없다.
- tokenization 으로 인해 GPT-3 는 단어의 순서를 뒤집는 작업, 길이가 긴 문장을 다루는 작업 등에 다소 아쉬운 성능을 보인다.
- GPT-3 가 잘 수행하지 못하는 다른 예시는 캐릭터를 조합하는 작업인데, 이를 수행하기 위해서는 GPT-3 에게 [단계적 수행을 위한 알고리즘](https://twitter.com/npew/status/1525900849888866307)을 학습시켜야 한다.

![alt text](mlops/images/fsdl_7_10.png)

- 또 다른 prompt engineering 의 예시는 “Let’s Think Step by Step”. **[Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916.pdf)** 라는 논문에서 처음 등장했으며, GPT-3 모델에 “Let’s Think Step by Step” 이라는 내용을 추가하는 것 만으로 수학 문제에 대한 정답율을 17% 에서 78% 로 향상하는 것을 관찰한 내용을 담고있다.

![alt text](mlops/images/fsdl_7_11.png)

- GPT 는 또한 긴 문맥을 파악하는 작업을 잘 수행할 수 있으며, csv 파일을 생성하거나 Python 코드 등을 작성하는 [예시](https://twitter.com/goodside/status/1557381916109701120)를 쉽게 찾을 수 있다.
- GPT 는 문맥에 따라 원하는 작업을 수행하지 않을 수 있기 때문에 주의가 필요하다. GPT-3 를 기반으로 제작된 어플리케이션에서 또한 발생하는 문제이니 사전 조치가 필요할 수 있는 영역 (**[prompt injection attacks](https://simonwillison.net/2022/Sep/12/prompt-injection/), [possess your AI](https://twitter.com/goodside/status/1564112369806151680)**).

![alt text](mlops/images/fsdl_7_12.png)

- PromptSource, OpenPrompt 등의 prompt engineering 툴이 존재하지만, 실제 배포 모델 개발을 위해서는 커뮤니티 내 더욱 개선된 툴셋이 필요한 상황.

## Other Applications

### Code

![alt text](mlops/images/fsdl_7_13.png)

- 대규모 모델의 주요 활용 예시 중 하나는 **코드 생성** 영역. DeepMind 팀은 약 400 억 개의 파라미터를 가진 트랜스포머 모델 **[Alphacode](https://www.deepmind.com/blog/competitive-programming-with-alphacode)** 를 통해 Codeforce 대회에서 평균 이상의 성적을 기록한 적이 있으며, 이를 위해 코드를 생성하는 모델과 생성된 코드를 검증하는 모델을 개발.
- 모델의 아웃풋을 필터링 하는 작업을 실제 성능 향상에 상당한 도움을 줄 수 있다. OpenAI 또한 수학 문제를 풀기 위해 유사한 프로세스를 적용한 적이 있으니 [관련 자료](https://openai.com/blog/grade-school-math/)를 참고하면 좋을 것.

![alt text](mlops/images/fsdl_7_14.png)

- 코드 생성 모델은 **[Github Copilot](https://github.blog/2022-06-21-github-copilot-is-generally-available-to-all-developers/)** 과 같은 실제 제품으로도 개발 되었으며, 유사한 제품으로는 [replit](https://blog.replit.com/ai) 이 존재한다.
- 프로그래밍 환경에 ML 모델을 적용하는 것은 이제 시작 단계에 있는 상태이며, **[A recent paper](https://arxiv.org/pdf/2207.14502.pdf)** 에서는 자체적으로 생성한 문제를 기반으로 코딩 능력을 향상시키는 모델 또한 공개한 바가 있다.

![alt text](mlops/images/fsdl_7_15.png)

### Semantic Search

- **Semantic Search (의미 기반 검색)** 는 또 다른 유망한 적용 영역이다. 하나의 문서를 기반으로 생성한 embedding vector 를 활용해 유사성을 탐색하는 기능인데, cosine similarity 를 기반으로 해당 vector 간 의미가 얼마나 유사한지를 명확하게 판별할 수 있다.

![alt text](mlops/images/fsdl_7_16.png)

- 정보가 많은, float type 의 벡터 연산에는 많은 자원이 필요한데, Google 과 Facebook 같은 회사는 이러한 문제를 해결하기 위해 [FAISS](https://towardsdatascience.com/using-faiss-to-search-in-multidimensional-spaces-ccc80fcbf949), [ScaNN](https://cloud.google.com/blog/topics/developers-practitioners/find-anything-blazingly-fast-googles-vector-search-technology) 과 같은 라이브러리를 개발.
- 오픈소스로는 [Haystack from DeepSet](https://www.deepset.ai/haystack), [Jina.AI](http://Jina.AI) 등을 참고할 것. Pinecone, Weaviate, Milvus, Qdrant, Qdrant, Google Vector AI Matching Engine 또한 관련 서비스를 제공한다.

### Going Cross-Modal

- 새로운 모델들은 비전, 텍스트 간 서로 다른 형태의 데이터를 취합하는 기능을 제공한다. 예시로 **[Flamingo model](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/tackling-multiple-tasks-with-a-single-visual-language-model/flamingo.pdf)** 을 들 수 있는데, perceiver resampler 라는 구조를 활용해 이미지를 규격화된 토큰으로 전환할 수 있다.

![alt text](mlops/images/fsdl_7_17.png)

- 최근 공개된 **[Socratic Model](https://socraticmodels.github.io/)** 은 각자 별도의 학습 과정을 거친 비전, 언어, 오디오 모델이 하나의 인터페이스로 통합되어, 자연어 prompt 를 통해 새로운 task 를 수행하는 구조를 가지고 있다.
- Foundation 모델이란 Stanford 에서 공개한 **[On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258)** 에서 처음 알려진 개념이다. 강사진은 Large Language Model, 또는 Large Neural Network 가 더욱 적합한 단어라고 생각.

## CLIP and Image Generation

- 해당 섹션에서는 비전 분야를 주제로 다룸.
- 2021 년 OpenAI 은 **[CLIP (Contrastive Language-Image Pre-Training)](https://arxiv.org/abs/2103.00020)** 을 공개했다. 트랜스포머를 활용해 인코딩 된 텍스트, ResNet 과 Visual Transformer 를 활용해 인코딩 된 이미지를 기반으로 대조 학습 (cosine similarity 기반 이미지, 텍스트 페어 매칭) 을 진행하는 구조.

![alt text](mlops/images/fsdl_7_18.png)

- OpenAI 팀은 이러한 간단한 구조를 활용해 학습되지 않은 경우에 대해서도 이미지와 텍스트 임베딩을 매칭할 수 있는 모델을 구축한다.
- 새로운 데이터를 예측하는 방법은 크게 linear probe (로지스틱 회귀), zero-shot learning 으로 나뉘어지며, 주로 zero-shot learning 방법론이 더 나은 성능을 보임.
- 모델의 주요한 점은 이미지와 텍스트가 직접 연계되는 것이 아니라, 임베딩 공간에서 매칭 된다는 점. 다른 포맷의 데이터를 페어링 할 때 유용하게 쓰이는 테크닉이다.
- 예시적으로 image-to-text 과제의 경우, CLIP 모델을 활용해 추출된 이미지 임베딩을 기반으로 GPT 모델이 텍스트를 생성하는 방식.

![alt text](mlops/images/fsdl_7_19.png)

- 이미지 생성 영역에서 가장 널리 알려진 모델은 **[DALL-E (unCLIP)](https://cdn.openai.com/papers/dall-e-2.pdf)** 이다.
- 구조적으로 기존 CLIP 시스템 대비 (1) 텍스트 임베딩을 이미지 임베딩과 맵핑하는 Prior 영역 (2) 이미지 임베딩을 실제 이미지로 전환하는 Decoder 영역을 추가적으로 가지고 있다.
- Prior 영역은 수많은 텍스트 임베딩이 모두 한개의 이미지에 적합할 수 있다는 문제를 해결한다.

![alt text](mlops/images/fsdl_7_20.png)

- DALL-E 2 의 경우 이러한 Prior 구조로 **[Diffusion Model](https://towardsdatascience.com/diffusion-models-made-easy-8414298ce4da)** 을 차용. Diffusion Model 이란 Noisy 한 데이터에 단계적으로 학습함으로서 효과적인 Data Denoising 을 가능하게 한다.

![alt text](mlops/images/fsdl_7_21.png)

- Diffusion 시스템은 인코딩된 텍스트, CLIP 텍스트 임베딩, Diffusion Timestamp, Noised CLIP Embedding 에 순차적으로 Denoising 을 적용, 노이즈가 제거된 CLIP 이미지 예측을 가능하게 한다.
- 이러한 방법을 통해 실제 텍스트와 모델, CLIP 이미지 임베딩 공간 간 간극을 좁히는 효과를 가짐.

![alt text](mlops/images/fsdl_7_22.png)

- Decoder 는 Prior 의 아웃풋인 이미지 임베딩을 실제 이미지로 변환하는 역할을 수행한다. U-Net 구조를 차용해 인풋 이미지 임베딩에서 점차 노이즈를 제거하는 방식.

![alt text](mlops/images/fsdl_7_23.png)

- 결과적으로 CLIP 임베딩만을 활용해 이미지를 생성하거나, 복수의 CLIP 임베딩을 활용해 이미지를 합치는 것 또한 가능하다.

![alt text](mlops/images/fsdl_7_24.png)

- 유사한 모델로는 Parti 와 StableDiffusion 이 있다.
    - **[Parti](https://parti.research.google/)** : DALL-E 2 공개 이후 구글이 공개한 모델이며, Diffusion 모델이 아닌 VQGAN 구조를 활용, 이미지를 고차원의 토큰으로 변환하는 과정을 거친다.
    - **[StableDiffusion](https://stability.ai/blog/stable-diffusion-public-release)** : 비교적 최근 [공개](https://github.com/CompVis/latent-diffusion)된 모델이며, latent diffusion 모델을 활용해 이미지를 저차원의 latent 공간으로 치환 후, 이미지를 픽셀 공간에 재생성하는 구조를 가진다.

![alt text](mlops/images/fsdl_7_25.png)

- image-to-image, video generation, photoshop plugin 등 다양한 분야에서 활용.
- 이러한 모델을 구동하는 것은 꽤 많은 작업을 필요로 할 수 있으며, 툴셋이나 코드베이스로 바뀔 가능성 또한 있다.