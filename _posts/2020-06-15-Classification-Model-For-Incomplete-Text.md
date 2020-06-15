---
title: Survey of Classification Models for Incomplete Text data
tags: NLP Embedding Erroneous_Sentence Denoising_Encoder
---

## 개요
- NLP 모델이 입력받는 텍스트 데이터의 에러(오탈자, 잘못된 띄어쓰기 등) 대응하는 모델 survey

<!--more-->

### Integrated Eojeol Embedding for Erroneous Sentence Classification in Korean Chatbots

***

- 논문: [https://arxiv.org/abs/2004.05744](https://arxiv.org/abs/2004.05744)
- 리뷰: [https://jeonsworld.github.io/NLP/iee/](https://jeonsworld.github.io/NLP/iee/)
- 한국어 논문은 HCLT 2019 (한국어 언어정보처리 학회) 논문집에 수록

#### Problem Statement
- 기존 카카오 미니에서 입력 문장의 의도 분류 모델 성능 개선 연구의 연장선
- 기존의 **형태소(morpheme) 기반 임베딩**은 erroneous inputs에 좋지 않은 성능을 보임
    - 모델이 문장을 분류하기 위한 <U>결정적 단서 (vital cue)</U>에 대한 정보가 erroneous input에서는 유실
    - 잘못된 문장 분류(intent classification)의 확률이 높아짐


#### Approach
- 어절 단위마다 자모 / 음절 / subword(BPE) / 형태소 정보를 각각 임베딩 한 후, 통합된 임베딩 벡터를 생성하여 분류 모델에 사용

    - 통합된 임베딩 벡터(Integrated Eojeol Embedding; IEE)
    - 저자의 기존 연구에서는 수집된 코퍼스로 GloVe 워드 임베딩하여 분류 모델에 입력으로 사용 (KoGloVe)
    - 기존 분류 모델에 IEE 벡터를 입력 워드 임베딩으로 사용 <br/><br/> 

    - **Integrated Eojeol Embedding 네트워크 구조**  
    ![IEE Network Architecture](/assets/images/IEE network architecture.png) <br/><br/> 

    - **어절 단위 기준의 통합 어절 임베딩 생성에 사용하는 subword 리스트**  
    ![IEE Subword list](/assets/images/iee subword list.png) <br/><br/> 

    - 기존 연구 분류 모델과 IEE 벡터 입력 활용 기반 분류 모델 변형 구조도

기존 연구 분류 모델           |  IEE 벡터를 입력으로 하는 기존 모델의 variant
:-------------------------:|:-------------------------:
![Base model](/assets/images/base model.png) | ![Model variant](/assets/images/model variant.png)
    
#### 코퍼스

- 기존 연구에서 사용한 127,322개 문장 규모의 intent classification corpus 사용 (48개의 intents)
    - weather / fortune / music / podcast 등 <br/><br/>    


- 기존 코퍼스는 WF(Well-Formed) corpus로 명칭
    - 에러 문장이 없는 코퍼스 <br/><br/> 

- **KM corpus**(Korean Misspelling)를 별도로 구성
    - Common / OOD(intent for meaningless)의 두 가지 intent는 제외
    - 작업자는 가이드라인에 따라, 에러 문장 생성<br/><br/> 
    ![KM corpus](/assets/images/km corpus.png) <br/><br/> 

- **SM corpus** (Space missing)는 WF / KM corpus에서 재구성
    - 기존 코퍼스의 테스트 셋에서 문장의 단어 간 0.5 확률로 띄어쓰기를 제거
    - 문장에 최소 하나의 띄어쓰기 제거

- - -