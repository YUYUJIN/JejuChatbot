# JejuChatbot
> 2024 빅데이터 콘테스트에서 제공된 신한 카드 사용 데이터를 기반으로 단순 검색형 질의 및 사용자의 취향에 맞는 가맹점 추천을 위한 챗봇 서비스  
> Langchain의 Multi-Chain 아키텍처를 기반으로 사용자의 자유로운 양식의 질문을 단순 검색형과 추천형 질문으로 분류하여 서로 다른 엔진 객체를 활용.  
> 검색 엔진에서는 사용자의 질문에서 타겟 데이터 상의 속성을 검출하고 스케일하여 데이터 검색을 진행. 임의의 사용자 질문에도 기존 쿼리문을 수행하듯 데이터 검색 서비스 지원.  
> 추천 엔진에서는 22차원으로 임베딩한 가게 정보를 기반으로 진행. 사용자의 질문을 활용하여 LLM의 결과를 파싱하고, 추가적인 문장을 생성하여 단어 유사도 기반으로 속성을 추출. 최종적으로 22차원의 속성으로 Normalize 되어 임베딩된 질문과 가게 정보 벡터들을 비교하여 추천할 가게 정보를 선정.  

<img src=https://img.shields.io/badge/python-3.10.0-green></img>
<img src=https://img.shields.io/badge/transformer-4.45.2-yellow></img>
<img src=https://img.shields.io/badge/Langchain-0.3.4-blue></img>
<img src=https://img.shields.io/badge/FAISS-1.7.2-orange></img>
<img src=https://img.shields.io/badge/pytorch-2.3.1-red></img>
<img src=https://img.shields.io/badge/GeminiFlash-1.5.0-purple></img>

## How to use
 Langchain과 임베딩 모델을 사용하기 위해 pytorch등 기반 dependency를 설치한다.  
 환경에 맞게 설치하되, 본 프로젝트의 개발 환경인 Linux에 맞는 설치는 다음과 같다.
```
pip3 install torch torchvision torchaudio
pip install transformer
```  
  
 사전준비 및 필요 dependency 설치   
```
git clone https://github.com/YUYUJIN/JejuChatbot
cd JejuChatbot
pip install -r requirements.txt
```

 workplace 내에 .env 파일을 만들어 Gemmini-Flash API 키의 정보생성
```
GOOGLE_API_KEY={API Key}
```

 이후 app을 구동하여 간이 API를 사용한다.  
```
python app.py
```

## Structure
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/structure.png></img>  
프로젝트의 구조도는 위와 같다.  
  
 크게 벡터 DB 업로드 단계와 Gemmini-Flash API를 활용해 사용자의 질문을 분류, 데이터를 검색하여 답변을 생성하여 서빙하는 단계로 구성된다.
 상세 내용은 아래에 목차별로 서술.

## Crowling Data
 신한 카드 결제 데이터 상에 존재하는 가맹점들을 네이버 플레이스에서 크롤링하여 데이터 구축.  
 아래와 같이 리뷰 내의 유효 키워드들만 추출하여 단위 벡터로 정규화.  
 선별된 키워드에서 가맹점 단위 벡터에서 영향력이 상위 300차원만 추출.  
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/crowlingdata1.png></img>  
<키워드 추출 및 분석>  
  
 300차원의 벡터를 보다 낮은 차원으로 축소.  
 PCA를 진행하여 유의미한 결과를 만들었지만, 사용자의 질문을 축소할 임베딩 모델을 만드는 것이 대회 규정에 어긋나 키워드 매칭 위주의 임베딩 알고리즘 구현.    
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/crowlingdata2.png></img>  
<차원 축소 예시>  

## Vector Store
 아래와 같은 임베딩 후보군 중에 intfloat/multilingual-e5-large-instruct 채택.      
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/embedingmodel.png></img>  
<임베딩 모델 선정>  
  
 단순 검색에 활용되는 벡터 스토어는 아래와 같이 질문, 파싱 결과를 스케일을 위해 준비.  
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/searchstore.png></img>  
<검색 엔진 관련 벡터 스토어>  

 추천 검색에 활용되는 벡터 스토어는 아래와 같이 최종 임베딩할 22차원의 속성으로 스케일하기 위해 준비.  
 최종적으로 가맹점과 질문의 유사도를 계산하기 위한 벡터는 Crowling Data 항목에 따라 생성한 22차원의 벡터 스토어를 활용.   
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/recommendstore.png></img>  
<추천 엔진 관련 벡터 스토어>  

## Search Engine
 LLM API와 프롬프트 엔지니어링을 활용하여 사용자의 질문에서 테이블 데이터의 속성 값을 추출/파싱.  
 추출된 속성 값들을 사전에 구성한 벡터 스토어를 활용하여 유효한 범위로 스케일링.  
 전체 데이터에서 해당 조건으로 데이터 추출을 진행하여 후보군을 추출.  
 프롬프트 엔지니어링을 활용하여 최종 답변 생성.      
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/searchengine1.png></img>   
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/searchengine2.png></img>  
<최종 아키텍처>
  
 단일 Retriever 객체로만 성능 평가를 진행.  
 제공된 데이터를 기반으로 코드와 사람이 직접 작성한 질문-정답 데이터셋을 구축하여 성능 평가에 활용.  
 k는 5를 기준으로 정밀도, 재현율, F1 스코어를 활용.  
 추가적으로 정답 가맹점이 n개일 때, n개의 출력인지 확인하기 위해 커스텀 평가 지표인 개수 유사도 점수를 활용.  
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/searchmetrics.png></img>  
<Test 데이터셋 기준 성능평가>  
 단순 cos-similarity, TF-IDF 알고리즘으로 구성된 Retriever가 재현율: 0.2였던 것보다 약 0.5 정도 향상된 결과 지표를 보임.(재현율, 정밀도, F1 모두 유사)  

## Recommend Engine
 LLM API와 프롬프트 엔지니어링을 활용하여 사용자의 질문과 가맹점 벡터 스토어 간 유사도 검증을 통해 후보군 선출.  
 최초의 파싱 데이터와 생성한 문장을 Normalize하고 더하여 질문 벡터를 구성.  
 이후 신한 카드 결제 데이터를 기반으로 후보군 내에서 계산된 가중치 점수와 합산하여 최종 추천 아이템 선정.  
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/recommendengine1.png></img>   
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/recommendengine2.png></img>  
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/recommendengine3.png></img>   
<최종 아키텍처>
  
 질문-답변의 정답 데이터를 작성하지 못해 10가지 테스트 질문을 정하고, 10명의 사용자에게 블라인드 테스트 진행.  
 최종 질문의 적합도를 확인하기 위해 단일 Retriever가 아닌 전체 체인에서 테스트 진행.    
<img src=https://github.com/YUYUJIN/JejuChatbot/blob/main/image/recommendmetrics.png></img>  
<Test 데이터셋 기준 성능평가> 

## Trouble Shooting
<details>
<summary>크롤링 데이터 처리</summary>

 네이버 플레이스에서 수집한 리뷰 데이터의 수가 가맹점 별로 편차가 많았다.

 항목 수가 적게는 10개에서 100개까지 차이가 존재하였고, 또한 확보된 리뷰 내용의 양이 달라 키워드를 추출하여 전체 데이터를 확인하면 소수의 가맹점 리뷰가 대부분을 차지하였다.

 이에 추출된 키워드에서 음식의 맛, 취향 등의 키워드를 수작업으로 필터링하였고, 가맹점 별로 검출 키워드 수로 벡터화한 후 가맹점 별 벡터를 크기가 1인 단위 벡터로 정규화하여 단순 합산에서 특정 가맹점이 높은 영향력을 가지는 것을 방지하였다.

 일차적으로 가맹점 별 키워드 영양력을 전체 데이터에서 합산하여 상위 300개의 키워드로 가맹점 리뷰 데이터를 벡터화하였다.

 최종에서는 300차원의 데이터는 차원의 저주, 이후 질문 쿼리 벡터와 유사도 연산을 위해 축소를 결정하였다.

 PCA 방식을 통해 15차원으로 축소 시 10개의 그룹으로 진행했을 때, 실루엣 스코어를 0.7까지 올릴 수 있었으나, 질문 쿼리를 임베딩할 수 없어 사용하지 않았다.

 300개의 키워드를 22개의 속성을 가지는 벡터로 다시 분해하여 최종적으로 22차원으로 가맹점 리뷰 데이터를 임베딩하였다.

</details>

<details>
<summary>벡터 스토어 사용 시 정확도 보완</summary>

 LLM의 결과를 파싱하여 얻은 단어 위주의 데이터를 스케일하기 위해 벡터 스토어를 활용하였다.

 초기 벡터 스토어는 임베딩 모델을 활용해 벡터화한 문장을 cos-similarity를 이용해 유사한 단어를 추출하는 방식으로 진행하였다.

 하지만 문장의 복잡도를 조금 올리거나 특정 단어들만 등장하는 것을 확인하였다.

 선정한 intfloat/multilingual-e5-large-instruct 모델은 유사 단어간 유사도를 보정하므로 단순 단어 위주로 구성한 page contents들을 의미론적 유사성이 높은 문장 구조로 변경하여 구성하였다.

 결과적으로 파싱한 단어들을 준비한 속성에 맞게 의미론적으로 유사한 단어로 스케일할 수 있었다.

 준비한 테스트 데이터 기준으로, 검출된 단어들을 분류할 때 정확도가 평균적으로 0.2~0.3에서 0.7~0.8로 향상되었다.
</details>

<details>
<summary>검색 엔진 정밀도 보완</summary>

 기존에는 벡터 스토어를 활용해 각 가맹점 별로 page contents를 구성하고 문서와 질문 쿼리 벡터 간 유사도로 후보군을 검출하였다.

 해당 방식에서는 단순 단어, 문장 간 유사도만을 활용해 수치의 높낮음, 범주에 속한 케이스를 잘 구별해내지 못하였다. 평가 데이터 기준, 정밀도 또한 0.2 이하였다.

 Text2Query를 위해 구현된 벡터 스토어, Chroma DB를 활용하였지만 질문에서 요구하는 가맹점을 잘 찾아내지 못해, 정밀도가 최초 케이스와 유사하였다.

 Text2Query를 간이로 구현하여 해결하였다. 문장에서 select를 진행하기 위한 속성들을 파싱하고 스케일하여 pandas 라이브러리를 사용하여 메모리 상의 데이터에서 적절한 데이터를 찾아 반환하였다.

 최종적으로 테스트 데이터 기준, 정밀도(k=5)를 0.2에서 0.74로 향상할 수 있었다.
</details>

<details>
<summary>추천 엔진 적합도 보완</summary>
  
 크롤링 데이터 처리 항목처럼 임베딩된 22차원의 속성 내에서 질문과 가맹점의 유사도를 비교하여 후보군을 검출하는 것을 목적으로 구현하였다.

 하지만 구현 시 단위 테스트에서 평가 질문들을 사람이 임의로 임베딩한 벡터와 같은 경향성이 적었다. 질문지를 기준으로 실제 사람이 만든 벡터와 내적이 평균적으로 5이하로 유사도가 낮다고 판단하였다(내적 최대값:22).
 
 1. LLM이 임베딩한 벡터 외에도 추가적인 파생 문장들을 생성하고 
 2. 위 문장들을 벡터 스토어를 활용하여 22차원의 속성으로 임베딩하였다.

 최종적으로 1번 벡터와 2번 벡터들의 평균 벡터를 활용하여 다시 평가 데이터로 테스트를 진행하였고, 내적이 평균적으로 7 정도로 유사도가 향상된 것을 확인하였다(평가 테스트인 질문은 약 10개의 데이터로 적은 표본이지만 향상된 결과로 채택하였다).
</details>  

<details>
<summary>추천 결과 평가 방안</summary>

 확보 데이터는 약 6만개의 데이터이고, 중복을 제거하고 가맹점 기준으로는 약 9천개의 데이터가 존재하였다.

 해당 데이터를 전부 검토하여 질문 당 정답 레이블을 만드는 것은 상당히 어려웠고, 사람이 직접 평가하는 것이기에 해석적인 요소가 포함되었다.

 이에 전체 아키텍처를 기반으로 10개의 질문에 대해 답변을 생성하고, 전체 데이터를 보지 않은 상태에서 10명의 지인에게 블라인드 테스트를 진행하였다.

 대체적으로 맵다/달콤하다/정갈하다 등 가맹점 벡터들이 명확히 구분되어 가지고 있는 속성에 대해서는 높은 만족도를 보이고, 부드럽다/바삭하다 등 표본적이 적어 속성을 가지는 벡터들이 두드러지지 않거나 보편적으로 넚게 분포한 속성이 포함된 질문에 대해서는 만족도가 낮았다.

 최종 평균은 7.7이었고, 최종 구현체에 대해서만 평가를 진행하여 이전 구현체와 비교점이 없는 것은 아쉬운 점이다.
</details>

