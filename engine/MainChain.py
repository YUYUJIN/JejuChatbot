import os
import torch
import pandas as pd
from dotenv import load_dotenv

from langchain.schema import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from .SearchEngine import SearchEngine
from .RecommendEngine import RecommendEngine
from utils import SEARCH_COLUMNS, RECOMMEND_COLUMNS

load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.environ.get('GOOGLE_API_KEY')

class JejuRestaurantRecommender:
    def __init__(self,
                 data_path : str,
                 store_path : str,
                 user_config : dict,
                 ):
        # 디바이스 설정 (cuda 사용 가능 여부 확인)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Gemini 설정
        self.llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

        # Hugging Face의 사전 학습된 임베딩 모델과 토크나이저 로드
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct",
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': False},
            show_progress=False
        )
        self.user_config=user_config
        self.search_data_table=pd.read_csv(data_path+'/jeju_mct_data_ko_v3.csv',encoding='cp949')
        self.recommend_data_table=pd.read_csv(data_path+'/recommend_data_v3.csv',encoding='utf-8-sig')
        self.search_engine=SearchEngine(llm_model=self.llm_model,
                                        embedding_model=self.embedding_model,
                                        type_store_path=store_path+'/type',
                                        address_store_path=store_path+'/address',
                                        range_store_path=store_path+'/range',
                                        column_store_path=store_path+'/columns',
                                        data_table_path=data_path+'/jeju_mct_data_ko_v3.csv',
                                        user_config=user_config,
                                        )
        self.recommend_engine=RecommendEngine(llm_model=self.llm_model,
                                        embedding_model=self.embedding_model,
                                        attribute_store_path=store_path+'/attribute',
                                        mct_vec_path=store_path+'/mct_vec',
                                        data_table_path=data_path+'/recommend_data_v3.csv',
                                        user_config=user_config,
                                        )
        
        # 분류 체인
        self.cl_prompt = ChatPromptTemplate.from_template(
            """
            주어진 사용자의 질문을 '검색형정보','추천형정보' 중 하나로 분류해주세요. '검색형정보', '추천형정보'로만 응답하세요.
            참고로,
            '검색형정보'는 가맹점명, 가맹점 주소, 이용건수, 이용금액, 건당 평균 이용금액, 요일별 이용건수 비중, 시간대별 이용건수 비중, 현지인 이용건수 비중, 성별/나이 비중의 정보를 포함하고 있습니다.
            '추천형정보'는 가맹점명, 가맹점 주소, 운영시간, 부가설명, 링크, 메뉴, 카테고리, 맛 표현 빈도수, 방문 목적, 리뷰내용 등을 포함하고 있습니다.
            <question>
            {question}
            </question>
            분류:
            """
        )
        self.cl_chain = (
            self.cl_prompt
            | self.llm_model
            | StrOutputParser()
        )
        self.current_question=''

        self.llm_prompt_search = ChatPromptTemplate.from_template(
            """
            당신은 제주 맛집 추천 전문가입니다.  \
            당신은 사용자 질문에 맞춰 맛집을 추천해주세요. 현재 사용자의 정보는 아래와 같습니다.
            지역 : {area}, 성별: {gender}, 나이대: {age}
            \
            추천할 때, 제주도 맛집 메타 데이터를 기준으로 추천해주세요.
            메타 데이터에는 아래와 같은 정보들을 포함하고 있습니다.

            가맹점명 : 맛집 가게 이름
            가맹점업종 : 가맹점 카테고리에 대한 내용 (단품요리, 한식, 일식 등)
            가맹점주소 : 가맹점이 위치한 주소
            이용건수 : 월별 업종별 이용건수 분위수 구간을 6개 구간으로 집계 시, 해당 가맹점의 이용건수가 포함되는 분위수 구간
            이용금액 : 월별 업종별 이용금액 분위수 구간을 6개 구간으로 집계 시, 해당 가맹점의 이용금액이 포함되는 분위수 구간
            건당평균이용금액 : 월별 업종별 건당평균이용금액 분위수 구간을 6개 구간으로 집계 시, 해당 가맹점의 건당평균이용금액이 포함되는 분위수 구간
            요일별 이용건수 비중 : 월요일부터 일요일까지 요일의 이용건수 비중
            시간대별 이용건수 비중 : 오전 5시부터 다음날 오전 4시까지 시간대의 이용건수 비중
            현지인이용건수 비중 : 현지인 이용건수 비중
            남성과 여성 성별 비중 : 남성, 여성 회원수 비중
            연령대별 회원수 비중 : 20대, 30대, 40대, 50대, 60대 회원수 비중

            \

            답변 할 때, "선택하신 설정(지역:{area},성별:{gender},나이:{age})에 해당하는 제주도 맛집 정보에 의하면.." 으로 시작하여 답변해주세요
            '추천이유' 항목에는 어떤 근거로 해당 가게가 검색되고 추천되었는지 설명해주고, 관련 가맹점 추가정보들을 포함하여 2-3줄 정도로 알려주세요.
            검색된 가게가 여러개인 경우, 여러 가게를 리스팅해서 1), 2), 3) 처럼 번호를 매겨 차례대로 응답해주세요.
            검색된 가게가 여러개인 경우, 최대 3개 가게까지만 알려주세요. 

            사용자 질문에 대한 응답 :{context}


            질문: {question}
            답변:

            \n- 가게명 :  가맹점명
            \n- 기준연월 :
            \n- 개설일자 :
            \n- 업종 :
            \n- 주소 :
            \n- 추천 이유 : 

            """ 
        )
        self.search_chain=(
            self.get_data_for_search
            | self.llm_prompt_search
            | self.llm_model
        )

        self.llm_prompt_recommend = ChatPromptTemplate.from_template(
            """
            당신은 제주 맛집 추천 전문가입니다.  \
            당신은 사용자 질문에 맞춰 맛집을 추천해주세요. 현재 사용자의 정보는 아래와 같습니다.
            지역 : {area}, 성별: {gender}, 나이대: {age}
            \
            추천할 때, 제주도 맛집 추천 데이터를 기준으로 추천해주세요.
            추천 데이터에는 아래와 같은 정보들을 포함하고 있습니다.
            가게명 : 맛집 가게 이름
            업종 : 가게 카테고리에 대한 내용 (단품요리, 한식, 일식 등)
            주소 : 가맹점이 위치한 주소
            운영시간 : 가게의 운영시간 
            번호 : 가게 전화번호 
            부가설명 : 포장, 단체, 예약 등 가게 관련 부가정보 
            링크 : 가게 링크 URL (전체 링크)
            메뉴 : 가게 메뉴
            \
            답변 할 때, "선택하신 설정(지역:{area},성별:{gender},나이:{age})에 해당하는 제주도 맛집 추천 정보에 의하면.." 으로 시작하여 답변해주세요
            '추천이유' 항목에는 어떤 근거로 해당 가게가 검색되고 추천되었는지 설명해주고, 관련 가맹점 추가정보들을 포함하여 2-3줄 정도로 알려주세요.
            검색된 가게가 여러개인 경우, 여러 가게를 리스팅해서 1), 2), 3) 처럼 번호를 매겨 차례대로 응답해주세요.
            검색된 가게가 여러개인 경우, 최대 3개 가게까지만 알려주세요. 
            가게 정보 중 링크 URL 은 데이터 그대로 출력해주세요. 
            사용자 질문에 대한 응답 :{context}
            질문: {question}
            답변:
            \n- 가게명 : 가맹점명
            \n
            \n- 업종 :
            \n- 주소 :
            \n- 운영시간 :
            \n- 번호 :
            \n- 부가설명 :
            \n- 링크 URL: *가지고 있는 URL 정보 그대로 출력하기
            \n- 메뉴 :
            \n- 추천이유 :
            """ )
        self.recommend_chain=(
            self.get_data_for_recommend
            | self.llm_prompt_recommend
            | self.llm_model
        )

        self.full_chain = (
            {"topic": self.cl_chain, "question": lambda x: x["question"]}
            | RunnableLambda( 
                # 라우팅 함수를 전달
                self.route,
            )
            | StrOutputParser()
        )

    def get_data_for_search(self, x):
        result = self.format_contents(self.search_engine(x["question"]))
        return {
            'question': result[0],
            'context': result[1],
            'area': self.user_config['area'],
            'gender': self.user_config['gender'],
            'age': self.user_config['age']
        }
    
    def get_data_for_recommend(self,x):
        result=self.format_contents(self.recommend_engine(x["question"]),flag=False)
        return {
            'question': x['question'],
            'context': result[1],
            'area': self.user_config['area'],
            'gender': self.user_config['gender'],
            'age': self.user_config['age']
        }
        

    # flag : True => search data, False => recommend data
    def format_contents(self, retriver_result, flag=True):
        question,data_index=retriver_result
        contents=''
        if flag:
            target_data=self.search_data_table
            for id in data_index:
                target=target_data.loc[id]
                for id in target.index:
                    if id in SEARCH_COLUMNS:
                        contents+=str(id)+': '+str(target.loc[id])+'\n'
            contents+='\n'
        else:
            target_data=self.recommend_data_table
            for ids in data_index:
                target=target_data[target_data['IDS']==ids]
                for id in target.index:
                    series=target.loc[id]
                    for i in series.index:
                        if i in RECOMMEND_COLUMNS:
                            contents+=str(i)+': '+str(series.loc[i])+'\n'
            contents+='\n\n'
        contents+='\n'
        self.current_question=question
        return question, contents
    
    def route(self, info):
        contents=info['topic'][:5]
        if contents=='검색형정보':
            return self.search_chain
        
        elif contents=='추천형정보':
            return self.recommend_chain
        # 이외 추천
        else:
            return self.recommend_chain

    def __call__(self, query):
        return self.full_chain.invoke({"question": query})