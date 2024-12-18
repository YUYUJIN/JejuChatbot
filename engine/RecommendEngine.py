import json
import pandas as pd
import numpy as np
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings.base import Embeddings
from sklearn.preprocessing import MinMaxScaler

from utils import map_usage_range

class SimpleVectorEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # 텍스트를 무시하고 벡터를 그대로 반환
        return [np.array(eval(text)) for text in texts]
    
    def embed_query(self, text):
        # 쿼리 텍스트를 벡터로 변환 (여기서는 텍스트가 이미 벡터 형태라고 가정)
        return np.array(eval(text))

class RecommendEngine:
    def __init__(self,llm_model: ChatGoogleGenerativeAI,
                 embedding_model: HuggingFaceEmbeddings,
                 attribute_store_path: str,
                 mct_vec_path: str,
                 data_table_path: str,
                 user_config: dict,
                 ):
        self.llm_model=llm_model
        self.embedding_model=embedding_model
        self.data_table=pd.read_csv(data_table_path, encoding='utf-8-sig')
        self.attribute_store=FAISS.load_local(attribute_store_path, embedding_model, allow_dangerous_deserialization=True)
        self.mct_store=FAISS.load_local(mct_vec_path, SimpleVectorEmbeddings(), allow_dangerous_deserialization=True)
        self.user_config=user_config
        self.score_scaler=MinMaxScaler()
        self.prompt_template=ChatPromptTemplate.from_template(
            """
            당신은 음식 추천 시스템의 일부로, 사용자의 요청을 분석하여 관련된 음식 속성을 식별하는 AI 어시스턴트입니다. 다음 지침에 따라 사용자의 질문을 분석해주세요:\n
            1. 주어진 질문을 주의 깊게 읽으세요.\n
            2. 아래 나열된 음식 속성 목록을 참고하세요:\n
            ["기름지다", "고소하다", "구수하다", "든든하다", "진하다", "실하다", "알차다", "짜다", "부드럽다", "담백하다", "신선하다", "얼큰하다", "맵다", 
            "바삭하다", "뜨근하다", "가볍다", "건강하다", "달콤하다", "쓰다", "시다", "정갈하다", "칼칼하다"]\n
            3. 각 속성에 대해 다음 사항을 판단하세요:\n
            a) 필요성: 해당 속성이 질문과 관련이 있는지 (True/False)\n
            b) 근거단어들: 관련성의 근거가 되는 질문 내 단어나 문구 (리스트 형태)\n
            c) 연상문장들: 근거단어들을 활용한 문장 (리스트 형태)\n
            d) 값: 속성의 예상 강도 또는 중요도 (0에서 1 사이의 float 값)\n
            4. 분석 시 다음 사항을 고려하세요:\n
            - 명시적으로 언급된 속성뿐만 아니라 암시된 속성도 고려하세요.\n
            - 사용자의 의도, 상황, 문맥을 파악하여 관련 속성을 추론하세요.\n
            - 특정 음식이나 요리 스타일과 연관된 일반적인 속성을 고려하세요.\n
            주어진 질문: {question}\n
            위의 지침에 따라 질문을 분석하고 결과를 JSON 형식으로 제공해주세요.\n

			""" )
        self.meta_chain = (
            {
                'question': lambda x: x["question"]
            }
            | self.prompt_template
            | self.llm_model
        )

    @lru_cache(maxsize=128)
    def get_attribute_scores(self, basis_word):
        return self.attribute_store.similarity_search(basis_word)[:2]
    
    def get_recommend_scores(self, targets):
        area=self.user_config['area']
        gender=self.user_config['gender']
        age=self.user_config['age']
        score=[]
        for _, target in targets.iterrows():
            acg=map_usage_range(target['이용건수구간'])               #이용건수구간
            uaptg=map_usage_range(target['건당평균이용금액구간'])     #건당평균이용금액구간
            
            usage_weight=acg*(1-uaptg*0.01)
            gender_weight=0.75*target[f'최근12개월{gender}회원수비중']+0.25*(1-target[f'최근12개월{gender}회원수비중']) if gender in(['남성','여성']) else 0.75
            age_weight=0.75*target[f'최근12개월{age}회원수비중']+0.25*(1-target[f'최근12개월{age}회원수비중']) if age in(['20대이하', '30대', '40대', '50대', '60대이상']) else 0.75
            if area=='전체':
                address_weight=1
            else:
                address_weight=1 if (target['분할구역']==area) else 0.5
            
            condition_weight=(gender_weight+age_weight)*address_weight
            score.append(usage_weight*condition_weight)
        return score
    
    def make_score_table(self,ids,candidates,scores):
        score_table={}
        score_table['RS']=self.get_recommend_scores(candidates)
        score_table['CS']=scores
        score_table=pd.DataFrame(data=score_table)
        score_table=self.score_scaler.fit_transform(score_table)
        
        score_table=pd.DataFrame(data=score_table,columns=['RS','CS'])
        score_table['IDS']=ids
        score_table['FS']=score_table['RS']-score_table['CS']
        score_table=score_table.sort_values(by=['FS'],ascending=[False]).reset_index()

        return score_table
    
    def __call__(self,question: str, top_k: int = 5):
        response=self.meta_chain.invoke({"question": question}).content.split('```')[1][4:]
        conditions=json.loads(response)
        candidates=self.data_table
        
        attributes={"기름지다":0, "고소하다":0, "구수하다":0, "든든하다":0, "진하다":0, "실하다":0, "알차다":0, "짜다":0, "부드럽다":0, "담백하다":0, "신선하다":0, "얼큰하다":0, "맵다":0, "바삭하다":0, "뜨근하다":0, "가볍다":0, "건강하다":0, "달콤하다":0, "쓰다":0, "시다":0, "정갈하다":0, "칼칼하다":0}
        query_attributes=[]
        words_attributes=[]
        basis_words=set()

        for attribute in attributes.keys():
            query_attributes.append(conditions[attribute]['값'])
            basis_words.update(conditions[attribute]['연상문장들'])
        for basis_word in basis_words:
            check=self.get_attribute_scores(basis_word)
            for c in check:
                attributes[c.metadata['topic']]+=1
        for key in attributes.keys():
            words_attributes.append(attributes[key])
        
        # L2 norm
        query_attributes=np.array(query_attributes)
        norms=np.linalg.norm(query_attributes)
        query_attributes/=norms
        norms=np.linalg.norm(words_attributes)
        words_attributes/=norms

        query_vector=query_attributes+words_attributes
        norms=np.linalg.norm(query_vector)
        query_vector/=norms
        
        results=self.mct_store.similarity_search_with_score_by_vector(query_vector,k=top_k*3)
        ids = [result[0].metadata['id'] for result in results]
        scores=[result[1] for result in results]
        candidates=candidates[candidates['IDS'].isin(ids)]

        score_table=self.make_score_table(ids,candidates,scores)
        return '', score_table['IDS'].loc[:top_k] if len(score_table)>0 else []
    
    def clear_cache(self):
        self.get_attribute_scores.cache_clear()
