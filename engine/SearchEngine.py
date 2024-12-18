import json
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

class SearchEngine:
    def __init__(self,llm_model: ChatGoogleGenerativeAI,
                 embedding_model: HuggingFaceEmbeddings,
                 type_store_path: str,
                 address_store_path: str,
                 range_store_path: str,
                 column_store_path: str,
                 data_table_path: str,
                 user_config: dict,
                 ):
        self.llm_model=llm_model
        self.embedding_model=embedding_model
        self.type_store=FAISS.load_local(type_store_path, embedding_model, allow_dangerous_deserialization=True)
        self.address_store=FAISS.load_local(address_store_path, embedding_model, allow_dangerous_deserialization=True)
        self.range_store=FAISS.load_local(range_store_path, embedding_model, allow_dangerous_deserialization=True)
        self.column_store=FAISS.load_local(column_store_path, embedding_model, allow_dangerous_deserialization=True)
        self.user_config=user_config
        self.data_table=pd.read_csv(data_table_path, encoding='cp949')
        self.prompt_template=ChatPromptTemplate.from_template(
            """
            다음 질문을 분석하여 각 데이터 속성에 대해 아래 사항을 판단해주세요. 신중히 한 번 두 번 세번 고민하고, 결과만 Json 형식으로 나타내주세요:\n
            {question}\n
            1. 해당 속성이 질문에 필요한지(필요성): True나 False\n
            2. 해당 속성에 근거가 되는 단어(근거단어들): List\n
            3. 범위 값(범위값): List\n
            4. 기준 값(기준값): float\n
            5. 비교 문구 (비교문구): str, 높다 or 같다 or 낮다\n
            다음 속성들에 대해 분석해주세요:\n
            기준연월, 가맹점명, 개설일자, 업종, 주소, 이용건수구간, 이용금액구간, 건당평균이용금액구간, 월요일이용건수비중, 화요일이용건수비중, 수요일이용건수비중, 목요일이용건수비중, 금요일이용건수비중, 토요일이용건수비중, 일요일이용건수비중, 5시11시이용건수비중, 12시13시이용건수비중, 14시17시이용건수비중, 18시22시이용건수비중, 23시4시이용건수비중, 현지인이용건수비중, 최근12개월남성회원수비중, 최근12개월여성회원수비중, 최근12개월20대이하회원수비중, 최근12개월30대회원수비중, 최근12개월40대회원수비중, 최근12개월50대회원수비중, 최근12개월60대이상회원수비중\n
            답변:\n
            """ )
        self.meta_chain = (
            {
                'question': lambda x: x["question"]
            }
            | self.prompt_template
            | self.llm_model
        )
        self.current_question=''

    def __call__(self,question: str, top_k: int = 5):
        response=self.meta_chain.invoke({"question": question}).content.split('```')[1][4:]
        query_conditions=json.loads(response)
        candidates=self.data_table
        self.current_question=''

        # column scaling
        conditions={}
        for key in query_conditions.keys():
            conditions[self.column_store.as_retriever().invoke(key)[0].metadata['topic']]=query_conditions[key]

        # 구간 검색 컬럼 : 업종, 주소, 이용건수구간, 이용금액구간, 건당평균이용금액구간 -> vector_store 이용
        sections=['업종', '이용건수구간', '이용금액구간', '건당평균이용금액구간', '주소']
        # 업종
        if conditions[sections[0]]['필요성']:
            condition=' '.join(conditions[sections[0]]['근거단어들'])
            condition=self.type_store.as_retriever().invoke(condition)[0].metadata['topic']
            candidates=candidates[candidates[sections[0]]==condition]
            self.current_question+=f'업종이 {condition}이고 '
            

        # 그 외
        for section in sections[1:4]:
            if conditions[section]['필요성']:
                condition=' '.join(conditions[section]['범위값'])
                if condition!='':
                    condition=self.range_store.as_retriever().invoke(condition)[0].metadata['topic']
                    candidates=candidates[candidates[section]==condition] 
                    self.current_question+=f'{section}이 {condition}에 속하고 '

        # 주소
        if conditions[sections[4]]['필요성']:
            clues=conditions[sections[4]]['근거단어들']
            if len(clues)>1:
                clues=' '.join(clues)
                address=self.address_store.as_retriever().invoke(clues)[0].metadata['topic'].split(' ')
            else:
                address=self.address_store.as_retriever().invoke(clues[0])[0].metadata['topic'].split(' ')
            if len(address)==1:
                candidates=candidates[candidates['시/군']==address[0]]
            elif len(address)==2:
                candidates=candidates[(candidates['시/군']==address[0]) & ((candidates['읍/면/구']==address[1]) | (candidates['동/리']==address[1]))]
            else:
                candidates=candidates[(candidates['시/군']==address[0]) & (candidates['읍/면/구']==address[1]) & (candidates['동/리']==address[2])]
            address=' '.join(address)
            self.current_question+=f'주소가 {address}이고 '

        # # 수치 검색 컴럼 : 월요일이용건수비중, 화요일이용건수비중, 수요일이용건수비중, 목요일이용건수비중, 금요일이용건수비중, 토요일이용건수비중, 일요일이용건수비중, 5시11시이용건수비중, 12시13시이용건수비중, 14시17시이용건수비중, 18시22시이용건수비중, 23시4시이용건수비중, 현지인이용건수비중, 최근12개월남성회원수비중, 최근12개월여성회원수비중, 최근12개월20대이하회원수비중, 최근12개월30대회원수비중, 최근12개월40대회원수비중, 최근12개월50대회원수비중, 최근12개월60대이상회원수비중
        sections=['월요일이용건수비중', '화요일이용건수비중', '수요일이용건수비중', '목요일이용건수비중', '금요일이용건수비중', '토요일이용건수비중', '일요일이용건수비중', '5시11시이용건수비중', '12시13시이용건수비중', '14시17시이용건수비중', '18시22시이용건수비중', '23시4시이용건수비중', '현지인이용건수비중', '최근12개월남성회원수비중', '최근12개월여성회원수비중', '최근12개월20대이하회원수비중', '최근12개월30대회원수비중', '최근12개월40대회원수비중', '최근12개월50대회원수비중', '최근12개월60대이상회원수비중']
        for section in sections:
            if conditions[section]['필요성']:
                if conditions[section]['기준값'] is None:
                    if conditions[section]['비교문구']=='높다':
                        candidates=candidates[candidates[section]==candidates[section].max()]
                        self.current_question+=f'{section}이 가장 높고 '
                    else:
                        candidates=candidates[candidates[section]==candidates[section].min()]
                        self.current_question+=f'{section}이 가장 낮고 '
                else:
                    standard=conditions[section]['기준값']
                    if conditions[section]['비교문구']=='높다':
                        candidates=candidates[candidates[section]>=standard]
                        self.current_question+=f'{section}이 {standard} 이상 '
                    elif conditions[section]['비교문구']=='낮다':
                        candidates=candidates[candidates[section]<=standard]
                        self.current_question+=f'{section}이 {standard} 이상 '
                    else:
                        candidates=candidates[candidates[section]==standard]
                        self.current_question+=f'{section}이 {standard}와 같고 '
        self.current_question+='를 만족하는 곳은?'
        # 지역
        if len(candidates)>1:
            if self.user_config['area']=='전체':
                pass
            else:
                candidates=candidates[candidates['분할구역']==self.user_config['area']]
            
            # 성별
            if self.user_config['gender']=='남성':
                candidates=candidates.sort_values(by=['최근12개월남성회원수비중'],ascending=[False])
            elif self.user_config['gender']=='여성':
                candidates=candidates.sort_values(by=['최근12개월여성회원수비중'],ascending=[False])

            # 나이
            if self.user_config['age']=='20대이하':
                candidates=candidates.sort_values(by=['최근12개월20대이하회원수비중'],ascending=[False])
            elif self.user_config['age']=='30대':
                candidates=candidates.sort_values(by=['최근12개월30대회원수비중'],ascending=[False])
            elif self.user_config['age']=='40대':
                candidates=candidates.sort_values(by=['최근12개월40대회원수비중'],ascending=[False])
            elif self.user_config['age']=='50대':
                candidates=candidates.sort_values(by=['최근12개월50대회원수비중'],ascending=[False])
            elif self.user_config['age']=='60대이상':
                candidates=candidates.sort_values(by=['최근12개월60대이상회원수비중'],ascending=[False])
        
        return self.current_question,candidates.index[:top_k] if len(candidates)>0 else []