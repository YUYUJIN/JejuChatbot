import os
import streamlit as st

from utils.function import *
from engine import JejuRestaurantRecommender

def main():
    st.set_page_config(page_title="Jeju food chatbot", page_icon="🤖",layout="wide")

    # Replicate Credentials
    with st.sidebar:
        st.title("🤖제주도 맛집 추천 챗봇")

        st.write(" ")

        st.subheader("제주도 지역")

        # selectbox 레이블 공백 제거
        st.markdown(
            """
            <style>
            .stSelectbox label {  /* This targets the label element for selectbox */
                display: none;  /* Hides the label element */
            }
            .stSelectbox div[role='combobox'] {
                margin-top: -20px; /* Adjusts the margin if needed */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        from PIL import Image
        image = Image.open("./image/jejumap.png")
        st.image(image, width=300)
        area = st.sidebar.selectbox("지역", ["전체", "제주시", "서귀포시", "동제주군", "서제주군"], key="area",label_visibility="hidden")

        st.write(" ")

        st.subheader("성별")

        # radio 레이블 공백 제거
        st.markdown(
            """
            <style>
            .stRadio > label {
                display: none;
            }
            .stRadio > div {
                margin-top: -20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        gender = st.radio(
            '성별',
            ('전체', '남성', '여성'),
            label_visibility="hidden"
        )

        st.write("")
        
        st.subheader("나이")

        # radio 레이블 공백 제거
        st.markdown(
            """
            <style>
            .stRadio > label {
                display: none;
            }
            .stRadio > div {
                margin-top: -20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        age = st.radio(
            '나이',
            ('전체', '20대이하', '30대', '40대', '50대', '60대이상'),
            label_visibility="hidden"
        )
        
        
        
    # chat-bot 페이지 디자인
    st.title(f"🍊취향저격 제주도 맛집 추천 챗봇")

    st.write("뜨끈한 국물?🍜 매콤한 음식?🌶️ 현재 어떤 음식을 먹고 싶은지 알려주세요")

    st.write("단순 검색 내용은 신한카드데이터를 기반으로 설정 지역으로 축소 제공하고, 성별/나이대 비중에 맞추어 우선 제공합니다.")
    st.write("추천 내용은 신한카드데이터를 기반으로 설정 지역/성별/나이대 비중을 고려하여 추천합니다.")

    st.write(" ")

    user_config={'area':area, 'gender': gender, 'age':age}

    # chatbot 입출력
    chain=JejuRestaurantRecommender(data_path='./data',
                                    store_path='./store',
                                    user_config=user_config)

    def clear_chat_history():
        st.session_state.clear()  # 히스토리 초기화 개선

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    def is_place_related_question(user_input):
        keywords = ["가게", "레스토랑", "식당", "추천", "음식점", "장소", "여기", "위치","곳","알려줘","찾아줘","말해줘"]
        return any(keyword in user_input for keyword in keywords)

    def generate_answer(user_input):
        res=chain(user_input)
        
        return res


    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("안녕하세요 👋 어떤 맛집을 추천해드릴까요?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if is_place_related_question(prompt):
            with st.spinner("맞춤형 맛집 정보를 확인하고 있습니다. 조금만 기다려 주세요..."):
                res = generate_answer(prompt)
        else:
            res = "먹고 싶은 음식을 말해주세요🤔"
        
        st.session_state.messages.append({"role": "assistant", "content": res})
        
        with st.chat_message("assistant"): 
            st.write_stream(stream_data(res))   

if __name__=='__main__':
    main()