AREA_EXAMPLE={
    '제주시':['일도일동', '일도이동', '이도일동', '이도이동', '삼도일동', '삼도이동', '용담일동', '용담이동', '건입동', '화북동', '삼양동', '봉개동', '아라동', '오라동', '연동', '노형동', '외도동', '이호동', '도두동'],
    '서귀포시':['송산동', '정방동', '중앙동', '천지동', '효돈동', '영천동', '동홍동', '서홍동', '대륜동', '대천동', '중문동', '예래동'],
    '동제주군':['구좌읍', '남원읍', '성산읍', '우도면', '조천읍', '표선면'],
    '서제주군':['대정읍','안덕면', '애월읍', '한림읍', '추자면', '한경면']
}

AREA_GROUP=['제주시', '서귀포시', '동제주군', '서제주군']

USAGE_SCORE={
    '상위 10% 이하': 1.0,
    '10~25%': 0.9,
    '25~50%': 0.75,
    '50~75%': 0.5,
    '75~90%': 0.25,
    '90% 초과(하위 10% 이하)': 0.1
}

SEARCH_COLUMNS=['기준연월', '가맹점명', '개설일자', '업종', '주소', '이용건수구간', '이용금액구간', '건당평균이용금액구간', 
                '월요일이용건수비중', '화요일이용건수비중', '수요일이용건수비중', '목요일이용건수비중', '금요일이용건수비중', 
                '토요일이용건수비중', '일요일이용건수비중', '5시11시이용건수비중', '12시13시이용건수비중', '14시17시이용건수비중', 
                '18시22시이용건수비중', '23시4시이용건수비중', '현지인이용건수비중', '최근12개월남성회원수비중', '최근12개월여성회원수비중', 
                '최근12개월20대이하회원수비중', '최근12개월30대회원수비중', '최근12개월40대회원수비중', '최근12개월50대회원수비중', '최근12개월60대이상회원수비중']

RECOMMEND_COLUMNS=['가맹점명', '업종', '주소', '별점', '운영시간', '번호', '부가설명', '링크', '메뉴']

def map_usage_range(value):
    ranges = value.split(',')
    mapped_values = [USAGE_SCORE[r.strip()] for r in ranges if r.strip() in USAGE_SCORE]
    return sum(mapped_values) / len(mapped_values)