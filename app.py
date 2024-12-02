import joblib
import streamlit as st
import pandas as pd

# 모델 및 스케일러 로드
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('./data/voting_regressor_model.pkl')
    scaler = joblib.load('./data/scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# 데이터 로드
@st.cache_data
def load_data():
    data_path = './data/전국연도별방문자추이및회계정보.csv'
    df = pd.read_csv(data_path)
    return df

df = load_data()

# 축제별 숫자형 데이터 평균값 생성
festival_means = df.select_dtypes(include=['number']).groupby(df['행사축제명']).mean()

# 기본값 생성
default_data = {
    '축체기간(일)': df['축체기간(일)'].mean(),
    '총비용': df['총비용'].mean(),
    '사업수익': df['사업수익'].mean(),
    '순원가': df['순원가'].mean(),
    '(현지인)방문자비율': df['(현지인)방문자비율'].mean(),
    '(외지인)방문자비율': df['(외지인)방문자비율'].mean(),
    '(외국인)방문자비율': df['(외국인)방문자비율'].mean(),
    '(전체)방문자증감': df['(전체)방문자증감'].mean(),
    '전년도 일평균 방문자수': df['전년도 일평균 방문자수'].mean(),
    '일평균 방문자수 증감률': df['일평균 방문자수 증감률'].mean(),
    '(이전)전체방문자': df['(이전)전체방문자'].mean()
}

# 연도별 데이터 조정 함수
def adjust_by_year(input_data, year, base_year=2023):
    growth_rate = 1.02  # 연도별 성장률
    years_diff = year - base_year
    for key in ['총비용', '사업수익', '순원가']:
        input_data[key] *= (growth_rate ** years_diff)
    return input_data

# 특정 축제 데이터 생성 및 스케일링
def prepare_future_data(region, year):
    input_data = default_data.copy()
    # 축제별 평균값 반영
    if region in festival_means.index:
        for key in input_data.keys():
            if key in festival_means.columns:
                input_data[key] = festival_means.loc[region, key]
    # 연도별 성장률 적용
    input_data = adjust_by_year(input_data, year)
    input_df = pd.DataFrame([input_data])
    # 스케일링 적용
    scaled_input_data = scaler.transform(input_df)
    return scaled_input_data


# Streamlit 앱 구성
st.title("미래 지역 축제 방문자 수 예측")

# 사용자 입력 - 연도
current_year = df['개최년도'].max()  # 데이터의 최신 연도
future_years = ['선택'] + [current_year + i for i in range(1, 6)] 
selected_year = st.selectbox("예측할 연도를 선택하세요:", future_years, index=0)  # 기본값을 '선택'으로 설정


# 사용자 입력 - 지역 축제
regions = df['행사축제명'].unique()
regions = ['선택'] + df['행사축제명'].unique().tolist()  
selected_region = st.selectbox("예측할 지역 축제를 선택하세요:", regions, index=0)  # 기본값을 '선택'으로 설정


past_data = pd.DataFrame()  

# 예측 버튼
if st.button("예측"):
    # 사용자 입력이 '선택' 상태인지 확인
    if selected_year != '선택' and selected_region != '선택':
        # 사용자 입력 기반 데이터 생성
        input_data = prepare_future_data(selected_region, int(selected_year))
        try:
            # 예측 수행
            predicted_visitors = model.predict(input_data)
            st.success(f"{selected_year}년 {selected_region}의 예측된 방문자 수: {int(predicted_visitors[0]):,}명")

            # 과거 데이터 추출 및 처리
            if selected_region in df['행사축제명'].values:
                past_data = df[df['행사축제명'] == selected_region][['개최년도', '(전체)방문자수']]
                past_data = past_data.sort_values(by='개최년도').reset_index(drop=True) 
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
    else:
        st.warning("연도와 지역 축제를 선택하세요.")

    # 과거 데이터 표시 (예측 버튼 클릭 후에만)
    if not past_data.empty:
        st.subheader("과거 방문자 수 데이터")
        st.dataframe(past_data)  
    else:
        pass