import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# =============================================
# 모델과 토크나이저 로드 함수
# =============================================
@st.cache_resource()
def load_model():
    tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
    model = BertForSequenceClassification.from_pretrained(
        'klue/bert-base',
        num_labels=6  # 클래스 수는 학습할 때와 동일해야 함!
    )
    model.load_state_dict(torch.load("hate_speech_model.pt", map_location='cpu'))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# =============================================
# 스트림릿 UI: 타이틀 & 안내문
# =============================================
st.title("혐오 표현 탐지기")

st.write(
    "🗨️ **온라인 커뮤니티나 소셜미디어에서 접한 혐오 표현이 의심되는 문장을 입력해보세요!**\n"
    "모델이 입력한 문장을 분석해 혐오 표현 카테고리를 예측해드립니다."
)

# =============================================
# 입력창 & 버튼
# =============================================
text = st.text_area("문장을 입력하세요:")

if st.button("분석하기"):
    if text.strip() == "":
        st.warning("문장을 입력해주세요!")
    else:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
        st.success(f"예측된 혐오 표현 카테고리 번호: {pred}")