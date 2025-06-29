import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import joblib

# ✅ 라벨 인코더 로드
category_encoder = joblib.load("category_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# ✅ 모델 & 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
model = torch.load("hate_speech_model.pt", map_location='cpu')
model.eval()

st.title("혐오 표현 탐지기")
st.write("🗨️ 온라인 커뮤니티나 소셜미디어에서 접한 혐오 표현이 의심되는 문장을 입력해보세요!")

text = st.text_input("문장을 입력하세요:")

if st.button("분석하기"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # ✅ category 출력
        predicted_category_idx = torch.argmax(logits[0]).item()
        predicted_category = category_encoder.inverse_transform([predicted_category_idx])[0]

        # ✅ target 출력 (만약 두 번째 출력이 target이라면)
        predicted_target_idx = torch.argmax(logits[1]).item()
        predicted_target = target_encoder.inverse_transform([predicted_target_idx])[0]

    st.write(f"✅ 예측된 혐오 표현 카테고리: **{predicted_category}**")
    st.write(f"✅ 예측된 혐오 표현 대상: **{predicted_target}**")