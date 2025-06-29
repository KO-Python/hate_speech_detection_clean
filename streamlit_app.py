# streamlit_app.py

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# ✅ 라벨 인코더 로드
category_encoder = joblib.load("category_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# ✅ 토크나이저 & 모델 로드
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

# torch.load로 모델 가중치 로드
model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=len(category_encoder.classes_) + len(target_encoder.classes_))
model.load_state_dict(torch.load("hate_speech_model.pt", map_location='cpu'))
model.eval()

st.title("🗨️ 혐오 표현 탐지기")
st.write("온라인 커뮤니티나 소셜미디어에서 접한 혐오 표현이 의심되는 문장을 입력해보세요!")

text = st.text_input("문장을 입력하세요:")

if st.button("분석하기"):
    if text.strip() == "":
        st.warning("문장을 입력해주세요.")
    else:
        # 입력 토크나이징
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # ✨ logits가 다중 출력이면 category, target으로 split
            # 예시: [batch_size, total_labels] -> 슬라이스 필요
            # 예: 앞쪽은 category, 뒤쪽은 target
            category_logits = logits[:, :len(category_encoder.classes_)]
            target_logits = logits[:, len(category_encoder.classes_):]

            pred_category_idx = torch.argmax(category_logits, dim=1).item()
            pred_target_idx = torch.argmax(target_logits, dim=1).item()

            pred_category = category_encoder.inverse_transform([pred_category_idx])[0]
            pred_target = target_encoder.inverse_transform([pred_target_idx])[0]

        st.success(f"✅ 예측된 혐오 표현 카테고리: **{pred_category}**")
        st.success(f"✅ 예측된 혐오 표현 대상: **{pred_target}**")