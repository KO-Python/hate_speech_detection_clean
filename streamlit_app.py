import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import joblib

# 사용자 정의 모델 클래스
class CategoryOnlyBERT(torch.nn.Module):
    def __init__(self, pretrained_model_name, num_category_labels):
        super(CategoryOnlyBERT, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.category_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_category_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        category_logits = self.category_classifier(pooled_output)
        return category_logits

# 라벨 인코더 로드
category_encoder = joblib.load("category_encoder.pkl")

# 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
model = CategoryOnlyBERT(pretrained_model_name="klue/bert-base", num_category_labels=len(category_encoder.classes_))
model.load_state_dict(torch.load("hate_speech_model.pt", map_location='cpu'))
model.eval()

# Streamlit UI
st.title("🗨️ 혐오 표현 탐지기")
st.write("온라인 커뮤니티나 소셜미디어에서 접한 혐오 표현이 의심되는 문장을 입력해보세요!")

# 사용자로부터 입력 받기
text = st.text_input("문장을 입력하세요:")

# 분석 버튼 클릭 시 동작
if st.button("분석하기"):
    if text.strip() == "":  # 입력이 비어있는 경우 경고 메시지
        st.warning("문장을 입력해주세요.")
    else:
        # 입력된 텍스트를 토크나이저로 처리
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # 예측 수행
        with torch.no_grad():
            # 모델에 입력
            category_logits = model(**inputs)

            # 예측된 카테고리 인덱스 추출
            pred_category_idx = torch.argmax(category_logits, dim=1).item()

            # 라벨 인코딩을 통해 카테고리 추출
            pred_category = category_encoder.inverse_transform([pred_category_idx])[0]

        # 예측된 카테고리 결과 출력
        st.success(f"✅ 예측된 혐오 표현 카테고리: **{pred_category}**")