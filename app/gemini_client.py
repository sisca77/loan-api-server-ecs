# pip install openai python-dotenv

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ANALYSIS_PROMPT = """\
사용자의 질문을 분석하고 답변해주세요.

다음 JSON 형식으로만 응답하세요 (```json 마크다운 없이 순수 JSON만):
{{
  "answer": "질문에 대한 답변",
  "sentiment": "긍정 | 부정 | 중립",
  "sentiment_score": 0.0~1.0 사이의 감성 점수 (1.0이 가장 긍정),
  "category": "카테고리명"
}}

카테고리는 다음 중 하나를 선택하세요:
- 대출/금융
- 부동산
- 신용/채무
- 일반상담
- 기타

사용자 질문: {question}
"""


def ask_gemini(question: str) -> str:
    """OpenAI API에 질문을 보내고 응답 텍스트를 반환한다."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


def analyze_question(question: str) -> dict:
    """질문에 대한 답변 + 감성분석 + 카테고리 분류를 수행한다."""
    prompt = ANALYSIS_PROMPT.format(question=question)
    text = ask_gemini(prompt).strip()
    # ```json ... ``` 마크다운 블록 제거
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    return json.loads(text)
