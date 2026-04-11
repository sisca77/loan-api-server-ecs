import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# 데이터 로드
train_df = pd.read_csv("data/loan_data.csv")
pred_df  = pd.read_csv("data/prediction_logs.csv")

feature_cols = ["나이", "연소득", "근속연수", "신용점수",
                "기존대출건수", "연간카드사용액", "부채비율",
                "대출신청액", "대출기간",
                "성별", "주거형태", "대출목적", "상환방식"]

ref = train_df[feature_cols]   # Reference (학습)
cur = pred_df[feature_cols]    # Current (운영)

# === 이 3줄이 전부입니다! ===
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=ref, current_data=cur)
snapshot.save_html("drift_report.html")