"""
모델 재학습 스크립트
드리프트가 감지된 후 실행합니다.
기존 학습 데이터(loan_data.csv) + 운영 로그(prediction_logs.csv - cloud watch data)를
합쳐서 모델을 새로 학습합니다.

실행 방법: python retrain.py
"""
import io
import os
import joblib
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


TRAIN_DATA = "data/loan_data.csv"
PRED_LOGS = "data/prediction_logs.csv"
MODEL_DIR = "models"

# S3 설정 — 본인 번호로 변경하세요
S3_BUCKET = "mlops-lab-shared-data"
S3_PREFIX = "student-17"                 # XX → 본인 번호
AWS_REGION = "ap-northeast-2"

FEATURE_COLS = [
    "나이", "성별", "연소득", "근속연수", "주거형태",
    "신용점수", "기존대출건수", "연간카드사용액", "부채비율",
    "대출신청액", "대출목적", "상환방식", "대출기간",
]
TARGET_COL = "승인여부"
CATEGORICAL_COLS = ["성별", "주거형태", "대출목적", "상환방식"]


def load_and_merge_data():
    """
    STEP 1: 기존 학습 데이터 + 운영 로그를 합칩니다.

    prediction_logs.csv는 컬럼명이 다르므로 맞춰줍니다:
      - approved -> 승인여부
      - request_id, timestamp 등 불필요한 컬럼 제거
    """
    print()
    print("  [STEP 1] 데이터 로드 및 병합")
    print()

    # 기존 학습 데이터
    train_df = pd.read_csv(TRAIN_DATA)
    print(f"  기존 학습 데이터: {len(train_df)}건")

    # 운영 로그
    pred_df = pd.read_csv(PRED_LOGS)
    print(f"  운영 로그 데이터: {len(pred_df)}건")

    # 운영 로그를 학습 데이터 형식에 맞추기
    # approved → 승인여부로 이름 변경
    pred_df = pred_df.rename(columns={"approved": TARGET_COL})

    # 학습에 필요한 컬럼만 남기기
    pred_df = pred_df[FEATURE_COLS + [TARGET_COL]]

    # 두 데이터 합치기
    merged_df = pd.concat([train_df, pred_df], ignore_index=True)
    print(f"\n  합친 데이터: {len(merged_df)}건")
    print(f"    - 승인: {merged_df[TARGET_COL].sum()}건")
    print(f"    - 거절: {len(merged_df) - merged_df[TARGET_COL].sum()}건")

    return merged_df


def train_model(df):
    """
    STEP 2~4: 전처리 -> 데이터 분할 -> 모델 학습
    """
    # ─── STEP 2: 피처/타겟 분리 + 인코딩 ───
    print()
    print("  [STEP 2] 전처리 (범주형 인코딩)")
    print()

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"  {col}: {list(le.classes_)}")

    # ─── STEP 3: 데이터 분할 ───
    print()
    print("  [STEP 3] 데이터 분할 (80% 학습 / 20% 테스트)")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"  학습 데이터: {len(X_train)}건")
    print(f"  테스트 데이터: {len(X_test)}건")

    # ─── STEP 4: 모델 학습 ───
    print()
    print("  [STEP 4] 모델 학습 (XGBoost)")
    print()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        )),
    ])

    pipeline.fit(X_train, y_train)

    # 평가
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  새 모델 정확도: {accuracy:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['거절', '승인'])}")

    return pipeline, label_encoders, accuracy, X_test, y_test


def get_s3_model_accuracy(X_test, y_test):
    """
    S3에 배포된 현재 운영 모델의 정확도를 측정합니다.
    같은 테스트 데이터로 평가하여 공정하게 비교합니다.
    """
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)

        # S3에서 운영 모델 다운로드
        resp = s3.get_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}/loan_pipeline.pkl")
        s3_pipeline = joblib.load(io.BytesIO(resp["Body"].read()))

        # 같은 테스트 데이터로 평가 (X_test는 이미 인코딩된 상태)
        y_pred = s3_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc

    except Exception as e:
        print(f"  S3 모델 로드 실패: {e}")
        print(f"  기존 모델 없음으로 처리합니다 (첫 배포)")
        return 0.0


def compare_and_save(pipeline, label_encoders, new_accuracy, X_test, y_test):
    """
    STEP 5: S3 운영 모델과 비교 후 로컬 저장
    """
    print()
    print("  [STEP 5] S3 운영 모델과 비교")
    print(f"  (S3: s3://{S3_BUCKET}/{S3_PREFIX}/)")
    print()

    # S3에서 현재 운영 모델의 정확도 측정
    old_accuracy = get_s3_model_accuracy(X_test, y_test)
    if old_accuracy > 0:
        print(f"  S3 운영 모델 정확도: {old_accuracy:.4f}")

    print(f"  새로 학습한 모델 정확도: {new_accuracy:.4f}")

    # 비교
    diff = new_accuracy - old_accuracy
    if diff > 0:
        print(f"\n  +{diff:.4f} 향상!")
    elif diff == 0:
        print(f"\n  동일한 성능.")
    else:
        print(f"\n  {diff:.4f} 하락.")

    # 로컬 저장 (deploy_model.py에서 S3 업로드 여부를 결정)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "loan_pipeline.pkl"))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_names.pkl"))

    with open(os.path.join(MODEL_DIR, "accuracy.txt"), "w") as f:
        f.write(f"{new_accuracy:.4f}")

    print(f"\n  로컬 저장 완료: {MODEL_DIR}/")
    print(f"    - loan_pipeline.pkl   (모델)")
    print(f"    - label_encoders.pkl  (인코더)")
    print(f"    - feature_names.pkl   (피처 목록)")
    print(f"    - accuracy.txt        (정확도 기록)")

    return new_accuracy > old_accuracy


def main():
    print()
    print("  모델 재학습 시작")
    print("  (기존 데이터 + 운영 로그 합쳐서 학습)")
    print()

    # 1. 데이터 로드 및 병합
    merged_df = load_and_merge_data()

    # 2~4. 전처리 → 분할 → 학습
    pipeline, label_encoders, accuracy, X_test, y_test = train_model(merged_df)

    # 5. S3 운영 모델과 비교 + 로컬 저장
    is_better = compare_and_save(pipeline, label_encoders, accuracy, X_test, y_test)

    print()
    print("  재학습 완료!")
    print()
    if is_better:
        print("  ✔ 새 모델이 운영 모델보다 좋습니다.")
        print("  다음 단계:")
        print("    python deploy_model.py  S3 배포  ECS 자동 재시작")
    else:
        print("  ✘ 새 모델이 운영 모델보다 좋지 않습니다.")
        print("  배포를 권장하지 않지만, 필요하다면:")
        print("    python deploy_model.py   S3 배포 (선택)")
    print()


if __name__ == "__main__":
    main()


# models 폴더에는 새로 학습한 모델과 관련 파일들이 저장됩니다.
# accuracy.txt, loanPipeline.pkl
# deploy_model.py에서 이 파일들을 S3로 업로드하여 배포할 수 있습니다.