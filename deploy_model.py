"""
모델 배포 스크립트

retrain.py로 재학습한 모델을 S3에 업로드합니다.
S3 업로드가 완료되면 s3 -> Lambda → ECS 자동 재시작이 트리거됩니다.

사용 순서:
  1. python detect_drift.py    드리프트 감지
  2. python retrain.py         재학습 (로컬 저장)
  3. python deploy_model.py    S3 업로드 → 자동 배포

실행 방법: python deploy_model.py

cloutwatch 로그에서 배포 과정과 결과를 확인할 수 있습니다.
"""
import os
import boto3

# ─────────────────────────────────
# S3 설정 — 본인 번호로 변경하세요
# ─────────────────────────────────
S3_BUCKET = "mlops-lab-shared-data"
S3_PREFIX = "student-17"                 # XX → 본인 번호
AWS_REGION = "ap-northeast-2"

MODEL_DIR = "models"
MODEL_FILES = [
    "loan_pipeline.pkl",  # 실제 ecs에서 로드하는 모델
    "label_encoders.pkl",
    "feature_names.pkl",
    "accuracy.txt",
]


def check_local_models():
    """로컬 모델 파일이 존재하는지 확인합니다."""
    print("[1/3] 로컬 모델 파일 확인")
    for fname in MODEL_FILES:
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"  ✘ {path} 파일이 없습니다.")
            print("  retrain.py를 먼저 실행하세요.")
            return False
        size = os.path.getsize(path)
        print(f"  {fname} ({size:,} bytes)")
    return True


def check_accuracy():
    """재학습 정확도를 확인하고, 배포 여부를 확인합니다."""
    print("\n[2/3] 재학습 결과 확인")

    accuracy_path = os.path.join(MODEL_DIR, "accuracy.txt")
    if os.path.exists(accuracy_path):
        with open(accuracy_path, "r") as f:
            accuracy = f.read().strip()
        print(f"  재학습 모델 정확도: {accuracy}")
    else:
        print("  정확도 파일 없음 (accuracy.txt)")

    # 사용자 확인
    answer = input("\n  이 모델을 S3에 배포하시겠습니까? (y/n): ").strip().lower()
    if answer != "y":
        print("  배포를 취소합니다.")
        return False
    return True


def upload_to_s3():
    """모델 파일을 S3에 업로드합니다."""
    print(f"\n[3/3] S3 업로드: s3://{S3_BUCKET}/{S3_PREFIX}/")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    for fname in MODEL_FILES:
        local_path = os.path.join(MODEL_DIR, fname)
        s3_key = f"{S3_PREFIX}/{fname}"

        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"  {fname} → s3://{S3_BUCKET}/{s3_key}")

    print()
    print(f"  S3 업로드 완료!")
    print(f"  Lambda -> ECS 자동 재시작 진행 중")
    print(f"  약 2~3분 후 새 모델로 서빙이 시작됩니다.")
    print()


def main():
    print()
    print("  모델 S3 배포")
    print(f"  대상: s3://{S3_BUCKET}/{S3_PREFIX}/")
    print()

    # 1. 로컬 모델 파일 확인
    if not check_local_models():
        return

    # 2. 정확도 확인 + 배포 확인
    if not check_accuracy():
        return

    # 3. S3 업로드
    upload_to_s3()


if __name__ == "__main__":
    main()
