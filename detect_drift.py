"""
데이터 드리프트 감지 

원리: 학습 데이터와 운영 데이터의 평균(수치형데이터) / 비율(범주형데이터)을 비교해서
      차이가 크면 "드리프트 발생"으로 판단합니다.

"""

import pandas as pd


def load_data():
    """학습 데이터와 예측 로그를 불러옵니다."""
    train_df = pd.read_csv("data/loan_data.csv")
    pred_df = pd.read_csv("data/prediction_logs.csv")

    print(f"학습 데이터: {len(train_df)}건")
    print(f"예측 로그:  {len(pred_df)}건")

    return train_df, pred_df


def check_numerical_drift(train_df, pred_df, columns, threshold=20):
    """
    수치형 피처의 드리프트를 감지합니다.

    방법: 학습 데이터 평균 vs 운영 데이터 평균을 비교해서
          차이가 threshold% 이상이면 드리프트로 판단합니다.

    예) threshold=20 → 평균이 20% 이상 차이나면 드리프트
    """
    print("\n[ 수치형 피처 드리프트 체크 ]")
    print("-" * 55)
    print(f"  {'피처':>10s} | {'학습 평균':>10s} | {'운영 평균':>10s} | {'차이(%)':>7s} | 결과")
    print("-" * 55)

    drift_count = 0

    for col in columns:
        if col not in train_df.columns or col not in pred_df.columns:
            continue

        train_mean = train_df[col].mean()
        pred_mean = pred_df[col].mean()

        # 평균 차이를 %로 계산
        if train_mean != 0:
            diff_pct = abs(train_mean - pred_mean) / abs(train_mean) * 100
        else:
            diff_pct = 0

        is_drift = diff_pct >= threshold
        status = "DRIFT!" if is_drift else "OK"

        if is_drift:
            drift_count += 1

        print(f"  {col:>10s} | {train_mean:>10.1f} | {pred_mean:>10.1f} | {diff_pct:>6.1f}% | {status}")

    print(f"\n  → 수치형 드리프트: {drift_count}개 발견")
    return drift_count


def check_categorical_drift(train_df, pred_df, columns, threshold=10):
    """
    범주형 피처의 드리프트를 감지합니다.

    방법: 각 카테고리의 비율을 비교해서
          가장 많이 변한 카테고리의 차이가 threshold%p 이상이면 드리프트

    예) 학습: 남성 60%, 여성 40%
        운영: 남성 45%, 여성 55%
        → 최대 차이 = 15%p → threshold(10)보다 크므로 DRIFT
    """
    print("\n[ 범주형 피처 드리프트 체크 ]")
    print("-" * 55)

    drift_count = 0

    for col in columns:
        if col not in train_df.columns or col not in pred_df.columns:
            continue

        # 각 카테고리의 비율 계산
        train_ratio = train_df[col].value_counts(normalize=True)
        pred_ratio = pred_df[col].value_counts(normalize=True)

        # 모든 카테고리에 대해 비율 차이 계산
        all_categories = set(train_ratio.index) | set(pred_ratio.index)
        max_diff = 0

        for cat in all_categories:
            t = train_ratio.get(cat, 0) * 100  # %로 변환
            p = pred_ratio.get(cat, 0) * 100
            diff = abs(t - p)
            max_diff = max(max_diff, diff)

        is_drift = max_diff >= threshold
        status = "DRIFT!" if is_drift else "OK"

        if is_drift:
            drift_count += 1

        print(f"  {col:>8s} | 최대 비율 차이: {max_diff:.1f}%p | {status}")

    print(f"\n  → 범주형 드리프트: {drift_count}개 발견")
    return drift_count


def check_prediction_drift(train_df, pred_df):
    """
    예측 결과(승인율)의 드리프트를 감지합니다.

    학습 데이터의 승인율과 운영 데이터의 승인율을 비교합니다.
    """
    print("\n[ 예측 결과 드리프트 체크 ]")
    print("-" * 55)

    train_rate = train_df["승인여부"].mean() * 100
    pred_rate = pred_df["approved"].astype(int).mean() * 100
    diff = abs(train_rate - pred_rate)

    print(f"  학습 데이터 승인율: {train_rate:.1f}%")
    print(f"  운영 데이터 승인율: {pred_rate:.1f}%")
    print(f"  차이: {diff:.1f}%p")

    return diff


def main():
    print("=" * 55)
    print("  데이터 드리프트 감지 리포트 ")
    print("=" * 55)

    # 1. 데이터 불러오기
    train_df, pred_df = load_data()

    # 2. 수치형 피처 체크 (평균 차이가 20% 이상이면 드리프트)
    numerical_cols = [
        "나이", "연소득", "근속연수", "신용점수",
        "기존대출건수", "연간카드사용액", "부채비율",
        "대출신청액", "대출기간",
    ]
    num_drifted = check_numerical_drift(train_df, pred_df, numerical_cols)

    # 3. 범주형 피처 체크 (비율 차이가 10%p 이상이면 드리프트)
    categorical_cols = ["성별", "주거형태", "대출목적", "상환방식"]
    cat_drifted = check_categorical_drift(train_df, pred_df, categorical_cols)

    # 4. 예측 결과 체크
    pred_diff = check_prediction_drift(train_df, pred_df)

    # 5. 종합 결론
    total_drifted = num_drifted + cat_drifted
    print("\n" + "=" * 55)
    print("  종합 결론")
    print("=" * 55)
    print(f"  드리프트 발생 피처: {total_drifted}개")
    print(f"    - 수치형: {num_drifted}개")
    print(f"    - 범주형: {cat_drifted}개")
    print(f"  승인율 차이: {pred_diff:.1f}%p")

    # 재학습 판단: 드리프트 3개 이상 또는 승인율 차이 10%p 이상
    if total_drifted >= 3 or pred_diff > 10:
        print("\n  ** 재학습이 필요합니다! **")
        print("  → train.py를 실행하여 모델을 재학습하세요.")
        return True
    else:
        print("\n  현재 모델이 안정적입니다.")
        return False


if __name__ == "__main__":
    needs_retrain = main()
