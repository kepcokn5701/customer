"""가상 CS 만족도 데이터 생성 스크립트"""
import random
import pandas as pd
from datetime import datetime

random.seed(42)

# ── 지사 (17개) ──
branches = [
    "진주지사", "직할", "마산지사", "김해지사", "창원지사",
    "거제지사", "통영고성지사", "밀양지사", "양산지사", "함안의령지사",
    "거창지사", "사천지사", "합천지사", "산청지사", "남해지사",
    "창녕지사", "하동지사"
]

# ── 계약종별 ──
contract_types = {
    "주택용전력": 0.55,
    "일반용(갑)저압": 0.17,
    "농사용(을)저압": 0.12,
    "일반용(갑)고압": 0.04,
    "산업용(갑)고압A": 0.03,
    "교육용(갑)저압": 0.02,
    "산업용(을)고압A": 0.02,
    "일반용(을)고압A": 0.01,
    "농사용(갑)저압": 0.01,
    "임시용(갑)저압": 0.01,
    "교육용(갑)고압": 0.005,
    "가로등(갑)": 0.005,
    "산업용(갑)저압": 0.005,
    "심야전력(을)": 0.005,
}

# ── 업무구분 ──
service_categories = {
    "정전": 0.24,
    "사용자기본사항변경": 0.18,
    "기타단순문의등": 0.10,
    "복지/대가족/수가구등": 0.08,
    "신규/증설": 0.08,
    "요금문의/안내": 0.07,
    "전기설비시험점검": 0.05,
    "이전/철거/폐지/감설": 0.05,
    "계기관련": 0.04,
    "납입방법변경": 0.03,
    "명의변경": 0.03,
    "요금감면/할인": 0.02,
    "전자고지/요금제": 0.02,
    "전력품질/설비개선": 0.005,
    "기타": 0.005,
}

# ── 서술 의견 (다양한 주관식) ──
positive_opinions = [
    "친절하게 응대해주셔서 감사합니다",
    "빠른 처리 감사드립니다",
    "직원분이 매우 친절하셨어요",
    "신속하게 처리해주셔서 만족합니다",
    "상세하게 설명해주셔서 좋았습니다",
    "추운 날씨에 고생 많으셨습니다 감사합니다",
    "항상 친절한 서비스 감사합니다",
    "불편함 없이 잘 처리되었습니다",
    "전화 응대가 매우 좋았습니다",
    "문제 해결이 빨라서 좋았어요",
    "기사님이 매우 성실하게 작업해주셨습니다",
    "덕분에 불편 없이 사용하고 있습니다",
    "설명이 정확하고 이해하기 쉬웠습니다",
    "방문 점검 후 깔끔하게 처리해주셨어요",
    "민원 접수부터 처리까지 매우 빨랐습니다",
    "야간에도 신속하게 대응해주셨습니다",
    "고객센터 직원분 목소리도 밝고 친절했어요",
    "오랜 문제를 한번에 해결해주셨습니다",
    "주말인데도 빨리 와주셔서 감사했습니다",
    "다른 공공기관보다 서비스가 훨씬 좋습니다",
    "이사하면서 걱정했는데 처리가 순조로웠습니다",
    "안전 점검까지 꼼꼼하게 해주셨어요",
    "전기 관련 궁금한 점도 친절히 알려주셨습니다",
    "약속한 시간에 정확히 방문하셨습니다",
    "어르신이라 걱정했는데 쉽게 설명해주셨어요",
    "정전 복구가 생각보다 훨씬 빨랐습니다",
    "온라인 신청도 편리하게 잘 되어있네요",
    "기사님이 작업 후 청소까지 깨끗이 해주셨습니다",
    "전화 한 통으로 간단히 해결되었습니다",
    "겨울철 난방 관련 조언도 해주셔서 좋았어요",
]

negative_opinions = [
    "정전 사전 공지가 없어서 불편했습니다",
    "전화 연결이 너무 오래 걸립니다",
    "처리 기간이 너무 길었습니다",
    "직원 태도가 불친절했습니다",
    "같은 문제가 반복되고 있습니다",
    "약속 시간에 오지 않아서 불편했습니다",
    "요금 고지서 내용이 이해하기 어렵습니다",
    "설명이 부족해서 아직도 잘 모르겠습니다",
    "정전이 너무 자주 발생합니다",
    "전화 대기 시간이 30분 이상이었습니다",
    "공사 안내가 너무 늦게 왔습니다",
    "민원 접수 후 회신이 없었습니다",
    "복구 시간 안내가 정확하지 않았습니다",
    "요금 청구에 오류가 있었습니다",
    "방문 시간 조율이 어려웠습니다",
    "온라인 시스템이 자주 오류가 납니다",
    "현장 직원과 상담원 말이 달랐습니다",
    "소음 문제를 해결해주지 않았습니다",
    "감면 신청 절차가 너무 복잡합니다",
    "처리 결과에 대한 안내가 없었습니다",
    "이전 설치 시 마감이 깔끔하지 않았어요",
    "계기 교체 후 요금이 갑자기 올랐습니다",
    "비가 오면 전기가 나가는데 근본 해결이 안됩니다",
    "낮시간 정전은 자영업자에게 큰 피해입니다",
    "민원 처리 번호만 주고 진행 상황을 모르겠어요",
]

neutral_opinions = [
    "특별한 의견 없습니다",
    "보통입니다",
    "무난했습니다",
    "그냥 그랬어요",
    "크게 불만은 없습니다",
    "평범한 서비스였습니다",
    "별다른 문제는 없었습니다",
    "괜찮았습니다",
    "딱히 할 말은 없습니다",
    "나쁘지는 않았어요",
]

suggestion_opinions = [
    "야간 정전 복구 인력을 좀 더 보강해주시면 좋겠습니다",
    "모바일 앱으로도 민원 접수가 가능하면 좋겠어요",
    "정전 예정 문자를 좀 더 일찍 보내주세요",
    "어르신들을 위한 방문 상담 서비스가 있으면 좋겠습니다",
    "요금 고지서를 더 쉽게 만들어주세요",
    "콜센터 운영 시간을 연장해주시면 좋겠습니다",
    "온라인 접수 시스템을 좀 더 간편하게 개선해주세요",
    "전기 안전 교육을 지역에서도 해주시면 좋겠어요",
    "복지 할인 대상을 좀 더 넓혀주셨으면 합니다",
    "자동 검침기를 조금 더 확대해주세요",
    "정전 시 복구 예상 시간을 문자로 알려주세요",
    "에너지 절약 팁 같은 안내가 있으면 좋겠습니다",
    "주말에도 민원 처리가 가능하면 좋겠어요",
    "전기차 충전 관련 상담도 해주시면 좋겠습니다",
    "태양광 관련 안내 서비스가 있으면 좋겠어요",
]


def weighted_choice(options_dict):
    items = list(options_dict.keys())
    weights = list(options_dict.values())
    return random.choices(items, weights=weights, k=1)[0]


def generate_score_set():
    """점수를 고르게 분포시키기 위한 함수"""
    # 점수 등급별 확률: 고르게 분포
    grade = random.choices(
        ["excellent", "good", "average", "below_avg", "poor"],
        weights=[0.25, 0.25, 0.25, 0.15, 0.10],
        k=1
    )[0]

    if grade == "excellent":
        base = random.randint(90, 100)
    elif grade == "good":
        base = random.randint(75, 89)
    elif grade == "average":
        base = random.randint(55, 74)
    elif grade == "below_avg":
        base = random.randint(30, 54)
    else:
        base = random.randint(0, 29)

    scores = []
    for _ in range(5):
        variation = random.randint(-10, 10)
        score = max(0, min(100, base + variation))
        # 10 단위로 반올림 (원본 데이터 패턴)
        score = round(score / 10) * 10
        scores.append(score)

    # 종합 점수: 5개 평균 (소수점 반올림)
    composite = round(sum(scores) / 5)
    return scores + [composite]


def generate_opinion(scores):
    """점수에 맞는 다양한 주관식 의견 생성"""
    avg = sum(scores[:5]) / 5

    # 55% 확률로 응답없음
    if random.random() < 0.40:
        return "응답없음"

    if avg >= 80:
        return random.choice(positive_opinions)
    elif avg >= 55:
        # 중간 점수: 건의, 중립, 긍정 혼합
        pool = neutral_opinions + suggestion_opinions
        return random.choice(pool)
    else:
        # 낮은 점수: 부정적 의견 위주
        pool = negative_opinions + suggestion_opinions[:5]
        return random.choice(pool)


def generate_receipt_number():
    prefix = random.choice(["C100", "C800", "C200", "C300"])
    date_str = f"2026{random.randint(1,2):02d}{random.randint(1,28):02d}"
    seq = f"{random.randint(1000, 99999):06d}"
    return f"{prefix}-{date_str}-{seq}"


# ── 데이터 생성 (1000건) ──
NUM_ROWS = 1000
data = []

for _ in range(NUM_ROWS):
    branch = random.choice(branches)
    contract = weighted_choice(contract_types)
    receipt = generate_receipt_number()
    service = weighted_choice(service_categories)
    scores = generate_score_set()
    opinion = generate_opinion(scores)

    data.append({
        "지사": branch,
        "계약종별": contract,
        "접수번호": receipt,
        "업무구분": service,
        "직원 친절도": scores[0],
        "전반적 만족": scores[1],
        "처리 신속도": scores[2],
        "처리 정확도": scores[3],
        "업무 개선도": scores[4],
        "종합 점수": scores[5],
        "서술 의견": opinion,
    })

df = pd.DataFrame(data)

# ── 엑셀 저장 ──
output_path = r"c:\Users\Admin\Desktop\customer\가상_CS만족도_샘플데이터.xlsx"
df.to_excel(output_path, index=False, sheet_name="Sheet1")

print(f"생성 완료: {output_path}")
print(f"총 {len(df)}건")
print(f"\n=== 점수 분포 ===")
print(df[["직원 친절도", "전반적 만족", "처리 신속도", "처리 정확도", "업무 개선도", "종합 점수"]].describe().round(1).to_string())
print(f"\n=== 서술 의견 분포 ===")
no_resp = (df["서술 의견"] == "응답없음").sum()
print(f"응답없음: {no_resp}건 ({no_resp/len(df)*100:.1f}%)")
print(f"실제 의견: {len(df)-no_resp}건 ({(len(df)-no_resp)/len(df)*100:.1f}%)")
print(f"\n=== 지사별 건수 ===")
print(df["지사"].value_counts().to_string())
