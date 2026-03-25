# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
import math

random.seed(42)
np.random.seed(42)

N = 200

# ── 지사 (weighted like real data) ──
지사_list = [
    ("경남본부", 15), ("진주지사", 12), ("마산지사", 12), ("거제지사", 7),
    ("밀양지사", 7), ("함안의령지사", 6), ("진해지사", 6), ("사천지사", 5),
    ("통영지사", 5), ("창녕지사", 4), ("산청지사", 4), ("거창지사", 4),
    ("남해지사", 3), ("합천지사", 3), ("하동지사", 3), ("고성지사", 3),
    ("함양지사", 2),
]
지사_names, 지사_weights = zip(*지사_list)
지사_weights = np.array(지사_weights, dtype=float)
지사_weights /= 지사_weights.sum()

# ── 계약종별 (weighted) ──
계약종별_list = [
    ("주택용전력", 70), ("일반용(갑)저압", 15), ("농사용(을)저압", 9),
    ("농사용(갑)", 1.5), ("산업용(갑)저압", 1.3), ("일반용(을)고압A", 0.9),
    ("일반용(갑)II고압A", 0.9), ("산업용(갑)II고압A", 0.6),
    ("산업용(을)고압A", 0.5), ("가로등(을)", 0.3),
]
계약종별_names, 계약종별_weights = zip(*계약종별_list)
계약종별_weights = np.array(계약종별_weights, dtype=float)
계약종별_weights /= 계약종별_weights.sum()

# ── 업무구분 (weighted) ──
업무구분_list = [
    ("기타단순문의등", 31), ("사용자기본사항변경", 18), ("자동이체", 15),
    ("정전", 10), ("요금수납관련", 7), ("청구서재발행", 5),
    ("복지/대가족/수가구등", 3), ("신규/증설", 3), ("계기업무", 2),
    ("청구서/세금계산서", 1.3), ("전기요금문의", 1.3), ("계약종별변경", 1.1),
    ("검침관련", 0.6), ("해지/재사용", 0.4), ("휴지/부활", 0.3),
]
업무구분_names, 업무구분_weights = zip(*업무구분_list)
업무구분_weights = np.array(업무구분_weights, dtype=float)
업무구분_weights /= 업무구분_weights.sum()

# ── 신청방법 (weighted) ──
신청방법_list = [
    ("전화", 70), ("사이버지점", 10), ("내방", 8), ("기타", 6),
    ("전화녹취", 1.4), ("FAX", 1.1), ("검침PDA(핸디터미널)", 0.9),
    ("우편", 0.9), ("모바일", 0.2),
]
신청방법_names, 신청방법_weights = zip(*신청방법_list)
신청방법_weights = np.array(신청방법_weights, dtype=float)
신청방법_weights /= 신청방법_weights.sum()

# ── 접수자구분 (weighted) ──
접수자구분_list = [
    ("직원", 25), ("서울고객센터", 12), ("부산고객센터", 9),
    ("한전ON", 8), ("경기고객센터", 8), ("대구고객센터", 7),
    ("인천고객센터", 5), ("전남고객센터", 5), ("충남고객센터", 4),
    ("경기북부고객센터", 4), ("경남고객센터", 3), ("충북고객센터", 3),
    ("전북고객센터", 2), ("강원고객센터", 2), ("검침사", 2),
    ("제주고객센터", 2), ("기타", 3),
]
접수자구분_names, 접수자구분_weights = zip(*접수자구분_list)
접수자구분_weights = np.array(접수자구분_weights, dtype=float)
접수자구분_weights /= 접수자구분_weights.sum()

# ── 접수번호 prefix mapping ──
prefix_for_접수자 = {
    "서울고객센터": "C100", "부산고객센터": "C800", "경기고객센터": "C350",
    "대구고객센터": "C700", "인천고객센터": "C200", "전남고객센터": "C650",
    "충남고객센터": "C550", "경기북부고객센터": "C300", "경남고객센터": "C850",
    "충북고객센터": "C500", "전북고객센터": "C600", "강원고객센터": "C400",
    "제주고객센터": "C900", "한전ON": None, "직원": None, "검침사": None,
    "기타": None,
}

지사_code_map = {
    "경남본부": "7812", "진주지사": "5740", "마산지사": "5690",
    "거제지사": "5768", "밀양지사": "5732", "함안의령지사": "5730",
    "진해지사": "5720", "사천지사": "5726", "통영지사": "5750",
    "창녕지사": "5760", "산청지사": "5783", "거창지사": "5770",
    "남해지사": "5747", "합천지사": "5778", "하동지사": "5771",
    "고성지사": "5713", "함양지사": "5782",
}


def gen_접수번호(지사, 접수자구분, idx):
    prefix = prefix_for_접수자.get(접수자구분)
    if prefix is None:
        prefix = 지사_code_map.get(지사, "5690")
    seq = f"{idx+1:06d}"
    return f"{prefix}-20260115-{seq}"


# ── Score generation with realistic distribution ──
def weighted_score(high_pct=0.6, mid_pct=0.25, low_pct=0.15):
    """Generate a score with controllable distribution."""
    r = random.random()
    if r < high_pct:
        return random.choice([100, 100, 100, 90])
    elif r < high_pct + mid_pct:
        return random.choice([70, 80, 80, 90])
    else:
        return random.choice([10, 20, 30, 40, 50, 60])


# ── 서술 의견 pool ──
# Positive opinions (diverse)
positive_opinions = [
    "친절하게 잘 안내해 주셔서 감사합니다.",
    "빠르고 정확한 업무처리에 만족합니다.",
    "직원분이 매우 친절하셔서 기분 좋았습니다.",
    "항상 한전 서비스에 만족하고 있습니다.",
    "궁금한 사항을 상세하게 설명해 주셔서 좋았어요.",
    "신속하게 처리해 주셔서 고맙습니다.",
    "전화 응대가 매우 친절했습니다. 감사해요.",
    "어려운 내용을 쉽게 풀어서 설명해 주셨습니다.",
    "업무 처리가 깔끔하고 빨라서 좋았습니다.",
    "한전ON 앱이 편리해서 잘 사용하고 있습니다.",
    "정전 복구가 빠르게 이루어져서 감사합니다.",
    "민원 접수 후 바로 연락 주셔서 좋았습니다.",
    "전기요금 관련 문의에 정확히 답변해 주셨어요.",
    "내방했는데 대기시간 없이 바로 처리해 주셨습니다.",
    "사이버지점 이용이 편리해서 만족합니다.",
    "검침원분이 항상 친절하세요. 감사합니다.",
    "직원분의 전문적인 안내에 감사드립니다.",
    "어려운 상황에서 적극적으로 도와주셨습니다.",
    "세금계산서 발급이 빠르고 정확해서 좋았습니다.",
    "휴일인데도 신속하게 대응해 주셔서 감동받았습니다.",
    "복잡한 요금 체계를 이해하기 쉽게 설명해 주셨어요.",
    "자동이체 변경이 간편하게 처리되었습니다.",
    "전반적으로 서비스가 많이 개선된 것 같아요.",
    "항상 친절하고 빠른 응대 감사드립니다.",
    "현장 직원분들이 고생 많으십니다. 감사해요.",
    "전기 관련 안전 점검도 꼼꼼히 해주셔서 좋았습니다.",
    "불편사항을 즉시 해결해 주셔서 만족합니다.",
    "고객 입장에서 생각해 주시는 게 느껴졌어요.",
    "명절인데도 불구하고 빠르게 처리해 주셨습니다.",
    "특별한 불만 없이 잘 이용하고 있습니다.",
]

# Negative/constructive opinions (diverse)
negative_opinions = [
    "전화 연결 대기시간이 너무 깁니다. 개선 바랍니다.",
    "고객센터 전화 연결이 너무 어려워요. 30분 넘게 기다렸습니다.",
    "ARS 메뉴가 너무 복잡해서 원하는 메뉴를 찾기 어렵습니다.",
    "상담원 연결까지 시간이 오래 걸려서 불편했습니다.",
    "전기요금이 왜 이렇게 많이 나왔는지 설명이 부족했어요.",
    "정전이 자주 발생하는데 원인 파악을 제대로 해주셨으면 합니다.",
    "직원분이 불친절해서 기분이 좋지 않았습니다.",
    "처리 과정에 대한 안내가 없어서 답답했습니다.",
    "인터넷으로 신청했는데 확인 전화가 와서 번거로웠어요.",
    "요금 고지서가 너무 늦게 와서 납부일을 놓칠 뻔했습니다.",
    "한전ON 앱이 자주 오류가 나서 불편합니다.",
    "전화했는데 담당자가 자리에 없다고 해서 다시 전화해야 했어요.",
    "민원 처리가 너무 느립니다. 일주일이 지나도 연락이 없었어요.",
    "계량기 교체 일정을 사전에 알려주지 않아 불편했습니다.",
    "AI 자동 응답이 제 질문을 이해하지 못해서 답답했어요.",
    "전화 상담 시 배경 소음이 너무 커서 통화가 어려웠습니다.",
    "같은 내용을 여러 번 설명해야 해서 피곤했습니다.",
    "주말에는 전화 상담이 안 되어서 불편합니다.",
    "요금 계산 방식이 너무 복잡해서 이해하기 어렵습니다.",
    "정전 복구 예상 시간을 알려주지 않아 답답했어요.",
]

# Neutral/mixed opinions
neutral_opinions = [
    "응답없음",
    "응답없음",
    "응답없음",
    "응답없음",
    "응답없음",
    "특이사항 없습니다.",
    "해당없음",
    "없음",
    "보통입니다.",
    "별다른 의견 없습니다.",
    "그냥 그렇습니다.",
    "대체적으로 만족합니다만 대기시간이 좀 길었습니다.",
    "친절하긴 했는데 전화 연결이 오래 걸렸어요.",
    "업무 처리는 빨랐는데 설명이 조금 부족했습니다.",
    "서비스는 괜찮은데 주차 공간이 부족해요.",
    "전반적으로 무난했습니다.",
    "빠른 처리 감사하지만 ARS 메뉴 개선이 필요해요.",
    "직원은 친절했는데 시스템이 복잡한 것 같아요.",
    "점심시간에도 운영되면 좋겠습니다.",
    "모바일 앱 UI가 좀 더 직관적이면 좋겠어요.",
]

# Short/colloquial opinions
short_opinions = [
    "굿",
    "좋아요",
    "감사합니다",
    "만족",
    "별로",
    "보통",
    "최고!",
    "수고하세요",
    "고마워요",
    "잘 처리됨",
    "OK",
    "친절함",
    "빨랐음",
    "불만없음",
    "괜찮음",
]


def pick_opinion(overall_score):
    """Pick an opinion based on the overall satisfaction score."""
    r = random.random()
    # ~35% 응답없음/neutral, rest depends on score
    if r < 0.30:
        return random.choice(neutral_opinions)
    elif r < 0.40:
        return random.choice(short_opinions)

    if overall_score >= 90:
        return random.choice(positive_opinions)
    elif overall_score >= 70:
        if random.random() < 0.5:
            return random.choice(positive_opinions)
        else:
            return random.choice(neutral_opinions)
    else:
        if random.random() < 0.7:
            return random.choice(negative_opinions)
        else:
            return random.choice(neutral_opinions)


# ── 급락 조합 보장용 시드 (업무, 접수자구분, 건수) ──
# 2~9건 범위로 각 급락 조합에 강제 건수 배정
_drop_seeds = [
    ("정전", "한전ON", 4),
    ("계기업무", "기타", 3),
    ("요금수납관련", "한전ON", 5),
    ("청구서재발행", "기타", 3),
    ("검침관련", "직원", 3),
]

# ── 실질적 리스크 보장용 시드 (>=10건 & 평균 이하) ──
# 이 조합들은 약간 낮은 점수(60~80)로 생성됨
_real_risk_seeds = [
    ("기타단순문의등", "서울고객센터", 15),   # 고객센터 그룹
    ("자동이체", "부산고객센터", 12),         # 고객센터 그룹
]

# ── Generate data ──
rows = []
_seed_queue = []
for _biz, _recv, _cnt in _drop_seeds:
    for _ in range(_cnt):
        _seed_queue.append((_biz, _recv, "drop"))
for _biz, _recv, _cnt in _real_risk_seeds:
    for _ in range(_cnt):
        _seed_queue.append((_biz, _recv, "real"))
random.shuffle(_seed_queue)

for i in range(N):
    지사 = np.random.choice(지사_names, p=지사_weights)
    계약종별 = np.random.choice(계약종별_names, p=계약종별_weights)

    # 시드 큐에서 급락/리스크 조합 우선 소진
    _seed_type = None
    if _seed_queue:
        업무구분, 접수자구분, _seed_type = _seed_queue.pop()
        신청방법 = np.random.choice(신청방법_names, p=신청방법_weights)
    else:
        업무구분 = np.random.choice(업무구분_names, p=업무구분_weights)
        신청방법 = np.random.choice(신청방법_names, p=신청방법_weights)
        접수자구분 = np.random.choice(접수자구분_names, p=접수자구분_weights)

    # Ensure consistency: 사이버지점 → 한전ON, 전화 → 고객센터 etc.
    if 신청방법 == "사이버지점":
        접수자구분 = "한전ON"
    elif 신청방법 in ("검침PDA(핸디터미널)",):
        접수자구분 = "검침사"
    elif 신청방법 == "내방":
        접수자구분 = "직원"

    접수번호 = gen_접수번호(지사, 접수자구분, i)

    # ── _group_channel 기준 채널그룹 결정 (앱 로직과 동일) ──
    _s = str(접수자구분).strip()
    if "검침" in _s:
        채널그룹 = None
    elif any(k in _s for k in ["직원", "방문", "현장"]):
        채널그룹 = "직원"
    elif any(k in _s for k in ["고객센터", "콜센터", "전화", "센터"]):
        채널그룹 = "고객센터"
    elif any(k in _s for k in ["ON", "온라인", "앱", "홈페이지", "인터넷", "모바일", "한전ON"]):
        채널그룹 = "한전ON"
    else:
        채널그룹 = "기타"

    # ── 시드 타입에 따라 점수 분포 결정 ──
    _is_drop_combo = _seed_type == "drop"
    _is_real_risk  = _seed_type == "real"

    # ── Scores ──
    # Determine which scores are "-" based on channel
    is_phone = 신청방법 in ("전화", "전화녹취")
    is_cyber = 신청방법 == "사이버지점"
    is_visit = 신청방법 == "내방"

    # 급락 조합: 매우 낮은 점수 (50점대 이하)
    # 실질적 리스크: 평균 부근이지만 약간 낮은 점수 (60~80점대)
    if _is_drop_combo:
        _score_fn = lambda: weighted_score(0.1, 0.2, 0.7)   # 70% 저점
    elif _is_real_risk:
        _score_fn = lambda: weighted_score(0.25, 0.50, 0.25) # 50% 중간, 넓은 분포
    else:
        _score_fn = lambda: weighted_score(0.6, 0.25, 0.15)  # 기본

    # 이용 편리성: 0 for phone-based (they don't use the system), else scored
    if is_phone:
        이용편리성 = 0
    else:
        이용편리성 = _score_fn() if _is_drop_combo else weighted_score(0.55, 0.30, 0.15)

    # 직원 친절도: scored for phone/visit, 0 for cyber
    if is_cyber:
        직원친절도 = 0
    else:
        직원친절도 = _score_fn() if _is_drop_combo else weighted_score(0.65, 0.25, 0.10)

    # 전반적 만족
    전반적만족 = _score_fn() if _is_drop_combo else weighted_score(0.55, 0.30, 0.15)

    # 사회적 책임: always "-"
    사회적책임 = "-"

    # 처리 신속도: "-" for most phone calls, scored for others
    if is_phone and random.random() < 0.78:
        처리신속도 = "-"
    else:
        처리신속도 = _score_fn() if _is_drop_combo else weighted_score(0.60, 0.25, 0.15)

    # 처리 정확도: "-" for cyber, else scored
    if is_cyber:
        처리정확도 = "-"
    else:
        처리정확도 = _score_fn() if _is_drop_combo else weighted_score(0.65, 0.25, 0.10)

    # 업무 개선도: "-" for cyber, else scored
    if is_cyber:
        업무개선도 = "-"
    else:
        업무개선도 = _score_fn() if _is_drop_combo else weighted_score(0.55, 0.30, 0.15)

    # 사용 추천도: "-" for phone/visit/기타, scored for cyber
    if is_cyber:
        사용추천도 = _score_fn() if _is_drop_combo else weighted_score(0.60, 0.25, 0.15)
    else:
        사용추천도 = "-"

    # ── 종합 점수: average of non-"-" and non-0 scores ──
    score_map = {
        "이용 편리성": 이용편리성,
        "직원 친절도": 직원친절도,
        "전반적 만족": 전반적만족,
        "처리 신속도": 처리신속도,
        "처리 정확도": 처리정확도,
        "업무 개선도": 업무개선도,
        "사용 추천도": 사용추천도,
    }
    valid_scores = [v for v in score_map.values() if v != "-" and v != 0]
    if valid_scores:
        종합점수 = round(sum(valid_scores) / len(valid_scores), 1)
        # Round to nearest even number like the real data (mostly even)
        종합점수 = round(종합점수 / 2) * 2
        종합점수 = float(종합점수)
    else:
        종합점수 = 0.0

    서술의견 = pick_opinion(전반적만족)

    rows.append({
        "순번": i + 1,
        "지사": 지사,
        "계약종별": 계약종별,
        "접수번호": 접수번호,
        "업무구분": 업무구분,
        "신청방법": 신청방법,
        "접수자구분": 접수자구분,
        "이용 편리성": 이용편리성,
        "직원 친절도": 직원친절도,
        "전반적 만족": 전반적만족,
        "사회적 책임": 사회적책임,
        "처리 신속도": str(처리신속도) if 처리신속도 != "-" else "-",
        "처리 정확도": str(처리정확도) if 처리정확도 != "-" else "-",
        "업무 개선도": str(업무개선도) if 업무개선도 != "-" else "-",
        "사용 추천도": str(사용추천도) if 사용추천도 != "-" else "-",
        "종합 점수": 종합점수,
        "서술 의견": 서술의견,
    })

df_out = pd.DataFrame(rows)
df_out.to_csv("dummy_data.csv", index=False, encoding="utf-8-sig")

# Also save as xlsx to match original format
df_out.to_excel("dummy_data.xlsx", index=False, engine="openpyxl")

print(f"Generated {len(df_out)} rows")
print()
print("=== Distribution check ===")
print(f"지사 unique: {df_out['지사'].nunique()}")
print(f"계약종별 unique: {df_out['계약종별'].nunique()}")
print(f"업무구분 unique: {df_out['업무구분'].nunique()}")
print(f"신청방법 unique: {df_out['신청방법'].nunique()}")
print()
print("=== 종합 점수 분포 ===")
print(df_out['종합 점수'].describe())
print()
print("=== 서술 의견 분포 ===")
opinion_counts = df_out['서술 의견'].value_counts()
print(f"응답없음: {(df_out['서술 의견'] == '응답없음').sum()}")
print(f"Unique opinions: {df_out['서술 의견'].nunique()}")
print()
print("=== Sample rows ===")
for idx in [0, 10, 50, 100, 150, 199]:
    row = df_out.iloc[idx]
    print(f"\nRow {idx}:")
    for col in df_out.columns:
        print(f"  {col}: {row[col]}")
