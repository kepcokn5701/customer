
# ==============================================================
#  AI 활용 고객경험관리시스템 조사결과 분석 웹 대시보드  v3.5
#  회사 CS 리포트 양식 반영 · Corporate Edition
#  실행법: python -m streamlit run cs_dashboard.py
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
import math
import io, re, os, json, ssl, textwrap

# ── SSL 방화벽 우회 (사내 프록시/방화벽 환경) ──────────────────
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass
try:
    import requests
    _orig_send = requests.Session.send
    def _ssl_bypass_send(self, *args, **kwargs):
        kwargs["verify"] = False
        return _orig_send(self, *args, **kwargs)
    requests.Session.send = _ssl_bypass_send
except Exception:
    pass

# ── 선택적 라이브러리 ──────────────────────────────────────────
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
except Exception:
    KONLPY_AVAILABLE = False

# WordCloud는 TF-IDF Action-Trigger 분석으로 대체됨

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except Exception:
    KEYBERT_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import requests as _req_lib
    _GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not _GEMINI_KEY:
        # .env 파일에서 읽기
        _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(_env_path):
            with open(_env_path, encoding="utf-8") as _ef:
                for _line in _ef:
                    _line = _line.strip()
                    if _line.startswith("GEMINI_API_KEY="):
                        _GEMINI_KEY = _line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if _GEMINI_KEY:
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except Exception:
    GEMINI_AVAILABLE = False

# ══════════════════════════════════════════════════════════════
#  지사별 환경 변수 사전 (KB)
# ══════════════════════════════════════════════════════════════
FULL_OFFICE_KB = {
    # ── 중부권 (창원 직할) ────────────────────────────────────
    "경남본부/직할/마산지사/진해지사": {
        "context": "[공단/도심형] 창원(성산 약21만·의창 약21만)+마산(약36만)+진해(약19만). "
                   "연령: 성산 3050 직장인 중심, 마산 고령화 진행(구도심), 진해 해군·가족 중심. "
                   "주요 산업: 방산·정밀기계·가전(성산 국가산단), 수출자유지역·전통제조(마산), 항만물류·해군기지(진해). "
                   "지형: 성산 평지형 대규모 공단, 마산 배후산+항만, 진해 항구. "
                   "기질: 강직함·기술 자부심(성산), 뚝심·마산항 자부심(마산), 절제·해양 기질(진해). "
                   "전력특성: [핵심 소비축] 국가산단 전력 수요 밀집·전압 품질 민감도 극상, 마산 노후 설비 현대화, 진해신항 물류자동화 전력수요 급증. "
                   "고객구성: 주택용 50~60%, 호수 多.",
        "action": "산단 전담 요금제 설명회, 야간 긴급 복구 알림 고도화, 기업 전용 상담 채널 강화, 마산 구도심 에너지 효율화 안내."
    },
    # ── 중부권 (별도 지사) ────────────────────────────────────
    "함안의령지사/함안지사/의령지사": {
        "context": "[공단/복합형] 인구 약8.6만(함안+의령). "
                   "연령: 고령화 높으나 산단 인구 상존. "
                   "주요 산업: 부품제조(함안), 농업(의령). 지형: 평야와 강이 만남. "
                   "기질: 넉넉함, 보수적. "
                   "전력특성: 중소 산단 분산 전원 및 농촌형 재생에너지 모델. 설비 수용성 민감 — 주민 상생 모델 중요. "
                   "고객구성: 농사용 20%↑, 호수 대비 산업용 비중 높음.",
        "action": "노후 공단 전력 설비 안전 점검 안내, 지역 커뮤니티 중심 대면 홍보, 산업용·농사용 복합 민원 원스톱 처리."
    },
    "창녕지사": {
        "context": "[농촌/공단형] 인구 약5.8만 / 초고령화. "
                   "주요 산업: 타이어(넥센산단, 전력 다소비), 마늘·양파 농사. 지형: 우포늪 등 습지와 평야. "
                   "기질: 소박함, 끈기. "
                   "전력특성: 대형 타이어 공장(전력 다소비)과 농업 전력 공존. "
                   "고객구성: 농사용 20%↑.",
        "action": "영농기 농사용 전기 대리 접수 활성화, 관광지 숙박업소 전용 요금 컨설팅."
    },
    # ── 남해안권 ──────────────────────────────────────────────
    "거제지사": {
        "context": "[공단/해안형] 인구 약23만 / 3040 비중 높음(활동적). "
                   "주요 산업: 조선, 해양플랜트. 지형: 섬, 리아스식 해안. "
                   "기질: 화끈함, 직설적. "
                   "전력특성: [수요 중심축] 조선소 아크 용접 등 고출력 전력망 필수, 경남 전력 소비 거대 지분. "
                   "고객구성: 주택용 50~60%, 호수 多.",
        "action": "결론 중심의 선제적 알림톡 발송, 조선소 부하 변동에 따른 맞춤형 컨설팅 제공."
    },
    "사천지사": {
        "context": "[공단/해안형] 인구 약10.5만 / 항공산업 종사자(전문직). "
                   "주요 산업: 항공우주(KAI), 위성. 지형: 해안선과 평야. "
                   "기질: 급함, 정이 많음. "
                   "전력특성: 항공기 정밀 제조를 위한 무정전 전원(UPS) 시스템 중요.",
        "action": "항공 정밀 제조 전용 '고품질 전력 관리 리포트' 발송, 신규 공장 신청 절차 간소화."
    },
    "통영지사": {
        "context": "[관광/해안형] 인구 약12만 / 고령화 및 관광 인구. "
                   "주요 산업: 수산물, LNG, 관광. 지형: 섬이 많은 다도해. "
                   "기질: 예술적, 자존심 강함. "
                   "전력특성: [공급 중심축] LNG 인수기지 기반 발전 및 수소 연계, 경남 전력 생산 토대.",
        "action": "도서(섬) 지역 정전 대응 매뉴얼 배포, 숙박시설·수산가공업 전용 요금 안내문 제작."
    },
    "고성지사": {
        "context": "[농촌/해안형] 인구 약4.9만 / 초고령화. "
                   "주요 산업: 조선부품, 화력발전. 지형: 해안 평지. "
                   "기질: 묵묵함, 내유외강. "
                   "전력특성: [공급 중심축] 삼천포 화력발전소 소재, 전력 계통의 시작점.",
        "action": "읍면사무소 협업 '찾아가는 창구' 운영, 발전소 주변 지역 상생 프로그램 안내."
    },
    "남해지사": {
        "context": "[관광/농촌형] 인구 약4.1만 / 초고령화(관광 중심). "
                   "주요 산업: 관광, 마늘 농업. 지형: 섬, 수려한 경관. "
                   "기질: 온화함, 외지인 수용적. "
                   "전력특성: 신재생에너지(태양광) 및 관광 지구 에너지 자립 섬. "
                   "고객구성: 농사용 20%↑.",
        "action": "귀촌인 대상 전력 사용 가이드 배부, 이장님 네트워크 활용 비대면 서비스 홍보."
    },
    "하동지사": {
        "context": "[농촌/발전소형] 인구 약4.2만 / 초고령화. "
                   "주요 산업: 화력발전, 차(茶) 농업. 지형: 섬진강 유역, 산악. "
                   "기질: 문학적, 지조 있음. "
                   "전력특성: [공급 중심축] 하동 화력발전소 소재, 대규모 송전 계통 중심. "
                   "고객구성: 농사용 20%↑.",
        "action": "발전소 주변 지역 지원 제도 상세 안내, 영농기 맞춤형 농사용 신청 알림톡 발송."
    },
    # ── 서부권 ────────────────────────────────────────────────
    "진주지사": {
        "context": "[도심형] 인구 약34만 / 교육·혁신도시(청년층). "
                   "주요 산업: 공공기관, 항공, 교육. 지형: 남강 흐르는 내륙 평탄. "
                   "기질: 선비 정신, 깐깐함. "
                   "전력특성: [정책 허브] 한국남동발전 본사 소재, 에너지 정책 및 R&D 중심. 경남 전력 정책 조율. "
                   "고객구성: 주택용 50~60%, 호수 多.",
        "action": "혁신도시 3040 타겟 디지털 서비스 강화, 공공기관 전용 에너지 효율화 상담."
    },
    "밀양지사": {
        "context": "[복합/농촌형] 인구 약10만 / 고령화 및 귀농 인구. "
                   "주요 산업: 나노융합산단, 농업. 지형: 산악과 평야 조화. "
                   "기질: 원칙주의, 보수적. "
                   "전력특성: [송전 핵심] 나노 국가산단 수요 및 전력 계통 통과 핵심지. 설비 수용성 민감 — 주민 상생 모델 중요. "
                   "고객구성: 농사용 20%↑.",
        "action": "시설원예 단지 전력 과부하 사전 예보, 고령 농민 대상 요금 감면 제도 현장 안내."
    },
    "거창지사": {
        "context": "[복합/농촌형] 인구 약6만 / 교육도시 특성(학생층). "
                   "주요 산업: 승강기, 사과 농업. 지형: 내륙 분지. "
                   "기질: 굳건함, 학구적. "
                   "전력특성: 승강기 테스트 타워 등 특수 산업용 전력 수요. "
                   "고객구성: 농사용 20%↑.",
        "action": "교육 시설 전용 전기 요금 컨설팅, 농사용 서류 간소화 및 온라인 접수 지원."
    },
    "산청지사": {
        "context": "[농촌형] 인구 약3.4만 / 초고령화. "
                   "주요 산업: 한방, 약초, 관광. 지형: 험준한 지리산. "
                   "기질: 강직함, 소박함. "
                   "전력특성: 지리산 양수 발전 및 소수력 등 청정 에너지원. 설비 수용성 민감 — 주민 상생 모델 중요. "
                   "고객구성: 농사용 20%↑.",
        "action": "약초 건조기 등 특정 부하 사용 시기 점검 안내, 마을회관 중심 대면 서비스 강화."
    },
    "함양지사": {
        "context": "[농촌형] 인구 약3.7만 / 초고령화. "
                   "주요 산업: 산양삼, 물류. 지형: 고산 지대. "
                   "기질: 구수한 정, 고집 있음. "
                   "전력특성: 에너지 소외 지역 마이크로그리드 구축 적합. "
                   "고객구성: 농사용 20%↑.",
        "action": "과수원용 저온저장고 요금 체계 안내, 농번기 현장 민원 우선 처리제 운영."
    },
    "합천지사": {
        "context": "[관광/농촌형] 인구 약4.1만 / 초고령화(경남 최고수준). "
                   "주요 산업: 농업, 관광(해인사). 지형: 산악 및 황강 유역. "
                   "기질: 보수적, 신중함. "
                   "전력특성: [공급 중심축] 합천댐 수력 발전 및 수상 태양광 발전 거점. 설비 수용성 민감 — 주민 상생 모델 중요. "
                   "고객구성: 농사용 20%↑.",
        "action": "사찰 및 관광지 조명 설비 효율화 상담, 축사 전용 정전 예보 서비스 강화."
    },
}

# ── 경남 전력 산업 권역별 요약 (AI 분석 배경 컨텍스트) ─────────
POWER_INDUSTRY_CONTEXT = (
    "공급 중심축: 고성(삼천포화력), 하동(하동화력), 통영(LNG), 합천(수력)이 경남 전력 생산의 물리적 토대. "
    "수요 중심축: 창원 성산(기계·방산), 거제(조선)가 경남 전체 전력 소비의 거대 지분, 전압 품질 최민감. "
    "정책 허브: 진주(혁신도시)에 에너지 공기업 본사 위치, 경남 전역 전력 정책 조율. "
    "계통·수용성: 밀양 등 송전 선로 통과 지역과 고령화 심한 의령·산청·합천 등은 전력 설비 수용성 민감 — 주민 상생 모델 핵심."
)

def _get_office_kb(office_name: str) -> dict:
    """지사명으로 KB를 검색. 복합 키(창원지사/마산지사/진해지사)도 매칭."""
    if not office_name:
        return {}
    for key, val in FULL_OFFICE_KB.items():
        if office_name in key:
            return val
    return {}

# ══════════════════════════════════════════════════════════════
#  2025 연간 CS 분석 KB (엑셀 원본 기반 검증 완료)
#  ※ 모든 수치는 25_상반기_종합점수_수정본.xlsx + 25.하반기 개인정보
#    삭제 취합본.xlsx에서 직접 계산한 값이며, 추측은 포함하지 않음
# ══════════════════════════════════════════════════════════════
ANNUAL_ANALYSIS_KB = {
    # ── 전체 현황 ──
    "summary": (
        "2025년 총 17,274건(상반기 9,164 + 하반기 8,504). "
        "전체 종합 평균 93.63점. "
        "상반기 93.03점 → 하반기 94.19점(+1.16점 상승). "
        "15개 지사 하반기 개선, 2개 지사만 역행."
    ),
    # ── 월별 추이 ──
    "monthly_trend": (
        "6월 91.52점(연간 최저, 전월 대비 -1.71점) → "
        "7월 94.01점(+2.49점 급반등, 하반기 설문 항목 변경 시점). "
        "11월 95.10점, 12월 96.29점(연말 최고점). "
        "상반기 92~93점대 횡보, 하반기 94~96점대로 한 단계 상승."
    ),
    # ── 세부항목 변화 (공통 항목) ──
    "detail_items": (
        "상반기 항목: 이용편리성(91.31), 응대친절성(94.12), 설명의충분(92.85), 사회적책임(94.11). "
        "하반기 항목: 응대친절성(95.32), 설명의충분(93.50), 처리신속도(94.49), 처리정확도(95.04), 종합편리성(92.61). "
        "공통 항목 변화: 응대친절성 +1.20점↑, 설명의충분 +0.66점↑."
    ),
    # ── 리스크 조합 TOP5 (지사×업무, 10건 이상) ──
    "risk_combos": (
        "1위 함양지사×청구서재발행 83.59점(78건), "
        "2위 창녕지사×청구서재발행 86.06점(33건), "
        "3위 통영지사×청구서재발행 86.90점(42건), "
        "4위 함안의령지사×전기공급관련 87.08점(24건), "
        "5위 경남본부×전기공급관련 87.50점(16건). "
        "→ 청구서재발행이 TOP5 중 3개 점유(구조적 문제)."
    ),
    # ── 업무유형 주목 포인트 ──
    "biz_highlights": (
        "정전 업무: 유일하게 의미 있는 하락(-2.85점, 하반기 응대 품질 악화). "
        "전기요금문의: +10.68점 급등이나 하반기 30건 소표본(표본 효과 가능). "
        "요금수납관련: +4.30점 실질적 개선."
    ),
    # ── 불만족 고객 (50점 이하) ──
    "dissatisfied": (
        "전체 472건(2.7%). 상반기 270건(2.9%) → 하반기 215건(2.5%) 개선. "
        "불만족 비율 상위: 창녕지사 4.1%, 통영지사 3.7%, 경남본부 3.5%. "
        "불만족 다발 업무: 자동이체(103건 최다), 청구서재발행, 요금안내 순."
    ),
}

# 지사별 2025 연간 데이터 (상→하반기 변화)
OFFICE_ANNUAL_DATA = {
    "산청지사":     {"rank": 1,  "avg": 95.37, "cnt": 559,  "h1": 94.32, "h2": 96.30, "delta": "+1.98"},
    "거창지사":     {"rank": 2,  "avg": 95.10, "cnt": 700,  "h1": 93.83, "h2": 96.13, "delta": "+2.30"},
    "밀양지사":     {"rank": 3,  "avg": 94.60, "cnt": 947,  "h1": 93.79, "h2": 95.44, "delta": "+1.65"},
    "진주지사":     {"rank": 4,  "avg": 94.25, "cnt": 1246, "h1": 93.88, "h2": 94.56, "delta": "+0.68"},
    "함양지사":     {"rank": 5,  "avg": 94.24, "cnt": 531,  "h1": 93.27, "h2": 95.16, "delta": "+1.89"},
    "김해지사":     {"rank": 6,  "avg": 94.04, "cnt": 1465, "h1": 93.38, "h2": 94.63, "delta": "+1.25"},
    "사천지사":     {"rank": 7,  "avg": 93.99, "cnt": 770,  "h1": 93.38, "h2": 94.56, "delta": "+1.18"},
    "남해지사":     {"rank": 8,  "avg": 93.95, "cnt": 525,  "h1": 94.02, "h2": 93.89, "delta": "-0.13"},
    "하동지사":     {"rank": 9,  "avg": 93.89, "cnt": 466,  "h1": 92.26, "h2": 95.25, "delta": "+2.99"},
    "양산지사":     {"rank": 10, "avg": 93.81, "cnt": 1238, "h1": 93.19, "h2": 94.40, "delta": "+1.21"},
    "경남본부":     {"rank": 11, "avg": 93.63, "cnt": 316,  "h1": 92.74, "h2": 94.24, "delta": "+1.50"},
    "합천지사":     {"rank": 12, "avg": 93.28, "cnt": 537,  "h1": 92.35, "h2": 94.07, "delta": "+1.72"},
    "거제지사":     {"rank": 13, "avg": 93.17, "cnt": 1286, "h1": 92.56, "h2": 93.71, "delta": "+1.15"},
    "의령지사":     {"rank": 14, "avg": 92.96, "cnt": 310,  "h1": 92.65, "h2": 93.26, "delta": "+0.61"},
    "함안의령지사": {"rank": 14, "avg": 92.96, "cnt": 310,  "h1": 92.65, "h2": 93.26, "delta": "+0.61"},
    "고성지사":     {"rank": 15, "avg": 92.56, "cnt": 703,  "h1": 91.92, "h2": 93.18, "delta": "+1.26"},
    "통영지사":     {"rank": 16, "avg": 92.54, "cnt": 979,  "h1": 91.99, "h2": 93.14, "delta": "+1.15"},
    "창녕지사":     {"rank": 17, "avg": 92.40, "cnt": 704,  "h1": 91.80, "h2": 92.91, "delta": "+1.11"},
}

# 월별 급변 지사 (전월 대비 ±3점 이상)
OFFICE_SUDDEN_CHANGES = {
    "합천지사": "5→6월 91.9→80.1점(-11.8점 급락) → 6→7월 80.1→95.6점(+15.5점 회복). 단월 대량 불만 집중.",
    "고성지사": "9→10월 96.3→88.9점(-7.4점 급락) → 10→11월 88.9→97.0점(+8.1점 회복). 단월 대량 불만 집중.",
}

def _get_office_annual(office_name: str) -> str:
    """지사명으로 연간 분석 데이터를 조회하여 텍스트로 반환."""
    if not office_name:
        return ""
    for key, val in OFFICE_ANNUAL_DATA.items():
        if office_name in key or key in office_name:
            parts = [
                f"2025 연간 순위: 17개 지사 중 {val['rank']}위 (평균 {val['avg']}점, {val['cnt']}건)",
                f"상반기 {val['h1']}점 → 하반기 {val['h2']}점 ({val['delta']}점)",
            ]
            # 급변 이력 추가
            for skey, stxt in OFFICE_SUDDEN_CHANGES.items():
                if office_name in skey or skey in office_name:
                    parts.append(f"급변 이력: {stxt}")
            return " / ".join(parts)
    return ""

# ══════════════════════════════════════════════════════════════
#  0. 한글 폰트
# ══════════════════════════════════════════════════════════════
FONT_PATH = None
FONT_PROP = None

for _c in [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    "/System/Library/Fonts/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
]:
    if os.path.exists(_c):
        FONT_PATH = _c
        try:
            fm.fontManager.addfont(_c)
        except Exception:
            pass
        FONT_PROP = fm.FontProperties(fname=_c)
        plt.rcParams["font.family"] = FONT_PROP.get_name()
        break

plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════════════════════════
#  1. 디자인 상수
# ══════════════════════════════════════════════════════════════
C = dict(
    navy   = "#1a3a6c",
    blue   = "#0055a5",
    sky    = "#2196f3",
    light  = "#e3f2fd",
    gold   = "#f0a500",
    teal   = "#00897b",
    red    = "#c62828",
    orange = "#e65100",
    green  = "#2e7d32",
    gray   = "#546e7a",
    white  = "#ffffff",
    bg     = "#f4f7fc",
    # 채널별 테마 색상 (리포트 양식)
    ch_employee = "#4caf50",   # 직원 응대 — 녹색
    ch_center   = "#1976d2",   # 고객센터 — 파란색
    ch_online   = "#fbc02d",   # 회사ON — 노란색
    ch_etc      = "#9e9e9e",   # 기타
)

PIE_COLORS   = ["#1a3a6c","#0055a5","#2196f3","#42a5f5","#90caf9",
                "#64b5f6","#1565c0","#0288d1","#0097a7","#00897b"]
MIXED_COLORS = ["#1a3a6c","#f0a500","#2196f3","#00897b","#c62828",
                "#7b1fa2","#e65100","#558b2f","#0288d1","#ad1457"]
# 점수 구간 색상 (리포트 양식)
BUCKET_COLORS = {"90점 이상": "#2e7d32", "70~90점": "#1976d2",
                 "50~70점": "#f0a500", "50점 미만": "#c62828"}
BUCKET_ORDER  = ["90점 이상", "70~90점", "50~70점", "50점 미만"]
OFFICE_ORDER  = ["경남본부", "직할", "진주지사", "마산지사", "거제지사", "밀양지사", "사천지사",
                 "통영지사", "거창지사", "함안의령지사", "창녕지사", "합천지사", "진해지사",
                 "하동지사", "고성지사", "산청지사", "남해지사", "함양지사"]
PLOTLY_TPL    = "plotly_white"

def _sort_offices(values):
    """지사 목록을 OFFICE_ORDER 기준으로 정렬. 목록에 없는 값은 뒤에 원래 순서로."""
    order_map = {v: i for i, v in enumerate(OFFICE_ORDER)}
    known = [v for v in values if v in order_map]
    unknown = [v for v in values if v not in order_map]
    return sorted(known, key=lambda v: order_map[v]) + sorted(unknown)

def _sort_df_by_office(df, office_col, ascending=True):
    """DataFrame을 OFFICE_ORDER 기준으로 정렬."""
    order_map = {v: i for i, v in enumerate(OFFICE_ORDER)}
    df = df.copy()
    df["_office_sort"] = df[office_col].map(order_map).fillna(999)
    df = df.sort_values("_office_sort", ascending=ascending).drop(columns=["_office_sort"])
    return df

# ══════════════════════════════════════════════════════════════
#  2. 키워드 사전
# ══════════════════════════════════════════════════════════════
NEGATIVE_KEYWORDS = [
    "불만","불편","항의","화남","짜증","느림","느려","오류","오작동",
    "불량","고장","재방문","지연","오래","기다림","실망","최악","별로",
    "안됨","문제","취소","환불","비싸다","비싸","과다","과금","폭탄",
    "불친절","무시","황당","어이없","답답","이해불가","납득불가","부당",
    "잘못","실수","착오","불합리","불공정","차별","반복","또다시","여전히",
    "아직도","힘듭니다","어렵습니다","불쾌","무성의","소홀","방치","지체",
    "정전","단전","누전","위험","안전","사고",
]

# VOC 하이라이팅 전용 — 시설·기술·제도 중심 (감정어 제외)
VOC_HIGHLIGHT_KW = [
    # 시설·설비
    "정전","단전","누전","감전","고장","불량","오작동","오류",
    "계량기","변압기","전주","전선","배전","차단기","개폐기","누수","누유",
    # 요금·제도
    "과금","과다","요금","청구","납부","감면","할인","연체","체납",
    "계약","해지","이전","명의변경","폐전","신증설",
    # 절차·처리
    "재방문","지연","지체","방치","미처리","미흡","누락","중복",
    "민원","접수","처리","답변","회신","약속",
    # 안전·사고
    "위험","안전","사고","화재","폭발","합선",
]

RUDE_KEYWORDS = [
    "불친절","무시","무성의","불쾌","반말","태도","무례","소홀","건방",
    "고압적","짜증","화남","면박","윽박","냉담","퉁명","귀찮","성의없",
    "막말","인상","불성실","함부로","권위적","비꼼","쏘아","차갑","냉소",
]

# 긍정 문맥 키워드 — 이 단어가 부정/불친절 키워드 근처에 있으면 긍정으로 재판정
POSITIVE_CONTEXT = [
    "감사","친절","좋","만족","잘","훌륭","경청","공감","인상깊","고마",
    "칭찬","최고","기쁘","따뜻","성실","편안","빠르","신속","정확","꼼꼼",
    "배려","존경","사려깊","세심","상냥","다정","협조","도움","반갑","행복",
    "깔끔","쾌적","개선","나아","발전",
]

_STOP = {
    "이","가","은","는","을","를","의","에","에서","으로","로","와","과","하",
    "것","수","있","없","않","못","더","또","그","저","있다","없다","됩니다",
    "합니다","했다","하는","하여","해서","입니다","이다","이고","이며","같은",
    "같이","너무","매우","정말","아주","조금","많이","항상","계속","때문",
    "경우","부분","관련","대한","위한","통해","에게","주세요","바랍니다",
    "부탁","감사","좋겠","드립니다","했습니다","있습니다","없습니다","이고",
    "이나","이라","해주세요","요청","드립니다","하겠습니다","하셨으면",
    "관리","시스템","문의","확인","처리","접수","신청","등록","변경","조회",
    "이용","서비스","고객","전화","상담","안내","답변","진행","완료","내용",
    "회사","회사","전력","전기","사용","사용량","검침","수도","가스",
    "홈페이지","인터넷","방문","센터","지사","지점","담당","담당자","직원",
    "설문","조사","응답","결과","평가","만족","점수","항목","기타","해당",
    "응답없음",
}

# ── Action-Trigger 키워드 추출 전용: 강화 불용어 ──
# CS 본질 개선점을 가리는 단순 감성어·인사말·범용어를 철저히 필터링
# ※ Okt 명사 추출 결과 + regex 2글자 이상 매칭 결과 모두 포괄
_ACTION_STOP = _STOP | {
    # ── 긍정 감성어 (원형 + 활용형 모두) ──
    "친절","친절하게","친절히","친절한","친절하고","친절합니다","친절했습니다",
    "감사","감사합니다","감사드립니다","감사했습니다","감사드려요",
    "만족","만족합니다","만족했습니다","만족스럽","만족도",
    "좋음","좋다","좋아","좋아요","좋겠습니다","좋았습니다","좋습니다",
    "빠름","빠르고","빠르게","빨리","빠른",
    "수고","수고하","수고하셨","수고하십니다","수고많으","수고합니다",
    "고맙","고마웠","고마워","고마운","고맙습니다","고마웠습니다",
    "칭찬","최고","훌륭","편안","편리","깔끔","깨끗","성실","정확","정확한",
    "꼼꼼","꼼꼼하게","꼼꼼히","배려","도움","추천","완벽","노력","발전",
    "유지","응원","믿음","행복","따뜻","경청","공감","상냥","반갑","기쁘",
    "굿","적극","적극적","적극적인","열심","열심히","나아","개선","유쾌","쾌적",
    "고생","덕분","잘해","괜찮","잘하","잘해주","잘해주셔서",
    "주셔서","해주셔서","드립니다","드려요","드리겠습니다",
    "신속","신속하게","신속한","신속히",
    # ── 부정 감성어 (원인 아닌 감정 표현) ──
    "불만","불쾌","화남","짜증","최악","실망","별로","황당","어이없","답답",
    "이해불가","납득불가","힘듭니다","어렵습니다",
    # ── 범용 서베이 필러 / 무의미 단어 ──
    "없음","있음","필요","바람","희망","생각","의견","마음","정도","특별",
    "나름","보통","그냥","일단","전반","전반적","종합","전체","모든","각각",
    "사항","방면","측면","분야","방향","차원","수준","단계","기본","일반",
    "너무","매우","정말","아주","조금","많이","항상","계속","때문","현재",
    "하고","있도록","해서","대로","그래서","그리고","때문에","에서","으로",
    "시간이","시간","기타","해당","부분","경우","관련","대한","위한","통해",
    "업무","업무처리","자세히","쉽게","바로","조금더","했는데","있습니다",
    "없습니다","합니다","입니다","됩니다","습니다","하는","한다",
    "바랍니다","같습니다","같은","같이","하겠습니다","하셨으면",
    "해주세요","주세요","부탁","요청",
    # ── 추가 필러 / 조사 활용형 ──
    "좋았어요","좋겠어요","있으면","없으면","있어","없어","해주면",
    "더욱","다소","정도","위해","봅니다","것이","데요","거든요",
    "필요한","있었","없었","해야","좋을것","좋을","좋을것같","해야할",
    "도움이","것을","대로","처럼","만큼","에서","으로","하면","하려","하도록",
    "상세하게","상세","자세","자세하게","올바른","올바르게",
    "알기쉽게","좀더","더욱더","하였","했음","하였음",
    # ── 한전/전기 일반 용어 (모든 VOC에 공통 → 변별력 없음) ──
    "한전","전기","전력","전기가","고객","고객이","서비스","이용","사용",
    "회사","지사","지점","센터","상담","상담원","전화","통화","연결",
    "직원","담당","담당자","기사",
    # ── 버블 차트 키워드 추출 시 제외할 일반 명사 ──
    "문의","민원","처리","확인","내용","결과","방법","진행","완료","요청사항",
    "감사","친절","만족","불만","개선","건의","의견","제안","답변","응대",
    "빈항목","빈문서",
}

# ── 고객여정 + 실무 카테고리 매핑 ──
# ※ 실제 VOC에서 Okt/regex로 추출되는 단어 형태를 기준으로 매핑
_JOURNEY_CATEGORIES = {
    "📞 안내·상담": [
        "설명", "안내", "안내문", "고지", "통보", "공지", "알림", "문자", "카톡", "메시지",
        "응대", "상담", "문의", "답변", "연락", "통화", "전화", "콜백", "회신", "통지",
        "매뉴얼", "스크립트", "고지서", "청구서",
        "소통", "커뮤니케이션", "공감", "경청", "태도", "말투", "어투",
        "인사", "존대", "존칭", "반말",
    ],
    "📋 접수·절차": [
        "접수", "신청", "서류", "서식", "양식", "절차", "단계", "구비", "증빙",
        "본인확인", "인증", "승인", "심사", "검토", "반려", "보완", "재접수",
        "온라인", "앱", "모바일", "홈페이지", "창구", "대기", "번호표", "예약",
        "대기시간", "대기시간이", "기다림", "기다", "번거로", "복잡", "어렵",
        "서명", "동의서", "서약", "첨부",
        "등록", "변경", "해지", "이전", "명의", "양도",
    ],
    "🔧 현장처리·시공": [
        "방문", "현장", "출동", "공사", "시공", "설치", "철거", "교체", "수리",
        "점검", "검사", "측정", "계량기", "변압기", "전주", "전선", "배전", "누전",
        "정전", "단전", "복구", "송전", "고장", "장애", "사고", "위험", "안전",
        "기사", "작업자", "외주", "협력", "하청",
        "충전소", "전기실", "모뎀", "통신", "수소", "태양광", "발전기",
        "차단기", "개폐기", "배전반", "인입선", "케이블", "전력선",
        "작업", "시공", "준공", "감리", "계약용량",
    ],
    "⏱️ 처리속도·지연": [
        "지연", "느림", "느려", "오래", "늦게", "늦은", "늦다", "오래걸",
        "신속", "빠르", "즉시", "지체", "소요", "일정",
        "기다", "기다림", "대기", "대기시간",
        "재방문", "재처리", "재발", "반복", "미해결", "미처리", "미완료",
    ],
    "📮 사후관리·피드백": [
        "후속", "피드백", "결과통보", "완료통보", "만족확인", "사후", "사과",
        "보상", "감면", "환불", "정산", "이의", "이의제기", "민원",
        "재발", "반복", "재차", "또다시", "여전히", "아직",
        "확인", "안내", "결과", "처리결과", "진행상황",
    ],
    "💰 요금·과금": [
        "요금", "납부", "청구", "고지서", "과금", "누진", "할인", "감면", "경감",
        "계약", "종별", "변경", "이전", "해지", "명의", "양도", "분할", "연체",
        "체납", "독촉", "제도", "규정", "기준", "약관", "정책",
        "비싸", "비싸다", "과다", "폭탄", "부당", "억울", "부과",
        "전기세", "전기요금", "사용량", "검침", "미터기",
        "계산", "산정", "산출", "적용", "누진제", "누진요금",
    ],
    "🏗️ 시설물·환경": [
        "전주", "전선", "변압기", "개폐기", "배전반", "계량기", "미터기",
        "소음", "진동", "악취", "먼지", "미관", "경관", "가지", "나뭇가지",
        "수목", "도로", "인도", "보도", "골목", "위치", "이설",
        "위험", "안전", "사고", "누전", "감전", "화재", "합선",
        "전봇대", "철탑", "송전탑", "지중화",
    ],
    "💻 디지털·시스템": [
        "홈페이지", "앱", "어플", "모바일", "인터넷", "시스템", "프로그램",
        "로그인", "비밀번호", "패스워드", "인증", "공인인증", "본인확인",
        "오류", "에러", "버그", "접속", "속도", "다운", "먹통",
        "결제", "카드", "자동이체", "납부방법",
    ],
}

# ══════════════════════════════════════════════════════════════
#  3. 유틸리티 함수
# ══════════════════════════════════════════════════════════════
def _extract_nouns_konlpy(texts):
    okt = Okt()
    words = []
    for t in texts:
        if not t or str(t).strip() in ("", "nan"):
            continue
        nouns = okt.nouns(str(t))
        words.extend([n for n in nouns if n not in _STOP and len(n) >= 2])
    return words


_VOC_EMPTY = {"", "nan", "응답없음", "없음", "의견없음", "빈문서", "빈항목"}

def extract_keywords(texts, top_n=60):
    valid = [str(t) for t in texts if t and str(t).strip() not in _VOC_EMPTY and len(str(t).strip()) > 2]
    if not valid:
        return []
    if KEYBERT_AVAILABLE and KONLPY_AVAILABLE:
        try:
            nouns = _extract_nouns_konlpy(valid)
            noun_freq = Counter(nouns)
            if noun_freq:
                noun_doc = " ".join(nouns)
                kw_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")
                candidates = list(noun_freq.keys())
                kw_results = kw_model.extract_keywords(
                    noun_doc, candidates=candidates,
                    top_n=min(top_n, len(candidates)),
                    use_mmr=True, diversity=0.5,
                )
                result = []
                for kw, _score in kw_results:
                    if kw in noun_freq and kw not in _STOP and len(kw) >= 2:
                        result.append((kw, noun_freq[kw]))
                result.sort(key=lambda x: x[1], reverse=True)
                if len(result) >= 5:
                    return result[:top_n]
        except Exception:
            pass
    if KONLPY_AVAILABLE:
        try:
            nouns = _extract_nouns_konlpy(valid)
            if nouns:
                return Counter(nouns).most_common(top_n)
        except Exception:
            pass
    words = []
    for t in valid:
        found = re.findall(r"[가-힣]{2,}", t)
        words.extend([w for w in found if w not in _STOP])
    return Counter(words).most_common(top_n)


# ── Action-Trigger TF-IDF 키워드 추출 ──
# 부분매칭 불용어 패턴: 이 문자열을 *포함*하는 단어도 제거
_ACTION_STOP_PARTIAL = [
    "감사", "친절", "만족", "좋겠", "좋았", "좋을", "수고", "고맙", "칭찬", "훌륭", "편안",
    "깔끔", "성실", "정확", "꼼꼼", "배려", "추천", "완벽", "노력", "발전",
    "따뜻", "경청", "상냥", "반갑", "적극", "열심", "신속", "쾌적",
    "불쾌", "짜증", "실망", "황당", "어이없", "답답",
    "합니다", "습니다", "입니다", "됩니다", "겠습니다", "드립니다",
    "주셔서", "해주셔", "주시", "하셔", "주었", "주셨",
    "바랍니다", "같습니다", "없습니다", "있습니다",
    "봅니다", "거든요", "데요", "네요", "어요", "아요",
]

def _is_action_stop(word):
    """강화 불용어 체크: 정확 매칭 + 부분 매칭"""
    if word in _ACTION_STOP:
        return True
    return any(pat in word for pat in _ACTION_STOP_PARTIAL)


def _extract_nouns_action(texts):
    """Action-Trigger용 명사 추출 (강화 불용어 적용)"""
    if KONLPY_AVAILABLE:
        okt = Okt()
        words = []
        for t in texts:
            if not t or str(t).strip() in ("", "nan"):
                continue
            nouns = okt.nouns(str(t))
            words.extend([n for n in nouns if not _is_action_stop(n) and len(n) >= 2])
        return words
    words = []
    for t in texts:
        if not t or str(t).strip() in ("", "nan"):
            continue
        found = re.findall(r"[가-힣]{2,}", str(t))
        words.extend([w for w in found if not _is_action_stop(w)])
    return words


def _tfidf_keywords_sklearn(doc_words_list, top_n=30):
    """sklearn TF-IDF로 문서별 고유 키워드 추출"""
    docs = [" ".join(ws) for ws in doc_words_list]
    docs = [d for d in docs if d.strip()]
    if not docs:
        return []
    try:
        vec = TfidfVectorizer(max_features=500, token_pattern=r"[가-힣]{2,}")
        mat = vec.fit_transform(docs)
        feature_names = vec.get_feature_names_out()
        # 전체 문서 합산 TF-IDF 스코어
        scores = mat.sum(axis=0).A1
        ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        return [(w, round(s, 3)) for w, s in ranked if not _is_action_stop(w)][:top_n]
    except Exception:
        return []


def _tfidf_keywords_manual(all_words, doc_words_list, top_n=30):
    """sklearn 없이 수동 TF-IDF 계산"""
    freq = Counter(all_words)
    if not freq:
        return []
    n_docs = len(doc_words_list)
    # DF: 각 단어가 등장하는 문서 수
    df_count = Counter()
    for dw in doc_words_list:
        df_count.update(set(dw))
    tfidf = {}
    total_words = len(all_words)
    for w, tf in freq.items():
        if _is_action_stop(w):
            continue
        idf = math.log((1 + n_docs) / (1 + df_count.get(w, 0))) + 1
        tfidf[w] = (tf / total_words) * idf
    ranked = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
    return [(w, round(s, 5)) for w, s in ranked][:top_n]


def _categorize_keyword(word):
    """키워드를 고객여정/실무 카테고리로 매핑"""
    for cat, kw_list in _JOURNEY_CATEGORIES.items():
        if any(kw in word or word in kw for kw in kw_list):
            return cat
    return "🔍 기타 이슈"


@st.cache_data(show_spinner="Action-Trigger 키워드 추출 중…")
def extract_action_keywords(texts_tuple, top_n=30):
    """TF-IDF 기반 Action-Trigger 키워드 추출 (감성어 제거, 실무 키워드만)
    Returns: list of (keyword, score, category)
    """
    texts = list(texts_tuple)
    valid = [str(t) for t in texts if t and str(t).strip() not in _VOC_EMPTY and len(str(t).strip()) > 2]
    if not valid:
        return []

    # 문서별 명사 추출
    doc_words_list = []
    for t in valid:
        doc_words_list.append(_extract_nouns_action([t]))
    all_words = [w for dw in doc_words_list for w in dw]
    if not all_words:
        return []

    # TF-IDF 키워드 추출
    if SKLEARN_AVAILABLE:
        ranked = _tfidf_keywords_sklearn(doc_words_list, top_n=top_n)
    else:
        ranked = _tfidf_keywords_manual(all_words, doc_words_list, top_n=top_n)

    if not ranked:
        # fallback: 빈도 기반
        freq = Counter(w for w in all_words if not _is_action_stop(w))
        ranked = freq.most_common(top_n)

    # 카테고리 매핑
    result = []
    for w, s in ranked:
        cat = _categorize_keyword(w)
        result.append((w, s, cat))
    return result


def extract_action_keywords_by_group(df, group_col, voc_col, top_n=15):
    """그룹(지사/업무)별 TF-IDF 키워드 추출 → 그룹 간 비교용"""
    groups = df[group_col].dropna().unique()
    result = {}
    for g in groups:
        texts = df.loc[df[group_col] == g, voc_col].dropna().astype(str).tolist()
        texts = [t for t in texts if t.strip() not in ("", "nan", "응답없음")]
        if texts:
            kws = extract_action_keywords(tuple(texts), top_n=top_n)
            result[str(g)] = kws
    return result


# ── 역접 패턴 감지: "~는데/~지만/~었는데" 뒤에 긍정이 오면 전체가 긍정 ──
_ADVERSATIVE_PATTERNS = re.compile(
    r"(는데|었는데|했는데|인데|지만|하지만|그런데|그래도|근데|걱정했는데|당황하였는데|"
    r"몰라|걱정|당황|힘들었|어려웠|불안했|막막했)"
)

def _has_positive_conclusion(text):
    """텍스트의 결론(뒷부분)이 긍정인지 확인
    한국어 VOC는 '~는데 잘 해주셔서 감사합니다' 패턴이 매우 빈번.
    역접 접속사 뒤에 긍정이 있거나, 문장 끝이 긍정이면 True.
    """
    s = str(text)
    # 1) 역접 패턴 찾기 → 뒷부분에 긍정 있는지
    match = _ADVERSATIVE_PATTERNS.search(s)
    if match:
        after = s[match.end():]
        if any(pw in after for pw in POSITIVE_CONTEXT):
            return True

    # 2) 문장 끝 40자에 긍정 키워드가 있는지
    tail = s[-40:] if len(s) > 40 else s
    tail_pos = sum(1 for pw in POSITIVE_CONTEXT if pw in tail)
    tail_neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in tail)
    if tail_pos > 0 and tail_pos > tail_neg:
        return True

    return False


def _has_positive_context(text, keyword):
    """키워드 주변(앞뒤 20자)에 긍정 문맥 단어가 있는지 확인"""
    s = str(text)
    idx = s.find(keyword)
    if idx == -1:
        return False
    start = max(0, idx - 20)
    end = min(len(s), idx + len(keyword) + 20)
    window = s[start:end]
    return any(pw in window for pw in POSITIVE_CONTEXT)


def check_negative(text):
    if not text or str(text).strip() in ("", "nan", "응답없음"):
        return False, []
    s = str(text)
    # 문장 전체가 긍정 결론이면 부정 키워드 무시
    if _has_positive_conclusion(s):
        return False, []
    found = [kw for kw in NEGATIVE_KEYWORDS if kw in s]
    real_neg = [kw for kw in found if not _has_positive_context(s, kw)]
    return bool(real_neg), real_neg


def check_rude(text):
    if not text or str(text).strip() in ("", "nan", "응답없음"):
        return False, []
    s = str(text)
    # 문장 전체가 긍정 결론이면 불친절 키워드 무시
    if _has_positive_conclusion(s):
        return False, []
    found = [kw for kw in RUDE_KEYWORDS if kw in s]
    real_rude = [kw for kw in found if not _has_positive_context(s, kw)]
    return bool(real_rude), real_rude


def classify_voc_3tier(text):
    """VOC를 [불친절 / 불만 / 긍정]으로 3단 분류 (문맥 인식 버전)"""
    if not text or str(text).strip() in ("", "nan", "응답없음"):
        return "긍정"
    s = str(text)
    # 역접 후 긍정 결론 → 바로 긍정 판정
    if _has_positive_conclusion(s):
        return "긍정"
    is_rude, _ = check_rude(s)
    if is_rude:
        return "불친절"
    is_neg, _ = check_negative(s)
    if is_neg:
        return "불만"
    return "긍정"


# ── 통제 불가(Out of Scope) 필터링 ──
_OOS_KEYWORDS = [
    "전화연결", "통화연결", "전화 연결", "통화 연결", "대기시간", "대기 시간",
    "연결이 안", "연결 안", "전화가 안", "전화 안", "연결되지", "연결 되지",
    "전화를 안 받", "전화안받", "전화를 안받", "전화 잘 안",
    "콜센터", "고객센터 전화", "고객센터 연결", "상담원 연결", "상담원연결",
    "자동응답", "ARS", "0번", "대기",
]
_OFFICE_DIRECT_KW = ["지사", "직원", "기사", "방문", "현장", "담당자"]


def _is_out_of_scope(text):
    """콜센터/전화 연결 관련 불만 → 지사 통제 불가 영역"""
    if not text or str(text).strip() in ("", "nan", "응답없음"):
        return False
    s = str(text)
    if not any(kw in s for kw in _OOS_KEYWORDS):
        return False
    if any(kw in s for kw in _OFFICE_DIRECT_KW):
        return False
    return True


def _classify_sentiment_3tier(text, score=None):
    """VOC 3단계 감성 분류: 긍정/중립/부정 (역접 패턴 + 결론 우선)"""
    if not text or str(text).strip() in ("", "nan", "응답없음", "없음"):
        # 의견 없음 + 100점 → 긍정
        if score is not None and score >= 100:
            return "긍정"
        return "중립"
    s = str(text)

    # 역접 후 긍정 결론 → 바로 긍정
    if _has_positive_conclusion(s):
        return "긍정"

    _POS_KW = ["감사", "친절", "좋", "만족", "잘", "훌륭", "경청", "공감", "고마",
               "칭찬", "최고", "따뜻", "성실", "편안", "빠르", "신속", "정확", "꼼꼼",
               "배려", "도움", "편리", "추천", "완벽", "수고", "노력", "발전", "유지",
               "응원", "믿음", "행복", "깔끔", "굿", "좋아요"]
    pos_cnt = sum(1 for kw in _POS_KW if kw in s)
    adjusted_neg = 0
    for kw in NEGATIVE_KEYWORDS + RUDE_KEYWORDS:
        if kw in s and not _has_positive_context(s, kw):
            adjusted_neg += 1
    if adjusted_neg > 0 and adjusted_neg >= pos_cnt:
        return "부정"
    elif pos_cnt > 0:
        return "긍정"
    return "중립"


# 2차 원인 태깅 사전
_CAUSE_TAGS = {
    "직원 태도·불친절": ["불친절", "무시", "무례", "반말", "막말", "냉담", "퉁명", "짜증",
                       "태도", "소홀", "무성의", "건방", "고압적"],
    "처리 지연·느림":  ["느림", "느려", "지연", "오래", "기다림", "늦게", "답답"],
    "처리 오류·부정확": ["잘못", "실수", "착오", "오류", "오작동", "불량", "고장", "문제",
                       "안됨", "안되", "못하"],
    "요금·제도 불만":  ["비싸", "과다", "과금", "폭탄", "누진", "억울", "부당", "요금",
                       "불합리", "불공정"],
    "절차 복잡·불편":  ["복잡", "어렵", "힘들", "번거로", "피곤", "불편", "서류", "절차"],
    "정전·안전 관련":  ["정전", "단전", "위험", "사고", "누전"],
    "정보 안내 부족":  ["안내", "설명", "모르", "몰라", "이해", "정보"],
    "기대 미충족":     ["실망", "최악", "별로", "황당", "불만", "불쾌", "부족"],
}

_CAUSE_UNMET_NEEDS = {
    "직원 태도·불친절": "🤝 고객 감정을 먼저 수용하는 **공감형 응대** (경청→공감→해결 3단계)",
    "처리 지연·느림":  "⏱️ 접수 후 24시간 내 **진행 상황 문자/콜백** 발송 체계 구축",
    "처리 오류·부정확": "✅ 재처리 없는 **1회 완결(FCR)** — 체크리스트 기반 처리 확인",
    "요금·제도 불만":  "💰 고객 눈높이의 **비유·시각자료**로 요금 구조 투명 설명",
    "절차 복잡·불편":  "📋 한 번에 끝나는 **원스톱 처리** 및 절차 간소화",
    "정전·안전 관련":  "⚡ 예정 정전 **72시간 전 사전 알림** 및 복구 완료 통지",
    "정보 안내 부족":  "📖 표준 안내 스크립트 마련 — **FAQ 기반 즉답 체계** 구축",
    "기대 미충족":     "🎯 처리 완료 후 **만족 확인 콜** (기대치 재확인)",
}


@st.cache_data(show_spinner="감성 분석 중…")
def batch_classify_sentiment(texts_tuple):
    """키워드 기반 감성 분류 (경량화 버전 — HF 모델 불필요)"""
    texts = list(texts_tuple)
    results = ["neutral"] * len(texts)
    for i, t in enumerate(texts):
        if not t or str(t).strip() in ("", "nan"):
            continue
        has_rude, _ = check_rude(t)
        has_neg, _ = check_negative(t)
        if has_rude or has_neg:
            results[i] = "negative"
        else:
            results[i] = "positive"
    return results



def df_to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="잠재민원고객")
    return buf.getvalue()


def normalize_score_100(series):
    """점수를 100점 만점으로 자동 변환"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.to_numeric(series, errors="coerce")
    mx = s.max()
    if mx <= 5:
        return pd.to_numeric(series, errors="coerce") * 20
    elif mx <= 10:
        return pd.to_numeric(series, errors="coerce") * 10
    else:
        return pd.to_numeric(series, errors="coerce")


def score_bucket(val):
    """100점 기준 4구간 분류 (리포트 양식)"""
    if pd.isna(val):
        return "미응답"
    if val >= 90:
        return "90점 이상"
    elif val >= 70:
        return "70~90점"
    elif val >= 50:
        return "50~70점"
    else:
        return "50점 미만"


def _channel_color(name):
    """채널명으로 테마 색상 자동 매핑"""
    name_s = str(name)
    if any(k in name_s for k in ["직원", "방문", "현장"]):
        return C["ch_employee"]
    elif any(k in name_s for k in ["고객센터", "콜센터", "전화", "센터"]):
        return C["ch_center"]
    elif any(k in name_s for k in ["ON", "온라인", "앱", "홈페이지", "인터넷", "모바일"]):
        return C["ch_online"]
    return C["ch_etc"]


def _group_channel(name):
    """접수자구분을 4개 범주로 통합 (직원/고객센터/한전ON/기타, 검침사 제외)"""
    s = str(name).strip()
    if s in ("", "nan", "응답없음"):
        return None
    if "검침" in s:
        return None  # 검침사 제외
    if any(k in s for k in ["직원", "방문", "현장"]):
        return "직원"
    if any(k in s for k in ["고객센터", "콜센터", "전화", "센터"]):
        return "고객센터"
    if any(k in s for k in ["ON", "온라인", "앱", "홈페이지", "인터넷", "모바일", "한전ON", "회사ON"]):
        return "한전ON"
    return "기타"


def _group_contract(name):
    """계약종별을 6개 대분류로 통합"""
    s = str(name).strip()
    if s in ("", "nan", "응답없음"):
        return None
    if "주택" in s:
        return "주택용"
    if "일반" in s:
        return "일반용"
    if "산업" in s:
        return "산업용"
    if "농사" in s:
        return "농사용"
    if "교육" in s:
        return "교육용"
    if "가로등" in s or "가로" in s:
        return "가로등"
    return s  # 원래 값 유지


INSIGHT_RULES = [
    (["요금","비용","청구","과금","납부","고지"],
     "💳 요금·청구 → 고객 눈높이 비유·시각자료로 요금 구조 설명, 청구서 발송 3일 전 예고 문자 발송"),
    (["설치","공사","계량","계기","시공"],
     "🔧 설치·공사 → 공사 48시간 전 사전 알림(문자+전화), 시공 완료 후 재확인 콜 실시"),
    (["정전","단전","고장","누전","장애"],
     "⚡ 정전·장애 → 예정 정전 72시간 전 사전 알림 + 복구 완료 즉시 통지, 인접 고객 선제 아웃바운드 콜"),
    (["직원","친절","응대","상담","담당자"],
     "👤 직원 응대 → DISC 고객 유형별 맞춤 응대 스크립트 카드 배포, 칭찬 VOC 우수사례 주간 공유"),
    (["앱","홈페이지","사이트","온라인","인터넷","모바일"],
     "📱 디지털 채널 → 고령·취약계층 전담 안내 채널 지정, 주요 메뉴 3클릭 내 도달 UX 개선"),
    (["대기","기다림","오래","느림","지연"],
     "⏱ 대기·지연 → VOC 72시간 피드백 루프 도입(접수→24시간 내 진행 상황 문자→완료 확인 콜)"),
    (["안전","위험","누전","화재"],
     "🦺 안전 → 서비스 회복 골든타임 프로토콜: 10분 내 원인 인정+사과→복구 후 재확인 콜→3일 내 재발 방지 안내"),
    (["불친절","무시","무성의","태도","반말"],
     "😡 불친절 → 3분 감정 리셋 루틴 제도화(강성 민원 후 직원 휴식 보장), 첫 30초 표준 오프닝 훈련 실시"),
    (["복잡","어렵","힘들","번거로","절차","서류"],
     "📋 절차 불편 → 원스톱 처리 체계 구축, 고객 여정 맵 워크숍으로 불편 접점 발굴 후 즉시 개선"),
    (["실수","잘못","오류","착오","부정확"],
     "✅ 처리 오류 → 1회 완결(FCR) 체크리스트 도입, 접수번호별 처리 완결도 주간 5분 스탠딩 미팅 운영"),
]


def generate_ai_recommendations(kw_list, category_name, max_recs=5):
    if not kw_list:
        return []
    kw_text = " ".join([k for k, _ in kw_list[:15]])
    recs = []
    seen = set()
    for rule_kws, rule_msg in INSIGHT_RULES:
        if any(rk in kw_text for rk in rule_kws):
            if rule_msg not in seen:
                recs.append(rule_msg)
                seen.add(rule_msg)
    if not recs:
        top_kws = ", ".join([k for k, _ in kw_list[:5]])
        recs.append(f"📋 [{category_name}] 주요 키워드: {top_kws} → 해당 분야 개선 필요")
    return recs[:max_recs]


def generate_auto_insight(df, score_col, indiv_scores, cat_cols):
    """데이터 기반 한 줄 인사이트 자동 생성 (특정 지사 지목 없이 전반적 요약)"""
    parts = []
    # 1. 전체 만족도 수준 판정
    if score_col and score_col in df.columns:
        avg = df[score_col].mean()
        n = len(df)
        if avg >= 95:
            parts.append(f"전체 {n:,}건 평균 **{avg:.1f}점**으로 매우 우수한 수준입니다.")
        elif avg >= 90:
            parts.append(f"전체 {n:,}건 평균 **{avg:.1f}점**으로 양호하나, 세부 항목별 편차 점검이 필요합니다.")
        elif avg >= 80:
            parts.append(f"전체 {n:,}건 평균 **{avg:.1f}점**으로 개선 여지가 있습니다.")
        else:
            parts.append(f"전체 {n:,}건 평균 **{avg:.1f}점**으로 집중 관리가 필요한 수준입니다.")
    # 2. 종합점수에 가장 영향력 큰 항목
    if score_col and score_col in df.columns and indiv_scores:
        num = df[indiv_scores + [score_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(num) > 3:
            corr = num.corr()[score_col].drop(score_col).sort_values(ascending=False)
            parts.append(f"종합 점수에 가장 큰 영향을 미치는 항목은 **'{corr.index[0]}'**(상관계수 {corr.iloc[0]:.3f})입니다.")
    if not parts:
        return "데이터가 업로드되었습니다. 각 탭에서 상세 분석을 확인하세요."
    return " ".join(parts)


def find_vulnerable_group(df, cat_col, score_col, percentile=30):
    """범주형 컬럼에서 종합 점수 하위 N% 그룹 탐색"""
    grp = df.groupby(cat_col)[score_col].agg(["mean","count"]).reset_index()
    grp.columns = [cat_col, "평균", "건수"]
    grp = grp.sort_values("평균")
    threshold = np.percentile(df[score_col].dropna(), percentile)
    weak = grp[grp["평균"] <= threshold]
    return weak, threshold, grp


# ══════════════════════════════════════════════════════════════
#  4. 페이지 설정 & CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CS 분석 대시보드",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  .stApp {{ background-color: {C['bg']}; }}

  .dash-header {{
    background: linear-gradient(120deg, {C['navy']} 0%, {C['blue']} 55%, {C['sky']} 100%);
    padding: 2rem 2.5rem; border-radius: 16px; color: white;
    margin-bottom: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.18);
  }}
  .dash-header h1 {{ font-size:1.9rem; font-weight:800; margin:0; letter-spacing:-0.5px; }}
  .dash-header p  {{ font-size:0.95rem; margin:0.4rem 0 0 0; opacity:0.85; }}
  .dash-badge {{
    display:inline-block; background:rgba(255,255,255,0.18);
    border:1px solid rgba(255,255,255,0.35); border-radius:20px;
    padding:0.2rem 0.8rem; font-size:0.78rem; margin-top:0.7rem; margin-right:0.4rem;
  }}

  [data-testid="stMetric"] {{
    background: {C['white']}; border-radius: 14px;
    padding: 1.1rem 1.2rem 0.9rem 1.2rem;
    box-shadow: 0 2px 12px rgba(0,85,165,0.10);
    border-top: 4px solid {C['blue']}; transition: transform 0.15s;
  }}
  [data-testid="stMetric"]:hover {{ transform: translateY(-2px); }}
  [data-testid="stMetricLabel"]  {{ font-size:0.82rem !important; color:{C['gray']} !important; font-weight:600; }}
  [data-testid="stMetricValue"]  {{ font-size:1.75rem !important; font-weight:800 !important; color:{C['navy']} !important; }}
  [data-testid="stMetricDelta"]  {{ font-size:0.82rem !important; }}

  .stTabs [data-baseweb="tab-list"] {{ gap:6px; background:{C['bg']}; border-bottom:2px solid #dde4ef; padding-bottom:0; }}
  .stTabs [data-baseweb="tab"] {{ font-size:0.95rem; font-weight:700; padding:0.6rem 1.4rem; border-radius:8px 8px 0 0; color:{C['gray']}; background:transparent; }}
  .stTabs [aria-selected="true"] {{ color:{C['navy']} !important; background:{C['white']} !important; border-bottom:3px solid {C['blue']} !important; box-shadow:0 -2px 8px rgba(0,85,165,0.08); }}

  .sec-head {{ font-size:1.1rem; font-weight:800; color:{C['navy']}; border-left:5px solid {C['blue']}; padding:0.2rem 0 0.2rem 0.8rem; margin:1.4rem 0 1rem 0; }}

  .card     {{ background:{C['white']}; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 12px rgba(0,85,165,0.09); margin-bottom:1rem; color:#333; }}
  .card-red {{ background:#fff5f5; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 12px rgba(198,40,40,0.10); border-left:5px solid {C['red']}; margin-bottom:1rem; color:#333; }}
  .card-gold {{ background:#fffbf0; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 12px rgba(240,165,0,0.12); border-left:5px solid {C['gold']}; margin-bottom:1rem; color:#333; }}
  .card-teal {{ background:#e8f5e9; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 12px rgba(0,137,123,0.10); border-left:5px solid {C['teal']}; margin-bottom:1rem; color:#333; }}
  .card-blue {{ background:{C['light']}; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 12px rgba(0,85,165,0.09); border-left:5px solid {C['sky']}; margin-bottom:1rem; color:#333; }}
  .card-rude {{ background:#fce4ec; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 12px rgba(198,40,40,0.15); border-left:5px solid #e91e63; margin-bottom:1rem; color:#333; }}

  /* ── 사이드바 ── */
  [data-testid="stSidebar"] {{ background: linear-gradient(180deg, {C['navy']} 0%, #1e4d8c 100%) !important; }}
  [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] .stMarkdown *, [data-testid="stSidebar"] .stCaption,
  [data-testid="stSidebar"] small {{ color: white !important; }}
  [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stMultiSelect label,
  [data-testid="stSidebar"] .stSlider label {{ font-size:0.82rem !important; color:white !important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] > div {{ background:rgba(255,255,255,0.15) !important; border:1px solid rgba(255,255,255,0.35) !important; border-radius:8px !important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] > div * {{ color:white !important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill:white !important; }}
  [data-testid="stSidebar"] [data-baseweb="tag"] {{ background:rgba(255,255,255,0.22) !important; border:1px solid rgba(255,255,255,0.4) !important; }}
  [data-testid="stSidebar"] [data-baseweb="tag"] * {{ color:white !important; }}
  [data-baseweb="popover"] li, [data-baseweb="popover"] ul, [data-baseweb="popover"] div {{ color:{C['navy']} !important; }}
  [data-testid="stSidebar"] .stExpander {{ background:rgba(255,255,255,0.08) !important; border:1px solid rgba(255,255,255,0.25) !important; border-radius:10px !important; }}
  [data-testid="stSidebar"] .stExpander summary span {{ color:white !important; }}
  [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {{ color:white !important; }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] {{ background:rgba(255,255,255,0.15) !important; border:2px dashed rgba(255,255,255,0.6) !important; border-radius:12px !important; padding:1rem !important; }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] * {{ color:white !important; }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] section {{ background:rgba(255,255,255,0.12) !important; border-radius:8px !important; }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] section > div {{ color:white !important; }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] small {{ color:rgba(255,255,255,0.8) !important; font-size:0.78rem !important; }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] button {{ background:{C['gold']} !important; color:{C['navy']} !important; font-weight:700 !important; border:none !important; border-radius:8px !important; padding:0.4rem 1.2rem !important; }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {{ opacity:0.85 !important; }}
  [data-testid="stSidebar"] hr {{ border-color:rgba(255,255,255,0.2) !important; }}

  .insight-box {{
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
    border-left: 6px solid {C['teal']}; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,137,123,0.12);
    font-size: 1.02rem; line-height: 1.7; color: #333;
  }}
  .insight-box b {{ color: {C['navy']}; }}

  .stDownloadButton > button {{ background:linear-gradient(90deg,{C['navy']},{C['blue']}) !important; color:white !important; font-weight:700 !important; border-radius:10px !important; border:none !important; padding:0.65rem 1.5rem !important; box-shadow:0 3px 10px rgba(0,85,165,0.3) !important; }}
  .stDownloadButton > button:hover {{ opacity:0.88 !important; }}
  [data-testid="stDataFrame"] th {{ background:{C['navy']} !important; color:white !important; }}
  .stExpander {{ border-radius:10px !important; border:1px solid #dde4ef !important; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  5. 사이드바
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚡ CS 분석 대시보드")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 📂 데이터 업로드")
    uploaded_file = st.file_uploader("파일을 여기에 드래그하세요", type=["xlsx","xls","csv"], label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("© CS 분석 시스템")

# ══════════════════════════════════════════════════════════════
#  6. 헤더 배너
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="dash-header">
  <h1>⚡ AI 활용 고객경험관리시스템 조사결과 분석</h1>
  <p>경남본부 CS 만족도 조사 데이터 기반 · AI 자동 분석 리포트</p>
  <span class="dash-badge">📊 종합 현황</span>
  <span class="dash-badge">📡 항목별 · 계약종별 · 업무유형별 분석</span>
  <span class="dash-badge">🏢 지사 맞춤형 CS 솔루션</span>
  <span class="dash-badge">🎯 민원 조기 경보 시스템</span>
  <span class="dash-badge">💌 경험고객 서한문 생성</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  7. 업로드 전 안내
# ══════════════════════════════════════════════════════════════
if uploaded_file is None:
    c_l, c_r = st.columns([1, 1])
    with c_l:
        st.markdown('<div class="card-blue">', unsafe_allow_html=True)
        st.markdown("### 📋 사용 방법")
        st.markdown("""
1. **왼쪽 사이드바**에서 엑셀(.xlsx) 또는 CSV 파일을 업로드하세요.
2. 업로드 즉시 컬럼이 자동 인식되어 분석이 시작됩니다.
3. 사이드바 필터(지사·채널 등)로 데이터 범위를 조정하세요.
4. **📊 종합 현황** — 구간별 비중·사업소별 점수 등 전체 현황
5. **📡 항목별 · 계약종별 · 업무유형별 분석** — 교차 분석 및 사분면 비교
6. **🏢 지사 맞춤형 CS 솔루션** — 지사 선택 후 단계별 정밀 진단
7. **🎯 민원 조기 경보 시스템** — 잠재 민원고객 사전케어 리스트
8. **💌 경험고객 서한문 생성** — 지사별 맞춤 서한문 생성 및 기념품 추천
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with c_r:
        st.markdown('<div class="card-teal">', unsafe_allow_html=True)
        st.markdown("### 📌 권장 엑셀 컬럼 구성")
        st.markdown("""
| 컬럼명 예시 | 내용 |
|---|---|
| 사업소 | 직할, 진주, 마산 등 |
| 계약종별 | 주택용/일반용/산업용 등 |
| 업무구분 | 요금·신증설·정전 등 |
| 만족도점수 | 숫자 |
| 주관식답변 | VOC 텍스트 |
        """)
        st.warning("⚠️ **개인정보 주의** — 고객명·고객번호·연락처 등 개인식별 정보는 업로드 전에 반드시 제거하세요.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════
#  8. 데이터 로드
# ══════════════════════════════════════════════════════════════
_KNOWN_HEADERS = {"지사","사업소","계약종별","접수종류","업무구분","신청방법",
                   "접수자구분","종합 점수","종합점수","서술 의견","서술의견",
                   "이용 편리성","직원 친절도","전반적 만족","처리 신속도",
                   "접수일자","조사일자","접수일","조사일","일자","날짜","등록일",
                   "접수번호"}

def _detect_header_row(raw_bytes, max_scan=10):
    """엑셀 상위 N행을 스캔하여 헤더 행 자동 감지"""
    try:
        preview = pd.read_excel(io.BytesIO(raw_bytes), header=None,
                                nrows=max_scan, engine="openpyxl")
    except Exception:
        preview = pd.read_excel(io.BytesIO(raw_bytes), header=None, nrows=max_scan)
    for i in range(min(max_scan, len(preview))):
        row_vals = set(str(v).strip() for v in preview.iloc[i] if pd.notna(v))
        if row_vals & _KNOWN_HEADERS:
            return i
    return 0  # 못 찾으면 첫 행을 헤더로

def _check_pii(df):
    """개인정보(PII) 포함 여부를 검사하여 탐지 항목 리스트 반환"""
    import re
    warnings = []
    col_names = [str(c).strip() for c in df.columns]

    # 1단계: 컬럼명 키워드 검사
    _PII_COL_KW = ["고객번호", "전화", "휴대폰", "핸드폰", "연락처",
                    "주소", "성명", "명의자", "신청자명", "고객명",
                    ]
    for col in col_names:
        for kw in _PII_COL_KW:
            if kw in col:
                warnings.append(f"컬럼 '{col}' — '{kw}' 개인정보 컬럼 탐지")
                break

    # 2단계: 데이터 패턴 검사 (상위 100행 샘플)
    sample = df.head(100)
    _re_custno = re.compile(r'^09\d{8}$')
    _re_phone = re.compile(r'01[016789]-?\d{3,4}-?\d{4}')
    for col in sample.columns:
        vals = sample[col].dropna().astype(str).str.strip()
        if vals.empty:
            continue
        # 고객번호 패턴
        if vals.apply(lambda v: bool(_re_custno.match(v))).sum() >= 3:
            msg = f"컬럼 '{col}' — 고객번호 패턴(09XXXXXXXX) 탐지"
            if msg not in warnings:
                warnings.append(msg)
        # 전화번호 패턴
        if vals.apply(lambda v: bool(_re_phone.search(v))).sum() >= 3:
            msg = f"컬럼 '{col}' — 전화번호 패턴(010-XXXX-XXXX) 탐지"
            if msg not in warnings:
                warnings.append(msg)
    return warnings

@st.cache_data(show_spinner=False)
def load_data(raw_bytes, file_name=""):
    if file_name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding="cp949")
    else:
        header_row = _detect_header_row(raw_bytes)
        try:
            df = pd.read_excel(io.BytesIO(raw_bytes), header=header_row, engine="openpyxl")
        except Exception:
            df = pd.read_excel(io.BytesIO(raw_bytes), header=header_row)
    orig = len(df)
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    df.columns = [str(c).strip() for c in df.columns]
    return df, orig

if uploaded_file is None:
    st.info("📂 좌측 사이드바에서 엑셀(.xlsx) 또는 CSV 파일을 업로드해 주세요.")
    st.markdown("---")
    st.markdown("#### 사용 방법")
    st.markdown("1. 좌측 **파일 업로드** 영역에 엑셀(.xlsx) 또는 CSV 파일을 드래그하거나 선택하세요.\n2. 업로드가 완료되면 자동으로 분석이 시작됩니다.")
    import sys
    sys.exit(0)

with st.spinner("데이터를 불러오는 중…"):
    df_raw, orig_len = load_data(uploaded_file.read(), uploaded_file.name)

# ── 개인정보(PII) 차단 ──
_pii_warnings = _check_pii(df_raw)
if _pii_warnings:
    st.error("🚫 **개인정보가 탐지되어 업로드할 수 없습니다.**")
    for _pw in _pii_warnings:
        st.warning(_pw)
    st.info("고객번호·전화번호·성명·주소 등 개인정보 컬럼을 제거한 후 다시 업로드해주세요.")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  9. 컬럼 자동 매핑 (엑셀 컬럼명 기반)
# ══════════════════════════════════════════════════════════════
# 순번 컬럼: 원본 엑셀 행 번호 보존 후 내부 분석용 제거
if "순번" in df_raw.columns:
    df_raw["_원본순번"] = df_raw["순번"]
    df_raw.drop(columns=["순번"], inplace=True)
else:
    df_raw["_원본순번"] = range(1, len(df_raw) + 1)

def _find_col(keywords):
    """엑셀 컬럼명에 키워드가 포함되면 해당 컬럼명 반환"""
    for col in df_raw.columns:
        c = str(col).strip()
        for kw in keywords:
            if kw in c:
                return col
    return None

M = {
    "office":    _find_col(["지사", "사업소"]),
    "channel":   _find_col(["접수자구분"]),          # 직원/고객센터/회사ON
    "method":    _find_col(["신청방법"]),             # 전화/내방/사이버지점
    "reception": _find_col(["접수종류"]),             # 청구서재발행/자동이체 등
    "contract":  _find_col(["계약종별"]),
    "business":  _find_col(["업무구분", "업무유형"]),
    "score":     _find_col(["종합 점수", "종합점수"]),
    "voc":       _find_col(["서술 의견", "서술의견"]),
    "age":       _find_col(["연령"]),
    "date":      _find_col(["업무처리완료일", "처리완료일", "완료일", "접수일자", "조사일자", "접수일", "조사일", "일자", "날짜", "등록일"]),
    "receipt_no": _find_col(["접수번호"]),
    "response":  _find_col(["응답여부", "응답 여부"]),
    "id":        None,
    "name":      None,
    "contact":   None,
}

# ── 날짜 파싱: 접수번호에서 추출 또는 날짜 컬럼 직접 사용 ──
def _parse_date_from_receipt(val):
    """접수번호(예: 5726-20260131-010017)에서 가운데 8자리 날짜 추출"""
    s = str(val).strip()
    # 하이픈/슬래시 구분자가 있으면 가운데 파트 추출
    parts = re.split(r'[-/]', s)
    for p in parts:
        p = p.strip()
        if len(p) == 8 and p.isdigit():
            try:
                return pd.to_datetime(p, format="%Y%m%d")
            except Exception:
                continue
    return pd.NaT

# 우선순위: 날짜 컬럼(업무처리완료일 등) → 접수번호
_date_parsed = False
if M["date"]:
    _dc = M["date"]
    # Excel 시리얼 넘버 감지 (숫자 5자리 = Excel 날짜)
    _sample = df_raw[_dc].dropna().head(5)
    _is_serial = False
    if _sample.dtype in ("int64", "float64") or all(str(v).replace(".", "").isdigit() for v in _sample):
        try:
            _test = pd.to_numeric(_sample, errors="coerce").dropna()
            if not _test.empty and 40000 < _test.iloc[0] < 55000:
                _is_serial = True
        except Exception:
            pass
    if _is_serial:
        df_raw[_dc] = pd.to_timedelta(pd.to_numeric(df_raw[_dc], errors="coerce") - 2, unit="D") + pd.Timestamp("1900-01-01")
    else:
        df_raw[_dc] = pd.to_datetime(df_raw[_dc], errors="coerce")
    # 연도 검증 (2020~2030 범위 밖이면 파싱 실패로 판단)
    _valid = df_raw[_dc].dropna()
    if not _valid.empty and 2020 <= _valid.iloc[0].year <= 2030:
        _date_parsed = True
    else:
        df_raw[_dc] = pd.NaT

if not _date_parsed and M["receipt_no"]:
    df_raw["_접수일"] = df_raw[M["receipt_no"]].apply(_parse_date_from_receipt)
    if df_raw["_접수일"].dropna().any():
        M["date"] = "_접수일"
        _date_parsed = True

if not _date_parsed:
    M["date"] = None

if M["date"]:
    _valid_dates = df_raw[M["date"]].dropna()
    if not _valid_dates.empty:
        df_raw["_년월"] = df_raw[M["date"]].dt.to_period("M").astype(str)
        df_raw["_분기"] = df_raw[M["date"]].dt.to_period("Q").astype(str)
        df_raw["_월"] = df_raw[M["date"]].dt.month
        _season_map = {12:"겨울",1:"겨울",2:"겨울",3:"봄",4:"봄",5:"봄",
                       6:"여름",7:"여름",8:"여름",9:"가을",10:"가을",11:"가을"}
        df_raw["_계절"] = df_raw["_월"].map(_season_map)
    else:
        M["date"] = None

# ── 계약종별 대분류 그룹핑 ──
if M["contract"]:
    df_raw["_계약종별"] = df_raw[M["contract"]].apply(_group_contract)
    if df_raw["_계약종별"].dropna().any():
        M["contract"] = "_계약종별"
    else:
        del df_raw["_계약종별"]

# ── 개별 점수 컬럼 자동 탐지 (부분 매칭) ──
_SCORE_KEYWORDS = [
    "전반적 만족", "직원 친절", "처리 신속",
    "처리 정확", "업무 개선", "업무개선",
    "이용 편리", "사용 추천",
]

def _find_score_cols():
    """엑셀 컬럼 중 점수 키워드가 포함된 컬럼을 순서대로 반환 (중복 제거)"""
    found = []
    used = set()
    for kw in _SCORE_KEYWORDS:
        for col in df_raw.columns:
            c = str(col).strip()
            if kw.replace(" ", "") in c.replace(" ", "") and col not in used:
                # 숫자 데이터가 포함된 컬럼만 (점수 컬럼 확인)
                if pd.to_numeric(df_raw[col], errors="coerce").dropna().shape[0] > 0:
                    found.append(col)
                    used.add(col)
                    break
    return found

individual_scores = _find_score_cols()

# 개별 점수도 100점 환산
for _sc in individual_scores:
    df_raw[_sc] = normalize_score_100(df_raw[_sc])

# 범주형 컬럼 리스트
available_cats = [v for k, v in M.items()
                  if v is not None and k in ("office","channel","method","reception","contract","business")]


# ══════════════════════════════════════════════════════════════
#  10. 사이드바 필터
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 🔍 데이터 필터")
    df_f = df_raw.copy()

    if M["office"]:
        office_opts = _sort_offices(df_raw[M["office"]].dropna().astype(str).unique().tolist())
        sel_office = st.multiselect("🏬 지사", office_opts, default=office_opts, key="f_office")
        if sel_office and len(sel_office) < len(office_opts):
            df_f = df_f[df_f[M["office"]].astype(str).isin(sel_office)]
    if M["channel"]:
        chan_opts = sorted(df_raw[M["channel"]].dropna().astype(str).unique().tolist())
        sel_chan = st.multiselect("📡 접수자구분", chan_opts, default=chan_opts, key="f_chan")
        if sel_chan and len(sel_chan) < len(chan_opts):
            df_f = df_f[df_f[M["channel"]].astype(str).isin(sel_chan)]
    if M["method"]:
        method_opts = sorted(df_raw[M["method"]].dropna().astype(str).unique().tolist())
        sel_method = st.multiselect("📞 신청방법", method_opts, default=method_opts, key="f_method")
        if sel_method and len(sel_method) < len(method_opts):
            df_f = df_f[df_f[M["method"]].astype(str).isin(sel_method)]
    if M["reception"]:
        recep_opts = sorted(df_raw[M["reception"]].dropna().astype(str).unique().tolist())
        sel_recep = st.multiselect("📝 접수종류", recep_opts, default=recep_opts, key="f_recep")
        if sel_recep and len(sel_recep) < len(recep_opts):
            df_f = df_f[df_f[M["reception"]].astype(str).isin(sel_recep)]
    if M["contract"]:
        cont_opts = sorted(df_raw[M["contract"]].dropna().astype(str).unique().tolist())
        sel_cont = st.multiselect("📋 계약종별", cont_opts, default=cont_opts, key="f_cont")
        if sel_cont and len(sel_cont) < len(cont_opts):
            df_f = df_f[df_f[M["contract"]].astype(str).isin(sel_cont)]
    if M["business"]:
        biz_opts = sorted(df_raw[M["business"]].dropna().astype(str).unique().tolist())
        sel_biz = st.multiselect("🏢 업무구분", biz_opts, default=biz_opts, key="f_biz")
        if sel_biz and len(sel_biz) < len(biz_opts):
            df_f = df_f[df_f[M["business"]].astype(str).isin(sel_biz)]
    if M["age"]:
        age_opts = sorted(df_raw[M["age"]].dropna().astype(str).unique().tolist())
        sel_age = st.multiselect("👥 연령대", age_opts, default=age_opts, key="f_age")
        if sel_age and len(sel_age) < len(age_opts):
            df_f = df_f[df_f[M["age"]].astype(str).isin(sel_age)]
    if M["score"]:
        sc_s = pd.to_numeric(df_raw[M["score"]], errors="coerce").dropna()
        if not sc_s.empty:
            s_min, s_max = float(sc_s.min()), float(sc_s.max())
            sc_rng = st.slider("만족도 점수 범위", min_value=s_min, max_value=s_max,
                               value=(s_min, s_max), step=0.1, key="f_sc")
            df_f[M["score"]] = pd.to_numeric(df_f[M["score"]], errors="coerce")
            df_f = df_f[df_f[M["score"]].between(sc_rng[0], sc_rng[1])]
    st.caption(f"필터 적용 결과: **{len(df_f):,}건** / 전체 {len(df_raw):,}건")

# ══════════════════════════════════════════════════════════════
#  11. 점수 정규화 & 사전 계산
# ══════════════════════════════════════════════════════════════
score_100 = None
avg_score_100 = None
if M["score"]:
    score_100 = normalize_score_100(df_f[M["score"]])
    df_f["_점수100"] = score_100
    avg_score_100 = score_100.mean()
    df_f["_점수구간"] = score_100.apply(score_bucket)

# VOC 감성 분석
voc_texts_all = []
voc_sentiments = []
pos_cnt = neg_cnt = neu_cnt = 0
_row_sentiments = ["neutral"] * len(df_f)
if M["voc"]:
    raw_voc = df_f[M["voc"]].astype(str).str.strip()
    all_texts = raw_voc.tolist()
    voc_texts_all = [t for t in all_texts if t and t != "nan" and t != "응답없음"]
    if voc_texts_all:
        voc_sentiments = batch_classify_sentiment(tuple(voc_texts_all))
        vi = 0
        for i, t in enumerate(all_texts):
            if t and t != "nan" and t != "응답없음" and vi < len(voc_sentiments):
                _row_sentiments[i] = voc_sentiments[vi]
                vi += 1
        for sent in voc_sentiments:
            if sent == "negative":   neg_cnt += 1
            elif sent == "positive": pos_cnt += 1
            else:                    neu_cnt += 1
    voc_response_rate = len(voc_texts_all) / max(len(df_f), 1) * 100
    df_f["_VOC분류"] = df_f[M["voc"]].apply(classify_voc_3tier)
    if "_점수100" in df_f.columns:
        df_f["_VOC감성"] = df_f.apply(lambda r: _classify_sentiment_3tier(r[M["voc"]], r.get("_점수100")), axis=1)
    else:
        df_f["_VOC감성"] = df_f[M["voc"]].apply(_classify_sentiment_3tier)
    df_f["_is_oos"] = df_f[M["voc"]].apply(_is_out_of_scope)           # 통제 불가 필터링
else:
    voc_response_rate = 0.0

neg_ratio = neg_cnt / max(len(voc_texts_all), 1) * 100

# ── 잠재 민원고객 수 사전 계산 (KPI + 탭5 공용) ──
_pre_neg_n = 0
if M["voc"]:
    _pre_neg_res = df_f[M["voc"]].apply(check_negative)
    _pre_neg_kw_mask = _pre_neg_res.apply(lambda x: x[0])
    _pre_low_score_mask = pd.Series(False, index=df_f.index)
    if M["score"] and "_점수100" in df_f.columns:
        _pre_low_score_mask = df_f["_점수100"] <= 50
    _pre_neg_n = int((_pre_low_score_mask & _pre_neg_kw_mask).sum())
_pre_neg_r = _pre_neg_n / max(len(df_f), 1) * 100

# ══════════════════════════════════════════════════════════════
#  12. 자동 인사이트 & KPI 메트릭
# ══════════════════════════════════════════════════════════════
# ── 데이터 기반 한 줄 인사이트 ──
_insight_text = generate_auto_insight(
    df_f, M.get("score"), individual_scores, available_cats)
st.markdown(
    f'<div class="insight-box">💡 <b>AI 인사이트:</b> {_insight_text}</div>',
    unsafe_allow_html=True)

st.markdown('<p class="sec-head">📌 핵심 요약 지표 (100점 환산)</p>', unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    if avg_score_100 is not None and not np.isnan(avg_score_100):
        st.metric("⭐ 종합만족도", f"{avg_score_100:.1f}점")
    else:
        st.metric("⭐ 종합만족도", "컬럼 미선택")
with m2:
    st.metric("📋 분석 건수", f"{len(df_f):,}건",
              delta=f"전체 {len(df_raw):,}건 중" if len(df_f) != len(df_raw) else None)
with m3:
    if M["voc"]:
        st.metric("💬 VOC 응답률", f"{voc_response_rate:.1f}%", delta=f"총 {len(voc_texts_all):,}건")
    else:
        st.metric("💬 VOC 응답률", "미선택")
with m4:
    if M["voc"]:
        st.metric("😊 긍정 비율", f"{100 - neg_ratio:.1f}%", delta=f"긍정 {pos_cnt:,}건")
    else:
        st.metric("😊 긍정 비율", "미선택")
with m5:
    if M["voc"]:
        st.metric("🎯 조기 경보 감지", f"{_pre_neg_n:,}명",
                  delta=f"전체의 {_pre_neg_r:.1f}%", delta_color="inverse")
    else:
        st.metric("🎯 조기 경보 감지", "미선택")

st.markdown(
    '<div style="background:#f8f9fa;border-radius:8px;padding:8px 14px;margin:-8px 0 8px;font-size:0.82em;color:#666;">'
    '💬 <b>VOC 응답률</b> = 서술형 의견을 작성한 고객 비율 (전체 조사 건수 대비) &nbsp;│&nbsp; '
    '🎯 <b>조기 경보 감지</b> = 민원 조기 경보 시스템 탭의 잠재 민원 대상자 수'
    '</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  주간 리포트 기준일 계산
#  - 업로드된 데이터 전체 = 금주 (목~수 고정 주기)
#  - 전주 = 금주 시작일 - 7일 ~ 금주 시작일 - 1일
# ══════════════════════════════════════════════════════════════
_wr_week_available = False
_wr_this_week = _wr_last_week = _wr_month = pd.DataFrame()
_wr_ref_date = _wr_week_start = _wr_week_end = _wr_last_start = _wr_last_end = None
if M["date"]:
    _wr_dates = df_f[M["date"]].dropna()
    if not _wr_dates.empty:
        # 업로드 데이터의 실제 날짜 범위 = 금주
        _wr_date_min = _wr_dates.min()
        _wr_date_max = _wr_dates.max()
        _wr_ref_date = _wr_date_max
        _wr_week_start = _wr_date_min
        _wr_week_end = _wr_date_max
        _wr_last_end = _wr_week_start - pd.Timedelta(days=1)
        _wr_last_start = _wr_last_end - pd.Timedelta(days=6)
        _wr_month_start = _wr_date_min.replace(day=1)
        # 금주 = 업로드 데이터 전체
        _wr_this_week = df_f.copy()
        # 전주 = 금주 시작 전 7일 (데이터에 있으면)
        _wr_last_week = df_f[(df_f[M["date"]] >= _wr_last_start) & (df_f[M["date"]] < _wr_week_start)]
        _wr_month = df_f[(df_f[M["date"]] >= _wr_month_start) & (df_f[M["date"]] <= _wr_week_end)]
        _wr_week_available = len(_wr_this_week) > 0

# ══════════════════════════════════════════════════════════════
#  13. 탭 구성
# ══════════════════════════════════════════════════════════════
tab1, tab3, tab_sol, tab_weekly, tab5, tab_letter = st.tabs([
    "📊  종합 현황",
    "📡  항목별 · 계약종별 · 업무유형별 분석",
    "🏢  지사 맞춤형 CS 솔루션",
    "📋  주간 리포트",
    "🎯  민원 조기 경보 시스템",
    "💌  경험고객 서한문 생성",
])

# ─────────────────────────────────────────────────────────────
#  TAB WEEKLY  주간 리포트
# ─────────────────────────────────────────────────────────────
def _fv(v):
    """float → '85.345' / None → '-'"""
    return f"{v:.3f}" if v is not None else "-"

def _fv_delta(cur, prev):
    """금주 점수 + 증감을 한 셀에: '94.321 (△3.512)' / 점수만 / '-'"""
    if cur is None:
        return "-"
    s = f"{cur:.3f}"
    if prev is not None:
        d = cur - prev
        c = "#1565C0" if d >= 0 else "#C62828"
        arr = "△" if d >= 0 else "▽"
        s += f' <span style="color:{c};font-size:0.88em;">({arr}{abs(d):.3f})</span>'
    return s

with tab_weekly:
    if not _wr_week_available:
        st.warning("날짜 정보가 없거나 금주 데이터가 없어 주간 리포트를 생성할 수 없습니다. 엑셀에 접수일자/접수번호 컬럼이 있는지 확인해주세요.")
    else:
        _wr_score = M["score"]
        _wr_office = M["office"]
        _wr_biz = M["business"]
        _wr_voc = M["voc"]

        _wr_ws = _wr_week_start.strftime("%m/%d")
        _wr_we = _wr_week_end.strftime("%m/%d")
        _wr_ls = _wr_last_start.strftime("%m/%d")
        _wr_le = _wr_last_end.strftime("%m/%d")
        st.markdown(
            f'<div style="background:#eef3fb;border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:0.88em;color:#333;">'
            f'📅 <b>기준일:</b> {_wr_ref_date.strftime("%Y-%m-%d")} &nbsp;│&nbsp; '
            f'<b>금주:</b> {_wr_ws}~{_wr_we} ({len(_wr_this_week):,}건) &nbsp;│&nbsp; '
            f'<b>전주:</b> {_wr_ls}~{_wr_le} ({len(_wr_last_week):,}건) &nbsp;│&nbsp; '
            f'<b>월 누계:</b> {len(_wr_month):,}건'
            f'</div>', unsafe_allow_html=True)

        # ── 지사 선택 (전체 / 개별 지사) ──
        _wr_offices_all = _sort_offices(df_f[_wr_office].dropna().unique().tolist()) if _wr_office else []
        _wr_view_opts = ["전체 (본부 종합)"] + _wr_offices_all
        _wr_sel_view = st.selectbox("📌 조회 범위", _wr_view_opts, key="wr_view_sel")

        if _wr_sel_view == "전체 (본부 종합)":
            _wr_tw_view = _wr_this_week
            _wr_lw_view = _wr_last_week
            _wr_mo_view = _wr_month
        else:
            _wr_tw_view = _wr_this_week[_wr_this_week[_wr_office] == _wr_sel_view]
            _wr_lw_view = _wr_last_week[_wr_last_week[_wr_office] == _wr_sel_view]
            _wr_mo_view = _wr_month[_wr_month[_wr_office] == _wr_sel_view]

        # ── Section 1: 주간 조사 결과 ──
        st.markdown('<p class="sec-head">1. 주간 조사 결과</p>', unsafe_allow_html=True)

        if _wr_office and _wr_score:
            _hdr = "#d6e4f0"
            _bdr = "#b0b0b0"

            if _wr_sel_view == "전체 (본부 종합)":
                # ── 전체: 지사별 행 ──
                _offices_wr = _sort_offices(df_f[_wr_office].dropna().unique().tolist())
                # 금주 점수 dict (순위 계산용)
                _ofc_tw_scores = {}
                _s1_raw = []
                for ofc in _offices_wr:
                    _tw = _wr_this_week[_wr_this_week[_wr_office] == ofc]
                    _lw = _wr_last_week[_wr_last_week[_wr_office] == ofc]
                    _mo = _wr_month[_wr_month[_wr_office] == ofc]
                    _tw_total = len(_tw)
                    _tw_resp = int(_tw["_점수100"].dropna().count()) if "_점수100" in _tw.columns else _tw_total
                    _tw_sc = _tw["_점수100"].mean() if "_점수100" in _tw.columns and not _tw["_점수100"].dropna().empty else None
                    _lw_sc = _lw["_점수100"].mean() if "_점수100" in _lw.columns and not _lw["_점수100"].dropna().empty else None
                    _mo_sc = _mo["_점수100"].mean() if "_점수100" in _mo.columns and not _mo["_점수100"].dropna().empty else None
                    _ofc_tw_scores[ofc] = _tw_sc
                    _s1_raw.append((ofc, _tw_total, _tw_resp, _tw_sc, _lw_sc, _mo_sc))

                # 순위 계산 (금주 점수 기준, 높을수록 1위)
                _valid_scores = {k: v for k, v in _ofc_tw_scores.items() if v is not None}
                _ranked = sorted(_valid_scores.items(), key=lambda x: x[1], reverse=True)
                _rank_map = {}
                _total_ranked = len(_ranked)
                for idx, (k, _) in enumerate(_ranked, 1):
                    _rank_map[k] = idx

                _s1_rows = []
                for ofc, tt, tr, tw_sc, lw_sc, mo_sc in _s1_raw:
                    _rank_str = f"{_rank_map[ofc]}/{_total_ranked}위" if ofc in _rank_map else "-"
                    _s1_rows.append((ofc, tt, tr, tw_sc, lw_sc, mo_sc, _rank_str))
            else:
                # ── 개별 지사: 요약 1행만 ──
                _s1_rows = []
                _tw_total = len(_wr_tw_view)
                _tw_resp = int(_wr_tw_view["_점수100"].dropna().count()) if "_점수100" in _wr_tw_view.columns else _tw_total
                _tw_sc = _wr_tw_view["_점수100"].mean() if "_점수100" in _wr_tw_view.columns and not _wr_tw_view["_점수100"].dropna().empty else None
                _lw_sc = _wr_lw_view["_점수100"].mean() if "_점수100" in _wr_lw_view.columns and not _wr_lw_view["_점수100"].dropna().empty else None
                _mo_sc = _wr_mo_view["_점수100"].mean() if "_점수100" in _wr_mo_view.columns and not _wr_mo_view["_점수100"].dropna().empty else None
                _s1_rows.append((_wr_sel_view, _tw_total, _tw_resp, _tw_sc, _lw_sc, _mo_sc, "-"))

            # 합계 행 — 단순 평균
            _tw_all_t = sum(r[1] for r in _s1_rows)
            _tw_all_r = sum(r[2] for r in _s1_rows)
            _tw_all_s = _wr_tw_view["_점수100"].mean() if "_점수100" in _wr_tw_view.columns and not _wr_tw_view["_점수100"].dropna().empty else None
            _lw_vals = [r[4] for r in _s1_rows if r[4] is not None]
            _lw_all_s = sum(_lw_vals) / len(_lw_vals) if _lw_vals else None
            _mo_all_s = _wr_mo_view["_점수100"].mean() if "_점수100" in _wr_mo_view.columns and not _wr_mo_view["_점수100"].dropna().empty else None

            _row_label = "구분" if _wr_sel_view == "전체 (본부 종합)" else "업무유형"
            html_s1 = '<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;font-size:0.85em;text-align:center;">'
            # 1단 헤더: 조사결과 colspan=2
            html_s1 += f'<tr style="background:{_hdr};font-weight:bold;">'
            html_s1 += f'<th rowspan="2" style="border:1px solid {_bdr};padding:6px 8px;">{_row_label}</th>'
            html_s1 += f'<th rowspan="2" style="border:1px solid {_bdr};padding:6px 8px;">업무처리<br>건수</th>'
            html_s1 += f'<th rowspan="2" style="border:1px solid {_bdr};padding:6px 8px;">응답<br>호수</th>'
            html_s1 += f'<th colspan="2" style="border:1px solid {_bdr};padding:6px 8px;">조사결과</th>'
            html_s1 += f'<th rowspan="2" style="border:1px solid {_bdr};padding:6px 8px;">월별<br>누계</th>'
            html_s1 += f'<th rowspan="2" style="border:1px solid {_bdr};padding:6px 8px;">비고</th>'
            html_s1 += '</tr>'
            # 2단 헤더: 금주 / 전주
            html_s1 += f'<tr style="background:{_hdr};font-weight:bold;font-size:0.92em;">'
            html_s1 += f'<th style="border:1px solid {_bdr};padding:4px 6px;">금주</th>'
            html_s1 += f'<th style="border:1px solid {_bdr};padding:4px 6px;">전주</th>'
            html_s1 += '</tr>'

            for name, tt, tr, tw_s, lw_s, mo_s, rank_str in _s1_rows:
                html_s1 += '<tr>'
                html_s1 += f'<td style="border:1px solid {_bdr};padding:5px 8px;font-weight:bold;background:#f9f9f9;">{name}</td>'
                html_s1 += f'<td style="border:1px solid {_bdr};padding:5px;">{tt:,}</td>'
                html_s1 += f'<td style="border:1px solid {_bdr};padding:5px;">{tr:,}</td>'
                html_s1 += f'<td style="border:1px solid {_bdr};padding:5px;font-weight:bold;">{_fv_delta(tw_s, lw_s)}</td>'
                html_s1 += f'<td style="border:1px solid {_bdr};padding:5px;">{_fv(lw_s)}</td>'
                html_s1 += f'<td style="border:1px solid {_bdr};padding:5px;">{_fv(mo_s)}</td>'
                html_s1 += f'<td style="border:1px solid {_bdr};padding:5px;">{rank_str}</td>'
                html_s1 += '</tr>'

            # 합계
            html_s1 += f'<tr style="background:#f0f4f8;font-weight:bold;">'
            html_s1 += f'<td style="border:1px solid {_bdr};padding:5px 8px;">합계</td>'
            html_s1 += f'<td style="border:1px solid {_bdr};padding:5px;">{_tw_all_t:,}</td>'
            html_s1 += f'<td style="border:1px solid {_bdr};padding:5px;">{_tw_all_r:,}</td>'
            html_s1 += f'<td style="border:1px solid {_bdr};padding:5px;">{_fv_delta(_tw_all_s, _lw_all_s)}</td>'
            html_s1 += f'<td style="border:1px solid {_bdr};padding:5px;">{_fv(_lw_all_s)}</td>'
            html_s1 += f'<td style="border:1px solid {_bdr};padding:5px;">{_fv(_mo_all_s)}</td>'
            html_s1 += f'<td style="border:1px solid {_bdr};padding:5px;">-</td>'
            html_s1 += '</tr></table>'
            html_s1 += '<div style="text-align:right;font-size:0.8em;margin-top:4px;color:#555;">(단위 : 건, 점)</div></div>'
            st.markdown(html_s1, unsafe_allow_html=True)

            # 엑셀 다운로드
            def _fv_delta_txt(c, p):
                if c is None: return "-"
                s = f"{c:.3f}"
                if p is not None:
                    d = c - p
                    s += f' ({"△" if d>=0 else "▽"}{abs(d):.3f})'
                return s
            _s1_dl_rows = [(n, tt, tr, _fv_delta_txt(tw, lw), _fv(lw), _fv(mo), rk) for n, tt, tr, tw, lw, mo, rk in _s1_rows]
            _s1_dl = pd.DataFrame(_s1_dl_rows, columns=[_row_label,"업무처리건수","응답호수","금주(증감)","전주","월별누계","비고(순위)"])
            _s1_dl.loc[len(_s1_dl)] = ["합계", _tw_all_t, _tw_all_r, _fv_delta_txt(_tw_all_s, _lw_all_s), _fv(_lw_all_s), _fv(_mo_all_s), ""]
            st.download_button("📥 주간 조사 결과 다운로드", df_to_excel_bytes(_s1_dl),
                               "주간_조사결과.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="dl_weekly_s1")
        else:
            st.info("지사 컬럼과 점수 컬럼이 모두 매핑되어야 주간 조사 결과를 표시할 수 있습니다.")

        st.markdown("---")

        # ── Section 2: 업무유형별 조사결과 분석 ──
        st.markdown('<p class="sec-head">2. 업무유형별 조사결과 분석</p>', unsafe_allow_html=True)

        if _wr_biz and _wr_score:
            _biz_types = sorted(_wr_tw_view[_wr_biz].dropna().unique().tolist()) if not _wr_tw_view.empty else sorted(df_f[_wr_biz].dropna().unique().tolist())

            # 업무유형별 데이터 수집
            _s2_data = {}  # {업무유형: (tw_cnt, tw_score, lw_score, delta, mo_score)}
            for bt in _biz_types:
                _tw_b = _wr_tw_view[_wr_tw_view[_wr_biz] == bt]
                _lw_b = _wr_lw_view[_wr_lw_view[_wr_biz] == bt]
                _mo_b = _wr_mo_view[_wr_mo_view[_wr_biz] == bt]
                _tw_cnt = len(_tw_b)
                _tw_bs = _tw_b["_점수100"].mean() if "_점수100" in _tw_b.columns and not _tw_b["_점수100"].dropna().empty else None
                _lw_bs = _lw_b["_점수100"].mean() if "_점수100" in _lw_b.columns and not _lw_b["_점수100"].dropna().empty else None
                _mo_bs = _mo_b["_점수100"].mean() if "_점수100" in _mo_b.columns and not _mo_b["_점수100"].dropna().empty else None
                _s2_data[bt] = (_tw_cnt, _tw_bs, _lw_bs, _mo_bs)

            # 가로형 테이블: 열=업무유형, 행=응답건수/금주(증감)/전주/월누계
            html_s2 = '<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;font-size:0.85em;text-align:center;">'
            html_s2 += f'<tr style="background:{_hdr};font-weight:bold;">'
            html_s2 += f'<th style="border:1px solid {_bdr};padding:6px 10px;min-width:90px;">구분</th>'
            for bt in _biz_types:
                html_s2 += f'<th style="border:1px solid {_bdr};padding:6px 6px;">{bt}</th>'
            html_s2 += '</tr>'

            # 행: 응답건수(금주)
            html_s2 += f'<tr><td style="border:1px solid {_bdr};padding:5px 8px;font-weight:bold;background:#f9f9f9;">응답건수(금주)</td>'
            for bt in _biz_types:
                html_s2 += f'<td style="border:1px solid {_bdr};padding:5px;">{_s2_data[bt][0]:,}</td>'
            html_s2 += '</tr>'

            # 행: 금주 점수
            html_s2 += f'<tr><td style="border:1px solid {_bdr};padding:5px 8px;font-weight:bold;background:#f9f9f9;">금주</td>'
            for bt in _biz_types:
                html_s2 += f'<td style="border:1px solid {_bdr};padding:5px;font-weight:bold;">{_fv(_s2_data[bt][1])}</td>'
            html_s2 += '</tr>'

            html_s2 += '</table>'
            html_s2 += '<div style="text-align:right;font-size:0.8em;margin-top:4px;color:#555;">(단위 : 건, 점)</div></div>'
            st.markdown(html_s2, unsafe_allow_html=True)

            # 엑셀 다운로드
            _s2_dl_data = {"구분": ["응답건수(금주)", "금주"]}
            for bt in _biz_types:
                d = _s2_data[bt]
                _s2_dl_data[bt] = [d[0], _fv(d[1])]
            _s2_dl = pd.DataFrame(_s2_dl_data)
            st.download_button("📥 업무유형별 분석 다운로드", df_to_excel_bytes(_s2_dl),
                               "업무유형별_조사결과.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="dl_weekly_s2")
        else:
            st.info("업무구분 컬럼과 점수 컬럼이 모두 매핑되어야 업무유형별 분석을 표시할 수 있습니다.")

        st.markdown("---")

        # ── Section 3: 조사결과 피드백 ──
        st.markdown('<p class="sec-head">3. 조사결과 피드백</p>', unsafe_allow_html=True)

        if _wr_voc:
            _fb_df = _wr_tw_view.copy()
            if "_점수100" in _fb_df.columns:
                _fb_df["_감성분류"] = _fb_df.apply(lambda r: _classify_sentiment_3tier(r[_wr_voc], r.get("_점수100")), axis=1)
            else:
                _fb_df["_감성분류"] = _fb_df[_wr_voc].apply(_classify_sentiment_3tier)
            _fb_pos = int((_fb_df["_감성분류"] == "긍정").sum())
            _fb_neg = int((_fb_df["_감성분류"] == "부정").sum())
            _fb_neu = int((_fb_df["_감성분류"] == "중립").sum())

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("😊 긍정", f"{_fb_pos}건")
            mc2.metric("😐 중립", f"{_fb_neu}건")
            mc3.metric("😠 부정", f"{_fb_neg}건")

            # 부정 우선 정렬
            _sent_order = {"부정": 0, "중립": 1, "긍정": 2}
            _fb_df["_감성순서"] = _fb_df["_감성분류"].map(_sent_order)
            _fb_df = _fb_df.sort_values("_감성순서")

            _fb_cols = []
            _fb_rename = {}
            if _wr_office:
                _fb_cols.append(_wr_office)
                _fb_rename[_wr_office] = "지사"
            _fb_cols.append("_감성분류")
            _fb_rename["_감성분류"] = "감성분류"
            if _wr_biz:
                _fb_cols.append(_wr_biz)
                _fb_rename[_wr_biz] = "업무구분"
            if "_점수100" in _fb_df.columns:
                _fb_cols.append("_점수100")
                _fb_rename["_점수100"] = "종합점수"
            _fb_cols.append(_wr_voc)
            _fb_rename[_wr_voc] = "서술의견"

            _fb_show = _fb_df[_fb_cols].rename(columns=_fb_rename).reset_index(drop=True)

            # HTML 테이블 (부정 행 빨간 배경 강조)
            _fb_html = '<div style="overflow-x:auto;max-height:420px;overflow-y:auto;"><table style="border-collapse:collapse;width:100%;font-size:0.84em;text-align:left;">'
            _fb_html += f'<tr style="background:#d6e4f0;font-weight:bold;position:sticky;top:0;">'
            for col in _fb_show.columns:
                _fb_html += f'<th style="border:1px solid #b0b0b0;padding:6px 8px;">{col}</th>'
            _fb_html += '</tr>'
            for _, row in _fb_show.iterrows():
                _sent = row.get("감성분류", "")
                if _sent == "부정":
                    _row_bg = "background:#ffebee;"
                    _badge = '<span style="background:#C62828;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.88em;">부정</span>'
                elif _sent == "긍정":
                    _row_bg = ""
                    _badge = '<span style="background:#1565C0;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.88em;">긍정</span>'
                else:
                    _row_bg = ""
                    _badge = '<span style="background:#757575;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.88em;">중립</span>'
                _fb_html += f'<tr style="{_row_bg}">'
                for col in _fb_show.columns:
                    v = row[col]
                    if col == "감성분류":
                        _fb_html += f'<td style="border:1px solid #b0b0b0;padding:5px 8px;text-align:center;">{_badge}</td>'
                    elif col == "종합점수":
                        _fb_html += f'<td style="border:1px solid #b0b0b0;padding:5px 8px;text-align:center;">{_fv(v)}</td>'
                    else:
                        _fb_html += f'<td style="border:1px solid #b0b0b0;padding:5px 8px;">{v}</td>'
                _fb_html += '</tr>'
            _fb_html += '</table></div>'
            st.markdown(_fb_html, unsafe_allow_html=True)
        else:
            st.info("VOC(서술의견) 컬럼을 선택해야 피드백 목록을 표시할 수 있습니다.")

        st.markdown("---")

        # ── Section 4: 협조요청 사항 ──
        st.markdown('<p class="sec-head">4. 협조요청 사항</p>', unsafe_allow_html=True)

        st.markdown(
            '<div style="background:#f8f9fb;border:1px solid #d0d7de;border-radius:10px;'
            'padding:20px 24px;margin:8px 0;font-size:0.93em;line-height:1.9;">'
            '<b>📌 고객 우호 활동 독려</b><br>'
            '&nbsp;&nbsp;① 고객 응대 시 <b>친절하고 정중한 응대</b>를 통해 고객 만족도 향상에 적극 협조 바랍니다.<br>'
            '&nbsp;&nbsp;② 고객 방문 시 <b>홍보용품 증정</b> 등 고객 감동 활동을 적극 실시해 주시기 바랍니다.<br>'
            '&nbsp;&nbsp;③ CS 조사 대상 고객에게 <b>조사 참여를 당부</b>하여 응답률 향상에 협조 부탁드립니다.<br><br>'
            '<b>⚠️ 특이 민원 및 긴급 사안</b><br>'
            '&nbsp;&nbsp;• 고객 불만 사항 및 특이 민원 발생 시 <b>즉시 보고</b>하여 주시기 바랍니다.<br>'
            '&nbsp;&nbsp;• 반복 민원 및 법적 분쟁 가능성이 있는 건은 <b>사전 보고 후 대응</b>해 주시기 바랍니다.'
            '</div>', unsafe_allow_html=True)

        # AI 분석 버튼
        _wr_ai_key = "_ai_weekly_coop"
        if st.button("🤖 AI 주간 협조요청 분석", key="btn_weekly_ai", use_container_width=True):
            # 금주 데이터 요약 생성
            _ai_summary_parts = []
            if _wr_office and "_점수100" in _wr_tw_view.columns:
                _ofc_summary = _wr_tw_view.groupby(_wr_office)["_점수100"].agg(["mean","count"]).reset_index()
                _ofc_summary.columns = ["지사","평균점수","건수"]
                _ai_summary_parts.append("■ 지사별 금주 현황:\n" + _ofc_summary.to_string(index=False))
            if _wr_biz and "_점수100" in _wr_tw_view.columns:
                _biz_summary = _wr_tw_view.groupby(_wr_biz)["_점수100"].agg(["mean","count"]).reset_index()
                _biz_summary.columns = ["업무유형","평균점수","건수"]
                _ai_summary_parts.append("■ 업무유형별 금주 현황:\n" + _biz_summary.to_string(index=False))
            if _wr_voc:
                _neg_vocs = _wr_tw_view[_wr_tw_view[_wr_voc].apply(lambda x: _classify_sentiment_3tier(x) == "부정")][_wr_voc].head(10).tolist()
                if _neg_vocs:
                    _ai_summary_parts.append("■ 금주 부정 VOC (최대 10건):\n" + "\n".join(f"- {v}" for v in _neg_vocs))

            _ai_weekly_prompt = f"""당신은 한국전력공사 경남본부 CS 담당자입니다.
아래 금주 CS 조사 데이터를 분석하여 주간 협조요청 사항을 작성해주세요.

{chr(10).join(_ai_summary_parts)}

다음 4개 항목을 개조식(bullet point)으로 작성해주세요:

#### 1. 금주 핵심 이슈
- 이번 주 가장 주목해야 할 CS 이슈 3가지

#### 2. 지사별 주의사항
- 점수가 낮거나 부정 VOC가 많은 지사 중심으로 개선 방향 제시

#### 3. 업무유형별 개선 포인트
- 업무유형별 취약점과 구체적 개선 방안

#### 4. 금주 실행 과제
- 이번 주 즉시 실행 가능한 CS 개선 과제 3-5개

[필수 규칙]
- '전기세' 표현 절대 금지. 반드시 '전기요금'으로 표기.
- 줄글(산문체) 금지. 반드시 개조식(bullet, 짧은 문장) 보고서 형태로 작성.
- 한 bullet에 2줄 이상 금지. 핵심만 짧게."""

            with st.spinner("AI가 주간 협조요청 분석 중…"):
                try:
                    import urllib.request
                    _models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemma-3-12b-it"]
                    _payload = {"contents": [{"parts": [{"text": _ai_weekly_prompt}]}],
                                 "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192}}
                    _ctx = ssl._create_unverified_context()
                    _body = None
                    for _model in _models:
                        _api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{_model}:generateContent?key={_GEMINI_KEY}"
                        _req = urllib.request.Request(_api_url, data=json.dumps(_payload).encode("utf-8"),
                                                       headers={"Content-Type": "application/json"}, method="POST")
                        try:
                            with urllib.request.urlopen(_req, context=_ctx, timeout=90) as _resp:
                                _body = json.loads(_resp.read().decode("utf-8"))
                            break
                        except urllib.error.HTTPError as _http_err:
                            if _http_err.code in (429, 500, 502, 503):
                                continue
                            raise
                    if _body is None:
                        st.error("모든 AI 모델의 일일 한도가 소진되었습니다. 내일 다시 시도해주세요.")
                    else:
                        st.session_state[_wr_ai_key] = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
                except Exception as e:
                    st.error(f"AI 분석 중 오류: {e}")

        # 캐시된 결과 표시
        if _wr_ai_key in st.session_state:
            st.markdown(
                '<div style="background:#f8f9fb;border:1px solid #d0d7de;border-radius:10px;'
                'padding:24px 28px;margin:8px 0;font-size:0.93em;line-height:1.85;">\n\n'
                f'{st.session_state[_wr_ai_key]}\n\n</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  TAB 1  구간별 비중 · 종합 현황
# ─────────────────────────────────────────────────────────────
with tab1:
    if M["score"] and avg_score_100 is not None and not np.isnan(avg_score_100):
        # ── 게이지 + 구간별 비중 통계 (일직선) ──
        g_col, b_col = st.columns([1, 1])
        with g_col:
            st.markdown('<p class="sec-head">🎯 종합 만족도 게이지</p>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(avg_score_100, 1),
                number={"suffix": "점", "font": {"size": 40, "color": C["navy"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"size": 11}},
                    "bar": {"color": C["blue"], "thickness": 0.28},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 50], "color": "#ffcdd2"},
                        {"range": [50, 70], "color": "#fff9c4"},
                        {"range": [70, 90], "color": "#c8e6c9"},
                        {"range": [90, 100], "color": "#a5d6a7"},
                    ],
                },
                title={"text": "종합 만족도 (100점 환산)",
                       "font": {"size": 14, "color": C["navy"]}},
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=70, b=20, l=30, r=30), paper_bgcolor="white")
            st.plotly_chart(fig_gauge, use_container_width=True, config={'staticPlot': True})
        with b_col:
            st.markdown('<p class="sec-head">📊 만족도 점수 구간별 비중 통계</p>', unsafe_allow_html=True)
            bucket_cnt = df_f["_점수구간"].value_counts()
            bucket_data = pd.DataFrame({
                "구간": BUCKET_ORDER,
                "건수": [bucket_cnt.get(b, 0) for b in BUCKET_ORDER],
            })
            bucket_data["비율(%)"] = (bucket_data["건수"] / max(bucket_data["건수"].sum(), 1) * 100).round(1)
            bucket_data["라벨"] = bucket_data.apply(lambda r: f"{r['구간']} ({int(r['건수']):,}건)", axis=1)
            _bk_color_map = {f"{k} ({int(bucket_data.loc[bucket_data['구간']==k, '건수'].values[0]):,}건)": v
                             for k, v in BUCKET_COLORS.items() if k in bucket_data["구간"].values}
            fig_bp = px.pie(bucket_data, names="라벨", values="건수", color="라벨",
                            color_discrete_map=_bk_color_map, hole=0.45, template=PLOTLY_TPL)
            fig_bp.update_traces(textposition="outside", textinfo="percent+label", textfont_size=12,
                                  marker=dict(line=dict(color="#ffffff", width=2)),
                                  hovertemplate="%{label}<br>%{percent}<extra></extra>")
            fig_bp.update_layout(height=300, margin=dict(t=30, b=20, l=20, r=20), showlegend=True)
            st.plotly_chart(fig_bp, use_container_width=True, config={'staticPlot': True})

        # ── 사업소별 평균 만족도 ──
        if M["office"]:
            st.markdown("---")
            st.markdown('<p class="sec-head">🏢 사업소별 평균 만족도</p>', unsafe_allow_html=True)
            _ofc_grp_bar = df_f.groupby(M["office"])["_점수100"].agg(["mean", "count"]).reset_index()
            _ofc_grp_bar.columns = ["사업소", "평균만족도", "응답수"]
            _ofc_grp_bar = _ofc_grp_bar.sort_values("평균만족도", ascending=True)
            _bar_avg = df_f["_점수100"].mean()
            _ofc_grp_bar["그룹"] = _ofc_grp_bar["평균만족도"].apply(
                lambda v: "⬆ 본부평균 이상" if v >= _bar_avg else "⬇ 본부평균 미달")
            _ofc_grp_bar["_차이"] = _ofc_grp_bar["평균만족도"] - _bar_avg
            _ofc_grp_bar["_표시"] = _ofc_grp_bar.apply(
                lambda r: f"{r['평균만족도']:.1f} ({'+' if r['_차이']>=0 else ''}{r['_차이']:.1f}점)", axis=1)
            fig_bench = px.bar(_ofc_grp_bar, x="평균만족도", y="사업소", color="그룹",
                               color_discrete_map={"⬆ 본부평균 이상": C["teal"], "⬇ 본부평균 미달": C["red"]},
                               orientation="h", text="_표시", template=PLOTLY_TPL,
                               title=f"사업소별 평균 만족도 — 본부 평균: {_bar_avg:.1f}점")
            fig_bench.update_traces(texttemplate="%{text}", textposition="outside",
                                    hovertemplate="%{y}<br>평균: %{x:.1f}점<br>본부 대비: %{customdata[0]:+.1f}점<extra></extra>",
                                    customdata=_ofc_grp_bar[["_차이"]].values)
            fig_bench.add_vline(x=_bar_avg, line_color=C["navy"], line_dash="dash", line_width=2.5,
                                annotation_text=f"▼ 본부 평균 {_bar_avg:.1f}",
                                annotation_font_size=12, annotation_font_color=C["navy"],
                                annotation_position="top", annotation_yshift=10)
            fig_bench.update_layout(height=max(350, len(_ofc_grp_bar) * 35 + 80),
                                     margin=dict(t=80, b=20, l=10, r=160), legend_title_text="",
                                     title_font=dict(size=14, color=C["navy"]))
            st.plotly_chart(fig_bench, use_container_width=True, config={'staticPlot': True})


    # ── 분포 차트 ──
    has_pie = any([M["age"], M["contract"], M["business"]])
    if has_pie:
        st.markdown('<p class="sec-head">🍩 응답 분포 현황</p>', unsafe_allow_html=True)
        pie_cols = [c for c in [M["age"], M["contract"], M["business"]] if c]
        pc_list = st.columns(len(pie_cols))
        titles_map = {}
        if M["age"]: titles_map[M["age"]] = "연령대"
        if M["contract"]: titles_map[M["contract"]] = "계약종별"
        if M["business"]: titles_map[M["business"]] = "업무유형"
        for idx, col_nm in enumerate(pie_cols):
            counts = df_f[col_nm].dropna().astype(str).value_counts()
            _total = counts.sum()
            with pc_list[idx]:
                _title = titles_map.get(col_nm, col_nm)
                if col_nm == M["business"]:
                    # 업무유형: 가로 막대그래프, 퍼센트 높은 순 정렬
                    _biz_df = pd.DataFrame({"유형": counts.index, "건수": counts.values})
                    _biz_df["비율(%)"] = (_biz_df["건수"] / max(_total, 1) * 100).round(1)
                    _biz_df = _biz_df.sort_values("비율(%)", ascending=True)
                    fig_biz = px.bar(_biz_df, x="비율(%)", y="유형", orientation="h",
                                     text=[f"{r:.1f}% ({int(c):,}건)" for r, c in zip(_biz_df["비율(%)"], _biz_df["건수"])],
                                     color_discrete_sequence=[C["sky"]], template=PLOTLY_TPL,
                                     title=f"{_title} 분포")
                    fig_biz.update_traces(textposition="outside", textfont_size=11,
                                          hovertemplate="%{y}: %{x:.1f}% (%{customdata[0]:,}건)<extra></extra>",
                                          customdata=_biz_df[["건수"]].values)
                    _biz_x_max = _biz_df["비율(%)"].max() * 1.45
                    fig_biz.update_layout(height=max(360, len(_biz_df) * 30 + 80),
                                           margin=dict(t=50, b=20, l=20, r=20), showlegend=False,
                                           title_font=dict(size=15, color=C["navy"]),
                                           xaxis=dict(title="비율(%)", range=[0, _biz_x_max]), yaxis_title="")
                    st.plotly_chart(fig_biz, use_container_width=True, config={'staticPlot': True})
                else:
                    # 연령대, 계약종별: 파이차트 (건수 포함)
                    _pie_labels = [f"{nm} ({cnt:,}건)" for nm, cnt in zip(counts.index, counts.values)]
                    fig_pie = px.pie(names=_pie_labels, values=counts.values, color_discrete_sequence=PIE_COLORS,
                                     hole=0.42, title=f"{_title} 분포", template=PLOTLY_TPL)
                    fig_pie.update_traces(textposition="outside", textinfo="percent+label", textfont_size=11,
                                           marker=dict(line=dict(color="#ffffff", width=2)),
                                           hovertemplate="%{label}<br>%{percent}<extra></extra>")
                    fig_pie.update_layout(height=360, margin=dict(t=50, b=20, l=20, r=20), showlegend=False,
                                           title_font=dict(size=15, color=C["navy"]))
                    st.plotly_chart(fig_pie, use_container_width=True, config={'staticPlot': True})


# ─────────────────────────────────────────────────────────────
#  TAB 3  계약종별 · 업무유형별 분석
# ─────────────────────────────────────────────────────────────
def _render_category_section(df, cat_col, cat_label, office_col, score_col, overall_avg):
    """범주별 분석 공통 렌더링: 전체 막대 + 지사별 히트맵 + 하위 3개 카드"""
    grp = df.groupby(cat_col)[score_col].agg(["mean","count"]).reset_index()
    grp.columns = [cat_label, "평균만족도", "응답수"]
    grp = grp.sort_values("평균만족도", ascending=True)

    bottom3 = grp.head(3)
    bottom3_names = bottom3[cat_label].tolist()
    grp["구분"] = grp[cat_label].apply(lambda x: "하위 3" if x in bottom3_names else "일반")

    # ── 전체 평균 막대 차트 ──
    fig = px.bar(grp, x="평균만족도", y=cat_label, color="구분",
                 color_discrete_map={"하위 3": C["red"], "일반": C["sky"]},
                 orientation="h", text="평균만족도", template=PLOTLY_TPL,
                 title=f"{cat_label}별 평균 만족도 (빨간색 = 하위 3개)")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                      hovertemplate="%{y}<br>평균: %{x:.1f}점<br>응답: %{customdata[0]:,}건<extra></extra>",
                      customdata=grp[["응답수"]].values)
    fig.add_vline(x=overall_avg, line_color=C["navy"], line_dash="dash", line_width=2,
                  annotation_text=f"▼ 전체 평균 {overall_avg:.1f}",
                  annotation_font_size=11, annotation_font_color=C["navy"],
                  annotation_position="top", annotation_yshift=10)
    fig.update_layout(height=max(300, len(grp) * 30 + 80),
                       margin=dict(t=80, b=20, l=10, r=100), legend_title_text="",
                       title_font=dict(size=14, color=C["navy"]))
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    # ── 하위 3개 카드 ──
    b3_cols = st.columns(min(3, len(bottom3)))
    for i, (_, row) in enumerate(bottom3.iterrows()):
        if i < len(b3_cols):
            with b3_cols[i]:
                st.markdown(
                    f'<div class="card-red">'
                    f'<b style="font-size:1.05rem;">🔴 {row[cat_label]}</b><br><br>'
                    f'평균: <b>{row["평균만족도"]:.1f}점</b><br>'
                    f'응답: {row["응답수"]:,}건<br>'
                    f'전체 대비: <b style="color:{C["red"]}">{row["평균만족도"] - overall_avg:+.1f}점</b>'
                    f'</div>', unsafe_allow_html=True)

    # ── 지사별 × 범주별 점수 테이블/히트맵 ──
    if office_col:
        st.markdown(f'<p class="sec-head">🏢 지사별 {cat_label} 평균 만족도</p>', unsafe_allow_html=True)
        # 점수 pivot
        pivot = df.pivot_table(values=score_col, index=office_col,
                               columns=cat_col, aggfunc="mean").round(1)
        # 건수 pivot
        pivot_cnt = df.pivot_table(values=score_col, index=office_col,
                                   columns=cat_col, aggfunc="count").fillna(0).astype(int)
        _offices_sorted = _sort_offices(pivot.index.tolist())
        pivot = pivot.reindex(_offices_sorted)
        pivot_cnt = pivot_cnt.reindex(_offices_sorted).fillna(0).astype(int)

        _is_contract = (cat_label == "계약종별")
        # 계약종별 컬럼 순서 고정
        if _is_contract:
            _ct_order = ["주택용", "일반용", "산업용", "농사용", "교육용", "가로등"]
            _ordered = [c for c in _ct_order if c in pivot.columns]
            _rest = [c for c in pivot.columns if c not in _ct_order]
            pivot = pivot[_ordered + _rest]
            pivot_cnt = pivot_cnt.reindex(columns=pivot.columns).fillna(0).astype(int)

        if not pivot.empty:
            # 점수 범위 (색상 스케일용)
            _all_vals = pivot.values.flatten()
            _all_vals = _all_vals[~np.isnan(_all_vals)]
            _vmin = float(_all_vals.min()) if len(_all_vals) > 0 else 0
            _vmax = float(_all_vals.max()) if len(_all_vals) > 0 else 100

            def _score_color(v):
                """점수 → RdYlGn 배경색 (plotly 히트맵과 동일 채도)"""
                if pd.isna(v):
                    return ""
                t = max(0, min(1, (v - _vmin) / (_vmax - _vmin))) if _vmax > _vmin else 0.5
                # plotly RdYlGn 5-stop: 빨강(0) → 주황(0.25) → 노랑(0.5) → 연두(0.75) → 초록(1)
                stops = [
                    (0.0, 215, 48, 39),
                    (0.25, 252, 141, 89),
                    (0.5, 255, 255, 191),
                    (0.75, 145, 207, 96),
                    (1.0, 26, 152, 80),
                ]
                for i in range(len(stops) - 1):
                    t0, r0, g0, b0 = stops[i]
                    t1, r1, g1, b1 = stops[i + 1]
                    if t <= t1:
                        p = (t - t0) / (t1 - t0) if t1 > t0 else 0
                        r = int(r0 + (r1 - r0) * p)
                        g = int(g0 + (g1 - g0) * p)
                        b = int(b0 + (b1 - b0) * p)
                        return f"background:rgba({r},{g},{b},0.85);"
                return f"background:rgba(26,152,80,0.85);"

            _hdr = "#d6e4f0"
            _bdr = "#b0b0b0"
            _ylw = "#fef9e7"
            _cols = list(pivot.columns)

            html = '<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;font-size:0.85em;text-align:center;">'

            if _is_contract:
                # ── 양식2: 2단 헤더 (계약종 > 종합점수/응답호수) ──
                html += f'<tr style="background:{_hdr};font-weight:bold;">'
                html += f'<th rowspan="2" style="border:1px solid {_bdr};padding:6px 10px;min-width:70px;">지사</th>'
                for c in _cols:
                    html += f'<th colspan="2" style="border:1px solid {_bdr};padding:6px 4px;">{c}</th>'
                html += f'<th colspan="2" style="border:1px solid {_bdr};padding:6px 4px;">합계</th>'
                html += '</tr>'
                html += f'<tr style="background:{_hdr};font-weight:bold;font-size:0.9em;">'
                for _ in _cols:
                    html += f'<th style="border:1px solid {_bdr};padding:4px;">종합<br>점수</th>'
                    html += f'<th style="border:1px solid {_bdr};padding:4px;">응답<br>호수</th>'
                html += f'<th style="border:1px solid {_bdr};padding:4px;">종합<br>점수</th>'
                html += f'<th style="border:1px solid {_bdr};padding:4px;">응답<br>호수</th>'
                html += '</tr>'
            else:
                # ── 양식3: 1단 헤더 (업무유형/접수채널) ──
                html += f'<tr style="background:{_hdr};font-weight:bold;">'
                html += f'<th style="border:1px solid {_bdr};padding:6px 10px;min-width:70px;">구분</th>'
                for c in _cols:
                    html += f'<th style="border:1px solid {_bdr};padding:6px 4px;">{c}</th>'
                html += f'<th style="border:1px solid {_bdr};padding:6px 4px;">합계</th>'
                html += '</tr>'

            # 본부 합계 행
            _dbl = f"border-bottom:3px double {_bdr};"
            html += f'<tr style="background:#e8eef5;font-weight:bold;">'
            html += f'<td style="border:1px solid {_bdr};{_dbl}padding:5px 8px;font-weight:bold;background:#dce6f0;">본부</td>'
            for c in _cols:
                _hq_v = df[cat_col].eq(c).astype(int)
                _hq_score = df.loc[_hq_v == 1, score_col].mean()
                _hq_str = f"{_hq_score:.1f}" if pd.notna(_hq_score) else ""
                if _is_contract:
                    _hq_cnt = int(df.loc[_hq_v == 1, score_col].count())
                    html += f'<td style="border:1px solid {_bdr};{_dbl}padding:4px;">{_hq_str}</td>'
                    html += f'<td style="border:1px solid {_bdr};{_dbl}padding:4px;">{_hq_cnt}</td>'
                else:
                    html += f'<td style="border:1px solid {_bdr};{_dbl}padding:4px;">{_hq_str}</td>'
            # 본부 합계 총합
            _hq_total = df[score_col].mean()
            _hq_total_str = f"{_hq_total:.1f}" if pd.notna(_hq_total) else ""
            if _is_contract:
                _hq_total_cnt = int(df[score_col].count())
                html += f'<td style="border:1px solid {_bdr};{_dbl}padding:4px;font-weight:bold;">{_hq_total_str}</td>'
                html += f'<td style="border:1px solid {_bdr};{_dbl}padding:4px;">{_hq_total_cnt}</td>'
            else:
                html += f'<td style="border:1px solid {_bdr};{_dbl}padding:4px;font-weight:bold;">{_hq_total_str}</td>'
            html += '</tr>'

            # 데이터 행
            for ofc in pivot.index:
                html += '<tr>'
                html += f'<td style="border:1px solid {_bdr};padding:5px 8px;font-weight:bold;background:#f9f9f9;">{ofc}</td>'
                for c in _cols:
                    v = pivot.loc[ofc, c]
                    v_str = f"{v:.1f}" if pd.notna(v) else ""
                    _bg = _score_color(v)
                    if _is_contract:
                        cnt = int(pivot_cnt.loc[ofc, c])
                        cnt_str = str(cnt) if cnt > 0 else ""
                        html += f'<td style="border:1px solid {_bdr};padding:4px;{_bg}">{v_str}</td>'
                        html += f'<td style="border:1px solid {_bdr};padding:4px;">{cnt_str}</td>'
                    else:
                        html += f'<td style="border:1px solid {_bdr};padding:4px;{_bg}">{v_str}</td>'
                # 합계 열
                _t_score = df[df[office_col] == ofc][score_col].mean()
                _t_str = f"{_t_score:.1f}" if pd.notna(_t_score) else ""
                _t_bg = _score_color(_t_score)
                if _is_contract:
                    _t_cnt = int(df[df[office_col] == ofc][score_col].count())
                    html += f'<td style="border:1px solid {_bdr};padding:4px;font-weight:bold;">{_t_str}</td>'
                    html += f'<td style="border:1px solid {_bdr};padding:4px;">{_t_cnt}</td>'
                else:
                    html += f'<td style="border:1px solid {_bdr};padding:4px;font-weight:bold;">{_t_str}</td>'
                html += '</tr>'

            html += '</table>'
            _unit = "(단위 : 호, 점)" if _is_contract else "(단위 : 점)"
            html += f'<div style="text-align:right;font-size:0.8em;margin-top:4px;color:#555;">{_unit}</div></div>'
            st.markdown(html, unsafe_allow_html=True)

            # ── 엑셀 다운로드 ──
            _dl_key = f"dl_{cat_label.replace(' ','')}"
            if _is_contract:
                # 계약종별: 종합점수+응답호수 병합 테이블
                _dl_rows = []
                for ofc in pivot.index:
                    _r = {"지사": ofc}
                    for c in pivot.columns:
                        _r[f"{c}_종합점수"] = pivot.loc[ofc, c] if pd.notna(pivot.loc[ofc, c]) else ""
                        _r[f"{c}_응답호수"] = int(pivot_cnt.loc[ofc, c])
                    _r["합계_종합점수"] = df[df[office_col] == ofc][score_col].mean()
                    _r["합계_응답호수"] = int(df[df[office_col] == ofc][score_col].count())
                    _dl_rows.append(_r)
                _dl_df = pd.DataFrame(_dl_rows)
            else:
                # 업무유형/접수채널: 점수만
                _dl_df = pivot.copy()
                _dl_df.index.name = "지사"
                _dl_df["합계"] = [df[df[office_col] == ofc][score_col].mean() for ofc in _dl_df.index]
                _dl_df = _dl_df.reset_index()
            _dl_bytes = df_to_excel_bytes(_dl_df)
            st.download_button(
                label=f"📥 {cat_label} 테이블 엑셀 다운로드",
                data=_dl_bytes,
                file_name=f"지사별_{cat_label}_만족도.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=_dl_key, use_container_width=True)

        # ── 저점수 건 상세 조회 (60점 이하 전체 표) ──
        _uid = cat_label.replace(" ", "")
        with st.expander(f"🔍 60점 이하 저점수 건 상세 조회 ({cat_label})"):
            _filtered = df[df[score_col] <= 60].copy()
            if len(_filtered) > 0:
                _filtered = _filtered.sort_values(score_col)
                st.markdown(
                    f'총 <b style="color:{C["red"]}">{len(_filtered):,}건</b>이 60점 이하입니다.',
                    unsafe_allow_html=True)
                _show_cols = []
                if M.get("receipt_no") and M["receipt_no"] in _filtered.columns:
                    _show_cols.append(M["receipt_no"])
                if office_col and office_col in _filtered.columns:
                    _show_cols.append(office_col)
                _show_cols.append(cat_col)
                if score_col in _filtered.columns:
                    _show_cols.append(score_col)
                if M.get("voc") and M["voc"] in _filtered.columns:
                    _show_cols.append(M["voc"])
                _show_cols = [c for c in _show_cols if c in _filtered.columns]
                st.dataframe(_filtered[_show_cols].reset_index(drop=True),
                             use_container_width=True, height=400, hide_index=True)
            else:
                st.success("60점 이하 데이터가 없습니다.")


with tab3:
    if not M["score"]:
        st.warning("만족도 점수 컬럼이 필요합니다.")
    else:
        # ── ① 항목별 결과 (양식1) ──
        if M["office"] and individual_scores and "_점수100" in df_f.columns:
            st.markdown('<p class="sec-head">📋 사업소별 만족도 조사결과 — 항목별 결과</p>', unsafe_allow_html=True)
            _item_offices = _sort_offices(df_f[M["office"]].dropna().unique().tolist())
            _resp_col = M.get("response")
            _item_rows = []
            for _ofc in _item_offices:
                _odf = df_f[df_f[M["office"]] == _ofc]  # 응답(점수 있는) 데이터
                _raw_ofc = df_raw[df_raw[M["office"]] == _ofc]  # 전체 원본
                _completed = len(_raw_ofc)
                if _resp_col and _resp_col in df_raw.columns:
                    _rv = _raw_ofc[_resp_col].astype(str).str.strip()
                    _sent = int(_rv.isin(["응답", "미응답"]).sum())
                    _responded = int((_rv == "응답").sum())
                else:
                    _sent = "-"
                    _responded = len(_odf)
                _resp_rate = round(_responded / _sent * 100, 1) if isinstance(_sent, int) and _sent > 0 else "-"
                _row = {"구분": _ofc, "업무처리완료고객": f"{_completed:,}",
                        "발송호수": f"{_sent:,}" if isinstance(_sent, int) else _sent,
                        "응답호수": f"{_responded:,}" if isinstance(_responded, int) else _responded,
                        "응답률(%)": _resp_rate}
                for _sc in individual_scores:
                    if _sc in _odf.columns:
                        _val = _odf[_sc].dropna()
                        _row[_sc] = round(_val.mean(), 1) if len(_val) > 0 else ""
                    else:
                        _row[_sc] = ""
                _row["종합점수"] = round(_odf["_점수100"].mean(), 1) if len(_odf) > 0 else ""
                _item_rows.append(_row)

            # 본부 합계 행
            _all_completed = len(df_raw[df_raw[M["office"]].notna()])
            if _resp_col and _resp_col in df_raw.columns:
                _raw_with_ofc = df_raw[df_raw[M["office"]].notna()]
                _rv_all = _raw_with_ofc[_resp_col].astype(str).str.strip()
                _all_sent = int(_rv_all.isin(["응답", "미응답"]).sum())
                _all_responded = int((_rv_all == "응답").sum())
            else:
                _all_sent = "-"
                _all_responded = len(df_f)
            _all_resp_rate = round(_all_responded / _all_sent * 100, 1) if isinstance(_all_sent, int) and _all_sent > 0 else "-"
            _hq_row = {"구분": "본부", "업무처리완료고객": f"{_all_completed:,}",
                       "발송호수": f"{_all_sent:,}" if isinstance(_all_sent, int) else _all_sent,
                       "응답호수": f"{_all_responded:,}" if isinstance(_all_responded, int) else _all_responded,
                       "응답률(%)": _all_resp_rate}
            for _sc in individual_scores:
                if _sc in df_f.columns:
                    _val = df_f[_sc].dropna()
                    _hq_row[_sc] = round(_val.mean(), 1) if len(_val) > 0 else ""
                else:
                    _hq_row[_sc] = ""
            _hq_row["종합점수"] = round(df_f["_점수100"].mean(), 1)
            _item_rows.insert(0, _hq_row)

            _item_df = pd.DataFrame(_item_rows)
            # HTML 테이블 (양식1 스타일)
            _hdr_bg = "#d6e4f0"
            _border = "#b0b0b0"
            _f1_html = '<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;font-size:0.85em;text-align:center;">'
            _f1_html += f'<tr style="background:{_hdr_bg};font-weight:bold;">'
            for _col in _item_df.columns:
                _f1_html += f'<th style="border:1px solid {_border};padding:6px 4px;">{_col}</th>'
            _f1_html += '</tr>'
            for _ri, (_, _row) in enumerate(_item_df.iterrows()):
                _is_hq = (_ri == 0)
                _bg = "background:#e8eef5;font-weight:bold;" if _is_hq else ""
                _dbl_b = f"border-bottom:3px double {_border};" if _is_hq else ""
                _f1_html += f'<tr style="{_bg}">'
                for _col in _item_df.columns:
                    _v = _row[_col]
                    if _col == "구분":
                        _cell_bg = "background:#dce6f0;" if _is_hq else "background:#f9f9f9;"
                        _f1_html += f'<td style="border:1px solid {_border};{_dbl_b}padding:5px 8px;font-weight:bold;{_cell_bg}">{_v}</td>'
                    elif _col == "종합점수":
                        _f1_html += f'<td style="border:1px solid {_border};{_dbl_b}padding:4px;font-weight:bold;">{_v}</td>'
                    else:
                        _f1_html += f'<td style="border:1px solid {_border};{_dbl_b}padding:4px;">{_v}</td>'
                _f1_html += '</tr>'
            _f1_html += '</table>'
            _f1_html += f'<div style="text-align:right;font-size:0.8em;margin-top:4px;color:#555;">(단위 : 호, 점)</div>'
            _f1_html += '</div>'
            st.markdown(_f1_html, unsafe_allow_html=True)
            _f1_xl = df_to_excel_bytes(_item_df)
            st.download_button("📥 항목별 결과 Excel 다운로드", _f1_xl,
                               file_name="항목별_결과.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.markdown("---")

        # ── ② 계약종별 분석 ──
        if M["contract"]:
            st.markdown('<p class="sec-head">📋 계약종별 만족도 분석</p>', unsafe_allow_html=True)
            _render_category_section(df_f, M["contract"], "계약종별",
                                     M["office"], "_점수100", avg_score_100)
            st.markdown("---")

        # ── ③ 업무유형별 분석 ──
        if M["business"]:
            st.markdown('<p class="sec-head">🏢 업무유형별 만족도 분석</p>', unsafe_allow_html=True)
            _render_category_section(df_f, M["business"], "업무유형",
                                     M["office"], "_점수100", avg_score_100)
            st.markdown("---")

        # ── ⑤ 업무유형별 사분면 버블 차트 ──
        if M["business"] and M["score"]:
            st.markdown('<p class="sec-head">⭕ 업무유형별 사분면 분석 — 처리 건수 × 평균 만족도 (버블 크기 = 불만족 건수)</p>', unsafe_allow_html=True)
            _bbl_grp = df_f.groupby(M["business"]).agg(
                처리건수=("_점수100", "count"),
                평균만족도=("_점수100", "mean"),
            ).reset_index()
            _bbl_grp.columns = ["업무유형", "처리건수", "평균만족도"]
            _bbl_grp["평균만족도"] = _bbl_grp["평균만족도"].round(1)
            _bbl_low = df_f[df_f["_점수100"] < 50].groupby(M["business"]).size().reset_index(name="불만족건수")
            _bbl_grp = _bbl_grp.merge(_bbl_low, left_on="업무유형", right_on=M["business"], how="left")
            if M["business"] in _bbl_grp.columns and M["business"] != "업무유형":
                _bbl_grp.drop(columns=[M["business"]], inplace=True)
            _bbl_grp["불만족건수"] = _bbl_grp["불만족건수"].fillna(0).astype(int)
            _bbl_grp["버블크기"] = _bbl_grp["불만족건수"] + 1

            # ── 사분면 기준값 ──
            _q_x_avg = _bbl_grp["처리건수"].mean()
            _q_y_avg = _bbl_grp["평균만족도"].mean()
            _total_n = _bbl_grp["처리건수"].sum()

            # 사분면 분류
            _q_names = {(True, True): "1사분면: 본부 견인형",
                        (False, True): "2사분면: 전문 특화형",
                        (False, False): "3사분면: 개별 개선형",
                        (True, False): "4사분면: 최우선 혁신"}
            _q_colors = {"1사분면: 본부 견인형": C["green"],
                         "2사분면: 전문 특화형": C["blue"],
                         "3사분면: 개별 개선형": C["gold"],
                         "4사분면: 최우선 혁신": C["red"]}
            _bbl_grp["사분면"] = _bbl_grp.apply(
                lambda r: _q_names[(r["처리건수"] >= _q_x_avg, r["평균만족도"] >= _q_y_avg)], axis=1)

            # 가중치: 해당 업무가 전체 평균을 얼마나 깎는지 (건수 비중 × 평균 차이)
            _bbl_grp["건수비중"] = (_bbl_grp["처리건수"] / _total_n * 100).round(1)
            _bbl_grp["점수갭"] = (_bbl_grp["평균만족도"] - _q_y_avg).round(1)
            _bbl_grp["가중영향"] = (_bbl_grp["처리건수"] / _total_n * _bbl_grp["점수갭"]).round(2)

            # 불만족 키워드 TOP3 (VOC 있을 때)
            _biz_kw_map = {}
            if M.get("voc"):
                _low_voc = df_f[df_f["_점수100"] < 70]
                for biz in _bbl_grp["업무유형"]:
                    _texts = _low_voc.loc[_low_voc[M["business"]] == biz, M["voc"]].dropna().astype(str).tolist()
                    _texts = [t for t in _texts if t.strip() not in ("", "nan", "응답없음", "없음", "의견없음", "빈문서", "빈항목") and len(t.strip()) > 2]
                    if _texts:
                        kws = extract_action_keywords(tuple(_texts), top_n=3)
                        _biz_kw_map[biz] = [w for w, s, c in kws][:3]

            # ── 지사별 업무 점수 (hover용: 하위3·우수1, 최소 응답 기준 적용) ──
            _MIN_N = 10  # 신뢰 가능한 최소 응답 건수
            _biz_ofc_stats = {}
            if M.get("office"):
                for biz in _bbl_grp["업무유형"]:
                    _bdf_raw = df_f[df_f[M["business"]] == biz].groupby(M["office"])["_점수100"]
                    _bdf_mean = _bdf_raw.mean().dropna()
                    _bdf_cnt = _bdf_raw.count()
                    if _bdf_mean.empty:
                        continue
                    # 신뢰 지사(N>=기준) vs 모수 부족 지사 분리
                    _reliable = _bdf_mean[_bdf_cnt >= _MIN_N]
                    _unreliable = _bdf_mean[_bdf_cnt < _MIN_N]
                    _sorted_r = _reliable.sort_values() if not _reliable.empty else pd.Series(dtype=float)
                    _bot3 = []
                    for n, v in (_sorted_r.head(3).items() if not _sorted_r.empty else []):
                        _bot3.append((n, round(v, 1), int(_bdf_cnt.get(n, 0))))
                    _top1 = None
                    if not _sorted_r.empty:
                        _t = _sorted_r.tail(1)
                        _top1 = (_t.index[0], round(_t.values[0], 1), int(_bdf_cnt.get(_t.index[0], 0)))
                    _unreliable_list = []
                    for n, v in _unreliable.items():
                        _unreliable_list.append((n, round(v, 1), int(_bdf_cnt.get(n, 0))))
                    _biz_ofc_stats[biz] = {"bottom3": _bot3, "top1": _top1, "unreliable": _unreliable_list}

            # ── Hover 텍스트 ──
            _hover_texts = []
            for _, row in _bbl_grp.iterrows():
                biz = row["업무유형"]
                _n = int(row["처리건수"])
                _dissat_pct = round(row["불만족건수"] / _n * 100, 1) if _n > 0 else 0

                _h = f"<b>📌 {biz}</b><br><br>"
                _h += f"<b>[데이터 현황]</b><br>"
                _h += f"처리건수 {_n:,}건 / 만족도 {row['평균만족도']:.1f}점 (본부 평균 대비 {row['점수갭']:+.1f}점)<br><br>"
                _h += f"<b>[리스크 진단]</b><br>"
                _h += f"불만족 응답 비중 {_dissat_pct}% ({int(row['불만족건수'])}건) · 전체 점수 영향 {row['가중영향']:+.2f}점<br><br>"

                kws = _biz_kw_map.get(biz, [])
                if kws:
                    _h += f"<b>[현장 목소리]</b><br>"
                    _h += f"불만 키워드 TOP3: {', '.join(kws)}<br><br>"

                _stats = _biz_ofc_stats.get(biz)
                if _stats and _stats["bottom3"]:
                    _bot_str = " / ".join([f"{n}({v}점, N={c})" for n, v, c in _stats["bottom3"]])
                    _h += f"<b>[관리 포인트]</b><br>"
                    _h += f"하위 지사: {_bot_str}<br>"
                    if _stats.get("unreliable"):
                        _ur_str = ", ".join([f"{n}({v}점, N={c}) ⚠신뢰도 낮음" for n, v, c in _stats["unreliable"]])
                        _h += f"모수 부족: {_ur_str}<br>"
                    _h += "<br>"

                if _stats and _stats.get("top1"):
                    _best_name, _best_score, _best_n = _stats["top1"]
                    if _best_n >= _MIN_N:
                        _h += f"<b>[한 줄 처방]</b><br>"
                        _h += f"{_best_name}({_best_score}점, N={_best_n})의 우수 사례를 참고하여 프로세스 개선 권고<br>"
                    else:
                        _h += f"<b>[한 줄 처방]</b><br>"
                        _h += f"{_best_name}({_best_score}점, N={_best_n}) [신뢰도: 매우 낮음] — 모수 확보 후 재평가 필요<br>"

                if "최우선" in row["사분면"]:
                    _h += f"<br>🚨 <b>[최우선 혁신 대상]</b><br>"
                    _h += f"건수 多 + 점수 低 → TF 가동 · 시스템 개선 우선 배치"
                _hover_texts.append(_h)

            # ── 차트 그리기 (사분면별 trace) ──
            fig_bbl = go.Figure()
            _q_order = ["1사분면: 본부 견인형", "2사분면: 전문 특화형", "3사분면: 개별 개선형", "4사분면: 최우선 혁신"]

            # 라벨 겹침 방지: 텍스트 너비 고려 + 데이터포인트 회피
            _ann_boxes = []   # (cx, cy, half_w, half_h) — 라벨 바운딩 박스
            _dot_positions = []  # 데이터 포인트 픽셀 좌표
            _bbl_annotations = []
            _x_range = max(_bbl_grp["처리건수"].max() - _bbl_grp["처리건수"].min(), 1)
            _y_range = max(_bbl_grp["평균만족도"].max() - _bbl_grp["평균만족도"].min(), 1)
            _CHART_W, _CHART_H = 800, 450  # 논리 차트 크기

            # 후보 오프셋: 아래 우선 (도형 밑에 라벨) → 겹칠 때만 주변으로
            _cand_offsets = [
                (0, 20),                                    # 1순위: 바로 아래
                (-30, 20), (30, 20),                        # 좌하, 우하
                (0, -20),                                   # 위
                (-30, -20), (30, -20),                      # 좌상, 우상
                (0, 35), (-40, 35), (40, 35),               # 더 아래
                (0, -35), (-40, -35), (40, -35),            # 더 위
                (-60, 15), (60, 15), (-60, -15), (60, -15), # 좌우
                (0, 50), (-50, 50), (50, 50),               # 원거리 아래
                (0, -50), (-50, -50), (50, -50),            # 원거리 위
                (-80, 0), (80, 0),                          # 좌우 수평
            ]

            # 모든 trace 먼저 그리기
            _all_rows = []
            for q_name in _q_order:
                _sub = _bbl_grp[_bbl_grp["사분면"] == q_name]
                if _sub.empty:
                    continue
                _sub_idx = _sub.index.tolist()
                fig_bbl.add_trace(go.Scatter(
                    x=_sub["처리건수"], y=_sub["평균만족도"],
                    mode="markers",
                    marker=dict(size=(_sub["버블크기"] * 8).clip(lower=22, upper=90),
                                color=_q_colors[q_name], opacity=0.7,
                                line=dict(width=2, color="white")),
                    hovertext=[_hover_texts[i] for i in _sub_idx],
                    hovertemplate="%{hovertext}<extra></extra>",
                    name=q_name,
                ))
                for _, _r in _sub.iterrows():
                    _all_rows.append(_r)

            # 데이터 포인트 픽셀 좌표 미리 계산
            _x_min_val = _bbl_grp["처리건수"].min()
            _y_min_val = _bbl_grp["평균만족도"].min()
            for _r in _all_rows:
                _dpx = (_r["처리건수"] - _x_min_val) / _x_range * _CHART_W
                _dpy = (1 - (_r["평균만족도"] - _y_min_val) / _y_range) * _CHART_H
                _dot_positions.append((_dpx, _dpy))

            # 밀집도 높은 버블 먼저 배치
            def _neighbor_cnt(row):
                cnt = 0
                for other in _all_rows:
                    if row.name == other.name:
                        continue
                    if abs(row["처리건수"] - other["처리건수"]) / _x_range < 0.25 and \
                       abs(row["평균만족도"] - other["평균만족도"]) / _y_range < 0.25:
                        cnt += 1
                return cnt
            _all_rows.sort(key=lambda r: _neighbor_cnt(r), reverse=True)

            def _boxes_overlap(cx1, cy1, hw1, hh1, cx2, cy2, hw2, hh2, pad=6):
                """두 라벨 박스가 겹치는지 (패딩 포함)"""
                return (abs(cx1 - cx2) < hw1 + hw2 + pad) and (abs(cy1 - cy2) < hh1 + hh2 + pad)

            for _r in _all_rows:
                _rx = _r["처리건수"]
                _ry = _r["평균만족도"]
                _base_px = (_rx - _x_min_val) / _x_range * _CHART_W
                _base_py = (1 - (_ry - _y_min_val) / _y_range) * _CHART_H
                # 텍스트 크기 추정 (한글 1자 ≈ 6px wide at font 11, 높이 ≈ 14px)
                _txt_len = len(_r["업무유형"])
                _half_w = max(_txt_len * 6, 25)
                _half_h = 8
                # 버블 크기(px) 추정 — 라벨이 버블 바깥에 붙도록
                _bbl_r = max(float(_r["버블크기"]) * 2.5, 7)
                _best_ax, _best_ay = _cand_offsets[0]
                _best_score = -99999
                for _cax, _cay in _cand_offsets:
                    _lbl_cx = _base_px + _cax
                    _lbl_cy = _base_py + _cay
                    # 1) 기존 라벨과 겹침 체크
                    _overlap = False
                    for _ocx, _ocy, _ohw, _ohh in _ann_boxes:
                        if _boxes_overlap(_lbl_cx, _lbl_cy, _half_w, _half_h, _ocx, _ocy, _ohw, _ohh):
                            _overlap = True
                            break
                    if _overlap:
                        continue
                    # 2) 점수: 가까울수록 좋음 (버블 근접 배치)
                    _dist_penalty = (abs(_cax) + abs(_cay)) * 0.5
                    _score = 100 - _dist_penalty
                    if _score > _best_score:
                        _best_score = _score
                        _best_ax, _best_ay = _cax, _cay
                _final_cx = _base_px + _best_ax
                _final_cy = _base_py + _best_ay
                _ann_boxes.append((_final_cx, _final_cy, _half_w, _half_h))
                # 화살표 항상 표시
                _bbl_annotations.append(dict(
                    x=_rx, y=_ry,
                    text=_r["업무유형"],
                    showarrow=True,
                    arrowhead=2, arrowwidth=1.2, arrowcolor="#000",
                    ax=_best_ax, ay=_best_ay,
                    font=dict(size=13, color="#222"),
                    bgcolor="rgba(255,255,255,0)", borderpad=1,
                ))

            # 사분면 기준선
            fig_bbl.add_hline(y=_q_y_avg, line_color=C["navy"], line_dash="dot", line_width=1,
                              annotation_text=f"만족도 평균 {_q_y_avg:.1f}", annotation_font_size=10,
                              annotation_position="top left")
            fig_bbl.add_vline(x=_q_x_avg, line_color=C["navy"], line_dash="dot", line_width=1,
                              annotation_text=f"건수 평균 {_q_x_avg:.0f}", annotation_font_size=10,
                              annotation_position="top right")

            # 사분면 영역 라벨
            _x_min, _x_max = _bbl_grp["처리건수"].min(), _bbl_grp["처리건수"].max()
            _y_min, _y_max = _bbl_grp["평균만족도"].min(), _bbl_grp["평균만족도"].max()
            _qlabels = [
                ((_q_x_avg + _x_max) / 2, _y_max + 0.5, "★ 본부 견인형", C["green"]),
                ((_x_min + _q_x_avg) / 2, _y_max + 0.5, "전문 특화형", C["blue"]),
                ((_x_min + _q_x_avg) / 2, _y_min - 1.0, "개별 개선형", C["gold"]),
                ((_q_x_avg + _x_max) / 2, _y_min - 1.0, "🚨 최우선 혁신", C["red"]),
            ]
            for _qx, _qy, _qt, _qc in _qlabels:
                fig_bbl.add_annotation(x=_qx, y=_qy, text=_qt, showarrow=False,
                                       font=dict(color=_qc, size=12, weight="bold"), xanchor="center")

            # 라벨 annotations 추가
            for _ann in _bbl_annotations:
                fig_bbl.add_annotation(**_ann)

            # y축 여유 (라벨이 잘리지 않도록)
            _y_pad = _y_range * 0.12
            fig_bbl.update_layout(
                height=640, margin=dict(t=60, b=60, l=60, r=60),
                xaxis_title="처리 건수", yaxis_title="평균 만족도 (100점 환산)",
                yaxis=dict(range=[_bbl_grp["평균만족도"].min() - _y_pad - 2,
                                  _bbl_grp["평균만족도"].max() + _y_pad + 2]),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=14)),
                template=PLOTLY_TPL)
            st.plotly_chart(fig_bbl, use_container_width=True)

            # ── 사분면별 분류 카드 ──
            _q1_list = _bbl_grp[_bbl_grp["사분면"].str.contains("견인")].sort_values("처리건수", ascending=False)
            _q2_list = _bbl_grp[_bbl_grp["사분면"].str.contains("특화")].sort_values("평균만족도", ascending=False)
            _q3_list = _bbl_grp[_bbl_grp["사분면"].str.contains("개별")].sort_values("평균만족도")
            _q4_list = _bbl_grp[_bbl_grp["사분면"].str.contains("최우선")].sort_values("가중영향")

            _col_l, _col_r = st.columns(2)
            with _col_l:
                if not _q1_list.empty:
                    _q1_names = ", ".join(_q1_list["업무유형"].tolist())
                    st.success(f"**★ 본부 견인형** (건수↑ 만족도↑ · 효자 업무): {_q1_names}")
                if not _q2_list.empty:
                    _q2_names = ", ".join(_q2_list["업무유형"].tolist())
                    st.info(f"**전문 특화형** (건수↓ 만족도↑ · 잘하는 업무): {_q2_names}")
            with _col_r:
                if not _q3_list.empty:
                    _q3_names = ", ".join(_q3_list["업무유형"].tolist())
                    st.warning(f"**개별 개선형** (건수↓ 점수↓ · 소외 업무): {_q3_names}")
                if not _q4_list.empty:
                    _q4_names = ", ".join(_q4_list["업무유형"].tolist())
                    st.error(f"**🚨 최우선 혁신** (건수↑ 점수↓ · 핵심 리스크): {_q4_names}")

            # ── AI 사분면 분석 버튼 ──
            _q_ss_key = "_ai_quadrant_result"
            if st.button("🤖 AI 업무유형 처방전", key="ai_quadrant_btn", type="primary", use_container_width=True):
                if not GEMINI_AVAILABLE:
                    st.error("Gemini API 키가 설정되지 않았습니다. `.env` 파일에 `GEMINI_API_KEY`를 설정해주세요.")
                else:
                    # 사분면별 데이터 요약
                    _ai_q_data = f"전체 평균 만족도: {_q_y_avg:.1f}점 / 전체 평균 처리건수: {_q_x_avg:.0f}건\n\n"
                    for q_label in ["1사분면: 본부 견인형", "2사분면: 전문 특화형", "3사분면: 개별 개선형", "4사분면: 최우선 혁신"]:
                        _q_sub = _bbl_grp[_bbl_grp["사분면"] == q_label]
                        if not _q_sub.empty:
                            _ai_q_data += f"[{q_label}]\n"
                            for _, qr in _q_sub.iterrows():
                                _kws = _biz_kw_map.get(qr["업무유형"], [])
                                _kw_txt = f" / 불만 키워드: {', '.join(_kws)}" if _kws else ""
                                _ai_q_data += f"  {qr['업무유형']}: {qr['평균만족도']:.1f}점, {int(qr['처리건수'])}건(비중 {qr['건수비중']}%), 불만족 {int(qr['불만족건수'])}건, 가중영향 {qr['가중영향']:+.2f}점{_kw_txt}\n"
                            _ai_q_data += "\n"

                    # 3·4사분면 VOC 수집
                    _ai_q_voc = []
                    if M.get("voc"):
                        for _, qr in _bbl_grp[_bbl_grp["사분면"].str.contains("최우선|개별")].iterrows():
                            _biz = qr["업무유형"]
                            _vocs = df_f[(df_f[M["business"]] == _biz) & (df_f["_점수100"] < 70)][M["voc"]].dropna().astype(str).tolist()
                            _vocs = [v for v in _vocs if v.strip() not in ("", "nan", "응답없음", "없음", "의견없음", "빈문서", "빈항목") and len(v.strip()) > 2][:5]
                            if _vocs:
                                _ai_q_voc.append(f"[{_biz}] ({qr['사분면']})")
                                for v in _vocs:
                                    _ai_q_voc.append(f"  - {v}")

                    _ai_q_prompt = f"""당신은 한국전력 경남본부 소속 **본부 CS 실무 코디네이터**입니다.
현장 지사를 총괄하는 본부 관점에서, 업무유형별 사분면 분석 결과를 보고 실행 가능한 처방을 내립니다.

[원칙]
- "TF 가동", "전사적 혁신", "시스템 전면 개편" 같은 비현실적 제안 금지.
- 본부가 지사에 실제로 지시·권고할 수 있는 수준의 액션만 제시.
- 예: 상담 스크립트 수정, 콜백 절차 추가, 월간 모니터링 항목 신설, 우수지사 사례 회람, 안내문 템플릿 개선 등.
- 이 데이터는 전월 1개월치 실적이며, 월초에 방향을 잡는 용도임. "이번 주 당장" 같은 급박한 톤 금지.
- 절대 금지: 액션 뒤에 "(이번 주)", "(다음 주)", "(다음 달)", "(이번 달)" 등 괄호 시기 표기를 절대 붙이지 마세요.
- 절대 금지: "멘토링", "코칭", "직원 교육", "담당자 교육", "롤플레이", "모니터링 평가", "직원 평가" 등 특정 직원을 대상으로 삼는 듯한 제안. 점수가 낮다고 직원을 저격하는 것이 아님. 직원 개인의 역량을 탓하는 방향 금지.
- 솔루션 방향: 직원 업무를 방해하지 않는 선에서 프로세스·절차·템플릿·안내 체계를 개선하는 방식 위주로 제안. 직원이 추가 부담 없이 자연스럽게 따를 수 있는 구조적 변화에 집중.
- 각 업무별 솔루션은 반드시 2개씩 제시.
- 반드시 업무명·VOC 키워드·수치 근거를 명시.

[이번 달 사분면 데이터]
{_ai_q_data}
[3·4사분면 업무의 불만족 VOC 원문]
{chr(10).join(_ai_q_voc) if _ai_q_voc else '- 해당 없음'}

[출력 형식 — 아래 4단계를 순서대로. 모든 분석은 오직 이번 달 데이터와 VOC만 근거로 사용할 것.]
#### 1. 중점 관리 업무 (4사분면: 건수↑ 점수↓)
- 해당 업무가 본부 평균을 얼마나 깎는지 수치로 진단
- VOC에서 드러나는 핵심 불만 포인트
- 개선 솔루션 2개

#### 2. 방치 리스크 (3사분면: 건수↓ 점수↓)
- 소량이라도 점수가 낮은 업무가 민원으로 번질 가능성
- 프로세스·안내 체계 개선 솔루션 2개

#### 3. 우수 사례 활용 (1사분면: 건수↑ 점수↑)
- 잘하는 업무의 강점을 구체적으로 짚고
- 다른 업무에 전파할 현실적 방법 2개

#### 4. 이번 달 개선 우선순위
- 본부 점수 상승 기여도 순으로 업무 나열
- 각 업무별 솔루션 2개씩

[필수 규칙]
- '전기세' 표현 절대 금지. 반드시 '전기요금'으로 표기.
- 줄글(산문체) 금지. 반드시 개조식(bullet, 짧은 문장) 보고서 형태로 작성.
- 한 bullet에 2줄 이상 금지. 핵심만 짧게."""

                    with st.spinner("AI가 업무유형별 처방전 생성 중…"):
                        try:
                            import urllib.request
                            _models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemma-3-12b-it"]
                            _payload = {"contents": [{"parts": [{"text": _ai_q_prompt}]}],
                                         "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192}}
                            _ctx = ssl._create_unverified_context()
                            _body = None
                            for _model in _models:
                                _api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{_model}:generateContent?key={_GEMINI_KEY}"
                                _req = urllib.request.Request(_api_url, data=json.dumps(_payload).encode("utf-8"),
                                                               headers={"Content-Type": "application/json"}, method="POST")
                                try:
                                    with urllib.request.urlopen(_req, context=_ctx, timeout=90) as _resp:
                                        _body = json.loads(_resp.read().decode("utf-8"))
                                    break
                                except urllib.error.HTTPError as _http_err:
                                    if _http_err.code in (429, 503):
                                        continue
                                    raise
                            if _body is None:
                                st.error("모든 AI 모델의 일일 한도가 소진되었습니다. 내일 다시 시도해주세요.")
                            else:
                                st.session_state[_q_ss_key] = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
                        except Exception as e:
                            st.error(f"AI 분석 중 오류: {e}")
            # 캐시된 결과 표시 (리런 후에도 유지)
            if _q_ss_key in st.session_state:
                st.markdown(
                    '<div style="background:#f8f9fb;border:1px solid #d0d7de;border-radius:10px;'
                    'padding:24px 28px;margin:8px 0;font-size:0.93em;line-height:1.85;">\n\n'
                    f'{st.session_state[_q_ss_key]}\n\n</div>',
                    unsafe_allow_html=True)

        st.markdown("---")


#  TAB 5  민원 조기 경보 시스템
# ─────────────────────────────────────────────────────────────
with tab5:
    #  SECTION B. 잠재 민원고객 사전케어 리스트
    # ══════════════════════════════════════════════════════════════
    st.markdown('<p class="sec-head">📋 잠재적 민원고객 사전케어 리스트 (AI 자동 추출)</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-red"><b>📌 추출 기준</b><br>'
        '① 종합 점수 50점 이하<br>'
        '② 서술 의견에 부정적 키워드가 포함<br>'
        '두 조건을 <b>모두 충족</b>하는 고객만 잠재 민원고객으로 추출합니다.<br>'
        '해당 고객에게 <b>72시간 이내</b> 선제적으로 연락하여 민원 발생을 사전에 차단하세요.</div>',
        unsafe_allow_html=True)

    if not M["voc"]:
        st.warning("VOC 컬럼을 선택해야 리스트를 추출할 수 있습니다.")
    else:
        with st.spinner("잠재 민원고객 추출 중…"):
            neg_res = df_f[M["voc"]].apply(check_negative)
            neg_kw_s = neg_res.apply(lambda x: ", ".join(x[1]) if x[1] else "")
            # 조건1: 종합 점수 50점 이하
            low_score_mask = pd.Series(False, index=df_f.index)
            if M["score"] and "_점수100" in df_f.columns:
                low_score_mask = df_f["_점수100"] <= 50
            # 조건2: 부정 키워드 감지
            neg_kw_mask = neg_res.apply(lambda x: x[0])
            # 교집합 (두 조건 모두 충족)
            neg_mask = low_score_mask & neg_kw_mask

        df_neg = df_f[neg_mask].copy()
        df_neg["감지된_부정키워드"] = neg_kw_s[neg_mask].values
        # 추출 유형 표시 (교집합이므로 항상 두 조건 모두 충족)
        df_neg["추출유형"] = "50점이하 + 부정키워드"
        neg_n = len(df_neg)
        neg_r = neg_n / max(len(df_f), 1) * 100

        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            st.metric("🚨 잠재 민원고객 수", f"{neg_n:,}명")
        with nc2:
            st.metric("📊 전체 대비 비율", f"{neg_r:.1f}%",
                      delta=f"긍정 {100 - neg_r:.1f}%", delta_color="normal")
        with nc3:
            if M["score"] and neg_n > 0:
                neg_avg = df_neg["_점수100"].mean() if "_점수100" in df_neg.columns else 0
                st.metric("⭐ 민원고객 평균", f"{neg_avg:.1f}점",
                          delta=f"{neg_avg - avg_score_100:+.1f} vs 전체", delta_color="inverse")

        if neg_n > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            # 불만 유형별 전문 용어 우선 추출 (일반 명사 제외)
            _DOMAIN_KW = [
                "요금","검침","대기","절차","불친절","고지서","정전","단전","누전",
                "과금","과다","연체","체납","감면","청구","납부",
                "계량기","변압기","전주","전선","차단기","개폐기",
                "재방문","지연","지체","방치","미처리","누락","중복",
                "위험","안전","사고","화재","고장","불량","오작동",
                "해지","폐전","명의변경","이전","신증설",
                "민원","항의","실망","최악","부당","불합리",
            ]
            _GENERIC_EXCLUDE = {"문의","직원","고객","전화","상담","안내","처리",
                                "접수","신청","확인","이용","서비스","방문","센터",
                                "담당","답변","진행","완료","내용","빈문서","응답없음"}
            all_neg_flat = []
            for voc_text in df_neg[M["voc"]]:
                s = str(voc_text).strip()
                if not s or s in ("nan", "응답없음", ""):
                    continue
                # 1순위: 도메인 전문 키워드 매칭
                for dkw in _DOMAIN_KW:
                    if dkw in s:
                        all_neg_flat.append(dkw)
                # 2순위: 감지된 부정키워드 중 일반명사 제외
            for kws in df_neg["감지된_부정키워드"]:
                for k in kws.split(","):
                    k = k.strip()
                    if k and k not in _GENERIC_EXCLUDE:
                        all_neg_flat.append(k)
            neg_kw_cnt = Counter(all_neg_flat).most_common(10)
            if neg_kw_cnt:
                nkw_df = pd.DataFrame(neg_kw_cnt, columns=["부정키워드", "감지횟수"])
                nk_l, nk_r = st.columns([3, 2])
                with nk_l:
                    fig_neg = px.bar(nkw_df, x="부정키워드", y="감지횟수", color_discrete_sequence=[C["red"]],
                                     text="감지횟수", template=PLOTLY_TPL, title="불만 유형별 핵심 키워드 Top 10")
                    fig_neg.update_traces(texttemplate="%{text}", textposition="outside",
                                          hovertemplate="%{x}: %{y}회<extra></extra>")
                    fig_neg.update_layout(height=340, margin=dict(t=50, b=70, l=60, r=20), xaxis_tickangle=-25,
                                           title_font=dict(size=14, color=C["navy"]))
                    st.plotly_chart(fig_neg, use_container_width=True, config={'staticPlot': True})
                with nk_r:
                    fig_don = px.pie(nkw_df.head(10), names="부정키워드", values="감지횟수", hole=0.5,
                                     color_discrete_sequence=px.colors.sequential.Reds[::-1],
                                     title="불만 유형별 키워드 비중", template=PLOTLY_TPL)
                    fig_don.update_traces(textinfo="percent+label", textfont_size=11,
                                           marker=dict(line=dict(color="white", width=2)),
                                           hovertemplate="%{label}<br>%{value}회 (%{percent})<extra></extra>")
                    fig_don.update_layout(height=340, margin=dict(t=50, b=20, l=20, r=20), showlegend=False,
                                           title_font=dict(size=14, color=C["navy"]))
                    st.plotly_chart(fig_don, use_container_width=True, config={'staticPlot': True})

            # 리스트 테이블
            display_cols = []
            # 순번·접수번호를 맨 앞에 배치
            if "_원본순번" in df_neg.columns:
                display_cols.append("_원본순번")
            if M["receipt_no"] and M["receipt_no"] in df_neg.columns:
                display_cols.append(M["receipt_no"])
            for key in ["id","name","contact","age","office","channel","contract","business","score","voc"]:
                if M[key] and M[key] in df_neg.columns:
                    display_cols.append(M[key])
            display_cols.extend(["감지된_부정키워드", "추출유형"])
            df_disp = df_neg[[c for c in display_cols if c in df_neg.columns]].reset_index(drop=True)
            if "_원본순번" in df_disp.columns:
                df_disp.rename(columns={"_원본순번": "순번"}, inplace=True)

            st.markdown(f'<p class="sec-head">📋 잠재 민원고객 — 총 <span style="color:{C["red"]}">{neg_n:,}명</span></p>',
                        unsafe_allow_html=True)
            st.dataframe(df_disp, use_container_width=True, height=440, hide_index=True)

            excel_bytes = df_to_excel_bytes(df_disp)
            st.download_button(label="📥  잠재 민원고객 엑셀 다운로드", data=excel_bytes,
                               file_name="잠재민원고객_사전케어리스트.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        else:
            st.markdown("""<div class="card-teal">🎉 <b>잠재 민원고객이 없습니다!</b><br>
현재 고객 만족 수준이 양호합니다.</div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  SECTION C. 민원 예측 AI 리포트
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<p class="sec-head">🔮 민원 예측 리포트 — 공식 민원 발전 가능성 TOP 3</p>', unsafe_allow_html=True)

    # 업무유형별 리스크 스코어 산정 (건수비중 × 불만족비율 × 반복성)
    if M.get("business") and M.get("score") and "_점수100" in df_f.columns:
        _pred_data = []
        _total_n_pred = len(df_f)
        for biz in df_f[M["business"]].dropna().unique():
            _bdf = df_f[df_f[M["business"]] == biz]
            _bn = len(_bdf)
            _b_low = (_bdf["_점수100"] < 50).sum()
            _b_avg = _bdf["_점수100"].mean()
            _b_dissat_pct = _b_low / max(_bn, 1) * 100
            _b_weight = _bn / max(_total_n_pred, 1)
            # 리스크 점수 = 건수비중 × 불만족비율 × (100 - 평균점수) / 100
            _risk_score = round(_b_weight * _b_dissat_pct * (100 - _b_avg) / 100, 2)
            _pred_data.append((biz, _bn, round(_b_avg, 1), _b_low, round(_b_dissat_pct, 1), _risk_score))

        _pred_data.sort(key=lambda x: -x[5])
        _top3_pred = _pred_data[:3]

        # ── 도메인 키워드 사전 (잠재민원 차트와 동일 방식) ──
        _PRED_DOMAIN_KW = [
            "요금","검침","대기","절차","불친절","고지서","정전","단전","누전",
            "과금","과다","연체","체납","감면","청구","납부",
            "계량기","변압기","전주","전선","차단기","개폐기",
            "재방문","지연","지체","방치","미처리","누락","중복",
            "위험","안전","사고","화재","고장","불량","오작동",
            "해지","폐전","명의변경","이전","신증설",
            "민원","항의","실망","최악","부당","불합리",
        ]
        _PRED_GENERIC_EXCLUDE = {"문의","직원","고객","전화","상담","안내","처리",
                                  "접수","신청","확인","이용","서비스","방문","센터",
                                  "담당","답변","진행","완료","내용","빈문서","응답없음"}

        if _top3_pred:
            for _rank, (_biz, _bn, _bavg, _blow, _bdp, _rsk) in enumerate(_top3_pred, 1):
                _icon = "🔴" if _rank == 1 else ("🟠" if _rank == 2 else "🟡")
                # 해당 업무의 불만 키워드 추출 (도메인 사전 방식)
                _biz_kws = []
                if M.get("voc"):
                    _biz_low_voc = df_f[(df_f[M["business"]] == _biz) & (df_f["_점수100"] < 70)]
                    _biz_texts = _biz_low_voc[M["voc"]].dropna().astype(str).tolist()
                    _biz_texts = [t for t in _biz_texts if t.strip() not in ("", "nan", "응답없음")]
                    _biz_kw_flat = []
                    for _t in _biz_texts:
                        for _dkw in _PRED_DOMAIN_KW:
                            if _dkw in _t:
                                _biz_kw_flat.append(_dkw)
                    _biz_kws = [w for w, _ in Counter(_biz_kw_flat).most_common(3)]
                _kw_str = f" · 핵심 키워드: **{', '.join(_biz_kws)}**" if _biz_kws else ""
                st.markdown(f"{_icon} **{_rank}위 — {_biz}** | 처리 {_bn:,}건 · 만족도 {_bavg}점 · 불만족 {_blow}건({_bdp}%) · 리스크 점수 {_rsk}{_kw_str}")

    else:
        st.info("업무유형 컬럼과 점수 컬럼이 필요합니다.")

    # ══════════════════════════════════════════════════════════════
    #  SECTION D. AI 본부 민원 리스크 종합 진단 (통합)
    # ══════════════════════════════════════════════════════════════
    if neg_n > 0:
        st.markdown("---")
        st.markdown('<p class="sec-head">⚠️ 본부 민원 리스크 종합 진단</p>', unsafe_allow_html=True)

        # 부정 VOC 샘플 수집 (AI 프롬프트용)
        _neg_voc_samples = []
        if M.get("voc"):
            _neg_voc_raw = df_neg[M["voc"]].dropna().astype(str).tolist()
            _neg_voc_samples = [t for t in _neg_voc_raw if t.strip() not in ("", "nan", "응답없음")][:30]
        _neg_kw_top = [k for k, _ in neg_kw_cnt] if neg_kw_cnt else []
        _neg_office_dist = ""
        if M.get("office") and M["office"] in df_neg.columns:
            _ofc_cnt = df_neg[M["office"]].value_counts().head(5)
            _neg_office_dist = ", ".join([f"{n}({v}건)" for n, v in _ofc_cnt.items()])
        _neg_biz_dist = ""
        if M.get("business") and M["business"] in df_neg.columns:
            _biz_cnt = df_neg[M["business"]].value_counts().head(5)
            _neg_biz_dist = ", ".join([f"{n}({v}건)" for n, v in _biz_cnt.items()])

        _pc_ss_key = "_ai_precare_result"
        if st.button("🤖 AI 본부 민원 리스크 종합 진단", key="ai_precare_btn", type="primary", use_container_width=True):
            if not GEMINI_AVAILABLE:
                st.error("Gemini API 키가 설정되지 않았습니다. `.env` 파일에 `GEMINI_API_KEY`를 설정해주세요.")
            else:
                # ── TOP3 리스크 업무 데이터 수집 ──
                _rc_top3_lines = ""
                _rc_top3_voc = ""
                if M.get("business") and M.get("score") and "_점수100" in df_f.columns:
                    _rc_data = []
                    _rc_total = len(df_f)
                    for _biz in df_f[M["business"]].dropna().unique():
                        _bdf = df_f[df_f[M["business"]] == _biz]
                        _bn = len(_bdf)
                        _b_low = (_bdf["_점수100"] < 50).sum()
                        _b_avg = _bdf["_점수100"].mean()
                        _b_dp = _b_low / max(_bn, 1) * 100
                        _b_w = _bn / max(_rc_total, 1)
                        _rsk = round(_b_w * _b_dp * (100 - _b_avg) / 100, 2)
                        _rc_data.append((_biz, _bn, round(_b_avg, 1), _b_low, round(_b_dp, 1), _rsk))
                    _rc_data.sort(key=lambda x: -x[5])
                    _rc_top3 = _rc_data[:3]
                    for _biz, _bn, _bavg, _blow, _bdp, _rsk in _rc_top3:
                        _rc_top3_lines += f"- {_biz}: 처리 {_bn}건, 만족도 {_bavg}점, 불만족 {_blow}건({_bdp}%), 리스크점수 {_rsk}\n"
                    if M.get("voc"):
                        for _biz, *_ in _rc_top3:
                            _bvocs = df_f[(df_f[M["business"]] == _biz) & (df_f["_점수100"] < 70)]
                            _samples = _bvocs[M["voc"]].dropna().astype(str).tolist()
                            _samples = [t for t in _samples if t.strip() not in ("", "nan", "응답없음")][:10]
                            if _samples:
                                _rc_top3_voc += f"\n▶ {_biz}:\n"
                                for _s in _samples:
                                    _rc_top3_voc += f"  - {_s}\n"

                with st.spinner("Gemini AI가 본부 민원 리스크를 종합 진단 중…"):
                    _unified_prompt = f"""당신은 전력산업 고객만족(CS) 전문 컨설턴트입니다.
아래는 경남본부 고객 만족도 조사에서 추출된 데이터입니다.

[잠재 민원고객 현황]
- 총 {neg_n}명 (전체 대비 {neg_r:.1f}%)
- 민원고객 평균 점수: {df_neg["_점수100"].mean():.1f}점 / 전체 평균: {avg_score_100:.1f}점
- 부정 키워드 TOP10: {', '.join(_neg_kw_top)}
- 민원 집중 지사: {_neg_office_dist if _neg_office_dist else '정보 없음'}
- 민원 집중 업무: {_neg_biz_dist if _neg_biz_dist else '정보 없음'}

[리스크 TOP 3 업무]
{_rc_top3_lines if _rc_top3_lines else '- 데이터 없음'}

[업무별 불만족 VOC 원문]
{_rc_top3_voc if _rc_top3_voc else '- VOC 데이터 없음'}

[잠재 민원고객 부정 VOC 원문 (최대 30건)]
{chr(10).join([f'- {v}' for v in _neg_voc_samples]) if _neg_voc_samples else '- VOC 데이터 없음'}

# 출력 형식 (반드시 아래 3개 항목만 순서대로 출력. 추가 항목 금지)

#### 1. 부정 VOC 핵심 패턴
- 위 부정 키워드 TOP10과 VOC 원문에서 읽히는 본부 차원 공통 불만 패턴 2~3가지
- 어떤 업무·지사에 집중되는지 교차 분석
- 수치(건수, 점수, 비율) 근거 필수

#### 2. 리스크 TOP3 업무 진단
- 업무별 민원 발전 가능성 (상/중/하) + VOC 원문 기반 구체적 근거
- 가장 시급한 1개 업무의 최악 시나리오 한 줄

#### 3. 선제 대응 액션플랜
- 업무별 즉시 실행 가능한 프로세스·안내 체계 개선 조치 1~2개
- 예산·본사 승인 불필요한 '행동' 위주

[필수 규칙]
- 줄글(산문체) 절대 금지. 반드시 개조식 bullet 보고서 형태로 작성.
- 한 bullet에 1~2문장 이내. 핵심만 짧게 끊어 쓸 것.
- '전기세' 표현 절대 금지 → 반드시 '전기요금'으로 표기.
- 뻔한 "친절 교육 실시", "매뉴얼 배포" 같은 추상적 제안은 절대 금지.
- "멘토링", "코칭", "직원 교육", "롤플레이", "모니터링 평가" 등 직원 저격성 제안 금지.
- 반드시 위 데이터의 키워드·지사명·업무명·수치를 근거로 구체적으로 작성."""

                    try:
                        import urllib.request
                        _models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemma-3-12b-it"]
                        _payload = {"contents": [{"parts": [{"text": _unified_prompt}]}],
                                     "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192}}
                        _ctx = ssl._create_unverified_context()
                        _body = None
                        for _model in _models:
                            _api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{_model}:generateContent?key={_GEMINI_KEY}"
                            _req = urllib.request.Request(_api_url, data=json.dumps(_payload).encode("utf-8"),
                                                           headers={"Content-Type": "application/json"}, method="POST")
                            try:
                                with urllib.request.urlopen(_req, context=_ctx, timeout=90) as _resp:
                                    _body = json.loads(_resp.read().decode("utf-8"))
                                break
                            except urllib.error.HTTPError as _http_err:
                                if _http_err.code in (429, 503):
                                    continue
                                raise
                        if _body is None:
                            st.error("모든 AI 모델의 일일 한도가 소진되었습니다. 내일 다시 시도해주세요.")
                        else:
                            st.session_state[_pc_ss_key] = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
                    except Exception as e:
                        st.error(f"AI 분석 중 오류가 발생했습니다: {e}")
        # 캐시된 결과 표시 (리런 후에도 유지)
        if _pc_ss_key in st.session_state:
            st.markdown(
                '<div style="background:#f8f9fb;border:1px solid #d0d7de;border-radius:10px;'
                'padding:24px 28px;margin:8px 0;font-size:0.93em;line-height:1.85;">\n\n'
                f'{st.session_state[_pc_ss_key]}\n\n</div>',
                unsafe_allow_html=True)
        else:
            st.caption("버튼을 누르면 Gemini AI가 부정 VOC 패턴 + 리스크 업무를 종합 진단합니다.")


# ─────────────────────────────────────────────────────────────
#  TAB SOL  지사별 정밀 진단 · AI 심층 분석 (3-Level UX)
# ─────────────────────────────────────────────────────────────
with tab_sol:

    if df_f.empty or avg_score_100 is None:
        st.info("데이터를 먼저 업로드하세요.")
    else:
        # ── 지사 선택 ────────────────────────────────────────
        _sol_offices = _sort_offices(df_f[M["office"]].dropna().unique().tolist()) if M.get("office") else []
        if not _sol_offices:
            st.info("지사 컬럼이 필요합니다.")
            st.stop()

        st.markdown("#### 🏢 분석할 지사를 선택하세요")
        _sel_off = st.selectbox("", _sol_offices, key="sol_office_sel", label_visibility="collapsed")
        # 지사 변경 시 카드 선택 초기화
        if st.session_state.get("_sol_prev_off") != _sel_off:
            st.session_state["_sol_prev_off"] = _sel_off
            st.session_state.pop("sol_cell_sel", None)
        _df_sel  = df_f[df_f[M["office"]] == _sel_off].copy()

        if _df_sel.empty:
            st.warning(f"{_sel_off}의 데이터가 없습니다.")
            st.stop()

        # ══════════════════════════════════════════════════════
        # LEVEL 2 — 지사별 정밀 진단 (Diagnosis)
        # ══════════════════════════════════════════════════════

        # ── 피어 그룹 (고정 분류) ─────────────────────────────
        _PEER_LARGE = {"직할", "진주지사", "마산지사", "거제지사", "밀양지사", "사천지사", "통영지사", "함안의령지사", "거창지사", "창녕지사"}
        _PEER_SMALL = {"합천지사", "진해지사", "하동지사", "고성지사", "산청지사", "남해지사", "함양지사"}
        _off_counts = df_f.groupby(M["office"])["_점수100"].count().sort_values()
        if _sel_off in _PEER_LARGE:
            _peer_label, _peer_tier = "대규모", 2
        elif _sel_off in _PEER_SMALL:
            _peer_label, _peer_tier = "소규모", 0
        else:
            _peer_label, _peer_tier = "대규모" if _sel_off in {"경남본부"} else "소규모", 2 if _sel_off in {"경남본부"} else 0
        _peer_offs = [o for o in _off_counts.index
                      if (o in _PEER_LARGE if _peer_tier == 2 else o in _PEER_SMALL)]
        _peer_avg = df_f[df_f[M["office"]].isin(_peer_offs)]["_점수100"].mean()

        # ── 기본 지표 ─────────────────────────────────────────
        _sel_avg    = _df_sel["_점수100"].mean()
        _sel_cnt    = len(_df_sel)
        _sel_gap    = _sel_avg - avg_score_100
        _all_avgs   = df_f.groupby(M["office"])["_점수100"].mean().sort_values(ascending=False)
        _rank       = int(list(_all_avgs.index).index(_sel_off)) + 1 if _sel_off in _all_avgs.index else "-"
        _total_offs = len(_all_avgs)
        _peer_avgs_s = (df_f[df_f[M["office"]].isin(_peer_offs)]
                        .groupby(M["office"])["_점수100"].mean().sort_values(ascending=False))
        _peer_rank   = int(list(_peer_avgs_s.index).index(_sel_off)) + 1 if _sel_off in _peer_avgs_s.index else "-"
        _reliability = "✅ 높음" if _sel_cnt >= 50 else "🟡 보통" if _sel_cnt >= 20 else "⚠️ 낮음"

        # ── 핵심 진단 ─────────────────────────────────────────
        if M.get("business"):
            _biz_sel  = _df_sel.groupby(M["business"])["_점수100"].agg(["mean", "count"]).reset_index()
            _biz_sel.columns = ["업무유형", "평균만족도", "건수"]
            _biz_sel["전체대비"] = (_biz_sel["평균만족도"] - avg_score_100).round(1)
            _biz_worst = _biz_sel.sort_values("전체대비").iloc[0] if not _biz_sel.empty else None
            if _biz_worst is not None and _biz_worst["전체대비"] < -3:
                _diag = (f"전반적으로 {'우수' if _sel_gap >= 0 else '개선 필요'}하나, "
                         f"'{_biz_worst['업무유형']}' 업무가 본부 평균보다 {abs(_biz_worst['전체대비']):.1f}점 낮아 집중 개선 필요.")
            elif _sel_gap >= 0:
                _diag = f"본부 평균을 {_sel_gap:.1f}점 상회하는 우수 지사입니다."
            else:
                _diag = f"전반적 만족도가 본부 평균을 {abs(_sel_gap):.1f}점 하회합니다. 전방위 개선이 필요합니다."
        else:
            _diag = f"본부 평균 {'상회' if _sel_gap >= 0 else '하회'}({_sel_gap:+.1f}점)."

        # ── 헤더 카드 ─────────────────────────────────────────
        _gap_color = "#80cbc4" if _sel_gap >= 0 else "#ef9a9a"
        st.markdown(
            '<div style="background:linear-gradient(135deg,#1a237e 0%,#283593 60%,#1565c0 100%);'
            'border-radius:12px;padding:20px 24px;color:white;margin-bottom:20px;">'
            f'<div style="font-size:1.25em;font-weight:800;margin-bottom:14px;">📍 {_sel_off} · 종합 컨디션 리포트</div>'
            '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px;">'
            '<div style="background:rgba(255,255,255,0.13);border-radius:8px;padding:12px;text-align:center;">'
            '<div style="font-size:0.72em;opacity:0.8;margin-bottom:4px;">종합 만족도</div>'
            f'<div style="font-size:1.9em;font-weight:900;">{_sel_avg:.1f}<span style="font-size:0.45em;">점</span></div>'
            f'<div style="font-size:0.78em;color:{_gap_color};">{_sel_gap:+.1f}점 (본부 {avg_score_100:.1f})</div>'
            '</div>'
            '<div style="background:rgba(255,255,255,0.13);border-radius:8px;padding:12px;text-align:center;">'
            '<div style="font-size:0.72em;opacity:0.8;margin-bottom:4px;">본부 랭킹</div>'
            f'<div style="font-size:1.9em;font-weight:900;">{_rank}<span style="font-size:0.45em;">위</span></div>'
            f'<div style="font-size:0.78em;opacity:0.75;">/ {_total_offs}개 지사</div>'
            '</div>'
            '<div style="background:rgba(255,255,255,0.13);border-radius:8px;padding:12px;text-align:center;">'
            f'<div style="font-size:0.72em;opacity:0.8;margin-bottom:4px;">{_peer_label} 그룹 랭킹</div>'
            f'<div style="font-size:1.9em;font-weight:900;">{_peer_rank}<span style="font-size:0.45em;">위</span></div>'
            f'<div style="font-size:0.78em;opacity:0.75;">/ {len(_peer_offs)}개 · 피어평균 {_peer_avg:.1f}점</div>'
            '</div>'
            '<div style="background:rgba(255,255,255,0.13);border-radius:8px;padding:12px;text-align:center;">'
            '<div style="font-size:0.72em;opacity:0.8;margin-bottom:4px;">데이터 신뢰도</div>'
            f'<div style="font-size:1.2em;font-weight:700;margin:4px 0;">{_reliability}</div>'
            f'<div style="font-size:0.78em;opacity:0.75;">응답 {_sel_cnt:,}건</div>'
            '</div></div>'
            '<div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:10px 14px;'
            'font-size:0.9em;border-left:4px solid #80cbc4;">'
            f'💡 핵심 진단: "{_diag}"'
            '</div></div>',
            unsafe_allow_html=True)

        # ── 업무별 강점/약점 레이더 + 페르소나 미스매치 ──────────
        if M.get("business"):
            _sol_mid_l, _sol_mid_r = st.columns([1, 1])

            # ── 좌측: 레이더 차트 ──────────────────────────────
            with _sol_mid_l:
                st.markdown("##### 🎯 업무별 강점 / 약점 — 본부 평균과 비교")
                _biz_sel_avg = _df_sel.groupby(M["business"])["_점수100"].mean()
                _biz_all_avg = df_f.groupby(M["business"])["_점수100"].mean()
                _radar_cats = sorted(set(_biz_sel_avg.index) & set(_biz_all_avg.index))
                if len(_radar_cats) >= 3:
                    _r_sel = [round(_biz_sel_avg.get(c, 0), 1) for c in _radar_cats]
                    _r_all = [round(_biz_all_avg.get(c, 0), 1) for c in _radar_cats]
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=_r_sel + [_r_sel[0]], theta=_radar_cats + [_radar_cats[0]],
                        fill="toself", fillcolor="rgba(255,215,0,0.15)",
                        line=dict(color="#FFD700", width=2.5),
                        name=_sel_off,
                        hovertemplate="%{theta}: %{r:.1f}점<extra></extra>"))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=_r_all + [_r_all[0]], theta=_radar_cats + [_radar_cats[0]],
                        fill="toself", fillcolor="rgba(144,164,174,0.08)",
                        line=dict(color="#90a4ae", width=1.5, dash="dash"),
                        name="본부 평균",
                        hovertemplate="%{theta}: %{r:.1f}점<extra></extra>"))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(range=[60, 105], dtick=10, showticklabels=True, tickfont_size=9)),
                        template=PLOTLY_TPL, height=340,
                        margin=dict(t=30, b=30, l=60, r=60),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                        showlegend=True)
                    st.plotly_chart(fig_radar, use_container_width=True, config={'staticPlot': True})

                    # 강점/약점 TOP 3
                    _biz_gap = pd.DataFrame({
                        "업무": _radar_cats,
                        "지사": _r_sel[:len(_radar_cats)],
                        "본부": _r_all[:len(_radar_cats)]})
                    _biz_gap["편차"] = _biz_gap["지사"] - _biz_gap["본부"]
                    _strengths = _biz_gap.sort_values("편차", ascending=False).head(3)
                    _weaknesses = _biz_gap.sort_values("편차").head(3)
                    _sw_l, _sw_r = st.columns(2)
                    with _sw_l:
                        _s_html = '<div style="background:#e8f5e9;border-radius:8px;padding:10px 14px;font-size:0.88em;"><b>💪 강점 TOP 3</b><br>'
                        for _, r in _strengths.iterrows():
                            _s_html += f'🟢 {r["업무"]} — {r["지사"]:.1f}점 (본부 대비 <b>{r["편차"]:+.1f}</b>점)<br>'
                        st.markdown(_s_html + '</div>', unsafe_allow_html=True)
                    with _sw_r:
                        _w_html = '<div style="background:#ffebee;border-radius:8px;padding:10px 14px;font-size:0.88em;"><b>🩹 약점 TOP 3</b><br>'
                        for _, r in _weaknesses.iterrows():
                            _w_html += f'🔴 {r["업무"]} — {r["지사"]:.1f}점 (본부 대비 <b style="color:#c62828">{r["편차"]:+.1f}</b>점)<br>'
                        st.markdown(_w_html + '</div>', unsafe_allow_html=True)
                else:
                    st.info("업무유형이 3개 이상이어야 레이더 차트를 그릴 수 있습니다.")

            # ── 우측: 페르소나별 미스매치 매트릭스 ──────────────
            with _sol_mid_r:
                _ct_col = M.get("contract")
                if _ct_col and _ct_col in _df_sel.columns:
                    st.markdown("##### 🎯 고객군별 미스매치 — 건수 비중 × 만족도")
                    _pm_grp = _df_sel.groupby(_ct_col)["_점수100"].agg(["mean", "count"]).reset_index()
                    _pm_grp.columns = ["고객군", "만족도", "건수"]
                    _pm_grp = _pm_grp[_pm_grp["건수"] >= 2]
                    _pm_total = _pm_grp["건수"].sum()
                    _pm_grp["비중(%)"] = (_pm_grp["건수"] / max(_pm_total, 1) * 100).round(1)
                    if not _pm_grp.empty:
                        _pm_x_mid = _pm_grp["비중(%)"].mean()
                        _pm_y_mid = avg_score_100
                        fig_pm = go.Figure()
                        fig_pm.add_trace(go.Scatter(
                            x=_pm_grp["비중(%)"], y=_pm_grp["만족도"],
                            mode="markers+text",
                            marker=dict(
                                size=(_pm_grp["건수"] / max(_pm_grp["건수"].max(), 1) * 40 + 12).tolist(),
                                color=_pm_grp["만족도"].tolist(),
                                colorscale="RdYlGn", cmin=max(60, _pm_grp["만족도"].min() - 3),
                                cmax=min(100, _pm_grp["만족도"].max() + 3),
                                showscale=True, colorbar=dict(title="만족도", len=0.6),
                                line=dict(width=1, color="white")),
                            text=_pm_grp["고객군"],
                            textposition="top center", textfont_size=10,
                            customdata=list(zip(_pm_grp["만족도"].round(1), _pm_grp["건수"], _pm_grp["비중(%)"])),
                            hovertemplate="%{text}<br>만족도: %{customdata[0]:.1f}점<br>건수: %{customdata[1]}건<br>비중: %{customdata[2]:.1f}%<extra></extra>"))
                        fig_pm.add_vline(x=_pm_x_mid, line_dash="dash", line_color="#bdbdbd", line_width=1)
                        fig_pm.add_hline(y=_pm_y_mid, line_dash="dash", line_color="#bdbdbd", line_width=1)
                        # 사분면 라벨
                        _pm_x_range = [0, max(_pm_grp["비중(%)"].max() * 1.3, _pm_x_mid * 2)]
                        _pm_y_range = [max(55, _pm_grp["만족도"].min() - 5), min(108, _pm_grp["만족도"].max() + 8)]
                        fig_pm.add_annotation(x=_pm_x_range[1] * 0.85, y=_pm_y_range[0] + 2,
                                              text="⚠️ 1순위 개선", showarrow=False,
                                              font=dict(size=10, color="#c62828"))
                        fig_pm.add_annotation(x=1, y=_pm_y_range[0] + 2,
                                              text="🔍 특이 리스크", showarrow=False,
                                              font=dict(size=10, color="#e65100"))
                        fig_pm.update_layout(
                            template=PLOTLY_TPL, height=340,
                            margin=dict(t=40, b=50, l=50, r=20),
                            xaxis=dict(title="건수 비중(%)", range=_pm_x_range),
                            yaxis=dict(title="평균 만족도", range=_pm_y_range),
                            showlegend=False)
                        st.plotly_chart(fig_pm, use_container_width=True, config={'staticPlot': True})

                        # 우하 사분면 경고
                        _pm_danger = _pm_grp[(_pm_grp["비중(%)"] >= _pm_x_mid) & (_pm_grp["만족도"] < _pm_y_mid)]
                        _pm_watch  = _pm_grp[(_pm_grp["비중(%)"] < _pm_x_mid) & (_pm_grp["만족도"] < _pm_y_mid)]
                        if not _pm_danger.empty:
                            _d_names = ", ".join(_pm_danger["고객군"].tolist())
                            st.markdown(
                                f'<div style="background:#ffebee;border-radius:8px;padding:8px 12px;font-size:0.85em;">'
                                f'🚨 <b>1순위 개선:</b> {_d_names} — 비중이 높은데 만족도가 낮습니다.</div>',
                                unsafe_allow_html=True)
                        if not _pm_watch.empty:
                            _w_names = ", ".join(_pm_watch["고객군"].tolist())
                            st.markdown(
                                f'<div style="background:#fff8e1;border-radius:8px;padding:8px 12px;font-size:0.85em;">'
                                f'⚡ <b>특이 리스크:</b> {_w_names} — 소수이지만 점수가 극도로 낮아 민원 위험</div>',
                                unsafe_allow_html=True)
                    else:
                        st.info("계약종별 데이터가 부족합니다.")
                else:
                    st.info("계약종 컬럼이 설정되지 않았습니다.")

        st.markdown("---")

        # ── 업무 × 계약종별 정밀 진단 ──────────────────────
        if M.get("business") and M.get("contract"):
            st.markdown("##### 🔥 업무 × 계약종별 리스크 히트맵 — " + _sel_off)
            st.caption("업무유형(행) × 계약종별(열)의 평균 만족도입니다. **가장 빨간 칸**이 우선 관리 대상입니다. 하단 카드를 클릭하면 상세 분석이 열립니다.")

            _sol_pivot = _df_sel.pivot_table(
                index=M["business"], columns=M["contract"],
                values="_점수100", aggfunc="mean"
            ).round(1)

            # 응답호수 pivot
            _sol_cnt = _df_sel.pivot_table(
                index=M["business"], columns=M["contract"],
                values="_점수100", aggfunc="count"
            ).fillna(0).astype(int)
            # 업무유형별 합계
            _sol_score_total = _df_sel.groupby(M["business"])["_점수100"].mean().round(1)
            _sol_cnt_total = _df_sel.groupby(M["business"])["_점수100"].count().astype(int)

            if not _sol_pivot.empty and _sol_pivot.size > 0:
                # 계약종별 컬럼 순서 고정
                _hm_ct_order = ["주택용", "일반용", "산업용", "농사용", "교육용", "가로등"]
                _hm_ordered = [c for c in _hm_ct_order if c in _sol_pivot.columns]
                _hm_rest = [c for c in _sol_pivot.columns if c not in _hm_ct_order]
                _hm_cols = _hm_ordered + _hm_rest
                _sol_pivot = _sol_pivot.reindex(columns=_hm_cols)
                _sol_cnt = _sol_cnt.reindex(columns=_hm_cols).fillna(0).astype(int)
                # 업무유형 행: 점수순 정렬
                _sol_pivot = _sol_pivot.reindex(_sol_pivot.mean(axis=1).sort_values().index)
                _sol_cnt = _sol_cnt.reindex(_sol_pivot.index)

                _hm_vals = _sol_pivot.values.flatten()
                _hm_vals = _hm_vals[~np.isnan(_hm_vals)]
                _hm_vmin = float(_hm_vals.min()) if len(_hm_vals) > 0 else 0
                _hm_vmax = float(_hm_vals.max()) if len(_hm_vals) > 0 else 100
                def _hm_color(v):
                    if pd.isna(v): return ""
                    t = max(0, min(1, (v - _hm_vmin) / (_hm_vmax - _hm_vmin))) if _hm_vmax > _hm_vmin else 0.5
                    stops = [(0.0,215,48,39),(0.25,252,141,89),(0.5,255,255,191),(0.75,145,207,96),(1.0,26,152,80)]
                    for i in range(len(stops)-1):
                        t0,r0,g0,b0 = stops[i]; t1,r1,g1,b1 = stops[i+1]
                        if t <= t1:
                            p = (t-t0)/(t1-t0) if t1>t0 else 0
                            return f"background:rgba({int(r0+(r1-r0)*p)},{int(g0+(g1-g0)*p)},{int(b0+(b1-b0)*p)},0.85);"
                    return "background:rgba(26,152,80,0.85);"

                _hm_bdr = "#b0b0b0"; _hm_hdr = "#d6e4f0"; _hm_ylw = "#fef9e7"
                _hm_html = '<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;font-size:0.85em;text-align:center;">'
                # ── 헤더 1줄: 업무유형 | 계약종별들... | 합계
                _hm_html += f'<tr style="background:{_hm_hdr};font-weight:bold;">'
                _hm_html += f'<th style="border:1px solid {_hm_bdr};padding:6px 6px;min-width:56px;max-width:80px;">업무유형</th>'
                for c in _hm_cols:
                    _hm_html += f'<th style="border:1px solid {_hm_bdr};padding:6px 4px;">{c}</th>'
                _hm_html += f'<th style="border:1px solid {_hm_bdr};padding:6px 4px;">합계</th>'
                _hm_html += '</tr>'
                # ── 데이터 행: 각 셀에 "점수 (호수건)"
                for biz in _sol_pivot.index:
                    _hm_html += '<tr>'
                    _hm_html += f'<td style="border:1px solid {_hm_bdr};padding:5px 6px;font-weight:bold;background:#f9f9f9;white-space:nowrap;font-size:0.92em;">{biz}</td>'
                    for c in _hm_cols:
                        v = _sol_pivot.loc[biz, c]
                        cnt = int(_sol_cnt.loc[biz, c]) if biz in _sol_cnt.index and c in _sol_cnt.columns else 0
                        if pd.notna(v):
                            _bg = _hm_color(v)
                            _hm_html += f'<td style="border:1px solid {_hm_bdr};padding:4px;{_bg}">{v:.1f} <span style="font-size:0.8em;color:#444;">({cnt}건)</span></td>'
                        else:
                            _hm_html += f'<td style="border:1px solid {_hm_bdr};padding:4px;">-</td>'
                    # 합계 열
                    t_score = _sol_score_total.get(biz, None)
                    t_cnt = _sol_cnt_total.get(biz, 0)
                    if pd.notna(t_score):
                        _hm_html += f'<td style="border:1px solid {_hm_bdr};padding:4px;font-weight:bold;">{t_score:.1f} <span style="font-size:0.8em;color:#444;">({t_cnt}건)</span></td>'
                    else:
                        _hm_html += f'<td style="border:1px solid {_hm_bdr};padding:4px;">-</td>'
                    _hm_html += '</tr>'
                # ── 하단 합계 행: 계약종별 평균 + 총 건수
                _hm_html += f'<tr style="background:{_hm_hdr};font-weight:bold;">'
                _hm_html += f'<td style="border:1px solid {_hm_bdr};padding:5px 6px;">평균</td>'
                _hm_total_all = 0
                for c in _hm_cols:
                    _col_mean = _sol_pivot[c].mean() if c in _sol_pivot.columns else None
                    _col_cnt = int(_sol_cnt[c].sum()) if c in _sol_cnt.columns else 0
                    _hm_total_all += _col_cnt
                    if pd.notna(_col_mean):
                        _bg = _hm_color(_col_mean)
                        _hm_html += f'<td style="border:1px solid {_hm_bdr};padding:4px;{_bg}">{_col_mean:.1f} <span style="font-size:0.8em;color:#444;">({_col_cnt}건)</span></td>'
                    else:
                        _hm_html += f'<td style="border:1px solid {_hm_bdr};padding:4px;">-</td>'
                _hm_all_mean = _df_sel["_점수100"].mean()
                _hm_html += f'<td style="border:1px solid {_hm_bdr};padding:4px;font-weight:bold;">{_hm_all_mean:.1f} <span style="font-size:0.8em;color:#444;">({_hm_total_all}건)</span></td>'
                _hm_html += '</tr>'
                _hm_html += '</table></div>'
                st.markdown(_hm_html, unsafe_allow_html=True)

                # ── 사전케어 대상 ──────────────────────────────
                st.markdown("##### 🚨 사전케어 대상 — " + _sel_off + " 50점 이하")
                _pc_df = _df_sel[_df_sel["_점수100"] <= 50].copy()
                if not _pc_df.empty:
                    _pc_df = _pc_df.sort_values("_점수100")
                    st.caption(f"해당 지사 50점 이하 **{len(_pc_df)}건** — 해피콜 우선 대상")
                    _pc_show_cols = []
                    if M.get("receipt_no") and M["receipt_no"] in _pc_df.columns:
                        _pc_show_cols.append(M["receipt_no"])
                    if M.get("business") and M["business"] in _pc_df.columns:
                        _pc_show_cols.append(M["business"])
                    _pc_show_cols.append("_점수100")
                    if M.get("voc") and M["voc"] in _pc_df.columns:
                        _pc_show_cols.append(M["voc"])
                    _pc_show = _pc_df[[c for c in _pc_show_cols if c in _pc_df.columns]].head(10)
                    _pc_show = _pc_show.rename(columns={"_점수100": "점수(100점)"})
                    st.dataframe(_pc_show.reset_index(drop=True), use_container_width=True, hide_index=True)
                else:
                    st.success("✅ 해당 지사에 50점 이하 건이 없습니다.")

                st.markdown("---")

                # ── 리스크 분류: 계약종별 × 업무 기준 ──────────
                _sol_score_cols = [c for c in individual_scores if c in _df_sel.columns]
                _sol_combos = _df_sel.groupby([M["contract"], M["business"]]).agg(
                    _avg=("_점수100", "mean"), _cnt=("_점수100", "count")).reset_index()
                _sol_combos.columns = [M["contract"], M["business"], "점수", "건수"]
                _sol_combos = _sol_combos[_sol_combos["점수"] < avg_score_100]

                _real_risk, _drop_risk = [], []
                for _, _cr in _sol_combos.iterrows():
                    _c_ct, _c_biz = _cr[M["contract"]], _cr[M["business"]]
                    _c_score, _c_cnt = round(float(_cr["점수"]), 1), int(_cr["건수"])
                    _c_sub = _df_sel[(_df_sel[M["contract"]] == _c_ct) & (_df_sel[M["business"]] == _c_biz)]
                    _c_worst_item, _c_worst_val = "", 100.0
                    for _sc in _sol_score_cols:
                        _sv = pd.to_numeric(_c_sub[_sc], errors="coerce").dropna()
                        if len(_sv) > 0 and float(_sv.mean()) < _c_worst_val:
                            _c_worst_val = float(_sv.mean())
                            _c_worst_item = _sc
                    _c_voc_sample = ""
                    if M.get("voc"):
                        _c_vocs = _c_sub[_c_sub["_점수100"] < 70][M["voc"]].dropna().astype(str).tolist()
                        _c_vocs = [v for v in _c_vocs if v.strip() not in _VOC_EMPTY and len(v.strip()) > 2]
                        if _c_vocs:
                            _c_voc_sample = _c_vocs[0][:40]
                    _item = {"계약종별": _c_ct, "업무": _c_biz, "점수": _c_score,
                             "건수": _c_cnt, "최저항목": _c_worst_item,
                             "최저점수": round(_c_worst_val, 1), "voc": _c_voc_sample,
                             "impact": _c_cnt * (avg_score_100 - _c_score)}
                    if _c_cnt >= 10:
                        _real_risk.append(_item)
                    else:
                        _drop_risk.append(_item)
                _real_top3 = sorted(_real_risk, key=lambda x: x["impact"], reverse=True)[:3]
                _drop_top3 = sorted(_drop_risk, key=lambda x: x["점수"])[:3]

                # ── 실질적 리스크 카드 (계약종별×업무) ──────────
                if _real_top3:
                    st.markdown("##### 🚨 실질적 리스크 TOP 3 — 건수 多 & 평균 이하 (우선 개선)")
                    st.caption("카드를 클릭하면 상세 분석이 열립니다.")
                    _rt_cols = st.columns(len(_real_top3))
                    for _ri, _rk in enumerate(_real_top3):
                        _rk_key = f"sol_real_{_ri}"
                        _cur_sel = st.session_state.get("sol_cell_sel")
                        _is_sel = (_cur_sel is not None
                                   and _cur_sel.get("계약종별") == _rk["계약종별"]
                                   and _cur_sel.get("업무") == _rk["업무"])
                        _badge = "①②③"[_ri]
                        _bg = "#fff3e0" if _is_sel else "#ffebee"
                        _bd = "#ff8f00" if _is_sel else "#ef9a9a"
                        with _rt_cols[_ri]:
                            _voc_line = f'<div style="font-size:0.75em;color:#888;margin-top:4px;">"{_rk["voc"]}…"</div>' if _rk["voc"] else ""
                            st.markdown(
                                f'<div style="background:{_bg};border:2px solid {_bd};'
                                'border-radius:10px;padding:12px;margin-bottom:8px;">'
                                f'<div style="font-size:0.78em;color:#c62828;font-weight:700;">리스크 {_badge}</div>'
                                f'<div style="font-size:1em;font-weight:800;margin:4px 0;">{_rk["계약종별"]} × {_rk["업무"]}</div>'
                                f'<div style="font-size:0.82em;color:#555;">최저: {_rk["최저항목"]} ({_rk["최저점수"]}점)</div>'
                                f'<div style="font-size:1.4em;font-weight:900;color:#c62828;">{_rk["점수"]:.1f}점</div>'
                                f'<div style="font-size:0.78em;color:#555;">{_rk["건수"]}건 · 임팩트 {_rk["impact"]:.0f}</div>'
                                f'{_voc_line}'
                                '</div>', unsafe_allow_html=True)
                            if st.button(
                                    "▲ 닫기" if _is_sel else "🔍 상세 원인 분석",
                                    key=_rk_key, use_container_width=True,
                                    type="secondary" if _is_sel else "primary"):
                                st.session_state["sol_cell_sel"] = (
                                    None if _is_sel else {"계약종별": _rk["계약종별"], "업무": _rk["업무"]})
                                st.rerun()

                # ── 급락 리스크 카드 ──────────────────────────────
                if _drop_top3:
                    st.markdown("##### ⚡ 급락 리스크 TOP 3 — 건수 少 & 점수 급락 (모니터링)")
                    st.caption("소수 응답이지만 점수가 급락한 조합입니다. 민원 전조 신호일 수 있으니 **추이를 주시**하세요.")
                    _dt_cols = st.columns(len(_drop_top3))
                    for _di, _dk in enumerate(_drop_top3):
                        _dk_key = f"sol_drop_{_di}"
                        _cur_sel = st.session_state.get("sol_cell_sel")
                        _is_sel = (_cur_sel is not None
                                   and _cur_sel.get("계약종별") == _dk["계약종별"]
                                   and _cur_sel.get("업무") == _dk["업무"])
                        _badge = "①②③"[_di]
                        _bg = "#fff3e0" if _is_sel else "#fff8e1"
                        _bd = "#ff8f00" if _is_sel else "#ffca28"
                        with _dt_cols[_di]:
                            _voc_line = f'<div style="font-size:0.75em;color:#888;margin-top:4px;">"{_dk["voc"]}…"</div>' if _dk["voc"] else ""
                            st.markdown(
                                f'<div style="background:{_bg};border:2px solid {_bd};'
                                'border-radius:10px;padding:12px;margin-bottom:8px;">'
                                f'<div style="font-size:0.78em;color:#e65100;font-weight:700;">급락 {_badge}</div>'
                                f'<div style="font-size:1em;font-weight:800;margin:4px 0;">{_dk["계약종별"]} × {_dk["업무"]}</div>'
                                f'<div style="font-size:0.82em;color:#555;">최저: {_dk["최저항목"]} ({_dk["최저점수"]}점)</div>'
                                f'<div style="font-size:1.4em;font-weight:900;color:#e65100;">{_dk["점수"]:.1f}점</div>'
                                f'<div style="font-size:0.78em;color:#555;">{_dk["건수"]}건 (소량)</div>'
                                f'{_voc_line}'
                                '</div>', unsafe_allow_html=True)
                            if st.button(
                                    "▲ 닫기" if _is_sel else "🔍 상세 원인 분석",
                                    key=_dk_key, use_container_width=True,
                                    type="secondary" if _is_sel else "primary"):
                                st.session_state["sol_cell_sel"] = (
                                    None if _is_sel else {"계약종별": _dk["계약종별"], "업무": _dk["업무"]})
                                st.rerun()

                if not _real_top3 and not _drop_top3:
                    st.success("✅ 모든 업무×계약종별 조합이 본부 평균 이상입니다.")

                # ══════════════════════════════════════════
                # 상세 분석 — 선택된 카드의 범인 특정 + VOC
                # ══════════════════════════════════════════
                # 기본값: 화면에 보이는 리스크 카드 1위 자동 선택
                # session_state에 이전 값이 있어도, 현재 지사에 데이터 없으면 리셋
                _first_card = None
                if _real_top3:
                    _first_card = {"계약종별": _real_top3[0]["계약종별"], "업무": _real_top3[0]["업무"]}
                elif _drop_top3:
                    _first_card = {"계약종별": _drop_top3[0]["계약종별"], "업무": _drop_top3[0]["업무"]}

                _cell = st.session_state.get("sol_cell_sel")
                if _cell is not None and _cell.get("계약종별"):
                    # 이전 선택값이 현재 지사에 데이터가 있는지 확인
                    _chk = _df_sel[
                        (_df_sel[M["contract"]] == _cell["계약종별"]) &
                        (_df_sel[M["business"]] == _cell["업무"])
                    ]
                    if len(_chk) == 0:
                        _cell = _first_card  # 데이터 없으면 카드 1위로 리셋
                if _cell is None:
                    _cell = _first_card
                if _cell and _cell.get("계약종별"):
                    _sel_ct = _cell["계약종별"]
                    _sel_biz = _cell["업무"]

                    st.markdown("---")
                    st.markdown(
                        '<div style="background:linear-gradient(90deg,#4a148c,#6a1b9a);'
                        'border-radius:10px;padding:14px 20px;color:white;margin:16px 0 12px;">'
                        f'<span style="font-size:1.1em;font-weight:800;">'
                        f'🔬 심층 진단 — {_sel_off} · {_sel_ct} × {_sel_biz}</span>'
                        '<span style="font-size:0.82em;opacity:.8;margin-left:10px;">범인 특정 → VOC → AI 처방전</span>'
                        '</div>', unsafe_allow_html=True)

                    _c3_df = _df_sel[
                        (_df_sel[M["contract"]] == _sel_ct) &
                        (_df_sel[M["business"]] == _sel_biz)
                    ]
                    _c3_n = len(_c3_df)

                    if _c3_n == 0:
                        st.info(f"[{_sel_ct}] × [{_sel_biz}] 조합의 데이터가 없습니다.")
                    elif _c3_n > 0:

                        st.markdown(f"**타겟**: [{_sel_ct}] 고객의 [{_sel_biz}] — **{_c3_n}건**")
                        if _c3_n < 10:
                            st.warning(
                                f"⚠️ 데이터 수가 적어(**{_c3_n}건**) 특정 사례에 의한 "
                                "왜곡 가능성이 있으니 **VOC 원문을 중심으로** 판단하세요.")

                        # ── 세부항목 점수 + VOC (2컬럼) ──────────────
                        _item_scores = {}
                        for _sc in _sol_score_cols:
                            _vals = pd.to_numeric(_c3_df[_sc], errors="coerce").dropna()
                            if len(_vals) > 0:
                                _item_scores[_sc] = round(float(_vals.mean()), 1)

                        if _item_scores:
                            _worst_item_name = min(_item_scores, key=_item_scores.get)
                            _worst_score = _item_scores[_worst_item_name]

                            _c3_l, _c3_r = st.columns([1, 1])

                            with _c3_l:
                                st.markdown("**📊 세부항목별 만족도 — 범인 특정**")
                                _c3_colors = [
                                    "#d32f2f" if k == _worst_item_name
                                    else "#f57c00" if v < avg_score_100
                                    else "#388e3c"
                                    for k, v in _item_scores.items()
                                ]
                                fig_c3 = go.Figure(go.Bar(
                                    y=list(_item_scores.keys()),
                                    x=list(_item_scores.values()),
                                    orientation="h", marker_color=_c3_colors,
                                    text=[f"{v:.1f}" for v in _item_scores.values()],
                                    textposition="outside",
                                    hovertemplate="%{y}<br>%{x:.1f}점<extra></extra>"))
                                fig_c3.add_vline(
                                    x=avg_score_100, line_dash="dash", line_color=C["navy"],
                                    annotation_text=f"본부 {avg_score_100:.1f}",
                                    annotation_position="top right")
                                _c3_x_min = max(0, min(_item_scores.values()) - 10)
                                fig_c3.update_layout(
                                    template=PLOTLY_TPL, height=max(250, len(_item_scores) * 45 + 60),
                                    margin=dict(t=10, b=10, l=10, r=90),
                                    xaxis=dict(range=[_c3_x_min, 110]))
                                st.plotly_chart(fig_c3, use_container_width=True, config={'staticPlot': True})

                                st.error(
                                    f"🎯 **범인 확정**: {_sel_ct} 계약종별 · {_sel_biz} 업무에서 "
                                    f"**{_worst_item_name}** 항목이 **{_worst_score}점** "
                                    f"(본부 대비 {_worst_score - avg_score_100:+.1f}점)")

                                _bm_best_txt = ""

                            with _c3_r:
                                st.markdown("**📝 실제 VOC 원문**")
                                _c3_voc_lines = ""
                                if M.get("voc") and not _c3_df.empty:
                                    _c3_voc_valid_idx = (
                                        _c3_df[M["voc"]].dropna()
                                        .apply(lambda x: str(x).strip())
                                        .loc[lambda s: (s.str.len() > 2) & (~s.isin(["응답없음", "nan", ""]))])
                                    _c3_vocs_idx = _c3_voc_valid_idx.head(10)
                                    if not _c3_vocs_idx.empty:
                                        _has_receipt = bool(M.get("receipt_no") and M["receipt_no"] in _c3_df.columns)
                                        with st.expander(f"VOC {len(_c3_vocs_idx)}건 보기", expanded=True):
                                            for _idx, _cv in _c3_vocs_idx.items():
                                                _cv_hl = str(_cv)
                                                for _nkw in VOC_HIGHLIGHT_KW:
                                                    if _nkw in _cv_hl:
                                                        _cv_hl = _cv_hl.replace(
                                                            _nkw,
                                                            '<mark style="background:#ffeb3b">'
                                                            + _nkw + "</mark>")
                                                _receipt_html = ""
                                                if _has_receipt:
                                                    _rn = str(_c3_df.at[_idx, M["receipt_no"]]).strip()
                                                    if _rn and _rn not in ("nan", ""):
                                                        _receipt_html = (
                                                            f'<span style="color:#1976d2;font-weight:600;'
                                                            f'font-size:0.8em;">[{_rn}]</span> ')
                                                st.markdown(
                                                    '<div style="border-left:3px solid #ef9a9a;'
                                                    'padding:4px 10px;margin-bottom:4px;font-size:0.87em;">'
                                                    + _receipt_html + _cv_hl + "</div>",
                                                    unsafe_allow_html=True)
                                        _c3_voc_lines = "\n".join(f"  - {v}" for v in _c3_vocs_idx)
                                    else:
                                        st.info("70점 미만 VOC가 없습니다.")
                                else:
                                    st.info("VOC 컬럼이 설정되지 않았습니다.")

                        else:
                            st.info("세부항목 점수 데이터가 없습니다.")

                # ── AI 종합 처방전 (실질+급락 전체) ──────────────
                if _real_top3 or _drop_top3:
                    st.markdown("---")
                    _office_kb = _get_office_kb(_sel_off)
                    _kb_ctx = _office_kb["context"] if _office_kb else "지역 특성 정보 없음"
                    _kb_act = _office_kb["action"] if _office_kb else ""

                    # 리스크 요약 텍스트 생성
                    _rx_lines = ""
                    if _real_top3:
                        _rx_lines += "■ 실질적 리스크 (건수 多 & 평균 이하):\n"
                        for _ri, _rk in enumerate(_real_top3, 1):
                            _rx_lines += f"  {_ri}. {_rk['계약종별']}×{_rk['업무']}: {_rk['점수']}점, {_rk['건수']}건, 최저항목={_rk['최저항목']}({_rk['최저점수']}점)"
                            if _rk["voc"]:
                                _rx_lines += f", VOC=\"{_rk['voc']}\""
                            _rx_lines += "\n"
                    if _drop_top3:
                        _rx_lines += "■ 급락 리스크 (건수 少 & 점수 급락):\n"
                        for _di, _dk in enumerate(_drop_top3, 1):
                            _rx_lines += f"  {_di}. {_dk['계약종별']}×{_dk['업무']}: {_dk['점수']}점, {_dk['건수']}건, 최저항목={_dk['최저항목']}({_dk['최저점수']}점)"
                            if _dk["voc"]:
                                _rx_lines += f", VOC=\"{_dk['voc']}\""
                            _rx_lines += "\n"

                    # 전체 VOC 수집 (80점 미만 + 넉넉히 30건)
                    _all_neg_vocs = ""
                    if M.get("voc"):
                        _voc_base = _df_sel[_df_sel["_점수100"] < 80].sort_values("_점수100")
                        _all_voc_df = _voc_base[M["voc"]].dropna().astype(str)
                        _all_voc_list = [v.strip() for v in _all_voc_df.tolist() if v.strip() not in _VOC_EMPTY and len(v.strip()) > 2]
                        if _all_voc_list:
                            _all_neg_vocs = "\n".join(f"- {v}" for v in _all_voc_list[:30])

                    _sol_ai_key = f"_ai_sol_{_sel_off}"
                    if st.button("🤖 AI 종합 처방전 생성", key="sol_ai_total_btn",
                                 type="primary", use_container_width=True):
                        if not GEMINI_AVAILABLE:
                            st.error("Gemini API 키가 설정되지 않았습니다.")
                        else:
                            _sol_prompt = (
                                f"# 역할\n"
                                f"한국전력 경남본부 CS 컨설턴트. 지사(사업소) 단위에서 내일 당장 실행 가능한 전략을 세우는 전문가.\n\n"
                                f"# 분석 대상\n"
                                f"- 지사: {_sel_off}\n"
                                f"- 지역 특성: {_kb_ctx}\n"
                                f"- 종합 평균: {_sel_avg:.1f}점 (본부 평균 {avg_score_100:.1f}점)\n\n"
                                f"# 리스크 요약 (참고용 — 제목에 그대로 쓰지 말 것)\n{_rx_lines}\n"
                                f"# 고객 불만 VOC 원문 (핵심 데이터 — 반드시 패턴 분석할 것)\n"
                                f"{_all_neg_vocs or '없음'}\n\n"
                                f"# 현장 현실 제약 (반드시 숙지)\n"
                                f"- 문자 발송 시스템은 이미 있음 (체크박스로 발송 가능). '문자 발송하라'는 제안 불필요.\n"
                                f"- 만족도 조사 링크는 본사에서 일괄 발송. 사업소에서 보내는 것 아님.\n"
                                f"- 해피콜은 이미 시행 중. '해피콜 도입'은 제안 불필요.\n"
                                f"- 전담 창구/핫라인 지정은 소규모 사업소에서 업무 과중 유발. 특정인에게 업무 몰리는 제안 금지.\n"
                                f"- 사업소는 콜센터가 아님. 전화 응대 전문 인력이 따로 없음.\n"
                                f"- 실현 가능한 방향: 사업소 내 안내물·게시물 배치, 고객 동선 개선, 대기 공간 활용, 간단한 서식·양식 비치 등 물리적 환경 개선 위주.\n\n"
                                f"# 요청\n"
                                f"1. VOC 불만 **공통 패턴** 2~3가지 묶기 (원문 인용 불필요)\n"
                                f"2. 위 현실 제약 반영, 사업소에서 진짜 할 수 있는 것만 제안\n\n"
                                f"# 문체 규칙 (가장 중요)\n"
                                f"- 극도로 짧은 개조식. 한 bullet = 명사형 종결 또는 짧은 서술 1개.\n"
                                f"- 좋은 예: '고령층 고려, 쉬운 용어·큰 글씨 안내문 창구 비치'\n"
                                f"- 나쁜 예: '고령층 고객을 고려하여 어려운 용어 대신 쉬운 언어를 사용하고 전화 상담 시에는 더욱 친절하고 상세한 안내를 제공합니다.'\n"
                                f"- '~합니다', '~하도록 합니다', '~를 마련합니다' 같은 장황한 종결 금지.\n"
                                f"- '세부 행동 1:', '세부 행동 2:' 같은 레이블 붙이지 말 것. 그냥 '-' bullet만.\n\n"
                                f"# 출력 형식\n"
                                f"### 📌 불만 패턴\n"
                                f"- (한 줄 요약)\n"
                                f"- ...\n\n"
                                f"### 🛠️ 실행 전략\n"
                                f"**1. (전략명)**\n"
                                f"- 행동 1\n"
                                f"- 행동 2\n\n"
                                f"**2. ...**\n"
                                f"**3. ...**\n"
                                f"(정확히 3개. 전략당 행동 2~3개)\n\n"
                                f"# 절대 금지\n"
                                f"- 교차 조합(일반용×요금수납) 제목 금지\n"
                                f"- TF 구성, 교육, 매뉴얼, 시스템 개편, 멘토링, 코칭\n"
                                f"- 문자 발송, 만족도 조사 링크, 해피콜, 전담 창구/핫라인\n"
                                f"- bullet 간 같은 말 반복\n"
                                f"- 특정 직원 지목·업무 과중 유발\n"
                                f"- '전기세' → '전기요금'\n"
                                f"- 줄글·산문·미사여구·AI 추임새\n"
                                f"- '(출처: ...)' 같은 인용 표기 금지. 출처 없이 행동만 쓸 것\n"
                                f"- 데이터에 없는 사실 창작\n"
                            )
                            with st.spinner("AI가 종합 처방전 생성 중…"):
                                try:
                                    import urllib.request
                                    _models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemma-3-12b-it"]
                                    _sol_pl = {
                                        "contents": [{"parts": [{"text": _sol_prompt}]}],
                                        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192}
                                    }
                                    _ctx = ssl._create_unverified_context()
                                    _body = None
                                    for _model in _models:
                                        _url = (f"https://generativelanguage.googleapis.com/v1beta/"
                                                f"models/{_model}:generateContent?key={_GEMINI_KEY}")
                                        _req = urllib.request.Request(
                                            _url, data=json.dumps(_sol_pl).encode("utf-8"),
                                            headers={"Content-Type": "application/json"}, method="POST")
                                        try:
                                            with urllib.request.urlopen(_req, context=_ctx, timeout=90) as _rsp:
                                                _body = json.loads(_rsp.read().decode("utf-8"))
                                            break
                                        except urllib.error.HTTPError as _he:
                                            if _he.code in (429, 500, 502, 503):
                                                continue
                                            raise
                                    if _body is None:
                                        st.error("모든 AI 모델의 일일 한도가 소진되었습니다.")
                                    else:
                                        st.session_state[_sol_ai_key] = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
                                except Exception as _e:
                                    st.error(f"AI 분석 중 오류: {_e}")

                    if _sol_ai_key in st.session_state:
                        st.markdown(
                            '<div style="background:#f8f9fb;border:1px solid #d0d7de;border-radius:12px;'
                            'padding:28px 32px;margin:12px 0;font-size:1.05em;line-height:2.0;">\n\n'
                            f'{st.session_state[_sol_ai_key]}\n\n</div>',
                            unsafe_allow_html=True)

        else:
            st.info("업무유형·계약종별 컬럼이 필요합니다.")


# (Tab SOL2 — Tab SOL로 통합됨)




# ─────────────────────────────────────────────────────────────
#  TAB LETTER  경험고객 서한문 생성
# ─────────────────────────────────────────────────────────────
with tab_letter:
    st.markdown('<p class="sec-head">💌 경험고객 서한문 생성 — 지사 맞춤형 문구 · 시각화 미리보기 · 기념품 추천</p>',
                unsafe_allow_html=True)

    # ── 계절 테마 정의 (인제지사 스타일 참고) ──────────────────
    _SEASON_THEMES = {
        "봄 (3~5월)": {
            "bg": "#fff8f0", "accent": "#e91e63", "sub": "#f8bbd0",
            "border": "#f48fb1", "title_color": "#ad1457",
            "greeting": "설레는 봄이 되시기를 바랍니다.",
            "closing": "새싹이 움트는 봄날입니다.\n항상 건강 유의하시고 고객님의 가정과 일터에 사랑과 행복이 넘쳐나길 기원드립니다.",
            "icon": "🌸", "season_name": "봄",
        },
        "여름 (6~8월)": {
            "bg": "#e8f5e9", "accent": "#2e7d32", "sub": "#a5d6a7",
            "border": "#66bb6a", "title_color": "#1b5e20",
            "greeting": "활기찬 여름이 되시기를 바랍니다.",
            "closing": "유난히 무더운 여름날입니다.\n항상 건강 유의하시고 고객님의 가정과 일터에 사랑과 행복이 넘쳐나길 기원드립니다.",
            "icon": "🍉", "season_name": "여름",
        },
        "가을 (9~11월)": {
            "bg": "#fff3e0", "accent": "#e65100", "sub": "#ffcc80",
            "border": "#ff9800", "title_color": "#bf360c",
            "greeting": "풍성한 가을 되시기를 바랍니다.",
            "closing": "일교차가 심한 가을날입니다.\n항상 건강 유의하시고 고객님의 가정과 일터에 사랑과 행복이 넘쳐나길 기원드립니다.",
            "icon": "🍂", "season_name": "가을",
        },
        "겨울 (12~2월)": {
            "bg": "#e3f2fd", "accent": "#1565c0", "sub": "#90caf9",
            "border": "#42a5f5", "title_color": "#0d47a1",
            "greeting": "따뜻한 겨울 되시기를 바랍니다.",
            "closing": "바람이 매서운 겨울날입니다.\n항상 건강 유의하시고 고객님의 가정과 일터에 사랑과 행복이 넘쳐나길 기원드립니다.",
            "icon": "❄️", "season_name": "겨울",
        },
    }

    # ── 기념품 추천 DB (계절×지역유형) ────────────────────────
    _GIFT_DB = {
        "봄": {
            "공통": [
                ("종량제 봉투 세트 (20L×10매)", "2,000원", "실용도 1위, 누구나 필요한 생필품"),
                ("미니 손소독제 + 물티슈 세트", "2,500원", "외출 필수품, 봄나들이 시즌 활용"),
                ("다용도 장바구니 (접이식)", "2,800원", "에코백 대용, 한전 로고 인쇄 가능"),
                ("국화차/캐모마일 티백 세트", "2,500원", "봄철 환절기 건강 관리"),
            ],
            "농촌형": [("원예용 면장갑 세트 (3켤레)", "2,000원", "영농기 앞두고 실용적")],
            "해안형": [("자외선 차단 쿨토시", "2,500원", "야외 작업 필수품")],
            "도심형": [("텀블러 (350ml 미니)", "3,000원", "직장인 필수템, 다회용 실천")],
        },
        "여름": {
            "공통": [
                ("종량제 봉투 세트 (20L×10매)", "2,000원", "실용도 1위, 누구나 필요한 생필품"),
                ("쿨링 넥밴드 (아이스타올)", "2,500원", "폭염 대비 필수품"),
                ("휴대용 미니 선풍기 (USB 충전)", "3,000원", "여름 필수 아이템"),
                ("모기 기피 팔찌 + 패치 세트", "2,000원", "여름 야외활동 실용품"),
            ],
            "농촌형": [("쿨링 아이스조끼 (간이형)", "3,000원", "영농기 폭염 대비")],
            "해안형": [("방수 파우치", "2,500원", "해양 레저 시즌 활용")],
            "도심형": [("아이스 텀블러 (콜드컵)", "3,000원", "사무실 필수템")],
        },
        "가을": {
            "공통": [
                ("종량제 봉투 세트 (20L×10매)", "2,000원", "실용도 1위, 누구나 필요한 생필품"),
                ("고급 양말 세트 (3켤레)", "2,500원", "환절기 보온, 남녀노소 실용"),
                ("핫초코/곡물차 티백 세트", "2,500원", "가을 환절기 따뜻한 음료"),
                ("다용도 극세사 행주 세트", "2,000원", "주방 필수품, 실용성 최고"),
            ],
            "농촌형": [("LED 미니 손전등", "2,500원", "일몰 빠른 가을철 야외 활동")],
            "해안형": [("방풍 넥워머", "2,800원", "해풍 대비 가을 필수품")],
            "도심형": [("보온 머그컵 (뚜껑형)", "3,000원", "사무실 가을겨울 필수템")],
        },
        "겨울": {
            "공통": [
                ("종량제 봉투 세트 (20L×10매)", "2,000원", "실용도 1위, 누구나 필요한 생필품"),
                ("핫팩 세트 (10개입)", "2,000원", "겨울 필수 보온용품"),
                ("기모 장갑 (터치스크린 호환)", "2,800원", "방한 + 스마트폰 사용 가능"),
                ("보온 텀블러 (350ml)", "3,000원", "따뜻한 음료 휴대, 실용성 높음"),
            ],
            "농촌형": [("방한 귀마개 + 핫팩", "2,500원", "겨울 야외 작업 보온 세트")],
            "해안형": [("방풍 비니 모자", "3,000원", "해풍 방한 필수품")],
            "도심형": [("USB 보온 컵받침", "3,000원", "사무실 데스크 보온 아이템")],
        },
    }

    # ── 불만 키워드 → 약속 문장 매핑 (서한문 프롬프트용) ──
    _COMPLAINT_TO_PROMISE = {
        "정전": "안정적인 전력공급에 더욱 만전을 기하겠습니다",
        "단전": "안정적인 전력공급에 더욱 만전을 기하겠습니다",
        "지연": "신속한 업무 처리로 고객님의 시간을 소중히 여기겠습니다",
        "오래": "대기 시간 단축을 위해 지속적으로 개선하겠습니다",
        "기다림": "대기 시간 단축을 위해 지속적으로 개선하겠습니다",
        "지체": "대기 시간 단축을 위해 지속적으로 개선하겠습니다",
        "불친절": "친절하고 정중한 응대를 최우선으로 하겠습니다",
        "무시": "고객님 한 분 한 분의 목소리에 귀 기울이겠습니다",
        "냉담": "고객님 한 분 한 분의 목소리에 귀 기울이겠습니다",
        "요금": "투명하고 정확한 요금 안내에 힘쓰겠습니다",
        "과금": "투명하고 정확한 요금 안내에 힘쓰겠습니다",
        "과다": "투명하고 정확한 요금 안내에 힘쓰겠습니다",
        "고장": "설비 점검과 유지보수에 더욱 힘쓰겠습니다",
        "불량": "설비 점검과 유지보수에 더욱 힘쓰겠습니다",
        "오류": "정확한 업무 처리를 위해 꼼꼼히 점검하겠습니다",
        "실수": "정확한 업무 처리를 위해 꼼꼼히 점검하겠습니다",
        "착오": "정확한 업무 처리를 위해 꼼꼼히 점검하겠습니다",
        "반복": "같은 불편이 되풀이되지 않도록 근본적으로 개선하겠습니다",
        "여전히": "같은 불편이 되풀이되지 않도록 근본적으로 개선하겠습니다",
        "위험": "안전한 전기 사용 환경을 만들기 위해 최선을 다하겠습니다",
        "안전": "안전한 전기 사용 환경을 만들기 위해 최선을 다하겠습니다",
    }

    # ── 불만 키워드 → 기념품 매핑 (VOC 맞춤 추천용) ──
    _COMPLAINT_GIFT = {
        "정전": [("LED 비상 랜턴", "3,000원", "정전 시 즉시 활용 가능한 실용 아이템")],
        "단전": [("LED 비상 랜턴", "3,000원", "정전 시 즉시 활용 가능한 실용 아이템")],
        "요금": [("에너지 절약 멀티탭", "3,500원", "대기전력 차단으로 요금 절감 도움")],
        "과금": [("에너지 절약 멀티탭", "3,500원", "대기전력 차단으로 요금 절감 도움")],
        "고장": [("휴대용 멀티 충전케이블", "2,500원", "전기 관련 실용 아이템으로 호감 형성")],
        "불친절": [("프리미엄 보온텀블러", "3,500원", "정성이 담긴 사과의 의미 전달")],
        "위험": [("가정용 소화기 (미니)", "3,500원", "전기 안전 관련 실용 아이템")],
        "안전": [("가정용 소화기 (미니)", "3,500원", "전기 안전 관련 실용 아이템")],
    }

    # ── 계약종별 → 톤 가이드 매핑 ──
    _CONTRACT_TONE = {
        "주택용": "가정과 일상을 중심으로 편안하고 따뜻한 톤",
        "농사용": "농번기의 수고를 격려하고 계절 변화에 공감하는 톤",
        "산업용": "사업 번창을 기원하며 안정적 전력공급의 신뢰를 강조하는 톤",
        "일반용": "사업 운영을 응원하며 효율적 서비스를 강조하는 톤",
        "교육용": "교육 현장의 가치를 존중하며 안정적 학습환경을 약속하는 톤",
    }

    # ── 지역유형 → 톤 가이드 매핑 ──
    _REGION_TONE = {
        "농촌형": "소박하고 정감 있는 톤, 자연·계절 묘사를 살짝 더",
        "해안형": "바다·포구 등 해안 정서를 은은히, 중간 톤",
        "도심형": "간결하고 신뢰감 있는 톤, 효율·서비스 품질 강조",
        "공통": "보편적이고 따뜻한 톤",
    }

    def _get_region_type(kb_context: str) -> str:
        """KB context에서 지역유형 추출"""
        if not kb_context:
            return "공통"
        if "농촌" in kb_context:
            return "농촌형"
        if "해안" in kb_context:
            return "해안형"
        if "도심" in kb_context or "공단" in kb_context:
            return "도심형"
        return "공통"

    # ── 서한문 시각화 (matplotlib, 인제 스타일) ────────────────
    def _render_letter_preview(office_name, season_key, body_text, theme):
        """matplotlib로 인제지사 스타일 서한문 미리보기 이미지 생성"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 11))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 140)
        ax.axis("off")
        fig.patch.set_facecolor(theme["bg"])

        _fp_title = fm.FontProperties(fname=FONT_PATH, size=14, weight="bold") if FONT_PATH else fm.FontProperties(size=14, weight="bold")
        _fp_body = fm.FontProperties(fname=FONT_PATH, size=9.5) if FONT_PATH else fm.FontProperties(size=9.5)
        _fp_body_bold = fm.FontProperties(fname=FONT_PATH, size=9.5, weight="bold") if FONT_PATH else fm.FontProperties(size=9.5, weight="bold")
        _fp_head = fm.FontProperties(fname=FONT_PATH, size=18, weight="bold") if FONT_PATH else fm.FontProperties(size=18, weight="bold")
        _fp_footer = fm.FontProperties(fname=FONT_PATH, size=10, weight="bold") if FONT_PATH else fm.FontProperties(size=10, weight="bold")
        _fp_small = fm.FontProperties(fname=FONT_PATH, size=8) if FONT_PATH else fm.FontProperties(size=8)

        # 상단 헤더 영역 (컬러 배경)
        from matplotlib.patches import FancyBboxPatch, Rectangle
        header_bg = FancyBboxPatch((0, 118), 100, 22, boxstyle="round,pad=0",
                                    facecolor=theme["accent"], alpha=0.12, edgecolor="none")
        ax.add_patch(header_bg)

        # KEPCO 로고 텍스트
        ax.text(5, 136, "KEPCO", fontsize=11, fontweight="bold", color=theme["accent"],
                fontproperties=_fp_title, va="top")

        # 타이틀
        ax.text(50, 131, f"세상에 빛을, 이웃에 사랑을", ha="center", va="top",
                fontproperties=_fp_head, color=theme["title_color"])
        ax.text(50, 124, f"한국전력공사입니다.", ha="center", va="top",
                fontproperties=_fp_head, color=theme["title_color"])

        # 계절 텍스트 (matplotlib은 이모지 렌더링 불가하므로 한글로 표시)
        _season_label = {"봄": "春", "여름": "夏", "가을": "秋", "겨울": "冬"}.get(
            theme.get("season_name", ""), "")
        ax.text(92, 135, _season_label, fontsize=22, ha="center", va="top",
                fontproperties=_fp_head, color=theme["accent"], alpha=0.4)

        # 본문 영역 테두리
        body_box = FancyBboxPatch((6, 18), 88, 96, boxstyle="round,pad=1",
                                   facecolor="white", edgecolor=theme["border"],
                                   linewidth=1.5, alpha=0.9)
        ax.add_patch(body_box)

        # 본문 텍스트 줄바꿈 처리
        _lines = []
        for paragraph in body_text.split("\n"):
            if paragraph.strip() == "":
                _lines.append("")
            else:
                wrapped = textwrap.wrap(paragraph, width=38)
                _lines.extend(wrapped if wrapped else [""])

        # 줄 수에 따라 간격 동적 조절 (텍스트 잘림 방지)
        _total_lines = len(_lines)
        _available_height = 88  # y=110 ~ y=22
        _line_gap = min(4.5, max(3.0, _available_height / max(_total_lines, 1)))

        y_pos = 110
        for _line in _lines:
            if y_pos < 22:
                break
            _fp_use = _fp_body_bold if _line.startswith("한국전력") else _fp_body
            _fs = min(9.5, 9.5 * (4.5 / max(_line_gap, 3.0)))
            ax.text(10, y_pos, _line, fontproperties=_fp_use, fontsize=_fs,
                    color="#333333", va="top")
            y_pos -= _line_gap

        # 날짜 + 지사명 (본문 아래)
        from datetime import datetime as _dt
        _now = _dt.now()
        _date_str = f"{_now.year}년 {_now.month}월"
        if y_pos >= 26:
            ax.text(50, y_pos - 2, _date_str, ha="center", va="top",
                    fontproperties=_fp_body, color="#555555")
            ax.text(50, y_pos - 7, f"한국전력공사 {office_name}", ha="center", va="top",
                    fontproperties=_fp_body_bold, color=theme["accent"])

        # 하단 연락처 바
        footer_bg = Rectangle((0, 0), 100, 16, facecolor=theme["accent"], alpha=0.85)
        ax.add_patch(footer_bg)
        ax.text(50, 11, "365일 24시간 전기상담 / 고장신고 : 국번없이 123", ha="center", va="center",
                fontproperties=_fp_footer, color="white", fontsize=10)
        ax.text(50, 5, "모바일·인터넷 업무신청 : 한전ON(online.kepco.co.kr)", ha="center", va="center",
                fontproperties=_fp_small, color="white", fontsize=8)

        plt.tight_layout(pad=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf

    # ── UI 레이아웃 ──────────────────────────────────────────
    st.markdown("""<div class="card-blue">
    <b>💡 사용법</b><br>
    ① 지사·계절 선택 → ② AI 서한문 생성 → ③ 미리보기 확인 & 문구 복사 → ④ 기념품 세트 확인<br>
    생성된 문구를 <a href="https://gamma.app/" target="_blank" style="color:#1a73e8;font-weight:bold;">감마AI</a> · <a href="https://www.miricanvas.com/" target="_blank" style="color:#1a73e8;font-weight:bold;">미리캔버스</a> 등 디자인 도구에 붙여넣어 완성하세요.
    </div>""", unsafe_allow_html=True)

    _lt_c1, _lt_c2 = st.columns([1, 1])

    with _lt_c1:
        # 지사 선택
        _lt_offices = list(FULL_OFFICE_KB.keys())
        _lt_office_display = []
        for _k in _lt_offices:
            # 복합 키에서 대표명 추출
            _lt_office_display.append(_k.split("/")[0] if "/" in _k else _k)
        _lt_sel_idx = st.selectbox("지사 선택", range(len(_lt_offices)),
                                    format_func=lambda i: _lt_office_display[i],
                                    key="letter_office")
        _lt_sel_key = _lt_offices[_lt_sel_idx]
        _lt_sel_name = _lt_office_display[_lt_sel_idx]
        _lt_kb = FULL_OFFICE_KB[_lt_sel_key]

    with _lt_c2:
        # 계절 선택
        _lt_season = st.selectbox("계절 선택", list(_SEASON_THEMES.keys()), key="letter_season")
        _lt_theme = _SEASON_THEMES[_lt_season]

    st.markdown("---")

    # ── 지사별 데이터 집계 (서한문 톤·기념품 반영용) ──────────
    _lt_region_type = _get_region_type(_lt_kb.get("context", ""))

    # (a) 상위 불만 키워드 (top 3) — 사전 계산된 _pre_neg_res 재활용
    _lt_top_complaints = []
    _lt_branch_df = df_f[df_f[M["office"]] == _lt_sel_name] if M.get("office") else df_f
    if M.get("voc") and len(_lt_branch_df) > 0:
        try:
            _lt_neg_kws = _pre_neg_res.loc[_lt_branch_df.index].apply(lambda x: x[1])
            _lt_kw_counter = Counter([kw for kws in _lt_neg_kws for kw in kws])
            _lt_top_complaints = [kw for kw, _ in _lt_kw_counter.most_common(3)]
        except Exception:
            pass

    # (b) 주요 계약종별
    _lt_dominant_contract = ""
    _lt_contract_pct = ""
    if M.get("contract") and M["contract"] in _lt_branch_df.columns and len(_lt_branch_df) > 0:
        try:
            _vc = _lt_branch_df[M["contract"]].value_counts()
            _lt_dominant_contract = _vc.index[0]
            _lt_contract_pct = f"{_vc.iloc[0] / len(_lt_branch_df) * 100:.0f}%"
        except Exception:
            pass

    # ── 기본 템플릿 문구 (인제 스타일 기반) ──────────────────
    _region_desc = _lt_kb.get("context", "").split("/")[0].strip().replace("[", "").replace("]", "") if _lt_kb.get("context") else ""
    _lt_region_name = _lt_sel_key.replace("지사", "").replace("/", "·")
    # 지사명에서 관할 지역 추출
    _lt_area = _lt_sel_name.replace("지사", "")

    from datetime import datetime as _dt_now
    _lt_now = _dt_now.now()
    _lt_date_str = f"{_lt_now.year}년 {_lt_now.month}월"

    _lt_default_body = (
        f"고객님, 안녕하십니까. 한국전력 {_lt_sel_name}입니다.\n"
        f"항상 한국전력에 보내주시는 성원에 진심으로 감사드리며\n"
        f"{_lt_theme['greeting']}\n"
        f"\n"
        f"한국전력 {_lt_sel_name}는 {_lt_area} 일원을 관할하며\n"
        f"안정적인 전력공급과 최상의 서비스 제공을 제1의 목표로 삼아 늘 고민하고 있습니다.\n"
        f"또한 {_lt_sel_name}에서는 언제나 고객님의 목소리에 귀 기울이기 위해 노력하고 있으며,\n"
        f"친절·정확·신속한 서비스로 고객님께서 \"매우만족\" 하실 수 있도록 항상 노력하겠습니다.\n"
        f"\n"
        f"전기 사용과 관련하여 미흡하거나 궁금하신 사항이 있으실 경우\n"
        f"한전 고객센터(국번없이 123) 또는 한전ON으로 언제든지 연락주시기 바랍니다.\n"
        f"\n"
        f"{_lt_theme['closing']}\n"
        f"\n"
        f"{_lt_date_str}\n"
        f"한국전력공사 {_lt_sel_name}"
    )

    # ── AI 서한문 생성 or 기본 템플릿 ────────────────────────
    _lt_body_key = f"letter_body_{_lt_sel_key}_{_lt_season}"
    _lt_ta_key = f"letter_textarea_{_lt_sel_key}_{_lt_season}"

    if _lt_body_key not in st.session_state:
        st.session_state[_lt_body_key] = _lt_default_body

    _lt_col_preview, _lt_col_text = st.columns([1, 1])

    with _lt_col_text:
        st.markdown(f"### {_lt_theme['icon']} {_lt_sel_name} · {_lt_theme['season_name']} 서한문")

        # 데이터 반영 상태 캡션
        _lt_caption_parts = []
        if _lt_dominant_contract:
            _lt_caption_parts.append(f"주요 고객: {_lt_dominant_contract}({_lt_contract_pct})")
        if _lt_top_complaints:
            _lt_caption_parts.append(f"주요 개선 요청: {', '.join(_lt_top_complaints)}")
        if _lt_caption_parts:
            st.caption(f"📊 {' │ '.join(_lt_caption_parts)}")

        if st.button("🤖 AI 맞춤 문구 생성", key="letter_ai_btn", type="primary", use_container_width=True):
            if not GEMINI_AVAILABLE:
                st.error("Gemini API 키가 설정되지 않았습니다. `.env` 파일에 `GEMINI_API_KEY`를 설정해주세요.")
            else:
                # 불만 키워드 → 약속 문장 힌트 (중복 제거)
                _lt_promises = list(dict.fromkeys(
                    _COMPLAINT_TO_PROMISE[kw] for kw in _lt_top_complaints if kw in _COMPLAINT_TO_PROMISE))
                _lt_promise_hint = ("특히 다음 약속을 자연스럽게 녹여주세요: " + " / ".join(_lt_promises)) if _lt_promises else ""
                _lt_prompt = (
                    f"당신은 따뜻하고 진심 어린 편지를 쓰는 작가입니다. "
                    f"한국전력공사 {_lt_sel_name}에서 경험고객에게 보내는 {_lt_theme['season_name']} 서한문(감사 편지)을 작성하세요.\n\n"
                    f"[작성 대상]\n"
                    f"- 지사: 한국전력 {_lt_sel_name} (관할: {_lt_area})\n"
                    f"- 계절: {_lt_theme['season_name']}\n"
                    f"- 날짜: {_lt_date_str}\n\n"
                    f"[서한문 구조 — 반드시 5개 문단 모두 포함, 문단 사이 빈 줄]\n"
                    f"1문단) 인사 + 계절감: '고객님, 안녕하십니까. 한국전력 {_lt_sel_name}입니다.'로 시작. "
                    f"{_lt_theme['season_name']}의 정취를 담은 서정적 인사 1~2문장 (해당 지역의 자연·풍경을 은은히 녹여주세요)\n"
                    f"2문단) 감사: 변함없이 한국전력을 믿고 이용해주시는 고객 여러분께 진심 어린 감사 표현 2~3문장. "
                    f"고객 한 분 한 분의 소중함을 느낄 수 있는 따뜻한 문장으로 작성\n"
                    f"3문단) 약속: {_lt_sel_name}는 {_lt_area} 일원을 관할하며 안정적 전력공급과 최상의 서비스를 약속. "
                    f"친절·정확·신속한 서비스로 고객님께서 '매우만족' 하실 수 있도록 늘 노력하겠다는 다짐 2~3문장. "
                    f"{_lt_promise_hint}\n"
                    f"4문단) 안내: 전기 사용 관련 미흡하거나 궁금하신 사항은 "
                    f"한전 고객센터(국번없이 123) 또는 한전ON(online.kepco.co.kr)으로 연락 안내\n"
                    f"5문단) 마무리: {_lt_theme['season_name']}에 맞는 건강·행복 기원 인사 + '감사합니다.'\n\n"
                    f"[형식]\n"
                    f"- 마지막에 반드시 '{_lt_date_str}' 과 '한국전력공사 {_lt_sel_name}'을 넣으세요\n"
                    f"- 전체 분량: 12~18줄 (충분히 정성스럽게)\n"
                    f"- 본문 텍스트만 출력 (제목, 번호, 이모지, 영어, 마크다운 서식 금지)\n\n"
                    f"[문체 지침]\n"
                    f"- 담백하고 격식 있는 공공기관 서한문 톤. 따뜻하되 절제된 표현\n"
                    f"- 계절감은 첫 인사에서 한두 문장 정도만 간결하게\n"
                    f"- 지역 자연환경은 1곳만 짧게 언급하거나 생략해도 무방\n"
                    f"- 대등한 위치에서 감사를 전하는 어조 (저자세·과잉 겸손 금지)\n"
                    f"- 고객층 톤: {_CONTRACT_TONE.get(_lt_dominant_contract, '보편적이고 따뜻한 톤')}\n"
                    f"- 지역 톤: {_REGION_TONE.get(_lt_region_type, '보편적이고 따뜻한 톤')}\n\n"
                    f"[절대 금지]\n"
                    f"- '국민기업', '국민 기업', '국민과 함께' 표현 금지\n"
                    f"- '감사할 따름', '송구', '부족하나마', '미력하나마' 등 과잉 겸손 표현 금지\n"
                    f"- '빛이 되어드리겠습니다', '든든한 에너지' 등 과도한 비유 금지\n"
                    f"- 발전소, 에너지 산업, 인구 구성 언급 금지\n"
                )
                with st.spinner("AI가 맞춤 문구를 작성하고 있습니다..."):
                    try:
                        import urllib.request
                        # gemini-2.5-pro: 감성적 글쓰기 최적 (무료 100회/일)
                        # gemini-2.5-flash-lite: thinking 없는 빠른 폴백 (무료 1000회/일)
                        _models = ["gemini-2.5-pro", "gemini-2.5-flash-lite"]
                        _payload = {"contents": [{"parts": [{"text": _lt_prompt}]}],
                                     "generationConfig": {"temperature": 0.75, "maxOutputTokens": 2048}}
                        _ctx = ssl._create_unverified_context()
                        _body = None
                        for _model in _models:
                            _api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{_model}:generateContent?key={_GEMINI_KEY}"
                            _req = urllib.request.Request(_api_url, data=json.dumps(_payload).encode("utf-8"),
                                                           headers={"Content-Type": "application/json"}, method="POST")
                            try:
                                with urllib.request.urlopen(_req, context=_ctx, timeout=90) as _resp:
                                    _body = json.loads(_resp.read().decode("utf-8"))
                                break
                            except urllib.error.HTTPError as _http_err:
                                if _http_err.code in (429, 503):
                                    continue
                                raise
                        if _body is None:
                            st.error("모든 AI 모델의 일일 한도가 소진되었습니다. 내일 다시 시도해주세요.")
                        else:
                            _ai_letter = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
                            # 마크다운 코드블럭 제거
                            _ai_letter = re.sub(r"^```[a-z]*\n?", "", _ai_letter)
                            _ai_letter = re.sub(r"\n?```$", "", _ai_letter.strip())
                            st.session_state[_lt_body_key] = _ai_letter
                            # text_area 위젯 키도 동기화 (Streamlit 위젯 키 우선 문제 방지)
                            if _lt_ta_key in st.session_state:
                                del st.session_state[_lt_ta_key]
                            st.rerun()
                    except Exception as e:
                        st.error(f"AI 서한문 생성 중 오류: {e}")

        if st.button("🔄 기본 템플릿으로 초기화", key="letter_reset_btn", use_container_width=True):
            st.session_state[_lt_body_key] = _lt_default_body
            if _lt_ta_key in st.session_state:
                del st.session_state[_lt_ta_key]
            st.rerun()

        # 편집 가능한 텍스트 영역
        _lt_edited = st.text_area(
            "서한문 내용 (직접 수정 가능)",
            value=st.session_state[_lt_body_key],
            height=350,
            key=f"letter_textarea_{_lt_sel_key}_{_lt_season}")
        st.session_state[_lt_body_key] = _lt_edited

        # 복사용 텍스트 출력
        st.markdown("**📋 복사용 텍스트**")
        st.code(_lt_edited, language=None)
        st.caption("위 텍스트 박스 우측 상단 복사 버튼(📋)을 눌러 복사하세요.")

    with _lt_col_preview:
        st.markdown(f"### 👁️ 미리보기")
        _lt_buf = _render_letter_preview(_lt_sel_name, _lt_season, _lt_edited, _lt_theme)
        st.image(_lt_buf, use_container_width=True,
                 caption=f"{_lt_sel_name} · {_lt_theme['season_name']} 서한문 레이아웃 미리보기")
        st.markdown("※ 실제 서한문은 <a href='https://gamma.app/' target='_blank' style='color:#1a73e8;'>감마AI</a> / <a href='https://www.miricanvas.com/' target='_blank' style='color:#1a73e8;'>미리캔버스</a>에서 계절 일러스트를 추가하여 완성하세요.", unsafe_allow_html=True)

    # ── 기념품 세트 추천 ─────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🎁 {_lt_theme['season_name']} 추천 기념품 세트 — {_lt_sel_name}")

    _lt_season_name = _lt_theme["season_name"]
    _lt_gifts_common = _GIFT_DB.get(_lt_season_name, {}).get("공통", [])
    _lt_gifts_region = _GIFT_DB.get(_lt_season_name, {}).get(_lt_region_type, [])

    st.markdown(f'<div class="card-teal">'
                f'<b>📍 {_lt_sel_name} 지역유형:</b> {_lt_region_type} '
                f'({_lt_kb.get("context", "").split(".")[0] if _lt_kb.get("context") else "정보 없음"})'
                f'</div>', unsafe_allow_html=True)

    _gc_list = st.columns(2)

    with _gc_list[0]:
        st.markdown("**공통 추천 (어떤 지사든 인기 보장)**")
        _gift_table = []
        for _gname, _gprice, _gdesc in _lt_gifts_common:
            _gift_table.append({"기념품": _gname, "예상 단가": _gprice, "추천 이유": _gdesc})
        if _gift_table:
            st.dataframe(pd.DataFrame(_gift_table), use_container_width=True, hide_index=True)

    with _gc_list[1]:
        st.markdown(f"**{_lt_region_type} 특화 추천**")
        _gift_region_table = []
        for _gname, _gprice, _gdesc in _lt_gifts_region:
            _gift_region_table.append({"기념품": _gname, "예상 단가": _gprice, "추천 이유": _gdesc})
        if _gift_region_table:
            st.dataframe(pd.DataFrame(_gift_region_table), use_container_width=True, hide_index=True)
        else:
            st.info("공통 추천을 활용하세요.")

    # 추천 조합 카드
    st.markdown("**💡 추천 조합 (2~3천원 예산)**")
    _combo_items = []
    if _lt_gifts_common:
        _combo_items.append(_lt_gifts_common[0][0])  # 종량제 봉투
    if _lt_gifts_region:
        _combo_items.append(_lt_gifts_region[0][0])
    elif len(_lt_gifts_common) > 1:
        _combo_items.append(_lt_gifts_common[1][0])

    st.markdown(
        f'<div style="background:linear-gradient(135deg,{_lt_theme["bg"]},{_lt_theme["sub"]}30);'
        f'border:2px solid {_lt_theme["border"]};border-radius:12px;padding:20px;'
        f'text-align:center;font-size:1.1em;">'
        f'{_lt_theme["icon"]} <b>{_lt_theme["season_name"]} 베스트 조합</b><br><br>'
        f'<span style="font-size:1.3em;">{"  +  ".join(_combo_items)}</span><br><br>'
        f'<span style="color:{_lt_theme["accent"]};font-weight:bold;">서한문과 함께 동봉 시 고객 감동 효과 극대화!</span>'
        f'</div>', unsafe_allow_html=True)

