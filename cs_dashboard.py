
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
import io, re, os, json, ssl

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
    "직할/마산지사/진해지사": {
        "context": "[공단형] 인구 100.5만 / 4050 중장년층 주류. 주요 산업: 원자력, 방산, 가전. 기질: 강직함, 원칙 준수.",
        "action": "산단 전담 요금제 설명회, 야간 긴급 복구 알림 고도화, 기업 전용 상담 채널 강화."
    },
    "김해지사": {
        "context": "[공단/도심형] 인구 53.5만 / 젊은 생산인구 유입 활발. 주요 산업: 중소 제조 산단 밀집. 기질: 실용적, 빠른 처리 중시.",
        "action": "중소기업 전용 결제 편의 서비스(카카오페이 등) 홍보, 비대면 채널 이탈 방지 가이드 제공."
    },
    "양산지사": {
        "context": "[도심/공단형] 인구 35.5만 / 부산 배후 신도시, 3040 비중 높음. 주요 산업: 자동차 부품, IT 제조. 기질: 변화에 민감, 합리적.",
        "action": "신규 아파트 단지 '이사 정산 QR' 배포, 한전ON 앱 활용 비대면 신청률 제고 캠페인."
    },
    "함안지사": {
        "context": "[공단/복합형] 인구 6.1만 / 고령층 비중 높음. 주요 산업: 기계, 자동차부품(창원 배후). 기질: 넉넉한 인심, 뚝심.",
        "action": "노후 공단 대상 전력 설비 안전 점검 안내, 지역 커뮤니티 중심 대면 홍보 병행."
    },
    "밀양지사": {
        "context": "[복합/농촌형] 인구 10.2만 / 고령화 진행 중. 주요 산업: 나노융합산단, 시설원예(딸기). 기질: 보수적, 선비 정신.",
        "action": "시설원예 단지 전력 과부하 사전 예보, 고령 농민 대상 요금 감면 제도 현장 안내."
    },
    "창녕지사": {
        "context": "[농촌/관광형] 인구 5.8만 / 초고령사회. 주요 산업: 마늘/양파 농사, 관광. 기질: 소박함, 끈기.",
        "action": "영농기 농사용 전기 대리 접수 활성화, 관광지 숙박업소 전용 요금 컨설팅."
    },
    "거제지사": {
        "context": "[공단/해안형] 인구 23.3만 / 젊은 층 비중 높음. 주요 산업: 조선, 해양플랜트. 기질: 화끈함, 직설적.",
        "action": "결론 중심의 선제적 알림톡 발송, 조선소 부하 변동에 따른 맞춤형 컨설팅 제공."
    },
    "통영지사": {
        "context": "[관광/해안형] 인구 12.0만 / 관광/수산 종사자 중심. 주요 산업: 관광, LNG 거점. 기질: 예술적 자부심, 깐깐함.",
        "action": "도서(섬) 지역 정전 대응 매뉴얼 배포, 숙박시설/수산가공업 전용 요금 안내문 제작."
    },
    "사천지사": {
        "context": "[공단/해안형] 인구 10.9만 / 3040 항공 전문가 유입. 주요 산업: 항공우주, 위성. 기질: 급한 성격, 정이 많음.",
        "action": "항공 정밀 제조 전용 '고품질 전력 관리 리포트' 발송, 신규 공장 신청 절차 간소화."
    },
    "고성지사": {
        "context": "[농촌/해안형] 인구 5.0만 / 초고령사회. 주요 산업: 조선 부품, 농수산업. 기질: 보수적, 협조적.",
        "action": "읍면사무소 협업 '찾아가는 창구' 운영, 농사용 전기 부정사용 방지 캠페인."
    },
    "남해지사": {
        "context": "[관광/농촌형] 인구 4.2만 / 은퇴 이주민 비중 높음. 주요 산업: 관광, 마늘 농사. 기질: 독립적, 조용함.",
        "action": "귀촌인 대상 전력 사용 가이드 배부, 이장님 네트워크 활용 비대면 서비스 홍보."
    },
    "하동지사": {
        "context": "[농촌형] 인구 4.1만 / 고령화 심화. 주요 산업: 화력발전소, 농업. 기질: 보수적, 정이 많음.",
        "action": "발전소 주변 지역 지원 제도 상세 안내, 영농기 맞춤형 농사용 신청 알림톡 발송."
    },
    "진주지사": {
        "context": "[도심형] 인구 34.2만 / 교육/혁신도시 중심. 주요 산업: 우주항공 배후, 세라믹. 기질: 체면 중시, 합리적.",
        "action": "혁신도시 3040 타겟 디지털 서비스 강화, 공공기관 전용 에너지 효율화 상담."
    },
    "산청지사": {
        "context": "[농촌형] 인구 3.4만 / 초고령사회. 주요 산업: 한방약초, 딸기 농사. 기질: 점잖음, 보수적.",
        "action": "약초 건조기 등 특정 부하 사용 시기 점검 안내, 마을회관 중심 대면 서비스 강화."
    },
    "함양지사": {
        "context": "[농촌형] 인구 3.7만 / 생산인구 감소세. 주요 산업: 산삼, 과수원(사과). 기질: 강직함, 뚝심.",
        "action": "과수원용 저온저장고 요금 체계 안내, 농번기 현장 민원 우선 처리제 운영."
    },
    "거창지사": {
        "context": "[복합/농촌형] 인구 6.0만 / 교육 도시(거창고 등). 주요 산업: 화강석, 사과 농사. 기질: 자부심 강함, 원칙적.",
        "action": "교육 시설 전용 전기 요금 컨설팅, 농사용 서류 간소화 및 온라인 접수 지원."
    },
    "의령지사": {
        "context": "[농촌형] 인구 2.5만 / 경남 최소 인구 지역. 주요 산업: 망개떡, 원예. 기질: 충성심 높음, 정이 많음.",
        "action": "전력 소외 계층 방문 서비스 확대, 노후 주택 배선 점검 및 안전 서비스 병행."
    },
    "합천지사": {
        "context": "[관광/농촌형] 인구 4.2만 / 초고령사회. 주요 산업: 해인사 관광, 축산업. 기질: 끈기, 묵묵함.",
        "action": "사찰 및 관광지 조명 설비 효율화 상담, 축사 전용 정전 예보 서비스 강화."
    }
}

def _get_office_kb(office_name: str) -> dict:
    """지사명으로 KB를 검색. 복합 키(창원지사/마산지사/진해지사)도 매칭."""
    if not office_name:
        return {}
    for key, val in FULL_OFFICE_KB.items():
        if office_name in key:
            return val
    return {}

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
]:
    if os.path.exists(_c):
        FONT_PATH = _c
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
OFFICE_ORDER  = ["직할", "진주지사", "마산지사", "거제지사", "밀양지사", "사천지사",
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
    "불만","불편","민원","항의","화남","짜증","느림","느려","오류","오작동",
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


def _classify_sentiment_3tier(text):
    """VOC 3단계 감성 분류: 긍정/중립/부정 (역접 패턴 + 결론 우선)"""
    if not text or str(text).strip() in ("", "nan", "응답없음"):
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
    """데이터 기반 한 줄 인사이트 자동 생성"""
    parts = []
    if score_col and score_col in df.columns and indiv_scores:
        num = df[indiv_scores + [score_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(num) > 3:
            corr = num.corr()[score_col].drop(score_col).sort_values(ascending=False)
            parts.append(f"종합 점수에 가장 큰 영향을 미치는 항목은 **'{corr.index[0]}'** (상관계수 {corr.iloc[0]:.3f})입니다.")
    if score_col and cat_cols:
        for cat in cat_cols:
            if cat not in df.columns:
                continue
            grp = df.groupby(cat)[score_col].mean()
            if grp.empty:
                continue
            worst = grp.idxmin()
            best = grp.idxmax()
            gap = grp.max() - grp.min()
            if gap >= 3:
                parts.append(
                    f"'{cat}' 기준 **'{worst}'**({grp.min():.1f}점)의 만족도가 가장 낮으며, "
                    f"**'{best}'**({grp.max():.1f}점)과 {gap:.1f}점 차이가 있습니다.")
                break
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
  <p>종합 현황 · 계약종별/채널별/업무별 분석 · 지사 맞춤형 CS 솔루션 · 민원 조기 경보 · CXO 딥 인사이트</p>
  <span class="dash-badge">📊 종합 현황</span>
  <span class="dash-badge">📡 계약종별·업무유형별 분석</span>
  <span class="dash-badge">🏢 지사 맞춤 솔루션</span>
  <span class="dash-badge">🎯 민원 조기 경보</span>
  <span class="dash-badge">🧠 CXO 딥 인사이트</span>
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
5. **📡 계약종별·업무유형별 분석** — 교차 분석 및 사분면 비교
6. **🏢 지사 맞춤형 CS 솔루션** — 지사 선택 후 단계별 정밀 진단
7. **🎯 민원 조기 경보** — 잠재 민원고객 사전케어 리스트
8. **🧠 CXO 딥 인사이트** — 경영진용 핵심 요약 및 AI 분석
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
    df.drop_duplicates(inplace=True)
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
    "contract":  _find_col(["계약종"]),
    "business":  _find_col(["업무구분", "업무유형"]),
    "score":     _find_col(["종합 점수", "종합점수"]),
    "voc":       _find_col(["서술 의견", "서술의견"]),
    "age":       _find_col(["연령"]),
    "date":      _find_col(["접수일자", "조사일자", "접수일", "조사일", "일자", "날짜", "등록일"]),
    "receipt_no": _find_col(["접수번호"]),
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

# 우선순위: 접수번호 → 날짜 컬럼
_date_parsed = False
if M["receipt_no"]:
    df_raw["_접수일"] = df_raw[M["receipt_no"]].apply(_parse_date_from_receipt)
    if df_raw["_접수일"].dropna().any():
        M["date"] = "_접수일"
        _date_parsed = True

if not _date_parsed and M["date"]:
    df_raw[M["date"]] = pd.to_datetime(df_raw[M["date"]], errors="coerce")
    if df_raw[M["date"]].dropna().empty:
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

# ── 개별 점수 컬럼 자동 탐지 ──
INDIVIDUAL_SCORE_NAMES = [
    "전반적 만족", "직원 친절도", "처리 신속도",
    "처리 정확도", "업무 개선도",
    "이용 편리성", "사용 추천도",
]
individual_scores = [c for c in INDIVIDUAL_SCORE_NAMES if c in df_raw.columns]

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
        if sel_office:
            df_f = df_f[df_f[M["office"]].astype(str).isin(sel_office)]
    if M["channel"]:
        chan_opts = sorted(df_raw[M["channel"]].dropna().astype(str).unique().tolist())
        sel_chan = st.multiselect("📡 접수자구분", chan_opts, default=chan_opts, key="f_chan")
        if sel_chan:
            df_f = df_f[df_f[M["channel"]].astype(str).isin(sel_chan)]
    if M["method"]:
        method_opts = sorted(df_raw[M["method"]].dropna().astype(str).unique().tolist())
        sel_method = st.multiselect("📞 신청방법", method_opts, default=method_opts, key="f_method")
        if sel_method:
            df_f = df_f[df_f[M["method"]].astype(str).isin(sel_method)]
    if M["reception"]:
        recep_opts = sorted(df_raw[M["reception"]].dropna().astype(str).unique().tolist())
        sel_recep = st.multiselect("📝 접수종류", recep_opts, default=recep_opts, key="f_recep")
        if sel_recep:
            df_f = df_f[df_f[M["reception"]].astype(str).isin(sel_recep)]
    if M["contract"]:
        cont_opts = sorted(df_raw[M["contract"]].dropna().astype(str).unique().tolist())
        sel_cont = st.multiselect("📋 계약종별", cont_opts, default=cont_opts, key="f_cont")
        if sel_cont:
            df_f = df_f[df_f[M["contract"]].astype(str).isin(sel_cont)]
    if M["business"]:
        biz_opts = sorted(df_raw[M["business"]].dropna().astype(str).unique().tolist())
        sel_biz = st.multiselect("🏢 업무구분", biz_opts, default=biz_opts, key="f_biz")
        if sel_biz:
            df_f = df_f[df_f[M["business"]].astype(str).isin(sel_biz)]
    if M["age"]:
        age_opts = sorted(df_raw[M["age"]].dropna().astype(str).unique().tolist())
        sel_age = st.multiselect("👥 연령대", age_opts, default=age_opts, key="f_age")
        if sel_age:
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
    df_f["_VOC감성"] = df_f[M["voc"]].apply(_classify_sentiment_3tier)  # 투트랙: 긍정/중립/부정
    df_f["_is_oos"] = df_f[M["voc"]].apply(_is_out_of_scope)           # 통제 불가 필터링
else:
    voc_response_rate = 0.0

neg_ratio = neg_cnt / max(len(voc_texts_all), 1) * 100

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
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    if avg_score_100 is not None and not np.isnan(avg_score_100):
        st.metric("⭐ 종합만족도", f"{avg_score_100:.1f}점",
                  delta=f"목표 대비 {avg_score_100 - 80:+.1f}", delta_color="normal")
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
    if M["voc"] and "_VOC분류" in df_f.columns:
        rude_cnt = len(df_f[df_f["_VOC분류"] == "불친절"])
        st.metric("😡 불친절 감지", f"{rude_cnt:,}건",
                  delta=f"{rude_cnt / max(len(df_f), 1) * 100:.1f}%", delta_color="inverse")
    else:
        st.metric("😡 불친절 감지", "미선택")
with m6:
    if M["voc"]:
        st.metric("🚨 잠재 민원", f"{neg_cnt:,}명",
                  delta=f"전체의 {neg_ratio:.1f}%", delta_color="inverse")
    else:
        st.metric("🚨 잠재 민원", "미선택")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  13. 탭 구성
# ══════════════════════════════════════════════════════════════
tab1, tab3, tab_sol, tab_sol2, tab5, tab10 = st.tabs([
    "📊  종합 현황",
    "📡  계약종별 · 업무유형별 · 항목별 분석",
    "🏢  지사 맞춤형 CS 솔루션",
    "🔬  3단계 정밀 진단",
    "🎯  민원 조기 경보 시스템",
    "🧠  CXO 딥 인사이트",
])

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
                mode="gauge+number+delta",
                value=round(avg_score_100, 1),
                number={"suffix": "점", "font": {"size": 40, "color": C["navy"]}},
                delta={"reference": 80, "increasing": {"color": C["green"]},
                       "decreasing": {"color": C["red"]}},
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
                    "threshold": {"line": {"color": C["red"], "width": 3},
                                  "thickness": 0.78, "value": 80},
                },
                title={"text": "종합 만족도 (100점 환산)<br><span style='font-size:0.8em;color:gray'>빨간 선 = 목표 80점</span>",
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
            fig_bp = px.pie(bucket_data, names="구간", values="건수", color="구간",
                            color_discrete_map=BUCKET_COLORS, hole=0.45, template=PLOTLY_TPL)
            fig_bp.update_traces(textposition="outside", textinfo="percent+label", textfont_size=13,
                                  marker=dict(line=dict(color="#ffffff", width=2)),
                                  hovertemplate="%{label}<br>%{value:,}건 (%{percent})<extra></extra>")
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
            fig_bench = px.bar(_ofc_grp_bar, x="평균만족도", y="사업소", color="그룹",
                               color_discrete_map={"⬆ 본부평균 이상": C["teal"], "⬇ 본부평균 미달": C["red"]},
                               orientation="h", text="평균만족도", template=PLOTLY_TPL,
                               title=f"사업소별 평균 만족도 — 본부 평균: {_bar_avg:.1f}점")
            fig_bench.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                                    hovertemplate="%{y}<br>평균: %{x:.1f}점<extra></extra>")
            fig_bench.add_vline(x=_bar_avg, line_color=C["navy"], line_dash="dash", line_width=2.5,
                                annotation_text=f"▼ 본부 평균 {_bar_avg:.1f}",
                                annotation_font_size=12, annotation_font_color=C["navy"],
                                annotation_position="top", annotation_yshift=10)
            fig_bench.update_layout(height=max(350, len(_ofc_grp_bar) * 35 + 80),
                                     margin=dict(t=80, b=20, l=10, r=100), legend_title_text="",
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
                                     text=_biz_df["비율(%)"].apply(lambda v: f"{v:.1f}%"),
                                     color_discrete_sequence=[C["sky"]], template=PLOTLY_TPL,
                                     title=f"{_title} 분포")
                    fig_biz.update_traces(textposition="outside", textfont_size=11,
                                          hovertemplate="%{y}: %{x:.1f}% (%{customdata[0]:,}건)<extra></extra>",
                                          customdata=_biz_df[["건수"]].values)
                    fig_biz.update_layout(height=max(360, len(_biz_df) * 30 + 80),
                                           margin=dict(t=50, b=20, l=20, r=90), showlegend=False,
                                           title_font=dict(size=15, color=C["navy"]),
                                           xaxis_title="비율(%)", yaxis_title="")
                    st.plotly_chart(fig_biz, use_container_width=True, config={'staticPlot': True})
                else:
                    # 연령대, 계약종별: 기존 파이차트 유지
                    fig_pie = px.pie(names=counts.index, values=counts.values, color_discrete_sequence=PIE_COLORS,
                                     hole=0.42, title=f"{_title} 분포", template=PLOTLY_TPL)
                    fig_pie.update_traces(textposition="outside", textinfo="percent+label", textfont_size=12,
                                           marker=dict(line=dict(color="#ffffff", width=2)),
                                           hovertemplate="%{label}<br>%{value:,}건 (%{percent})<extra></extra>")
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
    grp["구분"] = grp[cat_label].apply(lambda x: "🔴 하위 3" if x in bottom3_names else "일반")

    # ── 전체 평균 막대 차트 ──
    fig = px.bar(grp, x="평균만족도", y=cat_label, color="구분",
                 color_discrete_map={"🔴 하위 3": C["red"], "일반": C["sky"]},
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
                html += f'<th colspan="2" style="border:1px solid {_bdr};padding:6px 4px;background:{_ylw};">합계</th>'
                html += '</tr>'
                html += f'<tr style="background:{_hdr};font-weight:bold;font-size:0.9em;">'
                for _ in _cols:
                    html += f'<th style="border:1px solid {_bdr};padding:4px;">종합<br>점수</th>'
                    html += f'<th style="border:1px solid {_bdr};padding:4px;">응답<br>호수</th>'
                html += f'<th style="border:1px solid {_bdr};padding:4px;background:{_ylw};">종합<br>점수</th>'
                html += f'<th style="border:1px solid {_bdr};padding:4px;background:{_ylw};">응답<br>호수</th>'
                html += '</tr>'
            else:
                # ── 양식3: 1단 헤더 (업무유형/접수채널) ──
                html += f'<tr style="background:{_hdr};font-weight:bold;">'
                html += f'<th style="border:1px solid {_bdr};padding:6px 10px;min-width:70px;">구분</th>'
                for c in _cols:
                    html += f'<th style="border:1px solid {_bdr};padding:6px 4px;">{c}</th>'
                html += f'<th style="border:1px solid {_bdr};padding:6px 4px;background:{_ylw};">합계</th>'
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
                    html += f'<td style="border:1px solid {_bdr};padding:4px;background:{_ylw};font-weight:bold;">{_t_str}</td>'
                    html += f'<td style="border:1px solid {_bdr};padding:4px;background:{_ylw};">{_t_cnt}</td>'
                else:
                    html += f'<td style="border:1px solid {_bdr};padding:4px;background:{_ylw};font-weight:bold;">{_t_str}</td>'
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
        # ── ① 계약종별 분석 ──
        if M["contract"]:
            st.markdown('<p class="sec-head">📋 계약종별 만족도 분석</p>', unsafe_allow_html=True)
            _render_category_section(df_f, M["contract"], "계약종별",
                                     M["office"], "_점수100", avg_score_100)
            st.markdown("---")

        # ── ② 업무유형별 분석 ──
        if M["business"]:
            st.markdown('<p class="sec-head">🏢 업무유형별 만족도 분석</p>', unsafe_allow_html=True)
            _render_category_section(df_f, M["business"], "업무유형",
                                     M["office"], "_점수100", avg_score_100)
            st.markdown("---")

        # ── ④ 항목별 결과 (양식1) ──
        if M["office"] and individual_scores and "_점수100" in df_f.columns:
            st.markdown('<p class="sec-head">📋 사업소별 만족도 조사결과 — 항목별 결과</p>', unsafe_allow_html=True)
            _item_offices = _sort_offices(df_f[M["office"]].dropna().unique().tolist())
            _item_rows = []
            for _ofc in _item_offices:
                _odf = df_f[df_f[M["office"]] == _ofc]
                _row = {"구분": _ofc, "업무처리완료고객": "-", "발송호수": "-",
                        "응답호수": len(_odf), "응답률(%)": "-"}
                for _sc in individual_scores:
                    if _sc in _odf.columns:
                        _val = _odf[_sc].dropna()
                        _row[_sc] = round(_val.mean(), 1) if len(_val) > 0 else ""
                    else:
                        _row[_sc] = ""
                _row["종합점수"] = round(_odf["_점수100"].mean(), 1) if len(_odf) > 0 else ""
                _item_rows.append(_row)

            _item_df = pd.DataFrame(_item_rows)
            # HTML 테이블 (양식1 스타일)
            _hdr_bg = "#d6e4f0"
            _border = "#b0b0b0"
            _yellow = "#fef9e7"
            _f1_html = '<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;font-size:0.85em;text-align:center;">'
            _f1_html += f'<tr style="background:{_hdr_bg};font-weight:bold;">'
            for _col in _item_df.columns:
                _bg = f"background:{_yellow};" if _col == "종합점수" else ""
                _f1_html += f'<th style="border:1px solid {_border};padding:6px 4px;{_bg}">{_col}</th>'
            _f1_html += '</tr>'
            for _, _row in _item_df.iterrows():
                _f1_html += '<tr>'
                for _col in _item_df.columns:
                    _v = _row[_col]
                    if _col == "구분":
                        _f1_html += f'<td style="border:1px solid {_border};padding:5px 8px;font-weight:bold;background:#f9f9f9;">{_v}</td>'
                    elif _col == "종합점수":
                        _f1_html += f'<td style="border:1px solid {_border};padding:4px;background:{_yellow};font-weight:bold;">{_v}</td>'
                    else:
                        _f1_html += f'<td style="border:1px solid {_border};padding:4px;">{_v}</td>'
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

            # 라벨 겹침 방지: annotation으로 처리
            _ann_positions = []  # (x, y) 저장하여 겹침 체크
            _bbl_annotations = []

            for q_name in _q_order:
                _sub = _bbl_grp[_bbl_grp["사분면"] == q_name]
                if _sub.empty:
                    continue
                _sub_idx = _sub.index.tolist()
                fig_bbl.add_trace(go.Scatter(
                    x=_sub["처리건수"], y=_sub["평균만족도"],
                    mode="markers",
                    marker=dict(size=(_sub["버블크기"] * 5).clip(lower=14),
                                color=_q_colors[q_name], opacity=0.75,
                                line=dict(width=1.5, color="white")),
                    hovertext=[_hover_texts[i] for i in _sub_idx],
                    hovertemplate="%{hovertext}<extra></extra>",
                    name=q_name,
                ))
                # annotation으로 라벨 배치 (겹침 시 위치 조정)
                _x_range = max(_bbl_grp["처리건수"].max() - _bbl_grp["처리건수"].min(), 1)
                _y_range = max(_bbl_grp["평균만족도"].max() - _bbl_grp["평균만족도"].min(), 1)
                for _, _r in _sub.iterrows():
                    # 후보 위치: 아래→위→더아래→더위→좌하→우상 순서로 시도
                    _candidates = [(0, -22), (0, 24), (0, -40), (0, 42),
                                   (-30, -22), (30, 24), (-30, 24), (30, -22)]
                    _ax, _ay = _candidates[0]
                    for _cax, _cay in _candidates:
                        _conflict = False
                        for _px, _py, _pax, _pay in _ann_positions:
                            _dx = abs((_r["처리건수"] - _px) / _x_range)
                            _dy = abs((_r["평균만족도"] - _py) / _y_range)
                            _near = _dx < 0.18 and _dy < 0.15
                            _same_side = abs(_cay - _pay) < 15 and _dx < 0.18
                            if _near and _same_side:
                                _conflict = True
                                break
                        if not _conflict:
                            _ax, _ay = _cax, _cay
                            break
                    _ann_positions.append((_r["처리건수"], _r["평균만족도"], _ax, _ay))
                    _bbl_annotations.append(dict(
                        x=_r["처리건수"], y=_r["평균만족도"],
                        text=_r["업무유형"], showarrow=True,
                        arrowhead=0, arrowwidth=0.8, arrowcolor="#aaa",
                        ax=_ax, ay=_ay,
                        font=dict(size=9.5, color="#333"),
                        bgcolor="rgba(255,255,255,0.8)", borderpad=2,
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

            fig_bbl.update_layout(
                height=560, margin=dict(t=60, b=60, l=60, r=40),
                xaxis_title="처리 건수", yaxis_title="평균 만족도 (100점 환산)",
                title=dict(text="업무유형별 사분면 분석 — 처리 건수 vs 만족도 (버블 크기 = 불만족 건수)",
                           font=dict(size=14, color=C["navy"])),
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(size=11)),
                template=PLOTLY_TPL)
            st.plotly_chart(fig_bbl, use_container_width=True, config={'staticPlot': True})

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
- 예: 상담 스크립트 수정, 콜백 절차 추가, 월간 모니터링 항목 신설, 우수지사 사례 회람, 담당자 교육 등.
- 이 데이터는 전월 1개월치 실적이며, 월초에 방향을 잡는 용도임. "이번 주 당장" 같은 급박한 톤 금지.
- 절대 금지: 액션 뒤에 "(이번 주)", "(다음 주)", "(다음 달)", "(이번 달)" 등 괄호 시기 표기를 절대 붙이지 마세요.
- 각 업무별 솔루션은 반드시 2개씩 제시.
- 반드시 업무명·VOC 키워드·수치 근거를 명시.

[사분면 데이터]
{_ai_q_data}
[3·4사분면 업무의 불만족 VOC 원문]
{chr(10).join(_ai_q_voc) if _ai_q_voc else '- 해당 없음'}

[출력 형식 — 아래 4단계를 순서대로]
## 1. 중점 관리 업무 (4사분면: 건수↑ 점수↓)
- 해당 업무가 본부 평균을 얼마나 깎는지 수치로 진단
- VOC에서 드러나는 핵심 불만 포인트
- 개선 솔루션 2개

## 2. 방치 리스크 (3사분면: 건수↓ 점수↓)
- 소량이라도 점수가 낮은 업무가 민원으로 번질 가능성
- 담당자 코칭 솔루션 2개

## 3. 우수 사례 활용 (1사분면: 건수↑ 점수↑)
- 잘하는 업무의 강점을 구체적으로 짚고
- 다른 업무에 전파할 현실적 방법 2개

## 4. 이번 달 개선 우선순위
- 본부 점수 상승 기여도 순으로 업무 나열
- 각 업무별 솔루션 2개씩"""

                    with st.spinner("AI가 업무유형별 처방전 생성 중…"):
                        try:
                            import urllib.request
                            _models = ["gemini-2.0-flash", "gemma-3-12b-it", "gemma-3-27b-it"]
                            _payload = {"contents": [{"parts": [{"text": _ai_q_prompt}]}],
                                         "generationConfig": {"temperature": 0.7, "maxOutputTokens": 3072}}
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
                                    if _http_err.code == 429:
                                        continue
                                    raise
                            if _body is None:
                                st.error("모든 AI 모델의 일일 한도가 소진되었습니다. 내일 다시 시도해주세요.")
                            else:
                                _ai_text = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
                                st.markdown(
                                    '<div style="background:#f8f9fb;border:1px solid #d0d7de;border-radius:10px;'
                                    'padding:24px 28px;margin:8px 0;font-size:0.93em;line-height:1.85;">\n\n'
                                    f'{_ai_text}\n\n</div>',
                                    unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"AI 분석 중 오류: {e}")

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

            # ── AI 사전케어 행동 가이드 ──
            st.markdown("---")
            st.markdown(f'<p class="sec-head">⚠️ 사전케어 행동 가이드 — 서비스 회복 골든타임 프로토콜</p>', unsafe_allow_html=True)

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

            if st.button("🤖 AI 맞춤 사전케어 가이드 생성", key="ai_precare_btn", type="primary", use_container_width=True):
                if not GEMINI_AVAILABLE:
                    st.error("Gemini API 키가 설정되지 않았습니다. `.env` 파일에 `GEMINI_API_KEY`를 설정해주세요.")
                else:
                    with st.spinner("Gemini AI가 VOC를 분석하여 맞춤 가이드를 생성 중입니다…"):
                        _precare_prompt = f"""당신은 전력산업 고객만족(CS) 전문 컨설턴트입니다.
아래는 고객 만족도 조사에서 추출된 **잠재 민원고객 {neg_n}명**의 데이터입니다.

[잠재 민원고객 현황]
- 총 {neg_n}명 (전체 대비 {neg_r:.1f}%)
- 민원고객 평균 점수: {df_neg["_점수100"].mean():.1f}점
- 전체 평균 점수: {avg_score_100:.1f}점
- 주요 부정 키워드 TOP10: {', '.join(_neg_kw_top)}
- 민원 집중 지사: {_neg_office_dist if _neg_office_dist else '정보 없음'}
- 민원 집중 업무: {_neg_biz_dist if _neg_biz_dist else '정보 없음'}

[실제 부정 VOC 원문 (최대 30건)]
{chr(10).join([f'- {v}' for v in _neg_voc_samples]) if _neg_voc_samples else '- VOC 데이터 없음'}

[요청사항]
위 데이터를 기반으로 **서비스 회복 골든타임 프로토콜**을 작성하세요.

반드시 아래 형식을 따르세요:

## 📊 현장 진단 요약
- 이 데이터에서 읽히는 핵심 문제 2~3가지를 구체적으로 지적 (추상적 표현 금지)

## 🔴 1단계: 즉시 조치 (72시간 내)
- 가장 시급한 민원 유형별로 **구체적인 멘트·행동**을 제시
- 해당 지사명, 업무명을 반드시 언급

## 🟡 2단계: 처리 완료 후 확인
- 고객별 팔로업 방법 (유형별로 다르게)

## 🟢 3단계: 재발 방지 (1주 내)
- 반복되는 불만 패턴에 대한 프로세스 개선 제안
- 구체적인 업무/지사를 명시

## 💡 이 데이터만의 특이 인사이트
- 일반적인 CS 교과서에 없는, 이 데이터에서만 발견되는 패턴 1~2가지

※ 뻔한 "친절 교육 실시", "매뉴얼 배포" 같은 추상적 제안은 절대 금지합니다.
※ 반드시 위 데이터의 키워드·지사명·업무명을 근거로 구체적으로 작성하세요."""

                        try:
                            import urllib.request
                            _models = ["gemini-2.0-flash", "gemma-3-12b-it", "gemma-3-27b-it"]
                            _payload = {"contents": [{"parts": [{"text": _precare_prompt}]}],
                                         "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4096}}
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
                                    if _http_err.code == 429:
                                        continue
                                    raise
                            if _body is None:
                                st.error("모든 AI 모델의 일일 한도가 소진되었습니다. 내일 다시 시도해주세요.")
                            else:
                                _ai_text = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
                                st.markdown(
                                    '<div style="background:#f8f9fb;border:1px solid #d0d7de;border-radius:10px;'
                                    'padding:24px 28px;margin:8px 0;font-size:0.93em;line-height:1.85;">\n\n'
                                    f'{_ai_text}\n\n</div>',
                                    unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"AI 분석 중 오류가 발생했습니다: {e}")
            else:
                st.caption("버튼을 누르면 Gemini AI가 실제 VOC 데이터를 분석하여 맞춤형 사전케어 가이드를 생성합니다.")
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

        if _top3_pred:
            for _rank, (_biz, _bn, _bavg, _blow, _bdp, _rsk) in enumerate(_top3_pred, 1):
                _icon = "🔴" if _rank == 1 else ("🟠" if _rank == 2 else "🟡")
                # 해당 업무의 불만 키워드 추출
                _biz_kws = []
                if M.get("voc"):
                    _biz_low_voc = df_f[(df_f[M["business"]] == _biz) & (df_f["_점수100"] < 70)]
                    _biz_texts = _biz_low_voc[M["voc"]].dropna().astype(str).tolist()
                    _biz_texts = [t for t in _biz_texts if t.strip() not in ("", "nan", "응답없음")]
                    if _biz_texts:
                        _biz_kw_result = extract_action_keywords(tuple(_biz_texts), top_n=3)
                        _biz_kws = [w for w, s, c in _biz_kw_result][:3]
                _kw_str = f" · 핵심 키워드: **{', '.join(_biz_kws)}**" if _biz_kws else ""
                st.markdown(f"{_icon} **{_rank}위 — {_biz}** | 처리 {_bn:,}건 · 만족도 {_bavg}점 · 불만족 {_blow}건({_bdp}%) · 리스크 점수 {_rsk}{_kw_str}")

        # AI 심층 예측 버튼
        st.markdown("")
        if st.button("🤖 AI 민원 예측 심층 분석", key="ai_predict_btn", type="primary", use_container_width=True):
            if not GEMINI_AVAILABLE:
                st.error("Gemini API 키가 설정되지 않았습니다.")
            else:
                # TOP3 업무의 VOC 샘플 수집
                _pred_voc_samples = []
                for _biz, *_ in _top3_pred:
                    _bvocs = df_f[(df_f[M["business"]] == _biz) & (df_f["_점수100"] < 70)]
                    if M.get("voc"):
                        _samples = _bvocs[M["voc"]].dropna().astype(str).tolist()
                        _samples = [t for t in _samples if t.strip() not in ("", "nan", "응답없음")][:10]
                        _pred_voc_samples.append((_biz, _samples))

                _pred_prompt = f"""당신은 전력산업 CS 분석 전문가입니다.
아래는 고객 만족도 조사에서 공식 민원 발전 가능성이 가장 높은 상위 3개 업무 유형입니다.

[리스크 TOP 3 업무]
"""
                for _biz, _bn, _bavg, _blow, _bdp, _rsk in _top3_pred:
                    _pred_prompt += f"- {_biz}: 처리 {_bn}건, 만족도 {_bavg}점, 불만족 {_blow}건({_bdp}%), 리스크점수 {_rsk}\n"

                _pred_prompt += "\n[업무별 불만족 VOC 원문]\n"
                for _biz, _samples in _pred_voc_samples:
                    _pred_prompt += f"\n▶ {_biz}:\n"
                    for _s in _samples:
                        _pred_prompt += f"  - {_s}\n"

                _pred_prompt += f"""
[분석 요청]
1. 각 업무별로 일주일 이내 공식 민원(국민신문고·언론·본사 접수)으로 발전할 가능성을 상/중/하로 판단하고, 근거를 구체적으로 제시하세요.
2. 각 업무별 선제 대응 액션플랜을 1~2줄로 제시하세요.
3. 3개 업무 중 가장 시급한 1개를 선정하고, 해당 업무의 '최악의 시나리오'와 '최선의 대응'을 각각 한 문장으로 작성하세요.

※ '직원이', '내용에', '문의' 같은 일반 단어는 근거에서 제외하세요.
※ 추상적 표현 금지. 구체적인 업무명·키워드·수치를 반드시 포함하세요."""

                with st.spinner("Gemini AI가 민원 예측 분석 중…"):
                    try:
                        import urllib.request
                        _models = ["gemini-2.0-flash", "gemma-3-12b-it", "gemma-3-27b-it"]
                        _payload = {"contents": [{"parts": [{"text": _pred_prompt}]}],
                                     "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4096}}
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
                                if _http_err.code == 429:
                                    continue
                                raise
                        if _body is None:
                            st.error("모든 AI 모델의 일일 한도가 소진되었습니다.")
                        else:
                            _sol_text = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
                            st.markdown(
                                '<div style="background:#f8f9fb;border:1px solid #d0d7de;border-radius:10px;'
                                'padding:24px 28px;margin:8px 0;font-size:0.93em;line-height:1.85;">\n\n'
                                f'{_sol_text}\n\n</div>',
                                unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"AI 분석 중 오류: {e}")
        else:
            st.caption("버튼을 누르면 Gemini AI가 VOC를 분석하여 민원 발전 가능성과 선제 대응 방안을 제시합니다.")
    else:
        st.info("업무유형 컬럼과 점수 컬럼이 필요합니다.")



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
        _df_sel  = df_f[df_f[M["office"]] == _sel_off].copy()

        if _df_sel.empty:
            st.warning(f"{_sel_off}의 데이터가 없습니다.")
            st.stop()

        # ══════════════════════════════════════════════════════
        # LEVEL 2 — 지사별 정밀 진단 (Diagnosis)
        # ══════════════════════════════════════════════════════

        # ── 피어 그룹 ──────────────────────────────────────────
        _off_counts = df_f.groupby(M["office"])["_점수100"].count().sort_values()
        _n_off = len(_off_counts)
        _t1 = _off_counts.iloc[_n_off // 3] if _n_off >= 3 else 0
        _t2 = _off_counts.iloc[2 * _n_off // 3] if _n_off >= 3 else float("inf")
        _my_cnt = int(_off_counts.get(_sel_off, 0))
        if _my_cnt <= _t1:
            _peer_label, _peer_tier = "소규모", 0
        elif _my_cnt <= _t2:
            _peer_label, _peer_tier = "중규모", 1
        else:
            _peer_label, _peer_tier = "대규모", 2
        _peer_offs = [o for o in _off_counts.index
                      if (_off_counts[o] <= _t1 if _peer_tier == 0
                          else _off_counts[o] <= _t2 if _peer_tier == 1
                          else _off_counts[o] > _t2)]
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
                        _pm_y_range = [max(55, _pm_grp["만족도"].min() - 5), min(105, _pm_grp["만족도"].max() + 5)]
                        fig_pm.add_annotation(x=_pm_x_range[1] * 0.85, y=_pm_y_range[0] + 2,
                                              text="⚠️ 1순위 개선", showarrow=False,
                                              font=dict(size=10, color="#c62828"))
                        fig_pm.add_annotation(x=1, y=_pm_y_range[0] + 2,
                                              text="🔍 특이 리스크", showarrow=False,
                                              font=dict(size=10, color="#e65100"))
                        fig_pm.update_layout(
                            template=PLOTLY_TPL, height=340,
                            margin=dict(t=30, b=50, l=50, r=20),
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

        # ── 업무 × 세부항목 히트맵 (지사 특화) ──────────────────────
        if M.get("business") and individual_scores:
            st.markdown("##### 🔥 업무 × 세부항목 리스크 히트맵 — " + _sel_off)
            st.caption("빨간 칸이 **실질적 리스크**입니다. 하단 리스크 카드에서 [상세 원인 분석]으로 범인을 특정하세요.")

            # 업무유형별 × 세부항목별 평균 점수 pivot
            _score_cols_avail = [c for c in individual_scores if c in _df_sel.columns]
            if _score_cols_avail:
                _hm_rows = []
                for _biz_name in _df_sel[M["business"]].dropna().unique():
                    _biz_df = _df_sel[_df_sel[M["business"]] == _biz_name]
                    _row = {"업무유형": _biz_name}
                    for _sc in _score_cols_avail:
                        _vals = pd.to_numeric(_biz_df[_sc], errors="coerce").dropna()
                        _row[_sc] = round(float(_vals.mean()), 1) if len(_vals) > 0 else float("nan")
                    _row["_건수"] = len(_biz_df)
                    _hm_rows.append(_row)
                _hm_df_sol = pd.DataFrame(_hm_rows).set_index("업무유형")
                _hm_cnt_sol = _hm_df_sol[["_건수"]].copy()
                _hm_pivot_sol = _hm_df_sol.drop(columns=["_건수"])

                _hm_vals = _hm_pivot_sol.values.astype(float)
                _hm_nonzero = _hm_vals[~np.isnan(_hm_vals)]
                _hm_range = [max(60, float(_hm_nonzero.min()) - 3) if len(_hm_nonzero) else 60, 100]

                fig_hm = px.imshow(
                    _hm_pivot_sol, color_continuous_scale="RdYlGn",
                    aspect="auto", template=PLOTLY_TPL,
                    range_color=_hm_range,
                    labels=dict(x="세부항목", y="업무유형", color="만족도"))
                _hm_text, _hm_hover = [], []
                for _bi in _hm_pivot_sol.index:
                    _trow, _hrow = [], []
                    _cn = int(_hm_cnt_sol.loc[_bi, "_건수"])
                    for _ci in _hm_pivot_sol.columns:
                        _sc = _hm_pivot_sol.loc[_bi, _ci]
                        if pd.notna(_sc):
                            _gp = round(float(_sc) - avg_score_100, 1)
                            _trow.append(f"{_sc:.1f}")
                            _hrow.append(f"{_bi} × {_ci}<br>점수: {_sc:.1f}점<br>건수: {_cn}건<br>본부 대비: {_gp:+.1f}점")
                        else:
                            _trow.append("")
                            _hrow.append(f"{_bi} × {_ci}<br>데이터 없음")
                    _hm_text.append(_trow)
                    _hm_hover.append(_hrow)
                fig_hm.update_traces(
                    text=_hm_text, texttemplate="%{text}",
                    customdata=_hm_hover,
                    hovertemplate="%{customdata}<extra></extra>")
                fig_hm.update_layout(
                    height=max(320, len(_hm_pivot_sol) * 38 + 100),
                    margin=dict(t=10, b=60, l=120, r=20),
                    coloraxis_colorbar=dict(title="만족도"))
                st.plotly_chart(fig_hm, use_container_width=True, config={'staticPlot': True})

                # ── 벤치마킹 솔루션 + 사전케어 대상 ──────────────
                _sol_bt_l, _sol_bt_r = st.columns([1, 1])

                with _sol_bt_l:
                    st.markdown("##### 📚 벤치마킹 솔루션 — 옆 지사는 어떻게?")
                    _bm_biz_sel = _df_sel.groupby(M["business"])["_점수100"].mean()
                    _bm_biz_all = df_f.groupby(M["business"])["_점수100"].mean()
                    _bm_gap = (_bm_biz_sel - _bm_biz_all).sort_values()
                    _bm_weak3 = _bm_gap.head(3)
                    _bm_ofc_biz = df_f.groupby([M["office"], M["business"]])["_점수100"].agg(
                        ["mean", "count"]).reset_index()
                    _bm_ofc_biz.columns = [M["office"], "업무", "점수", "건수"]
                    _bm_ofc_biz = _bm_ofc_biz[_bm_ofc_biz["건수"] >= 3]
                    _bm_cards = []
                    for _bw_name, _bw_gap in _bm_weak3.items():
                        _bw_score = round(float(_bm_biz_sel.get(_bw_name, 0)), 1)
                        _bw_subset = _bm_ofc_biz[
                            (_bm_ofc_biz["업무"] == _bw_name) &
                            (_bm_ofc_biz[M["office"]] != _sel_off)]
                        if not _bw_subset.empty:
                            _bw_best = _bw_subset.sort_values("점수", ascending=False).iloc[0]
                            _bm_cards.append({
                                "업무": _bw_name, "내점수": _bw_score,
                                "편차": round(float(_bw_gap), 1),
                                "1위지사": _bw_best[M["office"]],
                                "1위점수": round(float(_bw_best["점수"]), 1),
                                "1위건수": int(_bw_best["건수"]),
                            })
                    if _bm_cards:
                        for _bmc in _bm_cards:
                            _gap_txt = f'{_bmc["1위점수"] - _bmc["내점수"]:+.1f}점 차이'
                            st.markdown(
                                '<div style="background:#e3f2fd;border:1px solid #90caf9;'
                                'border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:0.88em;">'
                                f'<b>📌 {_bmc["업무"]}</b> — 귀 지사 <b>{_bmc["내점수"]:.1f}점</b>'
                                f' → <b style="color:#1565c0">{_bmc["1위지사"]}</b>'
                                f' <b>{_bmc["1위점수"]:.1f}점</b> (본부 1위, {_bmc["1위건수"]}건)'
                                f'<br><span style="color:#1565c0;font-size:0.9em;">'
                                f'{_gap_txt} — {_bmc["1위지사"]}의 노하우를 벤치마킹하세요</span></div>',
                                unsafe_allow_html=True)
                    else:
                        st.info("비교 대상 지사가 없습니다.")

                with _sol_bt_r:
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
                        _pc_show = _pc_df[[c for c in _pc_show_cols if c in _pc_df.columns]].head(5)
                        _pc_show = _pc_show.rename(columns={"_점수100": "점수(100점)"})
                        st.dataframe(_pc_show.reset_index(drop=True), use_container_width=True, hide_index=True)
                    else:
                        st.success("✅ 해당 지사에 50점 이하 건이 없습니다.")

                st.markdown("---")

                # ── 리스크 분류: 업무 × 최저항목 기준 ──────────
                _real_risk = []
                _drop_risk = []
                for _bi in _hm_pivot_sol.index:
                    _cn = int(_hm_cnt_sol.loc[_bi, "_건수"])
                    _row_scores = _hm_pivot_sol.loc[_bi].dropna()
                    if _row_scores.empty:
                        continue
                    _worst_item = _row_scores.idxmin()
                    _worst_score = float(_row_scores.min())
                    if _worst_score < avg_score_100 and _cn >= 2:
                        _item = {"업무": _bi, "최저항목": _worst_item,
                                 "점수": round(_worst_score, 1), "건수": _cn,
                                 "impact": round((avg_score_100 - _worst_score) * _cn, 1)}
                        if _cn >= 10:
                            _real_risk.append(_item)
                        else:
                            _drop_risk.append(_item)
                _real_top3 = sorted(_real_risk, key=lambda x: x["impact"], reverse=True)[:3]
                _drop_top3 = sorted(_drop_risk, key=lambda x: x["점수"])[:3]

                # ── 실질적 리스크 카드 ──────────────────────────
                if _real_top3:
                    st.markdown("##### 🚨 실질적 리스크 TOP 3 — 건수 多 & 평균 이하 (우선 개선)")
                    st.caption("응답 건수가 충분하여 통계적으로 신뢰할 수 있는 **최우선 개선 대상**입니다.")
                    _rt_cols = st.columns(len(_real_top3))
                    for _ri, _rk in enumerate(_real_top3):
                        _rk_key = f"sol_real_{_ri}"
                        _cur_sel = st.session_state.get("sol_cell_sel")
                        _is_sel = (_cur_sel is not None
                                   and _cur_sel.get("업무") == _rk["업무"])
                        _badge = "①②③"[_ri]
                        _bg = "#fff3e0" if _is_sel else "#ffebee"
                        _bd = "#ff8f00" if _is_sel else "#ef9a9a"
                        with _rt_cols[_ri]:
                            st.markdown(
                                f'<div style="background:{_bg};border:2px solid {_bd};'
                                'border-radius:10px;padding:12px;margin-bottom:8px;">'
                                f'<div style="font-size:0.78em;color:#c62828;font-weight:700;">리스크 {_badge}</div>'
                                f'<div style="font-size:1em;font-weight:800;margin:4px 0;">{_rk["업무"]}</div>'
                                f'<div style="font-size:0.82em;color:#555;">최저: {_rk["최저항목"]}</div>'
                                f'<div style="font-size:1.4em;font-weight:900;color:#c62828;">{_rk["점수"]:.1f}점</div>'
                                f'<div style="font-size:0.78em;color:#555;">{_rk["건수"]}건 · 임팩트 {_rk["impact"]:.0f}</div>'
                                '</div>', unsafe_allow_html=True)
                            if st.button(
                                    "▲ 닫기" if _is_sel else "🔍 상세 원인 분석",
                                    key=_rk_key, use_container_width=True,
                                    type="secondary" if _is_sel else "primary"):
                                st.session_state["sol_cell_sel"] = (
                                    None if _is_sel else {"업무": _rk["업무"]})
                                st.rerun()

                # ── 급락 조합 카드 ──────────────────────────────
                if _drop_top3:
                    st.markdown("##### ⚡ 급락 업무 TOP 3 — 건수 少 & 점수 급락 (모니터링)")
                    st.caption("소수 응답이지만 점수가 급락한 업무입니다. 민원 전조 신호일 수 있으니 **추이를 주시**하세요.")
                    _dt_cols = st.columns(len(_drop_top3))
                    for _di, _dk in enumerate(_drop_top3):
                        _dk_key = f"sol_drop_{_di}"
                        _cur_sel = st.session_state.get("sol_cell_sel")
                        _is_sel = (_cur_sel is not None
                                   and _cur_sel.get("업무") == _dk["업무"])
                        _badge = "①②③"[_di]
                        _bg = "#fff3e0" if _is_sel else "#fff8e1"
                        _bd = "#ff8f00" if _is_sel else "#ffca28"
                        with _dt_cols[_di]:
                            st.markdown(
                                f'<div style="background:{_bg};border:2px solid {_bd};'
                                'border-radius:10px;padding:12px;margin-bottom:8px;">'
                                f'<div style="font-size:0.78em;color:#e65100;font-weight:700;">급락 {_badge}</div>'
                                f'<div style="font-size:1em;font-weight:800;margin:4px 0;">{_dk["업무"]}</div>'
                                f'<div style="font-size:0.82em;color:#555;">최저: {_dk["최저항목"]}</div>'
                                f'<div style="font-size:1.4em;font-weight:900;color:#e65100;">{_dk["점수"]:.1f}점</div>'
                                f'<div style="font-size:0.78em;color:#555;">{_dk["건수"]}건 (소량)</div>'
                                '</div>', unsafe_allow_html=True)
                            if st.button(
                                    "▲ 닫기" if _is_sel else "🔍 상세 원인 분석",
                                    key=_dk_key, use_container_width=True,
                                    type="secondary" if _is_sel else "primary"):
                                st.session_state["sol_cell_sel"] = (
                                    None if _is_sel else {"업무": _dk["업무"]})
                                st.rerun()

                # ══════════════════════════════════════════
                # LEVEL 3 — AI 심층 진단 (Action)
                # ══════════════════════════════════════════
                if _real_top3 or _drop_top3:
                    _cell = st.session_state.get("sol_cell_sel")
                    if _cell:
                        _c3_biz = _cell["업무"]

                        st.markdown(
                            '<div style="background:linear-gradient(90deg,#4a148c,#6a1b9a);'
                            'border-radius:10px;padding:14px 20px;color:white;margin:16px 0 12px;">'
                            f'<span style="font-size:1.1em;font-weight:800;">'
                            f'🔬 AI 심층 진단 — {_sel_off} · {_c3_biz}</span>'
                            '<span style="font-size:0.82em;opacity:.8;margin-left:10px;">세부항목별 원인 규명</span>'
                            '</div>', unsafe_allow_html=True)

                        _c3_df = _df_sel[_df_sel[M["business"]] == _c3_biz].copy()
                        _c3_n = len(_c3_df)

                        if _c3_n < 10:
                            st.warning(
                                f"⚠️ 데이터 수가 적어(**{_c3_n}건**) 특정 사례에 의한 "
                                "왜곡 가능성이 있으니 **VOC 원문을 중심으로** 판단하세요.")

                        # ── 세부항목 점수 + 범인 특정 ──────────────
                        _c3_l, _c3_r = st.columns([1, 1])

                        with _c3_l:
                            st.markdown("**📊 세부항목별 만족도 — 범인 특정**")
                            _item_scores = {}
                            for _sc in _score_cols_avail:
                                _vals = pd.to_numeric(_c3_df[_sc], errors="coerce").dropna()
                                if len(_vals) > 0:
                                    _item_scores[_sc] = round(float(_vals.mean()), 1)
                            if _item_scores:
                                _item_sdf = pd.DataFrame([
                                    {"항목": k, "만족도": v} for k, v in _item_scores.items()
                                ]).sort_values("만족도")
                                _c3_colors = _item_sdf["만족도"].apply(
                                    lambda x: "#d32f2f" if x < avg_score_100 - 5
                                    else "#f57c00" if x < avg_score_100
                                    else "#388e3c").tolist()
                                fig_c3 = go.Figure(go.Bar(
                                    y=_item_sdf["항목"], x=_item_sdf["만족도"],
                                    orientation="h", marker_color=_c3_colors,
                                    text=[f"{v:.1f}" for v in _item_sdf["만족도"]],
                                    textposition="outside",
                                    hovertemplate="%{y}<br>%{x:.1f}점<extra></extra>"))
                                fig_c3.add_vline(
                                    x=avg_score_100, line_dash="dash", line_color=C["navy"],
                                    annotation_text=f"본부 {avg_score_100:.1f}",
                                    annotation_position="top right")
                                _c3_x_min = max(0, _item_sdf["만족도"].min() - 10)
                                fig_c3.update_layout(
                                    template=PLOTLY_TPL, height=280,
                                    margin=dict(t=10, b=10, l=10, r=90),
                                    xaxis=dict(range=[_c3_x_min, 110]))
                                st.plotly_chart(fig_c3, use_container_width=True, config={'staticPlot': True})
                                _culprit = _item_sdf.iloc[0]
                                _cul_gap = round(_culprit["만족도"] - avg_score_100, 1)
                                st.markdown(
                                    '<div style="background:#ffebee;border:2px solid #ef9a9a;'
                                    'border-radius:8px;padding:10px 14px;font-size:0.88em;">'
                                    f'🎯 <b>범인 확정:</b> <b>{_c3_biz}</b> 업무에서 '
                                    f'<b style="color:#c62828">{_culprit["항목"]}</b> 항목이 '
                                    f'<b style="color:#c62828">{_culprit["만족도"]:.1f}점</b>'
                                    f' (본부 대비 {_cul_gap:+.1f}점)</div>',
                                    unsafe_allow_html=True)

                        with _c3_r:
                            st.markdown("**📋 실제 VOC 원문**")
                            if M.get("voc") and not _c3_df.empty:
                                _c3_voc_valid = (
                                    _c3_df[M["voc"]].dropna()
                                    .apply(lambda x: str(x).strip())
                                    .loc[lambda s: (s.str.len() > 2) & (~s.isin(["응답없음", "nan", ""]))])
                                _c3_vocs = _c3_voc_valid.head(10)
                                if not _c3_vocs.empty:
                                    with st.expander(f"VOC {len(_c3_vocs)}건 보기", expanded=True):
                                        for _cv in _c3_vocs:
                                            _cv_hl = str(_cv)
                                            for _nkw in VOC_HIGHLIGHT_KW:
                                                if _nkw in _cv_hl:
                                                    _cv_hl = _cv_hl.replace(
                                                        _nkw,
                                                        '<mark style="background:#ffeb3b">'
                                                        + _nkw + "</mark>")
                                            st.markdown(
                                                '<div style="border-left:3px solid #ef9a9a;'
                                                'padding:4px 10px;margin-bottom:4px;font-size:0.87em;">'
                                                + _cv_hl + "</div>",
                                                unsafe_allow_html=True)
                                else:
                                    st.info("해당 업무의 VOC가 없습니다.")
                            else:
                                st.info("VOC 컬럼이 설정되지 않았습니다.")

                        # ── AI 처방전 ──────────────────────────
                        st.markdown("---")
                        # 세부항목 점수 텍스트
                        _item_lines = "\n".join(
                            f"  - {k}: {v}점" for k, v in _item_scores.items()
                        ) if _item_scores else "데이터 없음"
                        _worst_item_name = min(_item_scores, key=_item_scores.get) if _item_scores else "없음"

                        # VOC 수집
                        _c3_voc_lines = ""
                        if M.get("voc"):
                            _c3_vl = (_c3_df[M["voc"]].dropna().apply(str)
                                      .loc[lambda s: (s.str.len() > 2) &
                                                     (~s.isin(["응답없음", "nan", ""]))]
                                      .head(8))
                            _c3_voc_lines = "\n".join(f"  - {v}" for v in _c3_vl)

                        # 벤치마킹 대상 지사
                        _bm_best_txt = ""
                        _bm_all_biz = df_f[df_f[M["business"]] == _c3_biz].groupby(M["office"])["_점수100"].agg(["mean","count"]).reset_index()
                        _bm_all_biz.columns = ["지사","점수","건수"]
                        _bm_all_biz = _bm_all_biz[(_bm_all_biz["건수"] >= 3) & (_bm_all_biz["지사"] != _sel_off)]
                        if not _bm_all_biz.empty:
                            _bm_top = _bm_all_biz.sort_values("점수", ascending=False).iloc[0]
                            _bm_best_txt = f"벤치마킹 대상: {_bm_top['지사']} ({_bm_top['점수']:.1f}점, {int(_bm_top['건수'])}건)"

                        # KB
                        _office_kb = _get_office_kb(_sel_off)
                        _kb_section = ""
                        if _office_kb:
                            _kb_section = (
                                f"\n[{_sel_off} 현장 환경]\n"
                                f"지역 특성: {_office_kb['context']}\n"
                                f"추천 액션: {_office_kb['action']}\n"
                            )

                        if st.button("🤖 AI 처방전 생성", key="sol_ai_cell_btn",
                                     type="primary", use_container_width=True):
                            if not GEMINI_AVAILABLE:
                                st.error("Gemini API 키가 설정되지 않았습니다.")
                            else:
                                _c3_prompt = (
                                    "# ROLE: 전력 산업 25년 차 '베테랑 CS 컨설팅 마스터'\n"
                                    "당신은 수천 건의 민원을 해결해온 현장 전문가입니다. "
                                    "뻔한 도덕책 같은 소리(친절 교육, 시스템 개선)는 쓰레기통에 버리세요. "
                                    "지사장이 월초 회의에서 바로 지시할 수 있는 '전술적 소스'를 제공합니다.\n\n"
                                    "[절대 금지]\n"
                                    "- 'TF 구성', '교육 실시', '매뉴얼 배포', '시스템 개편' 같은 뻔한 제안\n"
                                    "- 예산이나 본사 승인이 필요한 제안\n"
                                    "- 액션 뒤에 '(이번 주)', '(다음 달)' 등 괄호 시기 표기\n\n"
                                    f"[진단 대상]\n"
                                    f"지사: {_sel_off} | 업무유형: {_c3_biz}\n"
                                    f"지사 종합 평균: {_sel_avg:.1f}점 (본부 {avg_score_100:.1f}점 대비 {_sel_gap:+.1f}점)\n"
                                    f"데이터 건수: {_c3_n}건 ({'신뢰도 주의' if _c3_n < 10 else '신뢰도 충분'})\n"
                                    f"{_kb_section}\n"
                                    f"[세부항목별 만족도 (5개 항목)]\n{_item_lines}\n"
                                    f"→ 최저 항목(범인): {_worst_item_name}\n\n"
                                    f"[{_bm_best_txt}]\n\n" if _bm_best_txt else ""
                                    f"[불만 VOC 원문]\n{_c3_voc_lines or '없음'}\n\n"
                                    "# STEP 1: 3차원 심층 진단 (Deep Diagnosis)\n"
                                    f"1. 지사 특성 반영: {_sel_off} 지역 고객의 기질과 인구 특성을 고려한 원인 추측.\n"
                                    f"2. 항목별 결함 특정: 5개 세부 항목 중 '{_worst_item_name}'이 가장 낮은 이유를 VOC 근거로 진단.\n"
                                    "   (예: 친절한데 전문성이 낮다 = 실력 부족 / 전문적인데 신속성이 낮다 = 업무 과부하)\n\n"
                                    "# STEP 2: 실무형 처방전 작성 규칙 (Mandatory)\n"
                                    "1. 구체성: '친절 교육' 대신 '특정 시간대 창구 인원 2명 전진 배치'라고 할 것.\n"
                                    "2. 즉시성: 예산이나 본사 승인 없이 '지사장 전결'로 실행 가능한 것.\n"
                                    "3. 벤치마킹: 옆 지사의 우수 사례(Best Practice) 인용.\n\n"
                                    "# STEP 3: 출력 포맷\n"
                                    f"## 🎯 [{_sel_off}] 심층 진단 리포트\n"
                                    "### 1. 데이터가 말하는 '진짜 원인' (Why?)\n"
                                    f"- 3차원 교차 분석 결과, [{_c3_biz}]에서 [{_worst_item_name}] 점수가 낮은 이유는 [추정 원인]입니다.\n"
                                    "### 2. 지사장 즉시 실행 처방 (Action Plan)\n"
                                    "- **전술 1:** (구체적인 행동 지침)\n"
                                    "- **전술 2:** (고객 응대 멘트 변경 또는 자원 재배치)\n"
                                    "### 3. 옆 지사 성공 노하우 벤치마킹\n"
                                    "- (유사한 문제를 겪었던 지사의 해결 사례 제안)\n"
                                )
                                with st.spinner("AI가 심층 진단 중…"):
                                    try:
                                        import urllib.request
                                        _models = ["gemini-2.0-flash", "gemma-3-12b-it", "gemma-3-27b-it"]
                                        _c3_pl = {
                                            "contents": [{"parts": [{"text": _c3_prompt}]}],
                                            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 3072}
                                        }
                                        _ctx = ssl._create_unverified_context()
                                        _body = None
                                        for _model in _models:
                                            _url = (f"https://generativelanguage.googleapis.com/v1beta/"
                                                    f"models/{_model}:generateContent?key={_GEMINI_KEY}")
                                            _req = urllib.request.Request(
                                                _url,
                                                data=json.dumps(_c3_pl).encode("utf-8"),
                                                headers={"Content-Type": "application/json"},
                                                method="POST")
                                            try:
                                                with urllib.request.urlopen(_req, context=_ctx, timeout=90) as _rsp:
                                                    _body = json.loads(_rsp.read().decode("utf-8"))
                                                break
                                            except urllib.error.HTTPError as _he:
                                                if _he.code == 429:
                                                    continue
                                                raise
                                        if _body is None:
                                            st.error("모든 AI 모델의 일일 한도가 소진되었습니다.")
                                        else:
                                            _sol_text = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
                                            st.markdown(
                                                '<div style="background:#f8f9fb;border:1px solid #d0d7de;border-radius:10px;'
                                                'padding:24px 28px;margin:8px 0;font-size:0.93em;line-height:1.85;">\n\n'
                                                f'{_sol_text}\n\n</div>',
                                                unsafe_allow_html=True)
                                    except Exception as _c3e:
                                        st.error(f"AI 분석 중 오류: {_c3e}")
        else:
            st.info("업무유형 컬럼과 세부항목 점수가 필요합니다.")


# ─────────────────────────────────────────────────────────────
#  TAB SOL2  3단계 정밀 진단 (타겟팅 → 고통 분석 → 처방)
# ─────────────────────────────────────────────────────────────

with tab_sol2:

    if df_f.empty or avg_score_100 is None:
        st.info("데이터를 먼저 업로드하세요.")
    elif not M.get("office") or not M.get("business") or not M.get("contract"):
        st.info("지사·업무유형·계약종별 컬럼이 모두 필요합니다.")
    else:
        # ── 지사 선택 ────────────────────────────────────────
        _s2_offices = _sort_offices(df_f[M["office"]].dropna().unique().tolist())
        st.markdown("#### 🔬 3단계 정밀 진단 — 분석할 지사를 선택하세요")
        _s2_off = st.selectbox("", _s2_offices, key="sol2_office_sel", label_visibility="collapsed")
        _s2_df = df_f[df_f[M["office"]] == _s2_off].copy()

        if _s2_df.empty:
            st.warning(f"{_s2_off}의 데이터가 없습니다.")
        else:
            _s2_avg = float(_s2_df["_점수100"].mean())
            _s2_gap = _s2_avg - avg_score_100

            # ══════════════════════════════════════════════════
            # STEP 1 — 조망: 업무 × 계약종별 히트맵
            # ══════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### STEP 1. 조망 — 어디가 제일 아픈가?")
            st.caption("업무유형(행) × 계약종별(열)의 평균 만족도입니다. **가장 빨간 칸**이 우선 관리 대상입니다.")

            _s2_pivot = _s2_df.pivot_table(
                index=M["business"], columns=M["contract"],
                values="_점수100", aggfunc="mean"
            ).round(1)

            if _s2_pivot.empty or _s2_pivot.size == 0:
                st.info("데이터가 부족하여 히트맵을 생성할 수 없습니다.")
            else:
                # 최저점 자동 탐지
                _s2_min_val = _s2_pivot.min().min()
                _s2_min_pos = _s2_pivot.stack()
                if not _s2_min_pos.empty:
                    _s2_min_idx = _s2_min_pos.idxmin()
                    _s2_worst_biz = _s2_min_idx[0]
                    _s2_worst_ct = _s2_min_idx[1]
                else:
                    _s2_worst_biz = _s2_pivot.index[0]
                    _s2_worst_ct = _s2_pivot.columns[0]

                fig_s2_hm = px.imshow(
                    _s2_pivot,
                    color_continuous_scale="RdYlGn",
                    zmin=max(50, float(_s2_pivot.min().min()) - 5),
                    zmax=min(100, float(_s2_pivot.max().max()) + 2),
                    text_auto=".1f",
                    labels={"x": "계약종별", "y": "업무유형", "color": "만족도"},
                    aspect="auto",
                )
                fig_s2_hm.update_layout(
                    template=PLOTLY_TPL, height=max(300, len(_s2_pivot) * 40 + 80),
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(side="top"),
                )
                st.plotly_chart(fig_s2_hm, use_container_width=True, config={"staticPlot": True})

                # 최우선 타겟 안내
                _s2_min_score = round(float(_s2_min_val), 1)
                st.warning(
                    f"⚠️ **최우선 타겟**: [{_s2_worst_ct}] 고객의 [{_s2_worst_biz}] 업무 "
                    f"= **{_s2_min_score}점** (본부 평균 {avg_score_100:.1f}점 대비 "
                    f"{_s2_min_score - avg_score_100:+.1f}점)"
                )

                # ══════════════════════════════════════════════════
                # STEP 2 — 심층: 타겟의 5대 항목 분석
                # ══════════════════════════════════════════════════
                st.markdown("---")
                st.markdown("### STEP 2. 심층 — 도대체 '왜' 화가 났나?")
                st.caption("STEP 1에서 잡은 타겟의 세부 항목별 점수를 쪼개봅니다. 최저 항목이 **진짜 원인**입니다.")

                _s2_ct_list = sorted(_s2_pivot.columns.tolist())
                _s2_biz_list = sorted(_s2_pivot.index.tolist())

                _s2_col_l, _s2_col_r = st.columns(2)
                with _s2_col_l:
                    _s2_sel_ct = st.selectbox(
                        "계약종별", _s2_ct_list,
                        index=_s2_ct_list.index(_s2_worst_ct) if _s2_worst_ct in _s2_ct_list else 0,
                        key="sol2_ct_sel"
                    )
                with _s2_col_r:
                    _s2_sel_biz = st.selectbox(
                        "업무유형", _s2_biz_list,
                        index=_s2_biz_list.index(_s2_worst_biz) if _s2_worst_biz in _s2_biz_list else 0,
                        key="sol2_biz_sel"
                    )

                # 타겟 필터
                _s2_target = _s2_df[
                    (_s2_df[M["contract"]] == _s2_sel_ct) &
                    (_s2_df[M["business"]] == _s2_sel_biz)
                ]
                _s2_tn = len(_s2_target)

                if _s2_tn == 0:
                    st.info(f"[{_s2_sel_ct}] × [{_s2_sel_biz}] 조합의 데이터가 없습니다.")
                else:
                    st.markdown(f"**타겟**: [{_s2_sel_ct}] 고객의 [{_s2_sel_biz}] — **{_s2_tn}건**")

                    # 5대 항목 점수 계산
                    _s2_score_cols = [c for c in individual_scores if c in _s2_target.columns]
                    _s2_item_scores = {}
                    for _sc in _s2_score_cols:
                        _vals = pd.to_numeric(_s2_target[_sc], errors="coerce").dropna()
                        if len(_vals) > 0:
                            _s2_item_scores[_sc] = round(float(_vals.mean()), 1)

                    if _s2_item_scores:
                        _s2_worst_item = min(_s2_item_scores, key=_s2_item_scores.get)
                        _s2_worst_score = _s2_item_scores[_s2_worst_item]

                        # 바 차트
                        _s2_bar_colors = [
                            "#d32f2f" if k == _s2_worst_item else "#43a047"
                            for k in _s2_item_scores
                        ]
                        fig_s2_bar = go.Figure(data=[
                            go.Bar(
                                x=list(_s2_item_scores.values()),
                                y=list(_s2_item_scores.keys()),
                                orientation="h",
                                marker_color=_s2_bar_colors,
                                text=[f"{v}점" for v in _s2_item_scores.values()],
                                textposition="outside",
                            )
                        ])
                        fig_s2_bar.update_layout(
                            template=PLOTLY_TPL, height=max(250, len(_s2_item_scores) * 45 + 60),
                            margin=dict(l=10, r=60, t=10, b=10),
                            xaxis=dict(range=[0, 105], title="만족도"),
                        )
                        st.plotly_chart(fig_s2_bar, use_container_width=True, config={"staticPlot": True})

                        st.error(
                            f"🔍 **범인 특정**: [{_s2_worst_item}] = {_s2_worst_score}점 — "
                            f"이 항목이 [{_s2_sel_ct}] 고객의 [{_s2_sel_biz}] 만족도를 끌어내리는 핵심 원인입니다."
                        )

                        # VOC 원문
                        _s2_voc_lines = ""
                        if M.get("voc"):
                            _s2_vocs = _s2_target[_s2_target["_점수100"] < 70][M["voc"]].dropna().astype(str).tolist()
                            _s2_vocs = [v for v in _s2_vocs if v.strip() not in _VOC_EMPTY and len(v.strip()) > 2][:10]
                            if _s2_vocs:
                                st.markdown("**📝 불만 VOC 원문**")
                                for _v in _s2_vocs:
                                    st.markdown(f"- {_v}")
                                _s2_voc_lines = "\n".join(f"  - {v}" for v in _s2_vocs)

                        # 벤치마킹
                        _s2_bm_txt = ""
                        _s2_bm_all = df_f[df_f[M["business"]] == _s2_sel_biz].groupby(M["office"])["_점수100"].agg(["mean", "count"]).reset_index()
                        _s2_bm_all.columns = ["지사", "점수", "건수"]
                        _s2_bm_all = _s2_bm_all[(_s2_bm_all["건수"] >= 3) & (_s2_bm_all["지사"] != _s2_off)]
                        if not _s2_bm_all.empty:
                            _s2_bm_top = _s2_bm_all.sort_values("점수", ascending=False).iloc[0]
                            _s2_bm_txt = f"벤치마킹 대상: {_s2_bm_top['지사']} ({_s2_bm_top['점수']:.1f}점, {int(_s2_bm_top['건수'])}건)"
                            st.success(f"📚 {_s2_bm_txt}")

                        # ══════════════════════════════════════════
                        # STEP 3 — 결론: AI 맞춤 처방전
                        # ══════════════════════════════════════════
                        st.markdown("---")
                        st.markdown("### STEP 3. 결론 — 어떻게 고칠 것인가?")

                        # KB 조회
                        _s2_kb = _get_office_kb(_s2_off)
                        _s2_kb_ctx = _s2_kb["context"] if _s2_kb else "지역 특성 정보 없음"
                        _s2_kb_act = _s2_kb["action"] if _s2_kb else ""

                        _s2_item_lines = "\n".join([f"- {k}: {v}점" for k, v in _s2_item_scores.items()])

                        if st.button("🤖 AI 핀셋 처방전 생성", key="sol2_ai_btn",
                                     type="primary", use_container_width=True):
                            if not GEMINI_AVAILABLE:
                                st.error("Gemini API 키가 설정되지 않았습니다.")
                            else:
                                _s2_prompt = (
                                    "# ROLE: 전력 산업 25년 차 '현장 밀착형 CS 마스터 컨설턴트'\n"
                                    "당신은 지사장이 내일 아침 주간 회의에서 부서장들에게 바로 지시할 수 있는 "
                                    "'핀셋 처방전'을 작성합니다. 무난한 교육이나 시스템 개선안은 철저히 배제하고, "
                                    "현장 인력과 기존 채널을 활용한 '즉시 실행 전술'에 집중하세요.\n\n"
                                    "[절대 금지]\n"
                                    "- 'TF 구성', '교육 실시', '매뉴얼 배포', '시스템 개편' 같은 뻔한 제안\n"
                                    "- 예산이나 본사 승인이 필요한 제안\n"
                                    "- 액션 뒤에 '(이번 주)', '(다음 달)' 등 괄호 시기 표기\n\n"
                                    "# INPUT DATA (3차원 교차 분석 결과)\n"
                                    f"1. 대상 지사: {_s2_off} (지역 특성: {_s2_kb_ctx})\n"
                                    f"2. 타겟 그룹: [{_s2_sel_ct}] 고객의 [{_s2_sel_biz}] 업무\n"
                                    f"3. 지사 종합 평균: {_s2_avg:.1f}점 (본부 {avg_score_100:.1f}점 대비 {_s2_gap:+.1f}점)\n"
                                    f"4. 타겟 건수: {_s2_tn}건 ({'신뢰도 주의' if _s2_tn < 10 else '신뢰도 충분'})\n"
                                    f"5. 핵심 결함: 5개 세부 항목 중 [{_s2_worst_item}] = {_s2_worst_score}점 (최저)\n\n"
                                    f"[세부항목별 만족도]\n{_s2_item_lines}\n\n"
                                )
                                if _s2_bm_txt:
                                    _s2_prompt += f"[{_s2_bm_txt}]\n\n"
                                if _s2_kb_act:
                                    _s2_prompt += f"[지역 추천 액션]\n{_s2_kb_act}\n\n"
                                _s2_prompt += (
                                    f"[불만 VOC 원문]\n{_s2_voc_lines or '없음'}\n\n"
                                    "# STEP 1: 현장의 언어로 '진짜 원인' 진단 (Root Cause)\n"
                                    f"- [{_s2_sel_ct}]의 특성과 [{_s2_sel_biz}]의 성격을 결합하여, "
                                    f"왜 [{_s2_worst_item}] 점수가 낮은지 심리학적/실무적으로 해석하세요.\n"
                                    "- (예: '산업용 고객에게 정전 시 정확성이 낮은 것은, 복구 지연 자체보다 "
                                    "예정 시간 번복으로 인한 공장 가동 계획 차질이 더 큰 분노 포인트임')\n\n"
                                    "# STEP 2: 지사장 전결 '72시간 내 즉시 실행' 전술 (Action Plan)\n"
                                    "- 예산 필요 없고, 본사 승인 필요 없는 '행동' 위주로 2가지 제안.\n"
                                    "- 전술 1 (내부 관리): 직원의 응대 방식이나 자원 재배치\n"
                                    "- 전술 2 (외부 소통): 고객과의 소통 방식 변경 또는 안내 체계 개선\n\n"
                                    "# STEP 3: 글로벌/로컬 베스트 프랙티스 매칭 (Benchmarking)\n"
                                    "- 이 상황에 가장 적합한 우수 사례를 인용하여 처방의 신뢰도를 높이세요.\n\n"
                                    "# OUTPUT FORMAT (Brief & Sharp)\n"
                                    f"## 🎯 [{_s2_off}] 현장 핀셋 처방전\n"
                                    f"### 1. 분석 결과: \"지금 [{_s2_sel_ct}]-[{_s2_sel_biz}] 고객이 가장 화난 이유\"\n"
                                    "- (데이터 기반의 날카로운 원인 진단 1~2줄)\n"
                                    "### 2. 내일 아침 즉시 지시사항 (Action)\n"
                                    "- **[전술 A]**: 구체적인 행동과 기대 효과\n"
                                    "- **[전술 B]**: 구체적인 소통 방식과 타겟\n"
                                    "### 3. 한 줄 조언 (Expert Insight)\n"
                                    f"- \"지사장님, 이 고객들에겐 [핵심 가치]를 먼저 보여주는 것이 우선입니다.\"\n"
                                )

                                with st.spinner("AI가 핀셋 처방전 생성 중…"):
                                    try:
                                        import urllib.request
                                        _models = ["gemini-2.0-flash", "gemma-3-12b-it", "gemma-3-27b-it"]
                                        _s2_pl = {
                                            "contents": [{"parts": [{"text": _s2_prompt}]}],
                                            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 3072}
                                        }
                                        _ctx = ssl._create_unverified_context()
                                        _body = None
                                        for _model in _models:
                                            _url = (f"https://generativelanguage.googleapis.com/v1beta/"
                                                    f"models/{_model}:generateContent?key={_GEMINI_KEY}")
                                            _req = urllib.request.Request(
                                                _url,
                                                data=json.dumps(_s2_pl).encode("utf-8"),
                                                headers={"Content-Type": "application/json"},
                                                method="POST")
                                            try:
                                                with urllib.request.urlopen(_req, context=_ctx, timeout=90) as _resp:
                                                    _body = json.loads(_resp.read().decode("utf-8"))
                                                break
                                            except urllib.error.HTTPError as _http_err:
                                                if _http_err.code == 429:
                                                    continue
                                                raise
                                        if _body is None:
                                            st.error("모든 AI 모델의 일일 한도가 소진되었습니다. 내일 다시 시도해주세요.")
                                        else:
                                            _s2_text = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
                                            st.markdown(
                                                '<div style="background:#f8f9fb;border:1px solid #d0d7de;border-radius:10px;'
                                                'padding:24px 28px;margin:8px 0;font-size:0.93em;line-height:1.85;">\n\n'
                                                f'{_s2_text}\n\n</div>',
                                                unsafe_allow_html=True)
                                    except Exception as _s2e:
                                        st.error(f"AI 분석 중 오류: {_s2e}")
                    else:
                        st.info("세부항목 점수 데이터가 없습니다.")


#  TAB 10  CXO 딥 인사이트 (투트랙 VOC 분석)
# ─────────────────────────────────────────────────────────────

with tab10:
    st.markdown('<p class="sec-head">🧠 CXO 딥 인사이트 — 투트랙 VOC 분석 · 계약종별 상관관계 · 지사 페르소나</p>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # 0. OOS 필터링 (이미 전역에서 계산됨, 여기서는 참조만)
    # ══════════════════════════════════════════
    oos_cnt = df_f["_is_oos"].sum() if "_is_oos" in df_f.columns else 0
    df_pure = df_f[~df_f["_is_oos"]].copy() if "_is_oos" in df_f.columns else df_f.copy()

    # OOS 현황 요약 배너
    st.markdown(
        f'<div class="card-gold">'
        f'<b>🔒 통제 불가(Out of Scope) 필터링 완료</b><br><br>'
        f'콜센터 전화 연결 지연 등 <b>일선 지사가 통제할 수 없는</b> 중앙 인프라 관련 불만 '
        f'<b style="color:{C["red"]}">{oos_cnt}건</b>을 분석에서 제외합니다.<br>'
        f'아래 모든 분석은 <b>순수 지사 통제 가능 VOC {len(df_pure):,}건</b> 기준입니다.'
        f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════════════════
    # 1. 투트랙 VOC 감성·원인 분석
    # ══════════════════════════════════════════
    st.markdown('<p class="sec-head">1️⃣ 투트랙 VOC 분석 — 1차 감성 분류 + 2차 원인 태깅</p>',
                unsafe_allow_html=True)

    if M["voc"] and "_VOC감성" in df_pure.columns:
        # 전역 _VOC감성 컬럼 사용 (전체 행 기준 — 응답없음도 중립으로 포함)
        voc_cls_pure = df_pure["_VOC감성"].value_counts()
        pos_n = voc_cls_pure.get("긍정", 0)
        neu_n = voc_cls_pure.get("중립", 0)
        neg_n = voc_cls_pure.get("부정", 0)
        _total_pure = pos_n + neu_n + neg_n

        # 부정 VOC 원인 태깅용: 텍스트가 있는 부정 VOC만 별도 추출
        voc_valid = df_pure[M["voc"]].dropna()
        voc_valid = voc_valid[~voc_valid.isin(["응답없음", "nan", ""])]
        sentiment_series = voc_valid.apply(_classify_sentiment_3tier)

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("📝 분석 대상", f"{_total_pure:,}건")
        with sc2:
            st.metric("😊 긍정", f"{pos_n:,}건 ({pos_n / max(_total_pure, 1) * 100:.1f}%)")
        with sc3:
            st.metric("😐 중립", f"{neu_n:,}건 ({neu_n / max(_total_pure, 1) * 100:.1f}%)")
        with sc4:
            st.metric("😡 부정", f"{neg_n:,}건 ({neg_n / max(_total_pure, 1) * 100:.1f}%)")

        # 감성 분포 도넛 차트
        sent_df = pd.DataFrame({"감성": ["긍정", "중립", "부정"], "건수": [pos_n, neu_n, neg_n]})
        fig_sent = px.pie(sent_df, values="건수", names="감성", hole=0.45,
                          color="감성",
                          color_discrete_map={"긍정": "#27AE60", "중립": "#95A5A6", "부정": "#E74C3C"},
                          title="1차 감성 분류 (OOS 제외)")
        fig_sent.update_traces(hovertemplate="%{label}<br>%{value:,}건 (%{percent})<extra></extra>")
        fig_sent.update_layout(height=340, margin=dict(t=60, b=20, l=20, r=20),
                               title_font=dict(size=14, color=C["navy"]))

        # 2차: 부정 VOC 원인 태깅
        neg_vocs = voc_valid[sentiment_series == "부정"]
        cause_data = []
        cause_examples = {}
        for v in neg_vocs:
            s = str(v)
            tagged = False
            for cause, keywords in _CAUSE_TAGS.items():
                for kw in keywords:
                    if kw in s:
                        idx = s.find(kw)
                        window = s[max(0, idx - 15):min(len(s), idx + len(kw) + 15)]
                        if not any(pw in window for pw in POSITIVE_CONTEXT):
                            cause_data.append(cause)
                            if cause not in cause_examples:
                                cause_examples[cause] = []
                            if len(cause_examples[cause]) < 2 and len(s) > 10:
                                cause_examples[cause].append(s[:120])
                            tagged = True
                            break
                if tagged:
                    break
            if not tagged:
                cause_data.append("기타")

        cause_counts = Counter(cause_data)
        if cause_counts:
            df_cause = pd.DataFrame([{"원인": k, "건수": v} for k, v in cause_counts.most_common()])
            df_cause = df_cause[df_cause["건수"] > 0]
        else:
            df_cause = pd.DataFrame(columns=["원인", "건수"])

        if not df_cause.empty:
            fig_cause = px.bar(df_cause, x="건수", y="원인", orientation="h",
                               text="건수", template=PLOTLY_TPL,
                               color="건수",
                               color_continuous_scale=["#F9E79F", "#E74C3C"],
                               title="2차 원인 태깅 — 부정 VOC의 근본 원인 분류")
            fig_cause.update_traces(textposition="outside",
                                    hovertemplate="%{y}: %{x}건<extra></extra>")
            fig_cause.update_layout(height=max(300, len(df_cause) * 45 + 80),
                                    margin=dict(t=60, b=20, l=10, r=100),
                                    coloraxis_showscale=False,
                                    title_font=dict(size=14, color=C["navy"]))

            ch_l, ch_r = st.columns(2)
            with ch_l:
                st.plotly_chart(fig_sent, use_container_width=True, config={'staticPlot': True})
            with ch_r:
                st.plotly_chart(fig_cause, use_container_width=True, config={'staticPlot': True})

            # Unmet Needs 카드 (상위 3개 원인)
            st.markdown("**🎯 고객의 진짜 니즈 (Unmet Needs) — 부정 원인 TOP 3**")
            top3_causes = df_cause.head(3)
            need_cols = st.columns(3)
            for i, (_, row) in enumerate(top3_causes.iterrows()):
                cause = row["원인"]
                with need_cols[i]:
                    need_text = _CAUSE_UNMET_NEEDS.get(cause, "맞춤 개선 필요")
                    ex_text = "<br>".join(
                        f'<span style="font-size:0.85em;color:#666">"{e}"</span>'
                        for e in cause_examples.get(cause, []))
                    st.markdown(
                        f'<div class="card-blue" style="min-height:220px">'
                        f'<b>{cause}</b> ({row["건수"]}건)<br><br>'
                        f'<b>Unmet Need:</b><br>{need_text}<br><br>'
                        f'{ex_text}</div>',
                        unsafe_allow_html=True)
        else:
            st.plotly_chart(fig_sent, use_container_width=True, config={'staticPlot': True})

        # OOS 상세 (접기)
        if oos_cnt > 0:
            with st.expander(f"🔒 통제 불가(OOS) VOC 상세 보기 ({oos_cnt}건)"):
                oos_vocs = df_f[df_f["_is_oos"]][M["voc"]].dropna().head(20)
                for i, v in enumerate(oos_vocs, 1):
                    st.markdown(f"**[{i}]** {str(v)[:200]}")
                if oos_cnt > 20:
                    st.caption(f"... 외 {oos_cnt - 20}건")
    else:
        st.info("VOC(서술 의견) 컬럼이 필요합니다.")



