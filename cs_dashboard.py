
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
OFFICE_ORDER  = ["경남본부", "진주지사", "마산지사", "거제지사", "밀양지사", "사천지사",
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


def extract_keywords(texts, top_n=60):
    valid = [str(t) for t in texts if t and str(t).strip() not in ("", "nan", "응답없음")]
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
    docs = [d if d.strip() else "빈문서" for d in docs]
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
    valid = [str(t) for t in texts if t and str(t).strip() not in ("", "nan", "응답없음")]
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
    page_title="회사 CS 분석 대시보드",
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
    st.markdown("### ⚡ 회사 CS 분석 대시보드")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 📂 데이터 업로드")
    uploaded_file = st.file_uploader("엑셀 파일을 여기에 드래그하세요", type=["xlsx","xls"], label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ 시스템 상태")
    st.markdown(
        f"{'✅ KoNLPy' if KONLPY_AVAILABLE else '🟡 기본 분석'}  \n"
        f"✅ 키워드 감성분석 (경량)  \n"
        f"{'✅ KeyBERT' if KEYBERT_AVAILABLE else '🟡 빈도 키워드'}  \n"
        f"{'✅ sklearn TF-IDF' if SKLEARN_AVAILABLE else '🟡 수동 TF-IDF'}  \n"
        f"{'✅ Google Gemini AI' if GEMINI_AVAILABLE else '🟡 Gemini 미연결'}  \n"
        f"{'✅ 한글 폰트' if FONT_PATH else '🟡 기본 폰트'}"
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("© 2025 회사 CS 분석 시스템 v3.5")

# ══════════════════════════════════════════════════════════════
#  6. 헤더 배너
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="dash-header">
  <h1>⚡ AI 활용 고객경험관리시스템 조사결과 분석</h1>
  <p>회사 CS 리포트 양식 · 구간별 비중 · 사업소 벤치마킹 · 채널별/업무별 분석 · VOC 3단 분류 · 사전케어</p>
  <span class="dash-badge">📊 구간별 비중</span>
  <span class="dash-badge">🏢 사업소 벤치마킹</span>
  <span class="dash-badge">📡 채널별 분석</span>
  <span class="dash-badge">☁️ VOC AI 분석</span>
  <span class="dash-badge">🎯 사전케어</span>
  <span class="dash-badge">📈 교차분석</span>
  <span class="dash-badge">📅 시계열</span>
  <span class="dash-badge">🔬 패턴탐지</span>
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
1. **왼쪽 사이드바**에서 엑셀 파일(.xlsx)을 업로드하세요.
2. 컬럼 매핑 설정을 완료하세요.
3. 사이드바 필터로 원하는 데이터 범위를 선택하세요.
4. 각 **탭**을 클릭하며 분석 결과를 확인하세요.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with c_r:
        st.markdown('<div class="card-teal">', unsafe_allow_html=True)
        st.markdown("### 📌 권장 엑셀 컬럼 구성")
        st.markdown("""
| 컬럼명 예시 | 내용 |
|---|---|
| 고객번호 | 고객 식별 ID |
| 고객명 | 이름 |
| 사업소 | 직할, 진주, 마산 등 |
| 접수채널 | 직원방문, 고객센터, 회사ON |
| 계약종 | 주택용/일반용/산업용 |
| 업무구분 | 요금·설치·정전 등 |
| 만족도점수 | 숫자 (1~5, 1~10, 100점) |
| 주관식답변 | VOC 텍스트 |
        """)
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

@st.cache_data(show_spinner=False)
def load_data(raw_bytes):
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
    st.info("📂 좌측 사이드바에서 엑셀 파일을 업로드해 주세요.")
    st.markdown("---")
    st.markdown("#### 사용 방법")
    st.markdown("1. 좌측 **파일 업로드** 영역에 엑셀(.xlsx) 파일을 드래그하거나 선택하세요.\n2. 업로드가 완료되면 자동으로 분석이 시작됩니다.")
    import sys
    sys.exit(0)

with st.spinner("데이터를 불러오는 중…"):
    df_raw, orig_len = load_data(uploaded_file.read())

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
    "이용 편리성", "직원 친절도", "전반적 만족", "사회적 책임",
    "처리 신속도", "처리 정확도", "업무 개선도", "사용 추천도",
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
tab1, tab3, tab5, tab9, tab10, tab11 = st.tabs([
    "📊  구간별 비중 · 종합 현황",
    "📡  계약종별 · 접수채널별 · 업무유형별 분석",
    "🎯  민원 조기 경보 시스템",
    "🔬  지사 심층 · 패턴",
    "🧠  CXO 딥 인사이트",
    "🏢  지사 맞춤형 CS 솔루션",
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
            st.plotly_chart(fig_gauge, use_container_width=True)
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
            st.plotly_chart(fig_bp, use_container_width=True)

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
                                annotation_text=f"본부 평균 {_bar_avg:.1f}",
                                annotation_font_size=12, annotation_font_color=C["navy"])
            fig_bench.update_layout(height=max(350, len(_ofc_grp_bar) * 35 + 80),
                                     margin=dict(t=60, b=20, l=10, r=100), legend_title_text="",
                                     title_font=dict(size=14, color=C["navy"]))
            st.plotly_chart(fig_bench, use_container_width=True)

        # ── 지사별 박스 플롯 ──
        if M["office"]:
            st.markdown("---")
            st.markdown('<p class="sec-head">📦 지사별 만족도 점수 분포 (Box Plot)</p>', unsafe_allow_html=True)
            _ofc_mean_map = df_f.groupby(M["office"])["_점수100"].mean()
            _ofc_count_map = df_f.groupby(M["office"])["_점수100"].count()
            _ofc_order = _ofc_count_map.sort_values(ascending=False).index.tolist()
            _box_avg = df_f["_점수100"].mean()
            _below_avg = set(_ofc_mean_map[_ofc_mean_map < _box_avg].index)
            fig_box = go.Figure()
            for ofc in _ofc_order:
                ofc_data = df_f[df_f[M["office"]] == ofc]["_점수100"]
                _is_below = ofc in _below_avg
                fig_box.add_trace(go.Box(
                    y=ofc_data, name=ofc,
                    marker_color="#E8A0A0" if _is_below else C["sky"],
                    line_color="#D08080" if _is_below else "#5B9BD5",
                    fillcolor="rgba(232,160,160,0.4)" if _is_below else "rgba(91,155,213,0.4)",
                    boxmean=True,
                ))
            fig_box.update_layout(height=450, margin=dict(t=60, b=80, l=60, r=20),
                                   xaxis_tickangle=-25, xaxis_title="", yaxis_title="만족도 점수 (100점 환산)",
                                   title=dict(text="지사별 만족도 분포 (응답 건수 많은 순 · 붉은색 = 전체 평균 미달)",
                                              font=dict(size=14, color=C["navy"])),
                                   showlegend=False, template=PLOTLY_TPL)
            st.plotly_chart(fig_box, use_container_width=True)
            _top5 = ", ".join(_ofc_order[:5])
            st.info(f"💡 본부 전체 점수 향상을 위해 **응답 비중이 높은 상위 5개 지사({_top5})**의 불만족 요인을 우선 해결하고, 응답 수가 적어 점수가 왜곡될 수 있는 지사는 'VOC 내용' 중심의 개별 밀착 관리를 진행합니다.")

            with st.expander("📊 박스 플롯(Box Plot), 1분 만에 이해하기"):
                st.markdown("""
**"평균 점수 뒤에 숨겨진 '진짜 모습'을 보여주는 그래프입니다."**

---

**1. 각 부위의 의미** (무엇을 나타내나요?)

박스 플롯은 데이터를 4등분 하여 어디에 사람들이 가장 많이 몰려 있는지 보여줍니다.

- **가로선 (중앙값):** 응답자를 점수순으로 줄 세웠을 때 딱 중간에 있는 사람의 점수입니다. (평균보다 실제 체감 만족도에 가깝습니다.)
- **색칠된 상자:** 전체 응답자의 **중심부 50%** 가 모여 있는 구간입니다. 우리 지사 고객 대부분의 점수대라고 보시면 됩니다.
- **위아래 수염:** '정상 범위' 내에서의 최고점과 최저점입니다.
- **떨어져 있는 점 (이상치):** 유독 아주 낮거나 높은 점수를 준 '특이 케이스'입니다. (집중 관리가 필요한 민원 신호!)
""")
                import base64 as _b64
                _box_guide_img = _b64.b64decode("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")
                st.image(_box_guide_img, use_container_width=True)
                st.markdown("""

---

**2. 딱 2가지만 확인하세요!** (어떻게 해석하나요?)

**① 상자의 위치:** "우리 지사는 전반적으로 잘하나?"
- 상자가 **위쪽(100점 근처)** 에 붙어 있을수록 고객들이 전반적으로 만족하고 있다는 뜻입니다.
- **목표:** 상자를 전체적으로 위로 끌어올리기!

**② 상자의 길이:** "우리 지사는 서비스가 일정한가?"
- **상자가 짧음 (안정형):** 모든 고객이 비슷한 수준의 서비스를 받고 있습니다. (기복 없는 우수한 관리)
- **상자가 길음 (불안정형):** 고객마다 느끼는 서비스 편차가 큽니다. 어떤 고객은 대만족, 어떤 고객은 대불만족인 상태입니다.
- **목표:** 상자의 길이를 짧게 만들어 서비스 표준화하기!

---

**3. 이런 지사는 주의 깊게 보세요!**
- **상자는 높은데 아래로 수염이 긴 경우:** 전반적으로는 잘하지만, 특정 상황에서 고객이 큰 불만을 느꼈을 가능성이 큽니다. (돌발 악성 민원 주의)
- **상자 아래에 점(이상치)이 많은 경우:** 반복적으로 아주 낮은 점수를 주는 특정 업무나 시간대가 있는지 파악이 필요합니다.
""")
            st.markdown("---")

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
                    fig_biz.update_layout(height=max(300, len(_biz_df) * 30 + 80),
                                           margin=dict(t=50, b=20, l=20, r=60), showlegend=False,
                                           title_font=dict(size=15, color=C["navy"]),
                                           xaxis_title="비율(%)", yaxis_title="")
                    st.plotly_chart(fig_biz, use_container_width=True)
                else:
                    # 연령대, 계약종별: 기존 파이차트 유지
                    fig_pie = px.pie(names=counts.index, values=counts.values, color_discrete_sequence=PIE_COLORS,
                                     hole=0.42, title=f"{_title} 분포", template=PLOTLY_TPL)
                    fig_pie.update_traces(textposition="outside", textinfo="percent+label", textfont_size=12,
                                           marker=dict(line=dict(color="#ffffff", width=2)),
                                           hovertemplate="%{label}<br>%{value:,}건 (%{percent})<extra></extra>")
                    fig_pie.update_layout(height=360, margin=dict(t=50, b=20, l=20, r=20), showlegend=False,
                                           title_font=dict(size=15, color=C["navy"]))
                    st.plotly_chart(fig_pie, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  TAB 3  계약종별 · 접수채널별 · 업무유형별 분석
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
                  annotation_text=f"전체 평균 {overall_avg:.1f}", annotation_font_size=11)
    fig.update_layout(height=max(300, len(grp) * 30 + 80),
                       margin=dict(t=60, b=20, l=10, r=100), legend_title_text="",
                       title_font=dict(size=14, color=C["navy"]))
    st.plotly_chart(fig, use_container_width=True)

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

    # ── 지사별 × 범주별 점수 히트맵 ──
    if office_col:
        st.markdown(f'<p class="sec-head">🏢 지사별 {cat_label} 평균 만족도</p>', unsafe_allow_html=True)
        pivot = df.pivot_table(values=score_col, index=office_col,
                               columns=cat_col, aggfunc="mean").round(1)
        pivot = pivot.reindex(_sort_offices(pivot.index.tolist()))
        if not pivot.empty:
            fig_hm = px.imshow(pivot, color_continuous_scale="RdYlGn",
                               text_auto=".1f", aspect="auto", template=PLOTLY_TPL,
                               title=f"지사 × {cat_label} 만족도 (초록=높음 / 빨강=낮음)")
            fig_hm.update_traces(hovertemplate="지사: %{y}<br>" + cat_label + ": %{x}<br>점수: %{z:.1f}점<extra></extra>")
            fig_hm.update_layout(
                height=max(350, len(pivot.index) * 30 + 100),
                margin=dict(t=60, b=60, l=120, r=60),
                title_font=dict(size=14, color=C["navy"]))
            st.plotly_chart(fig_hm, use_container_width=True)

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

        # ── ② 접수채널별 분석 ──
        if M["channel"]:
            st.markdown('<p class="sec-head">📡 접수채널별 만족도 분석</p>', unsafe_allow_html=True)
            df_f["_채널그룹"] = df_f[M["channel"]].apply(_group_channel)
            df_chan = df_f[df_f["_채널그룹"].notna()].copy()
            _render_category_section(df_chan, "_채널그룹", "접수채널",
                                     M["office"], "_점수100", avg_score_100)
            st.markdown("---")

        # ── ③ 업무유형별 분석 ──
        if M["business"]:
            st.markdown('<p class="sec-head">🏢 업무유형별 만족도 분석</p>', unsafe_allow_html=True)
            _render_category_section(df_f, M["business"], "업무유형",
                                     M["office"], "_점수100", avg_score_100)
            st.markdown("---")

        # ── ④ 업무유형별 사분면 버블 차트 ──
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
                    _texts = [t for t in _texts if t.strip() not in ("", "nan", "응답없음")]
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
            for q_name in _q_order:
                _sub = _bbl_grp[_bbl_grp["사분면"] == q_name]
                if _sub.empty:
                    continue
                _sub_idx = _sub.index.tolist()
                fig_bbl.add_trace(go.Scatter(
                    x=_sub["처리건수"], y=_sub["평균만족도"],
                    mode="markers+text",
                    marker=dict(size=(_sub["버블크기"] * 5).clip(lower=14),
                                color=_q_colors[q_name], opacity=0.75,
                                line=dict(width=1.5, color="white")),
                    text=_sub["업무유형"], textposition="top center", textfont=dict(size=12),
                    hovertext=[_hover_texts[i] for i in _sub_idx],
                    hovertemplate="%{hovertext}<extra></extra>",
                    name=q_name,
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

            fig_bbl.update_layout(
                height=520, margin=dict(t=60, b=60, l=60, r=40),
                xaxis_title="처리 건수", yaxis_title="평균 만족도 (100점 환산)",
                title=dict(text="업무유형별 사분면 분석 — 처리 건수 vs 만족도 (버블 크기 = 불만족 건수)",
                           font=dict(size=14, color=C["navy"])),
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(size=11)),
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
                    st.success(f"**★ 본부 견인형** (건수↑ 만족도↑ · 효자 업무): {_q1_names}\n\n→ 성공 사례(Best Practice)로 선정, 전 지사 교육 자료 활용")
                if not _q2_list.empty:
                    _q2_names = ", ".join(_q2_list["업무유형"].tolist())
                    st.info(f"**전문 특화형** (건수↓ 만족도↑ · 잘하는 업무): {_q2_names}\n\n→ 담당자 상담 스크립트 · 노하우를 타 업무에 매뉴얼화")
            with _col_r:
                if not _q3_list.empty:
                    _q3_names = ", ".join(_q3_list["업무유형"].tolist())
                    st.warning(f"**개별 개선형** (건수↓ 점수↓ · 소외 업무): {_q3_names}\n\n→ 프로세스 점검 및 담당자 1:1 코칭 실시")
                if not _q4_list.empty:
                    _q4_names = ", ".join(_q4_list["업무유형"].tolist())
                    st.error(f"**🚨 최우선 혁신** (건수↑ 점수↓ · 핵심 리스크): {_q4_names}\n\n→ 전사적 태스크포스(TF) 가동, 시스템 개선 및 인력 우선 배치")

            # ── 4사분면 가중치 영향도 테이블 ──
            if not _q4_list.empty:
                st.markdown("##### 🚨 4사분면 업무 — 전체 점수 영향도 분석")
                _q4_disp = _q4_list[["업무유형", "처리건수", "건수비중", "평균만족도", "점수갭", "가중영향", "불만족건수"]].copy()
                _q4_disp.columns = ["업무유형", "처리건수", "건수비중(%)", "평균만족도", "평균 대비 갭", "가중 영향(점)", "불만족건수"]
                st.dataframe(_q4_disp.reset_index(drop=True), use_container_width=True, hide_index=True)
                _total_drag = _q4_list["가중영향"].sum()
                st.caption(f"4사분면 업무가 본부 전체 평균을 **{_total_drag:+.2f}점** 끌어내리고 있습니다. 이 업무들의 만족도를 평균 수준으로 끌어올리면 본부 전체 점수가 약 **{abs(_total_drag):.2f}점** 상승할 수 있습니다.")

            # ── 우선순위 로드맵 ──
            _road = _bbl_grp.copy()
            _road["개선우선순위점수"] = ((_road["처리건수"] / _total_n) * abs(_road["점수갭"].clip(upper=0)) * 100).round(1)
            _road = _road[_road["개선우선순위점수"] > 0].sort_values("개선우선순위점수", ascending=False)
            if not _road.empty:
                st.markdown("##### 📋 점수 향상 우선순위 로드맵 (가성비 순)")
                for _rank, (_, rr) in enumerate(_road.iterrows(), 1):
                    _kws = _biz_kw_map.get(rr["업무유형"], [])
                    _kw_str = f" · 핵심 키워드: **{', '.join(_kws)}**" if _kws else ""
                    _icon = "🔴" if _rank <= 2 else "🟡"
                    st.markdown(f"{_icon} **{_rank}순위 — {rr['업무유형']}** | 건수 {int(rr['처리건수']):,}건({rr['건수비중']}%) · 만족도 {rr['평균만족도']:.1f}점(갭 {rr['점수갭']:+.1f}) · 불만족 {int(rr['불만족건수'])}건{_kw_str}")

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════
        #  3차원 교차분석 — 계약종별 × 접수채널 × 업무유형
        # ══════════════════════════════════════════════════════════════
        st.markdown('<p class="sec-head">🔬 3차원 교차분석 — 계약종별 × 접수채널 × 업무유형</p>', unsafe_allow_html=True)

        # 채널그룹 준비 (tab3 내에서 이미 생성된 경우도 있지만 안전하게 재생성)
        _ch_col = M.get("channel")
        if _ch_col and _ch_col in df_f.columns:
            df_f["_채널그룹_cross"] = df_f[_ch_col].apply(_group_channel)
            _df_cross = df_f[df_f["_채널그룹_cross"].notna()].copy()
        else:
            _df_cross = pd.DataFrame()

        # ── ⑤ 페르소나별 서비스 경로 분석 (계약종별 × 접수채널 × 점수) ──
        if M.get("contract") and not _df_cross.empty:
            st.markdown("##### 🗺️ 페르소나별 서비스 경로 분석 (계약종별 × 접수채널)")
            st.caption("특정 계약 고객이 어떤 채널을 이용할 때 만족도가 급락하는 **'채널 미스매치'** 지점을 찾아냅니다.")
            _pv_mean = _df_cross.pivot_table(values="_점수100", index=M["contract"],
                                              columns="_채널그룹_cross", aggfunc="mean").round(1)
            _pv_cnt = _df_cross.pivot_table(values="_점수100", index=M["contract"],
                                             columns="_채널그룹_cross", aggfunc="count").fillna(0).astype(int)
            if not _pv_mean.empty and _pv_mean.size > 1:
                # 히트맵
                _hover_cross = []
                for r_idx in _pv_mean.index:
                    row_h = []
                    for c_idx in _pv_mean.columns:
                        _v = _pv_mean.loc[r_idx, c_idx] if pd.notna(_pv_mean.loc[r_idx, c_idx]) else 0
                        _cn = int(_pv_cnt.loc[r_idx, c_idx]) if r_idx in _pv_cnt.index and c_idx in _pv_cnt.columns else 0
                        _gap = round(_v - avg_score_100, 1) if pd.notna(_v) and _v > 0 else 0
                        row_h.append(f"{r_idx} × {c_idx}<br>평균: {_v:.1f}점 (전체 대비 {_gap:+.1f})<br>응답: {_cn}건")
                    _hover_cross.append(row_h)

                fig_cross_hm = px.imshow(_pv_mean, color_continuous_scale="RdYlGn",
                                          text_auto=".1f", aspect="auto", template=PLOTLY_TPL,
                                          title="계약종별 × 접수채널 만족도 히트맵 (초록=높음 / 빨강=낮음)")
                fig_cross_hm.update_traces(
                    customdata=np.array(_hover_cross),
                    hovertemplate="%{customdata}<extra></extra>")
                fig_cross_hm.update_layout(
                    height=max(300, len(_pv_mean.index) * 45 + 100),
                    margin=dict(t=60, b=40, l=120, r=60),
                    title_font=dict(size=14, color=C["navy"]),
                    xaxis_title="접수채널", yaxis_title="계약종별")
                st.plotly_chart(fig_cross_hm, use_container_width=True)

                # 채널 미스매치 탐지 (-5점 이상 급락)
                _mismatch = []
                for r_idx in _pv_mean.index:
                    for c_idx in _pv_mean.columns:
                        _v = _pv_mean.loc[r_idx, c_idx]
                        _cn = int(_pv_cnt.loc[r_idx, c_idx]) if r_idx in _pv_cnt.index and c_idx in _pv_cnt.columns else 0
                        if pd.notna(_v) and _cn >= 3:
                            _gap = _v - avg_score_100
                            if _gap <= -5:
                                _mismatch.append({"계약종": r_idx, "채널": c_idx,
                                                   "평균만족도": round(_v, 1), "응답수": _cn,
                                                   "전체대비": round(_gap, 1)})
                if _mismatch:
                    _mm_df = pd.DataFrame(_mismatch).sort_values("전체대비")
                    st.markdown("**🚨 채널 미스매치 탐지** (전체 평균 대비 -5점 이상 급락 조합)")
                    _mm_cols = st.columns(min(3, len(_mm_df)))
                    for i, (_, mmr) in enumerate(_mm_df.head(3).iterrows()):
                        with _mm_cols[i]:
                            st.markdown(
                                f'<div class="card-red">'
                                f'<b>{mmr["계약종"]} × {mmr["채널"]}</b><br><br>'
                                f'평균: <b>{mmr["평균만족도"]:.1f}점</b> ({mmr["전체대비"]:+.1f}점)<br>'
                                f'응답: {mmr["응답수"]:,}건<br><br>'
                                f'<span style="font-size:0.85em">→ 해당 고객군의 채널 상담 프로세스 점검 필요</span>'
                                f'</div>', unsafe_allow_html=True)
                else:
                    st.success("채널 미스매치 없음: 모든 계약종 × 채널 조합이 전체 평균 대비 -5점 이내입니다.")

                # ── AI 분석 버튼: 페르소나별 서비스 경로 ──
                if st.button("🤖 AI 채널 미스매치 심층 분석", key="ai_cross5_btn", type="primary", use_container_width=True):
                    if not GEMINI_AVAILABLE:
                        st.error("Gemini API 키가 설정되지 않았습니다. `.env` 파일에 `GEMINI_API_KEY`를 설정해주세요.")
                    else:
                        # VOC 수집: 미스매치 조합의 불만 VOC
                        _ai5_voc_lines = []
                        if _mismatch and M.get("voc"):
                            for mm in _mismatch[:5]:
                                _mm_sub = _df_cross[
                                    (_df_cross[M["contract"]] == mm["계약종"]) &
                                    (_df_cross["_채널그룹_cross"] == mm["채널"])
                                ]
                                _mm_vocs = _mm_sub[_mm_sub["_점수100"] < 70][M["voc"]].dropna().astype(str).tolist()
                                _mm_vocs = [v for v in _mm_vocs if v.strip() not in ("", "nan", "응답없음")][:5]
                                if _mm_vocs:
                                    _ai5_voc_lines.append(f"[{mm['계약종']} × {mm['채널']}] (평균 {mm['평균만족도']}점)")
                                    for v in _mm_vocs:
                                        _ai5_voc_lines.append(f"  - {v}")

                        _ai5_data = "히트맵 데이터:\n"
                        for r_idx in _pv_mean.index:
                            for c_idx in _pv_mean.columns:
                                _v = _pv_mean.loc[r_idx, c_idx]
                                _cn = int(_pv_cnt.loc[r_idx, c_idx]) if r_idx in _pv_cnt.index and c_idx in _pv_cnt.columns else 0
                                if pd.notna(_v) and _cn > 0:
                                    _ai5_data += f"  {r_idx} × {c_idx}: {_v:.1f}점 ({_cn}건)\n"

                        _ai5_prompt = f"""당신은 전력산업 고객만족(CS) 전문 컨설턴트입니다.
아래는 '계약종별 × 접수채널' 교차분석 데이터입니다. 전체 평균은 {avg_score_100:.1f}점입니다.

{_ai5_data}
[미스매치 조합 VOC 원문]
{chr(10).join(_ai5_voc_lines) if _ai5_voc_lines else '- 해당 없음'}

[분석 요청]
1. **채널 미스매치 원인 진단**: 특정 계약종이 특정 채널에서 만족도가 급락하는 근본 원인을 VOC 근거와 함께 구체적으로 분석하세요.
2. **고객 페르소나별 채널 전략**: 각 계약종별 고객 특성(산업용=전문적, 주택용=민감 등)을 고려한 채널 최적화 전략을 제시하세요.
3. **즉시 실행 가능한 액션플랜**: 각 미스매치 조합별로 72시간 내 조치 가능한 구체적 해결방안을 제시하세요.

※ 추상적 제안("교육 실시", "매뉴얼 배포") 금지. 반드시 데이터에 근거한 구체적 제안만 하세요."""

                        with st.spinner("Gemini AI가 채널 미스매치를 심층 분석 중…"):
                            try:
                                import urllib.request
                                _models = ["gemini-2.0-flash", "gemma-3-12b-it", "gemma-3-27b-it"]
                                _payload = {"contents": [{"parts": [{"text": _ai5_prompt}]}],
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
                                    st.markdown(_ai_text)
                            except Exception as e:
                                st.error(f"AI 분석 중 오류: {e}")
            st.markdown("---")

        # ── ⑥ 업무 성격별 채널 적합도 분석 (업무유형 × 접수채널 × 건수/점수) ──
        if M.get("business") and not _df_cross.empty:
            st.markdown("##### 🎯 업무 성격별 채널 적합도 분석 (업무유형 × 접수채널)")
            st.caption("동일 업무라도 채널에 따라 만족도가 크게 달라지는 **'채널 전환(Call Steering)'** 대상 업무를 식별합니다.")

            _biz_ch = _df_cross.groupby([M["business"], "_채널그룹_cross"])["_점수100"].agg(
                ["mean", "count"]).reset_index()
            _biz_ch.columns = ["업무유형", "채널", "평균만족도", "응답수"]
            _biz_ch = _biz_ch[_biz_ch["응답수"] >= 5]
            _biz_ch["평균만족도"] = _biz_ch["평균만족도"].round(1)

            if len(_biz_ch) >= 2:
                _ch_colors = {"직원": C["blue"], "고객센터": C["green"],
                              "한전ON": C["sky"], "기타": C["gold"]}
                _total_biz_ch = _biz_ch["응답수"].sum()
                _biz_ch["건수비중"] = (_biz_ch["응답수"] / _total_biz_ch * 100).round(1)
                _x_avg_bch = _biz_ch["응답수"].mean()
                _y_avg_bch = _biz_ch["평균만족도"].mean()

                fig_biz_ch = go.Figure()
                for ch_name in ["직원", "고객센터", "한전ON", "기타"]:
                    _sub_ch = _biz_ch[_biz_ch["채널"] == ch_name]
                    if _sub_ch.empty:
                        continue
                    fig_biz_ch.add_trace(go.Scatter(
                        x=_sub_ch["응답수"], y=_sub_ch["평균만족도"],
                        mode="markers+text",
                        marker=dict(size=(_sub_ch["건수비중"] * 3).clip(lower=12, upper=60),
                                    color=_ch_colors.get(ch_name, C["navy"]), opacity=0.75,
                                    line=dict(width=1.5, color="white")),
                        text=_sub_ch["업무유형"], textposition="top center", textfont=dict(size=11),
                        hovertemplate=("<b>%{text}</b> · " + ch_name +
                                       "<br>응답수: %{x:,}건<br>만족도: %{y:.1f}점"
                                       "<br>건수비중: %{customdata[0]:.1f}%<extra></extra>"),
                        customdata=_sub_ch[["건수비중"]].values,
                        name=ch_name,
                    ))

                fig_biz_ch.add_hline(y=_y_avg_bch, line_color=C["navy"], line_dash="dot", line_width=1,
                                      annotation_text=f"평균 만족도 {_y_avg_bch:.1f}", annotation_font_size=10,
                                      annotation_position="top left")
                fig_biz_ch.add_vline(x=_x_avg_bch, line_color=C["navy"], line_dash="dot", line_width=1,
                                      annotation_text=f"평균 건수 {_x_avg_bch:.0f}", annotation_font_size=10,
                                      annotation_position="top right")
                fig_biz_ch.update_layout(
                    height=520, margin=dict(t=60, b=60, l=60, r=40),
                    xaxis_title="처리 건수", yaxis_title="평균 만족도 (100점 환산)",
                    title=dict(text="업무유형 × 채널별 만족도 버블 차트 (색상 = 채널, 크기 = 건수 비중)",
                               font=dict(size=14, color=C["navy"])),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5,
                                font=dict(size=11)),
                    template=PLOTLY_TPL)
                st.plotly_chart(fig_biz_ch, use_container_width=True)

                # 채널 간 점수 갭이 큰 업무 TOP3
                _biz_gap = _biz_ch.groupby("업무유형")["평균만족도"].agg(["max", "min", "count"]).reset_index()
                _biz_gap.columns = ["업무유형", "최고채널점수", "최저채널점수", "채널수"]
                _biz_gap = _biz_gap[_biz_gap["채널수"] >= 2]
                _biz_gap["채널갭"] = (_biz_gap["최고채널점수"] - _biz_gap["최저채널점수"]).round(1)
                _biz_gap = _biz_gap.sort_values("채널갭", ascending=False)

                if not _biz_gap.empty:
                    st.markdown("**📊 채널 간 만족도 갭 TOP 업무** (같은 업무, 채널에 따른 점수 차이)")
                    for _, gr in _biz_gap.head(3).iterrows():
                        _biz_name = gr["업무유형"]
                        _detail = _biz_ch[_biz_ch["업무유형"] == _biz_name].sort_values("평균만족도")
                        _low = _detail.iloc[0]
                        _high = _detail.iloc[-1]
                        st.warning(
                            f"**{_biz_name}**: {_high['채널']} {_high['평균만족도']:.1f}점 vs "
                            f"{_low['채널']} {_low['평균만족도']:.1f}점 (갭 **{gr['채널갭']:.1f}점**) "
                            f"→ {_low['채널']} 채널 프로세스 개선 또는 채널 전환 검토")

                # ── AI 분석 버튼: 업무별 채널 적합도 ──
                if st.button("🤖 AI 채널 전환 전략 분석", key="ai_cross6_btn", type="primary", use_container_width=True):
                    if not GEMINI_AVAILABLE:
                        st.error("Gemini API 키가 설정되지 않았습니다. `.env` 파일에 `GEMINI_API_KEY`를 설정해주세요.")
                    else:
                        # 업무×채널 데이터 요약
                        _ai6_data = "업무유형 × 채널별 만족도 데이터:\n"
                        for _, r in _biz_ch.iterrows():
                            _ai6_data += f"  {r['업무유형']} × {r['채널']}: {r['평균만족도']:.1f}점 ({int(r['응답수'])}건)\n"

                        # 갭 큰 업무의 VOC 수집
                        _ai6_voc_lines = []
                        if M.get("voc") and not _biz_gap.empty:
                            for _, gr in _biz_gap.head(3).iterrows():
                                _bname = gr["업무유형"]
                                _detail = _biz_ch[_biz_ch["업무유형"] == _bname].sort_values("평균만족도")
                                _low_ch = _detail.iloc[0]["채널"]
                                _voc_sub = _df_cross[
                                    (_df_cross[M["business"]] == _bname) &
                                    (_df_cross["_채널그룹_cross"] == _low_ch) &
                                    (_df_cross["_점수100"] < 70)
                                ]
                                _vocs = _voc_sub[M["voc"]].dropna().astype(str).tolist()
                                _vocs = [v for v in _vocs if v.strip() not in ("", "nan", "응답없음")][:5]
                                if _vocs:
                                    _ai6_voc_lines.append(f"[{_bname} × {_low_ch}] (최저 채널)")
                                    for v in _vocs:
                                        _ai6_voc_lines.append(f"  - {v}")

                        _ai6_prompt = f"""당신은 전력산업 고객만족(CS) 전문 컨설턴트입니다.
아래는 '업무유형 × 접수채널' 교차분석 데이터입니다. 전체 평균은 {avg_score_100:.1f}점입니다.

{_ai6_data}
[채널 갭이 큰 업무의 저점수 채널 VOC 원문]
{chr(10).join(_ai6_voc_lines) if _ai6_voc_lines else '- 해당 없음'}

[분석 요청]
1. **업무 복잡도 분석**: 각 업무의 성격(단순 조회 vs 복잡한 해지/변경 등)을 파악하고, 어떤 채널이 해당 업무에 적합/부적합한지 근거와 함께 분석하세요.
2. **채널 전환(Call Steering) 전략**: 특정 업무를 특정 채널에서 다른 채널로 유도해야 하는 경우, 구체적인 전환 시나리오를 설계하세요.
   - 예: "○○ 업무는 한전ON 접수 시 → 즉시 전문상담원 콜백 연결"
3. **채널별 개선 로드맵**: UI/UX 문제인지, 상담원 전문성 문제인지, 대기시간 문제인지 VOC 근거로 가설을 세우고 각각의 해결방안을 제시하세요.

※ 추상적 제안 금지. 반드시 업무명·채널명·VOC 근거를 명시하세요."""

                        with st.spinner("Gemini AI가 채널 전환 전략을 분석 중…"):
                            try:
                                import urllib.request
                                _models = ["gemini-2.0-flash", "gemma-3-12b-it", "gemma-3-27b-it"]
                                _payload = {"contents": [{"parts": [{"text": _ai6_prompt}]}],
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
                                    st.markdown(_ai_text)
                            except Exception as e:
                                st.error(f"AI 분석 중 오류: {e}")
            st.markdown("---")

        # ── ⑦ 리스크 집중 구역 도출 (계약종별 × 업무유형 × 점수) ──
        if M.get("contract") and M.get("business"):
            st.markdown("##### 🎯 리스크 집중 구역 (계약종별 × 업무유형)")
            st.caption("우리 본부의 **'가장 아픈 손가락'** — 전사적 프로세스 개선이 필요한 타겟을 도출합니다.")

            _risk_cross = df_f.groupby([M["contract"], M["business"]])["_점수100"].agg(
                ["mean", "count"]).reset_index()
            _risk_cross.columns = ["계약종", "업무유형", "평균만족도", "응답수"]
            _risk_cross = _risk_cross[_risk_cross["응답수"] >= 3]
            _risk_cross["평균만족도"] = _risk_cross["평균만족도"].round(1)

            if len(_risk_cross) >= 2:
                _total_risk_n = _risk_cross["응답수"].sum()
                _risk_cross["건수비중"] = (_risk_cross["응답수"] / _total_risk_n * 100).round(1)
                _risk_cross["전체대비"] = (_risk_cross["평균만족도"] - avg_score_100).round(1)
                _risk_cross["리스크점수"] = (((100 - _risk_cross["평균만족도"]) / 100) * _risk_cross["건수비중"]).round(2)
                _risk_cross["조합명"] = _risk_cross["계약종"] + " × " + _risk_cross["업무유형"]

                # 트리맵
                fig_tree = px.treemap(
                    _risk_cross, path=["계약종", "업무유형"], values="응답수",
                    color="평균만족도", color_continuous_scale="RdYlGn",
                    range_color=[_risk_cross["평균만족도"].min() - 5, _risk_cross["평균만족도"].max() + 5],
                    template=PLOTLY_TPL,
                    title="계약종별 × 업무유형 트리맵 (면적 = 건수, 색상 = 만족도)")
                fig_tree.update_traces(
                    hovertemplate="<b>%{label}</b><br>응답수: %{value:,}건<br>만족도: %{color:.1f}점<extra></extra>",
                    textinfo="label+value")
                fig_tree.update_layout(
                    height=520, margin=dict(t=60, b=20, l=10, r=10),
                    title_font=dict(size=14, color=C["navy"]))
                st.plotly_chart(fig_tree, use_container_width=True)

                # TOP 5 사전케어 집중 타겟
                _top5_risk = _risk_cross.sort_values("리스크점수", ascending=False).head(5)
                st.markdown("**🚨 사전케어 집중 타겟 TOP 5** (리스크 점수 = 불만족도 × 건수비중)")
                _t5_cols = st.columns(min(5, len(_top5_risk)))
                for i, (_, tr) in enumerate(_top5_risk.iterrows()):
                    if i < len(_t5_cols):
                        _card_cls = "card-red" if tr["전체대비"] <= -5 else "card-blue"
                        with _t5_cols[i]:
                            st.markdown(
                                f'<div class="{_card_cls}" style="min-height:170px">'
                                f'<b style="font-size:0.95rem">#{i+1} {tr["계약종"]}<br>× {tr["업무유형"]}</b><br><br>'
                                f'만족도: <b>{tr["평균만족도"]:.1f}점</b><br>'
                                f'전체 대비: <b style="color:{C["red"]}">{tr["전체대비"]:+.1f}점</b><br>'
                                f'응답: {int(tr["응답수"]):,}건 ({tr["건수비중"]:.1f}%)<br>'
                                f'리스크: <b>{tr["리스크점수"]:.2f}</b>'
                                f'</div>', unsafe_allow_html=True)

                # 리스크 인덱스 테이블
                with st.expander("📋 종합 리스크 인덱스 전체 보기"):
                    _disp_risk = _risk_cross[["조합명", "응답수", "건수비중", "평균만족도", "전체대비", "리스크점수"]].copy()
                    _disp_risk.columns = ["계약종 × 업무유형", "응답수", "건수비중(%)", "평균만족도", "전체 대비", "리스크 점수"]
                    _disp_risk = _disp_risk.sort_values("리스크 점수", ascending=False)
                    st.dataframe(_disp_risk.reset_index(drop=True), use_container_width=True, hide_index=True)

                # ── AI 분석 버튼: 리스크 집중 구역 ──
                if st.button("🤖 AI 리스크 타겟 심층 분석", key="ai_cross7_btn", type="primary", use_container_width=True):
                    if not GEMINI_AVAILABLE:
                        st.error("Gemini API 키가 설정되지 않았습니다. `.env` 파일에 `GEMINI_API_KEY`를 설정해주세요.")
                    else:
                        # TOP5 리스크 조합 데이터 + VOC
                        _ai7_data = "리스크 TOP 5 조합:\n"
                        _ai7_voc_lines = []
                        for _, tr in _top5_risk.iterrows():
                            _ai7_data += f"  #{int(_ + 1) if isinstance(_, int) else ''} {tr['계약종']} × {tr['업무유형']}: {tr['평균만족도']:.1f}점 ({int(tr['응답수'])}건, 전체 대비 {tr['전체대비']:+.1f}점, 리스크 {tr['리스크점수']:.2f})\n"
                            if M.get("voc"):
                                _risk_sub = df_f[
                                    (df_f[M["contract"]] == tr["계약종"]) &
                                    (df_f[M["business"]] == tr["업무유형"]) &
                                    (df_f["_점수100"] < 70)
                                ]
                                _rvocs = _risk_sub[M["voc"]].dropna().astype(str).tolist()
                                _rvocs = [v for v in _rvocs if v.strip() not in ("", "nan", "응답없음")][:5]
                                if _rvocs:
                                    _ai7_voc_lines.append(f"[{tr['계약종']} × {tr['업무유형']}]")
                                    for v in _rvocs:
                                        _ai7_voc_lines.append(f"  - {v}")

                        _ai7_prompt = f"""당신은 전력산업 고객만족(CS) 전문 컨설턴트입니다.
아래는 '계약종별 × 업무유형' 교차분석에서 도출된 리스크 TOP 5 조합입니다. 전체 평균은 {avg_score_100:.1f}점입니다.

{_ai7_data}
[리스크 조합별 불만족 VOC 원문]
{chr(10).join(_ai7_voc_lines) if _ai7_voc_lines else '- 해당 없음'}

[분석 요청]
1. **근본 원인 진단**: 각 리스크 조합이 낮은 점수를 기록하는 근본 원인을 VOC 근거와 함께 분석하세요. 특정 지사만의 문제인지, 전사적 프로세스 문제인지 판별하세요.
2. **사전케어 집중 타겟 전략**: TOP 5 조합 각각에 대해 구체적인 사전케어 시나리오를 설계하세요.
   - 접수 즉시 해야 할 일 / 처리 중 주의사항 / 완료 후 팔로업 방법
3. **프로세스 개선 제안**: 고지 방식, 응대 스크립트, 처리 절차 중 어떤 부분을 바꿔야 하는지 구체적으로 제안하세요.
4. **우선순위 로드맵**: 리스크 점수 기준 어떤 조합부터 먼저 개선해야 효과가 큰지 순서를 매기고 근거를 제시하세요.

※ 추상적 제안 금지. 반드시 계약종명·업무명·VOC 근거를 명시하세요."""

                        with st.spinner("Gemini AI가 리스크 타겟을 심층 분석 중…"):
                            try:
                                import urllib.request
                                _models = ["gemini-2.0-flash", "gemma-3-12b-it", "gemma-3-27b-it"]
                                _payload = {"contents": [{"parts": [{"text": _ai7_prompt}]}],
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
                                    st.markdown(_ai_text)
                            except Exception as e:
                                st.error(f"AI 분석 중 오류: {e}")

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════
        #  ⑧ 4개 차원 통합 리스크 진단 (지사 × 계약종별 × 업무유형 × 접수채널)
        # ══════════════════════════════════════════════════════════════
        _has_4d = M.get("office") and M.get("contract") and M.get("business") and not _df_cross.empty
        if _has_4d:
            st.markdown('<p class="sec-head">🔎 4개 차원 통합 리스크 진단 — 지사 × 계약종별 × 업무유형 × 접수채널</p>', unsafe_allow_html=True)
            st.caption("4개 축을 동시에 교차 필터링하여 **독립적 리스크 구간**을 찾아내고, 본부 점수 상승 기여도 순으로 우선순위를 산정합니다.")

            _MIN_4D = 5  # 표본 신뢰도 최소 건수

            # ── 4차원 그룹화 ──
            _grp4 = _df_cross.groupby([M["office"], M["contract"], M["business"], "_채널그룹_cross"])["_점수100"].agg(
                ["mean", "count"]).reset_index()
            _grp4.columns = ["지사", "계약종", "업무유형", "채널", "평균만족도", "응답수"]
            _grp4["평균만족도"] = _grp4["평균만족도"].round(1)

            # 유효 리스크 구간: N >= 5 & 전체 평균 -20점 이상 급락
            _grp4_valid = _grp4[_grp4["응답수"] >= _MIN_4D].copy()
            _grp4_valid["전체대비"] = (_grp4_valid["평균만족도"] - avg_score_100).round(1)
            _total_4d_n = _grp4_valid["응답수"].sum()
            _grp4_valid["건수비중"] = (_grp4_valid["응답수"] / max(_total_4d_n, 1) * 100).round(2)
            # 점수 상승 기여도: 이 구간을 평균까지 끌어올렸을 때 전체 평균 상승분
            _grp4_valid["기여도"] = ((_grp4_valid["응답수"] / max(_total_4d_n, 1)) *
                                    (avg_score_100 - _grp4_valid["평균만족도"]).clip(lower=0)).round(2)
            _grp4_valid["조합"] = (_grp4_valid["지사"] + " · " + _grp4_valid["계약종"] + " · " +
                                   _grp4_valid["업무유형"] + " · " + _grp4_valid["채널"])

            _outliers = _grp4_valid[_grp4_valid["전체대비"] <= -20].sort_values("기여도", ascending=False)
            _risk_all = _grp4_valid[_grp4_valid["전체대비"] < 0].sort_values("기여도", ascending=False)

            # ── 메트릭 카드 ──
            _m1, _m2, _m3 = st.columns(3)
            _m1.metric("분석 대상 조합", f"{len(_grp4_valid):,}개", help=f"응답 {_MIN_4D}건 이상 유효 조합")
            _m2.metric("이상치 (-20점↓)", f"{len(_outliers):,}개",
                        delta=f"{len(_outliers)}개 독립 리스크" if len(_outliers) > 0 else "없음",
                        delta_color="inverse")
            _potential_gain = _risk_all["기여도"].sum()
            _m3.metric("최대 점수 상승 여력", f"+{_potential_gain:.1f}점",
                        help="모든 리스크 구간을 평균 수준으로 끌어올렸을 때")

            # ── 트리맵: 지사 > 업무유형 (색상=만족도, 호버에 계약종·채널 상세) ──
            # 지사×업무 단위로 집계 (트리맵 최상위 레벨)
            _tree4 = _grp4_valid.groupby(["지사", "업무유형"]).agg(
                평균만족도=("평균만족도", lambda x: np.average(x, weights=_grp4_valid.loc[x.index, "응답수"])),
                응답수=("응답수", "sum"),
                기여도=("기여도", "sum"),
            ).reset_index()
            _tree4["평균만족도"] = _tree4["평균만족도"].round(1)
            _tree4["기여도"] = _tree4["기여도"].round(2)

            # 호버에 계약종·채널 상세 추가
            _hover4_map = {}
            for _, row4 in _grp4_valid.iterrows():
                _key4 = (row4["지사"], row4["업무유형"])
                if _key4 not in _hover4_map:
                    _hover4_map[_key4] = []
                _hover4_map[_key4].append(f"{row4['계약종']}·{row4['채널']}: {row4['평균만족도']:.1f}점({int(row4['응답수'])}건)")

            _tree4["상세"] = _tree4.apply(
                lambda r: "<br>".join(_hover4_map.get((r["지사"], r["업무유형"]), ["-"])), axis=1)
            _tree4["hover_text"] = _tree4.apply(
                lambda r: (f"<b>{r['지사']} · {r['업무유형']}</b><br>"
                           f"평균: {r['평균만족도']:.1f}점 | 응답: {int(r['응답수'])}건<br>"
                           f"점수 상승 기여도: +{r['기여도']:.2f}점<br><br>"
                           f"[계약종·채널별 상세]<br>{r['상세']}"), axis=1)

            fig_tree4 = px.treemap(
                _tree4, path=["지사", "업무유형"], values="응답수",
                color="평균만족도", color_continuous_scale="RdYlGn",
                range_color=[max(0, _tree4["평균만족도"].min() - 5),
                             min(100, _tree4["평균만족도"].max() + 5)],
                template=PLOTLY_TPL,
                title="4차원 통합 트리맵 — 지사 × 업무유형 (면적 = 건수, 색상 = 만족도, 호버 = 계약종·채널 상세)")
            fig_tree4.update_traces(
                hovertemplate="%{customdata[0]}<extra></extra>",
                customdata=_tree4[["hover_text"]].values,
                textinfo="label+value")
            fig_tree4.update_layout(
                height=600, margin=dict(t=60, b=20, l=10, r=10),
                title_font=dict(size=14, color=C["navy"]))
            st.plotly_chart(fig_tree4, use_container_width=True)

            # ── 이상치 리스트 (독립 리스크 구간) ──
            if not _outliers.empty:
                st.markdown(f"##### 🚨 독립 리스크 구간 — 전체 평균 대비 -20점 이상 ({len(_outliers)}건)")

                # 원인 유형 분류
                _cause_diag = []
                for _, ol in _outliers.iterrows():
                    # 같은 업무+계약종+채널인데 다른 지사에서도 낮은지 체크
                    _same_biz = _grp4_valid[
                        (_grp4_valid["업무유형"] == ol["업무유형"]) &
                        (_grp4_valid["계약종"] == ol["계약종"]) &
                        (_grp4_valid["채널"] == ol["채널"]) &
                        (_grp4_valid["지사"] != ol["지사"])
                    ]
                    _same_biz_low = _same_biz[_same_biz["전체대비"] <= -10]

                    # 같은 지사+업무인데 다른 채널에서는 괜찮은지 체크
                    _same_ofc_ch = _grp4_valid[
                        (_grp4_valid["지사"] == ol["지사"]) &
                        (_grp4_valid["업무유형"] == ol["업무유형"]) &
                        (_grp4_valid["채널"] != ol["채널"])
                    ]
                    _same_ofc_ch_ok = _same_ofc_ch[_same_ofc_ch["전체대비"] > -5]

                    if len(_same_biz_low) >= 2:
                        _cause = "전사 프로세스"
                        _icon = "🔴"
                    elif not _same_ofc_ch_ok.empty:
                        _cause = "채널 UI/응대"
                        _icon = "🟡"
                    else:
                        _cause = "지사 맞춤 코칭"
                        _icon = "🔵"

                    _cause_diag.append({
                        "원인유형": _cause, "아이콘": _icon,
                        "조합": ol["조합"], "지사": ol["지사"], "계약종": ol["계약종"],
                        "업무유형": ol["업무유형"], "채널": ol["채널"],
                        "평균만족도": ol["평균만족도"], "응답수": int(ol["응답수"]),
                        "전체대비": ol["전체대비"], "기여도": ol["기여도"],
                    })

                _cause_df = pd.DataFrame(_cause_diag)

                # 원인 유형별 카운트
                _cc1, _cc2, _cc3 = st.columns(3)
                _n_proc = len(_cause_df[_cause_df["원인유형"] == "전사 프로세스"])
                _n_ch = len(_cause_df[_cause_df["원인유형"] == "채널 UI/응대"])
                _n_ofc = len(_cause_df[_cause_df["원인유형"] == "지사 맞춤 코칭"])
                _cc1.metric("🔴 전사 프로세스", f"{_n_proc}건", help="다수 지사에서 공통으로 낮음")
                _cc2.metric("🟡 채널 UI/응대", f"{_n_ch}건", help="특정 채널에서만 점수 급락")
                _cc3.metric("🔵 지사 맞춤 코칭", f"{_n_ofc}건", help="특정 지사에서만 점수 급락")

                # 기여도 순 테이블
                _disp4 = _cause_df[["아이콘", "원인유형", "지사", "계약종", "업무유형", "채널",
                                     "평균만족도", "응답수", "전체대비", "기여도"]].copy()
                _disp4.columns = ["", "원인 유형", "지사", "계약종", "업무유형", "채널",
                                  "평균만족도", "응답수", "전체 대비", "점수 상승 기여도"]
                st.dataframe(_disp4.reset_index(drop=True), use_container_width=True, hide_index=True)
            else:
                st.success("전체 평균 대비 -20점 이상 급락한 독립 리스크 구간이 없습니다.")

            # ── 점수 상승 기여도 TOP 10 (리스크 구간 전체) ──
            _top10_contrib = _risk_all.head(10)
            if not _top10_contrib.empty:
                st.markdown("##### 📋 본부 점수 상승 기여도 TOP 10")
                st.caption("해당 구간의 만족도를 본부 평균까지 끌어올렸을 때 전체 점수 상승 효과가 큰 순서입니다.")
                _disp_top10 = _top10_contrib[["조합", "응답수", "건수비중", "평균만족도", "전체대비", "기여도"]].copy()
                _disp_top10.columns = ["지사·계약종·업무·채널", "응답수", "건수비중(%)", "평균만족도", "전체 대비", "점수 상승 기여도"]
                st.dataframe(_disp_top10.reset_index(drop=True), use_container_width=True, hide_index=True)

            # ── AI 통합 진단 버튼 ──
            if st.button("🤖 AI 4차원 통합 리스크 진단", key="ai_cross8_btn", type="primary", use_container_width=True):
                if not GEMINI_AVAILABLE:
                    st.error("Gemini API 키가 설정되지 않았습니다. `.env` 파일에 `GEMINI_API_KEY`를 설정해주세요.")
                else:
                    # AI 프롬프트용 데이터 수집
                    _ai8_outlier_txt = ""
                    if not _outliers.empty:
                        for _, ol in _outliers.head(10).iterrows():
                            _ai8_outlier_txt += f"  [{ol['지사']}·{ol['계약종']}·{ol['업무유형']}·{ol['채널']}] {ol['평균만족도']:.1f}점 ({int(ol['응답수'])}건, 전체 대비 {ol['전체대비']:+.1f}, 기여도 +{ol['기여도']:.2f})\n"

                    _ai8_cause_txt = ""
                    if '_cause_df' in dir() and not _cause_df.empty:
                        for ct in ["전사 프로세스", "채널 UI/응대", "지사 맞춤 코칭"]:
                            _ct_rows = _cause_df[_cause_df["원인유형"] == ct]
                            if not _ct_rows.empty:
                                _ai8_cause_txt += f"\n[{ct}] ({len(_ct_rows)}건)\n"
                                for _, cr in _ct_rows.head(5).iterrows():
                                    _ai8_cause_txt += f"  {cr['지사']}·{cr['계약종']}·{cr['업무유형']}·{cr['채널']}: {cr['평균만족도']:.1f}점\n"

                    _ai8_voc_lines = []
                    if M.get("voc") and not _outliers.empty:
                        for _, ol in _outliers.head(5).iterrows():
                            _ol_sub = _df_cross[
                                (_df_cross[M["office"]] == ol["지사"]) &
                                (_df_cross[M["contract"]] == ol["계약종"]) &
                                (_df_cross[M["business"]] == ol["업무유형"]) &
                                (_df_cross["_채널그룹_cross"] == ol["채널"]) &
                                (_df_cross["_점수100"] < 70)
                            ]
                            _ol_vocs = _ol_sub[M["voc"]].dropna().astype(str).tolist()
                            _ol_vocs = [v for v in _ol_vocs if v.strip() not in ("", "nan", "응답없음")][:3]
                            if _ol_vocs:
                                _ai8_voc_lines.append(f"[{ol['지사']}·{ol['계약종']}·{ol['업무유형']}·{ol['채널']}]")
                                for v in _ol_vocs:
                                    _ai8_voc_lines.append(f"  - {v}")

                    _ai8_prompt = f"""당신은 전력산업 고객만족(CS) 전문 컨설턴트입니다.
아래는 [지사 × 계약종별 × 업무유형 × 접수채널] 4차원 교차분석 결과입니다. 전체 평균은 {avg_score_100:.1f}점, 분석 대상 유효 조합 {len(_grp4_valid)}개입니다.

[독립 리스크 구간 (전체 평균 대비 -20점↓)]
{_ai8_outlier_txt if _ai8_outlier_txt else '  해당 없음'}

[원인 유형 분류]
{_ai8_cause_txt if _ai8_cause_txt else '  해당 없음'}

[리스크 구간 VOC 원문]
{chr(10).join(_ai8_voc_lines) if _ai8_voc_lines else '- 해당 없음'}

[최대 점수 상승 여력: +{_potential_gain:.1f}점]

[분석 요청]
1. **입체적 원인 진단**:
   - 🔴 전사 프로세스 문제: 다수 지사에서 공통으로 낮은 조합 → 전사적 프로세스 개선안
   - 🟡 채널 UI/응대 문제: 특정 채널에서만 낮은 조합 → 디지털/전화 응대 개선안
   - 🔵 지사 맞춤 코칭: 특정 지사에서만 낮은 조합 → 해당 지사 코칭 시나리오

2. **우선순위 로드맵**: 점수 상승 기여도가 높은 순서대로 TOP 5 구간에 대해:
   - 즉시 조치(1주 내), 중기 개선(1개월), 장기 프로세스 혁신으로 구분
   - 예상 점수 상승 효과를 수치로 제시

3. **구체적 액션플랜**: 각 리스크 유형별로 실행 가능한 해결방안을 구체적으로 제시
   - 상담 스크립트 수정안, 채널 전환 규칙, 지사 교육 포인트 등

※ 추상적 제안 금지. 반드시 지사명·계약종·업무명·채널명·VOC 근거를 명시하세요."""

                    with st.spinner("Gemini AI가 4차원 통합 리스크를 진단 중…"):
                        try:
                            import urllib.request
                            _models = ["gemini-2.0-flash", "gemma-3-12b-it", "gemma-3-27b-it"]
                            _payload = {"contents": [{"parts": [{"text": _ai8_prompt}]}],
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
                                st.markdown(_ai_text)
                        except Exception as e:
                            st.error(f"AI 분석 중 오류: {e}")


# ─────────────────────────────────────────────────────────────
#  TAB 5  민원 조기 경보 시스템
# ─────────────────────────────────────────────────────────────
with tab5:
    # ══════════════════════════════════════════════════════════════
    #  SECTION A. 민원 조기 경보 대시보드
    # ══════════════════════════════════════════════════════════════
    st.markdown('<p class="sec-head">🚨 민원 조기 경보 대시보드</p>', unsafe_allow_html=True)

    # ── A-1. 임계치 기반 경보 등급 산정 ──
    _RISK_THRESHOLD = 3  # 50점 미만 N건 이상 → 경고
    _alert_results = []  # (대상, 유형, 등급, 사유, 건수, 평균)
    _CRITICAL_KW = {"법적", "소송", "고소", "고발", "언론", "보도", "기사", "국민신문고", "신문고",
                     "감사원", "감사", "국회", "청와대", "검찰", "경찰", "변호사", "인권"}

    if M.get("office") and M.get("score") and "_점수100" in df_f.columns:
        # 30점 이하 + 강력 키워드 → [위험] 등급 (개별 건 단위)
        _critical_alerts = []
        if M.get("voc"):
            _ultra_low = df_f[df_f["_점수100"] <= 30]
            for _, row in _ultra_low.iterrows():
                _voc_text = str(row.get(M["voc"], ""))
                _found_critical = [kw for kw in _CRITICAL_KW if kw in _voc_text]
                if _found_critical:
                    _ofc_name = str(row.get(M["office"], "미상"))
                    _biz_name = str(row.get(M["business"], "")) if M.get("business") else ""
                    _score_val = row["_점수100"]
                    _critical_alerts.append((_ofc_name, _biz_name, _found_critical, _score_val, _voc_text[:80]))

        # 지사별 50점 미만 집중도 분석
        for ofc in df_f[M["office"]].dropna().unique():
            _ofc_df = df_f[df_f[M["office"]] == ofc]
            _ofc_n = len(_ofc_df)
            _ofc_low = (_ofc_df["_점수100"] < 50).sum()
            _ofc_avg = _ofc_df["_점수100"].mean()

            if _ofc_low >= _RISK_THRESHOLD:
                _grade = "🔴 위험" if _ofc_low >= _RISK_THRESHOLD * 2 else "🟡 주의"
                _alert_results.append((ofc, "지사", _grade,
                    f"50점 미만 {_ofc_low}건 집중 (전체 {_ofc_n}건 중 {_ofc_low/_ofc_n*100:.0f}%)",
                    _ofc_low, _ofc_avg))

        # 업무유형별 분석
        if M.get("business"):
            for biz in df_f[M["business"]].dropna().unique():
                _biz_df = df_f[df_f[M["business"]] == biz]
                _biz_n = len(_biz_df)
                _biz_low = (_biz_df["_점수100"] < 50).sum()
                _biz_avg = _biz_df["_점수100"].mean()

                if _biz_low >= _RISK_THRESHOLD:
                    _grade = "🔴 위험" if _biz_low >= _RISK_THRESHOLD * 2 else "🟡 주의"
                    _alert_results.append((biz, "업무", _grade,
                        f"50점 미만 {_biz_low}건 집중 (전체 {_biz_n}건 중 {_biz_low/_biz_n*100:.0f}%)",
                        _biz_low, _biz_avg))

        # 영향력(가중치) 기준 정렬: 건수비중 × 점수갭
        _total_n = len(df_f)
        for i, r in enumerate(_alert_results):
            _weight = (r[4] / max(_total_n, 1)) * abs(r[5] - df_f["_점수100"].mean())
            _alert_results[i] = (*r, round(_weight, 2))
        _alert_results.sort(key=lambda x: -x[6])  # 가중치 높은 순

    # ── A-2. 부정 키워드 밀집도 분석 ──
    _HIGHRISK_KW = {"고지서", "오류", "단전", "강제", "폭탄", "누전", "정전", "화재", "위험",
                     "사고", "감전", "합선", "과금", "착오", "잘못", "엉터리", "횡포",
                     "고지서오류", "강제단전", "폭탄요금"}
    _kw_alerts = []
    if M.get("voc") and M.get("office"):
        for ofc in df_f[M["office"]].dropna().unique():
            _ofc_vocs = df_f.loc[df_f[M["office"]] == ofc, M["voc"]].dropna().astype(str)
            _ofc_text = " ".join(_ofc_vocs.tolist())
            _found_kw = [kw for kw in _HIGHRISK_KW if kw in _ofc_text]
            if len(_found_kw) >= 2:
                _kw_alerts.append((ofc, _found_kw, len(_found_kw)))

    # ── A-3. 경보 대시보드 렌더링 ──
    _critical_cnt = len(_critical_alerts) if '_critical_alerts' in dir() else 0
    _danger_cnt = sum(1 for r in _alert_results if "위험" in r[2])
    _warn_cnt = sum(1 for r in _alert_results if "주의" in r[2])

    # 종합 경보 등급
    if _critical_cnt > 0 or _danger_cnt > 0:
        _overall_grade = "🔴 위험"
        _overall_color = C["red"]
        _overall_msg = "즉각 대응이 필요한 위험 징후가 감지되었습니다. 30점 이하 강력 키워드 포함 건이 존재합니다." if _critical_cnt else "즉각 대응이 필요한 위험 징후가 감지되었습니다."
    elif _warn_cnt > 0 or len(_kw_alerts) >= 1:
        _overall_grade = "🟡 주의"
        _overall_color = C["orange"]
        _overall_msg = "주의가 필요한 경고 징후가 감지되었습니다."
    else:
        _overall_grade = "🟢 안전"
        _overall_color = C["green"]
        _overall_msg = "현재 특이 징후가 감지되지 않았습니다."

    _g1, _g2, _g3, _g4 = st.columns(4)
    with _g1:
        st.markdown(f'<div style="background:{_overall_color};color:white;padding:18px;border-radius:10px;text-align:center">'
                    f'<div style="font-size:28px;font-weight:bold">{_overall_grade}</div>'
                    f'<div style="font-size:12px;margin-top:4px">종합 경보 등급</div></div>', unsafe_allow_html=True)
    with _g2:
        st.metric("🔴 위험 (30점↓+강력KW)", f"{_critical_cnt}건")
    with _g3:
        st.metric("🟠 임계치 초과", f"{_danger_cnt + _warn_cnt}건")
    with _g4:
        st.metric("⚠️ 키워드 밀집", f"{len(_kw_alerts)}건")

    st.caption(_overall_msg)

    # ── 30점 이하 + 강력 키워드 [위험] 건 ──
    if _critical_cnt > 0:
        st.markdown("---")
        st.markdown("##### 🚨 [위험] 30점 이하 + 강력 키워드 감지 건")
        for _ofc, _biz, _ckws, _sc, _vtxt in _critical_alerts:
            _biz_str = f" · {_biz}" if _biz else ""
            st.error(f"**{_ofc}{_biz_str}** | {_sc:.0f}점 | 감지 키워드: `{', '.join(_ckws)}`\n\n> \"{_vtxt}…\"")

    # ── 임계치 기반 경보 테이블 (가중치 순) ──
    if _alert_results:
        st.markdown("---")
        st.markdown("##### 📋 임계치 기반 경보 목록 (영향력 가중치 순)")
        _alert_df = pd.DataFrame(_alert_results, columns=["대상", "유형", "등급", "사유", "불만족건수", "평균점수", "영향력가중치"])
        _alert_df["평균점수"] = _alert_df["평균점수"].round(1)
        st.dataframe(_alert_df[["등급", "유형", "대상", "사유", "불만족건수", "평균점수", "영향력가중치"]],
                     use_container_width=True, hide_index=True)

    # 키워드 밀집도 경보
    if _kw_alerts:
        st.markdown("---")
        st.markdown("##### 🔍 부정 키워드 밀집도 경보")
        st.markdown('<div class="card-red"><b>📌 고위험 키워드</b> (고지서 오류, 강제 단전, 폭탄 요금, 누전, 정전 등) 가 '
                    '특정 지사에 집중 출현할 경우 리스크 등급을 격상합니다.</div>', unsafe_allow_html=True)
        for ofc, kws, cnt in sorted(_kw_alerts, key=lambda x: -x[2]):
            _grade_kw = "🔴 위험" if cnt >= 4 else "🟡 경고"
            st.warning(f"{_grade_kw} **{ofc}** — 고위험 키워드 {cnt}종 감지: `{', '.join(kws)}`")

    # 골든타임 카운터
    if M.get("score") and "_점수100" in df_f.columns:
        st.markdown("---")
        st.markdown("##### ⏱ 사전케어 골든타임 관리")
        _low_total = (df_f["_점수100"] < 50).sum()
        _col_gt1, _col_gt2, _col_gt3 = st.columns(3)
        with _col_gt1:
            st.metric("📌 50점 미만 총 건수", f"{_low_total}건")
        with _col_gt2:
            st.metric("⏰ 골든타임 기준", "72시간 이내")
        with _col_gt3:
            st.metric("🎯 목표 해피콜 완료율", "100%")
        st.markdown(
            '<div class="card-blue">'
            '<b>⏱ 골든타임 프로토콜</b><br>'
            '불만족 응답 감지 → <b>24시간 이내</b> 최초 해피콜/조치 착수 → <b>72시간 이내</b> 처리 완료<br>'
            '24시간 초과 시 팀장에게 자동 에스컬레이션 · 72시간 초과 시 지사장 직접 관리 전환</div>',
            unsafe_allow_html=True)

        # 지사별 50점 미만 건수 현황 (해피콜 대상)
        if M.get("office"):
            _gt_ofc = df_f[df_f["_점수100"] < 50].groupby(M["office"]).size().reset_index(name="해피콜 대상")
            _gt_ofc = _sort_df_by_office(_gt_ofc, M["office"])
            _gt_ofc.columns = ["지사", "해피콜 대상(건)"]
            st.dataframe(_gt_ofc.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════
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
            all_neg_flat = []
            for kws in df_neg["감지된_부정키워드"]:
                all_neg_flat.extend([k.strip() for k in kws.split(",") if k.strip()])
            neg_kw_cnt = Counter(all_neg_flat).most_common(10)
            if neg_kw_cnt:
                nkw_df = pd.DataFrame(neg_kw_cnt, columns=["부정키워드", "감지횟수"])
                nk_l, nk_r = st.columns([3, 2])
                with nk_l:
                    fig_neg = px.bar(nkw_df, x="부정키워드", y="감지횟수", color_discrete_sequence=[C["red"]],
                                     text="감지횟수", template=PLOTLY_TPL, title="부정 키워드 Top 10")
                    fig_neg.update_traces(texttemplate="%{text}", textposition="outside",
                                          hovertemplate="%{x}: %{y}회<extra></extra>")
                    fig_neg.update_layout(height=340, margin=dict(t=50, b=70, l=60, r=20), xaxis_tickangle=-25,
                                           title_font=dict(size=14, color=C["navy"]))
                    st.plotly_chart(fig_neg, use_container_width=True)
                with nk_r:
                    fig_don = px.pie(nkw_df.head(10), names="부정키워드", values="감지횟수", hole=0.5,
                                     color_discrete_sequence=px.colors.sequential.Reds[::-1],
                                     title="부정 키워드 비중", template=PLOTLY_TPL)
                    fig_don.update_traces(textinfo="percent+label", textfont_size=11,
                                           marker=dict(line=dict(color="white", width=2)),
                                           hovertemplate="%{label}<br>%{value}회 (%{percent})<extra></extra>")
                    fig_don.update_layout(height=340, margin=dict(t=50, b=20, l=20, r=20), showlegend=False,
                                           title_font=dict(size=14, color=C["navy"]))
                    st.plotly_chart(fig_don, use_container_width=True)

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
                                st.markdown(_ai_text)
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
                            st.markdown(_body["candidates"][0]["content"]["parts"][0]["text"].strip())
                    except Exception as e:
                        st.error(f"AI 분석 중 오류: {e}")
        else:
            st.caption("버튼을 누르면 Gemini AI가 VOC를 분석하여 민원 발전 가능성과 선제 대응 방안을 제시합니다.")
    else:
        st.info("업무유형 컬럼과 점수 컬럼이 필요합니다.")


# ─────────────────────────────────────────────────────────────
#  TAB 9  지사 심층 분석 · 패턴 탐지
# ─────────────────────────────────────────────────────────────
with tab9:
    if not M["office"] or not M["score"]:
        st.warning("지사 컬럼과 만족도 점수 컬럼이 필요합니다.")
    else:
        # ── 9-1. 지사별 강점/약점 자동 추출 ──
        if individual_scores:
            st.markdown('<p class="sec-head">💪 지사별 강점 · 약점 항목</p>', unsafe_allow_html=True)
            st.markdown(
                '<div class="card-blue"><b>📌 분석 방법</b> — 각 지사의 개별항목 점수를 전체 평균과 비교하여 '
                '가장 잘하는 항목(강점)과 가장 부족한 항목(약점)을 자동 추출합니다.</div>',
                unsafe_allow_html=True)

            overall_means = df_f[individual_scores].mean()
            office_detail = df_f.groupby(M["office"])[individual_scores].mean()
            office_detail = office_detail.reindex(_sort_offices(office_detail.index.tolist()))

            # 지사별 강점/약점 카드
            _ofc_means = df_f.groupby(M["office"])["_점수100"].mean()
            offices_sorted = _ofc_means.reindex(_sort_offices(_ofc_means.index.tolist()))
            card_cols = st.columns(min(3, len(offices_sorted)))

            for i, (ofc, _) in enumerate(offices_sorted.items()):
                if ofc not in office_detail.index:
                    continue
                row = office_detail.loc[ofc]
                diff = row - overall_means
                strength = diff.idxmax()
                weakness = diff.idxmin()
                ofc_avg = df_f[df_f[M["office"]] == ofc]["_점수100"].mean()

                card_class = "card-red" if ofc_avg < avg_score_100 - 3 else "card-teal" if ofc_avg >= avg_score_100 else "card"
                with card_cols[i % len(card_cols)]:
                    st.markdown(
                        f'<div class="{card_class}">'
                        f'<b>🏢 {ofc}</b> (종합 {ofc_avg:.1f}점)<br><br>'
                        f'💪 강점: <b>{strength}</b> (평균 대비 <span style="color:{C["green"]}">{diff[strength]:+.1f}</span>)<br>'
                        f'⚠️ 약점: <b>{weakness}</b> (평균 대비 <span style="color:{C["red"]}">{diff[weakness]:+.1f}</span>)'
                        f'</div>', unsafe_allow_html=True)

            st.markdown("---")

            # ── 지사 간 격차가 큰 항목 ──
            st.markdown('<p class="sec-head">📊 지사 간 격차가 큰 항목 (편차 Top 5)</p>', unsafe_allow_html=True)
            office_std = office_detail.std().sort_values(ascending=False)
            gap_items = office_std.head(5)
            gap_df = pd.DataFrame({
                "항목": gap_items.index,
                "지사간 표준편차": gap_items.values.round(2),
                "최고 지사": [office_detail[item].idxmax() for item in gap_items.index],
                "최고 점수": [office_detail[item].max().round(1) for item in gap_items.index],
                "최저 지사": [office_detail[item].idxmin() for item in gap_items.index],
                "최저 점수": [office_detail[item].min().round(1) for item in gap_items.index],
            })
            gap_df["격차"] = gap_df["최고 점수"] - gap_df["최저 점수"]

            fig_gap = px.bar(gap_df, x="항목", y="격차", text="격차",
                             color_discrete_sequence=[C["gold"]], template=PLOTLY_TPL,
                             title="지사 간 격차가 큰 항목 (최고-최저 점수 차이)")
            fig_gap.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                                  hovertemplate="%{x}<br>격차: %{y:.1f}점<extra></extra>")
            fig_gap.update_layout(height=380, margin=dict(t=60, b=60, l=60, r=20),
                                   yaxis_title="점수 격차", title_font=dict(size=14, color=C["navy"]))
            st.plotly_chart(fig_gap, use_container_width=True)

            with st.expander("📋 지사 간 격차 상세 데이터"):
                st.dataframe(gap_df, use_container_width=True, hide_index=True)

            # 인사이트
            top_gap = gap_df.iloc[0]
            st.markdown(
                f'<div class="insight-box">📊 <b>격차 인사이트:</b> '
                f'<b>{top_gap["항목"]}</b> 항목에서 지사 간 격차가 가장 큽니다. '
                f'{top_gap["최고 지사"]}({top_gap["최고 점수"]}점) vs '
                f'{top_gap["최저 지사"]}({top_gap["최저 점수"]}점), '
                f'격차 <b>{top_gap["격차"]:.1f}점</b> — 하위 지사 대상 집중 교육이 필요합니다.</div>',
                unsafe_allow_html=True)

            st.markdown("---")

        # ── 9-2. 교차 패턴 탐지 (업무×채널 급락 조합) ──
        if M["business"] and M["channel"]:
            st.markdown('<p class="sec-head">🔍 업무 × 채널 교차 패턴 탐지</p>', unsafe_allow_html=True)
            st.markdown(
                '<div class="card-gold"><b>📌 분석 방법</b> — 업무와 채널의 모든 조합에서 '
                '평균 만족도가 전체 평균 대비 크게 낮은 조합을 자동 탐지합니다.</div>',
                unsafe_allow_html=True)

            df_f["_채널그룹_pat"] = df_f[M["channel"]].apply(_group_channel)
            cross_pat = df_f[df_f["_채널그룹_pat"].notna()].groupby(
                [M["business"], "_채널그룹_pat"])["_점수100"].agg(["mean","count"]).reset_index()
            cross_pat.columns = ["업무", "채널", "평균만족도", "응답수"]
            cross_pat = cross_pat[cross_pat["응답수"] >= 5]  # 최소 5건 이상

            if not cross_pat.empty:
                cross_pat["전체대비"] = cross_pat["평균만족도"] - avg_score_100
                danger = cross_pat[cross_pat["전체대비"] <= -5].sort_values("전체대비")

                # 히트맵
                pivot_pat = cross_pat.pivot_table(values="평균만족도", index="업무",
                                                  columns="채널", aggfunc="mean").round(1)
                fig_pat = px.imshow(pivot_pat, color_continuous_scale="RdYlGn",
                                    text_auto=".1f", aspect="auto", template=PLOTLY_TPL,
                                    title="업무 × 채널 평균 만족도 (빨강=낮음)")
                fig_pat.update_traces(hovertemplate="업무: %{y}<br>채널: %{x}<br>점수: %{z:.1f}점<extra></extra>")
                fig_pat.update_layout(
                    height=max(350, len(pivot_pat) * 35 + 100),
                    margin=dict(t=60, b=60, l=120, r=60),
                    title_font=dict(size=14, color=C["navy"]))
                st.plotly_chart(fig_pat, use_container_width=True)

                if len(danger) > 0:
                    st.markdown(
                        f'<p class="sec-head">🚨 급락 조합 (전체 평균 대비 -5점 이상)</p>',
                        unsafe_allow_html=True)
                    for _, row in danger.head(5).iterrows():
                        st.markdown(
                            f'<div class="card-red">'
                            f'🔴 <b>[{row["업무"]}] × [{row["채널"]}]</b> — '
                            f'평균 {row["평균만족도"]:.1f}점 '
                            f'(전체 대비 <b>{row["전체대비"]:+.1f}점</b>, 응답 {row["응답수"]:,}건)'
                            f'</div>', unsafe_allow_html=True)
                else:
                    st.success("전체 평균 대비 -5점 이상 급락한 조합이 없습니다.")
            st.markdown("---")

        # ── 9-3. 범주별 개별항목 평균 히트맵 (다차원 교차분석에서 이동) ──
        if individual_scores and available_cats:
            st.markdown("---")
            st.markdown('<p class="sec-head">🌡️ 범주별 개별항목 평균 히트맵</p>', unsafe_allow_html=True)
            cat_sel_hm = st.selectbox("히트맵 기준 범주 선택", available_cats, key="hm_cat_sel_tab9")
            pivot_cross = df_f.pivot_table(
                values=individual_scores, index=cat_sel_hm, aggfunc="mean"
            ).round(1)
            if cat_sel_hm == M.get("office"):
                pivot_cross = pivot_cross.reindex(_sort_offices(pivot_cross.index.tolist()))
            if not pivot_cross.empty:
                fig_hm_cross = px.imshow(
                    pivot_cross, color_continuous_scale="RdYlGn",
                    text_auto=".1f", aspect="auto", template=PLOTLY_TPL,
                    labels=dict(x="항목", y="범주", color="점수"),
                    title=f"{cat_sel_hm}별 개별항목 평균 점수 (초록=높음 / 빨강=낮음)",
                )
                fig_hm_cross.update_traces(hovertemplate="%{y}<br>%{x}<br>점수: %{z:.1f}점<extra></extra>")
                fig_hm_cross.update_layout(
                    height=max(350, len(pivot_cross) * 30 + 100),
                    margin=dict(t=60, b=60, l=120, r=60),
                    title_font=dict(size=14, color=C["navy"]),
                )
                st.plotly_chart(fig_hm_cross, use_container_width=True)

# ─────────────────────────────────────────────────────────────
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
        df_cause = pd.DataFrame([{"원인": k, "건수": v} for k, v in cause_counts.most_common()])
        df_cause = df_cause[df_cause["건수"] > 0]

        if not df_cause.empty:
            fig_cause = px.bar(df_cause, x="건수", y="원인", orientation="h",
                               text="건수", template=PLOTLY_TPL,
                               color="건수",
                               color_continuous_scale=["#F9E79F", "#E74C3C"],
                               title="2차 원인 태깅 — 부정 VOC의 근본 원인 분류")
            fig_cause.update_traces(textposition="outside",
                                    hovertemplate="%{y}: %{x}건<extra></extra>")
            fig_cause.update_layout(height=max(300, len(df_cause) * 45 + 80),
                                    margin=dict(t=60, b=20, l=10, r=80),
                                    coloraxis_showscale=False,
                                    title_font=dict(size=14, color=C["navy"]))

            ch_l, ch_r = st.columns(2)
            with ch_l:
                st.plotly_chart(fig_sent, use_container_width=True)
            with ch_r:
                st.plotly_chart(fig_cause, use_container_width=True)

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
            st.plotly_chart(fig_sent, use_container_width=True)

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

    st.markdown("---")

    # ══════════════════════════════════════════
    # 2. 계약종별-점수 상관관계 (OOS 제외 순수 데이터)
    # ══════════════════════════════════════════
    st.markdown('<p class="sec-head">2️⃣ 계약종별 — 개별항목↔종합점수 상관관계 (순수 데이터)</p>',
                unsafe_allow_html=True)

    if M["score"] and "_점수100" in df_pure.columns:
        score_item_cols = []
        for c in df_pure.columns:
            if c in ("_점수100", "_원본순번", "_접수일", "_is_oos"): continue
            if c.startswith("_"): continue
            vals = pd.to_numeric(df_pure[c], errors="coerce")
            valid_ratio = vals.notna().sum() / max(len(df_pure), 1)
            if valid_ratio > 0.1 and vals.nunique() > 2:
                corr = vals.corr(df_pure["_점수100"])
                if pd.notna(corr) and abs(corr) > 0.05 and c != M["score"]:
                    score_item_cols.append(c)

        contract_col = M.get("contract")
        if contract_col and "_계약종별" in df_pure.columns:
            contract_groups = df_pure["_계약종별"].dropna().unique()
            contract_groups = [g for g in ["주택용", "일반용", "산업용", "농사용", "교육용", "가로등"]
                               if g in contract_groups and len(df_pure[df_pure["_계약종별"] == g]) >= 5]

            if score_item_cols and contract_groups:
                corr_data = []
                for grp_name in contract_groups:
                    sub = df_pure[df_pure["_계약종별"] == grp_name]
                    for col in score_item_cols:
                        vals = pd.to_numeric(sub[col], errors="coerce")
                        both = pd.DataFrame({"item": vals, "total": sub["_점수100"]}).dropna()
                        if len(both) >= 5:
                            corr = both["item"].corr(both["total"])
                            if pd.notna(corr):
                                corr_data.append({"계약종": grp_name, "항목": col, "상관계수": round(corr, 3)})

                if corr_data:
                    df_corr = pd.DataFrame(corr_data)
                    pivot_corr = df_corr.pivot_table(index="항목", columns="계약종", values="상관계수")

                    fig_hm = px.imshow(pivot_corr, text_auto=".2f", aspect="auto",
                                       color_continuous_scale="RdYlGn",
                                       title="계약종별 개별항목↔종합점수 상관관계 히트맵 (OOS 제외)")
                    fig_hm.update_traces(hovertemplate="항목: %{y}<br>계약종: %{x}<br>상관계수: %{z:.3f}<extra></extra>")
                    fig_hm.update_layout(height=max(350, len(score_item_cols) * 40 + 100),
                                          margin=dict(t=60, b=20, l=10, r=20),
                                          title_font=dict(size=14, color=C["navy"]))
                    st.plotly_chart(fig_hm, use_container_width=True)

                    # 핵심 인사이트 자동 도출
                    insights = []
                    for grp_name in contract_groups:
                        grp_corr = df_corr[df_corr["계약종"] == grp_name].sort_values("상관계수", ascending=False)
                        if len(grp_corr) >= 2:
                            top = grp_corr.iloc[0]
                            if top["상관계수"] > 0.8:
                                insights.append(
                                    f"**{grp_name}** 고객은 <b>{top['항목']}</b>(상관 {top['상관계수']:.2f})이 "
                                    f"종합점수에 가장 큰 영향 → 이 항목 집중 개선 시 만족도 급상승 가능")
                            grp_sub = df_pure[df_pure["_계약종별"] == grp_name]
                            grp_mean = pd.to_numeric(grp_sub[top["항목"]], errors="coerce").mean()
                            overall_mean = pd.to_numeric(df_pure[top["항목"]], errors="coerce").mean()
                            if pd.notna(grp_mean) and pd.notna(overall_mean) and grp_mean < overall_mean - 2:
                                insights.append(
                                    f"  → {grp_name}의 {top['항목']} 평균 {grp_mean:.1f}점은 "
                                    f"전체 평균 {overall_mean:.1f}점 대비 **{overall_mean - grp_mean:.1f}점 낮음** (집중 관리 필요)")

                    if insights:
                        st.markdown(
                            '<div class="card-gold"><b>📌 계약종별 핵심 상관관계 인사이트</b><br><br>' +
                            "<br>".join(f"• {ins}" for ins in insights[:6]) +
                            '</div>', unsafe_allow_html=True)

                    # 계약종별 응대 가이드
                    GUIDE = {
                        "주택용": ("친절도 + 편리성", "감성적 공감이 점수를 좌우하는 고객층. 편안하고 따뜻한 응대가 핵심"),
                        "일반용": ("처리 정확성 + 절차 간소화", "사업자는 시간이 곧 비용. 정확하게 한 번에 처리하는 것이 최우선"),
                        "산업용": ("전문성 + 신속성", "수전설비·피크요금 등 전력 전문지식이 필수. 친절보다 정확한 답을 원함"),
                        "농사용": ("친절도 + 쉬운 용어", "디지털 채널 이용률 낮은 고객층. 따뜻한 안내와 쉬운 설명이 핵심"),
                        "교육용": ("안정적 전력 공급", "학교·교육시설 특성상 안정적 공급과 요금 체계 명확성이 중요"),
                        "가로등": ("신속한 고장 처리", "야간 안전과 직결. 고장 신고 후 빠른 복구가 최우선"),
                    }
                    st.markdown('<p class="sec-head">📘 계약종별 응대 매뉴얼 인사이트</p>',
                                unsafe_allow_html=True)
                    guide_cols = st.columns(min(3, len(contract_groups)))
                    for i, grp_name in enumerate(contract_groups[:6]):
                        with guide_cols[i % 3]:
                            focus, desc = GUIDE.get(grp_name, ("일반", ""))
                            n = len(df_pure[df_pure["_계약종별"] == grp_name])
                            avg = df_pure[df_pure["_계약종별"] == grp_name]["_점수100"].mean()
                            st.markdown(
                                f'<div class="card-blue" style="min-height:160px">'
                                f'<b>{grp_name}</b> (n={n:,}, 평균 {avg:.1f}점)<br><br>'
                                f'🎯 <b>핵심 가치:</b> {focus}<br><br>'
                                f'<span style="font-size:0.9em">{desc}</span></div>',
                                unsafe_allow_html=True)

        else:
            st.info("계약종별 컬럼이 필요합니다.")
    else:
        st.info("종합 점수 컬럼이 필요합니다.")

    st.markdown("---")

    # ══════════════════════════════════════════
    # 3. 지사별 페르소나 (OOS 제외)
    # ══════════════════════════════════════════
    st.markdown('<p class="sec-head">3️⃣ 지사별 페르소나 — 데이터 기반 캐릭터 프로파일링 (순수 데이터)</p>',
                unsafe_allow_html=True)

    office_col = M.get("office")
    if office_col and M["voc"] and "_점수100" in df_pure.columns:
        offices = _sort_offices(df_pure[office_col].dropna().unique().tolist())
        if len(offices) >= 2:
            persona_data = []
            for ofc in offices:
                sub = df_pure[df_pure[office_col] == ofc]
                if len(sub) < 5:
                    continue
                avg_score = sub["_점수100"].mean()
                n = len(sub)

                # 부정 VOC (문맥 인식)
                voc_vals = sub[M["voc"]].dropna()
                voc_vals = voc_vals[~voc_vals.isin(["응답없음", "nan", ""])]
                neg_cnt = 0
                top_neg_kws = Counter()
                top_cause_cnt = Counter()
                for v in voc_vals:
                    s = str(v)
                    is_neg_found = False
                    for kw in NEGATIVE_KEYWORDS:
                        if kw in s:
                            idx_kw = s.find(kw)
                            w = s[max(0, idx_kw - 15):min(len(s), idx_kw + len(kw) + 15)]
                            if not any(pw in w for pw in POSITIVE_CONTEXT):
                                top_neg_kws[kw] += 1
                                is_neg_found = True
                    if is_neg_found:
                        neg_cnt += 1
                        # 원인 태깅
                        for cause, keywords in _CAUSE_TAGS.items():
                            if any(kw in s and not any(
                                pw in s[max(0, s.find(kw) - 15):min(len(s), s.find(kw) + len(kw) + 15)]
                                for pw in POSITIVE_CONTEXT) for kw in keywords if kw in s):
                                top_cause_cnt[cause] += 1
                                break

                neg_ratio = neg_cnt / max(n, 1) * 100

                # 계약종 비중
                contract_dist = ""
                if "_계약종별" in sub.columns:
                    top_ct = sub["_계약종별"].value_counts(normalize=True).head(2)
                    contract_dist = ", ".join(f"{k} {v * 100:.0f}%" for k, v in top_ct.items())

                # 업무 비중
                biz_dist = ""
                if M.get("business") and M["business"] in sub.columns:
                    top_biz = sub[M["business"]].value_counts(normalize=True).head(2)
                    biz_dist = ", ".join(f"{k} {v * 100:.0f}%" for k, v in top_biz.items())

                # 주요 부정 원인
                top_cause_str = ", ".join(
                    f"{k}({v})" for k, v in top_cause_cnt.most_common(2)) if top_cause_cnt else "없음"
                top_neg_str = ", ".join(
                    f"{k}({v})" for k, v in top_neg_kws.most_common(3)) if top_neg_kws else "없음"

                # 약점 항목 (전체 대비 3점 이상 낮은 항목)
                weak_items = []
                for col in df_pure.columns:
                    if col.startswith("_") or col == M["score"]:
                        continue
                    ofc_val = pd.to_numeric(sub[col], errors="coerce").dropna()
                    all_val = pd.to_numeric(df_pure[col], errors="coerce").dropna()
                    if len(ofc_val) > 5 and len(all_val) > 5:
                        gap = ofc_val.mean() - all_val.mean()
                        if gap < -3:
                            weak_items.append(f"{col}({gap:+.1f})")

                # 페르소나 태그
                tags = []
                if neg_ratio >= 10:
                    tags.append("🔴 부정VOC 다발")
                elif neg_ratio <= 4:
                    tags.append("🟢 안정 지사")
                if avg_score < 92:
                    tags.append("⚠️ 점수 하위권")
                elif avg_score >= 95:
                    tags.append("⭐ 우수 지사")

                persona_data.append({
                    "지사": ofc, "건수": n, "평균점수": round(avg_score, 1),
                    "부정비율": round(neg_ratio, 1), "계약종구성": contract_dist,
                    "주요업무": biz_dist, "부정원인": top_cause_str,
                    "부정키워드": top_neg_str,
                    "약점항목": ", ".join(weak_items) if weak_items else "-",
                    "태그": " ".join(tags) if tags else "보통",
                })

            df_persona = _sort_df_by_office(pd.DataFrame(persona_data), "지사")

            # 상위 3개 지사 처방전 카드
            top3 = df_persona.head(3)
            st.markdown("**🏥 CS 처방전 — 부정 VOC 상위 3개 지사 (OOS 제외 순수 부정만)**")
            p_cols = st.columns(3)
            for i, (_, row) in enumerate(top3.iterrows()):
                with p_cols[i]:
                    # 주요 원인에 따라 Do/Don't 자동 매칭
                    main_cause = row["부정원인"].split(",")[0].split("(")[0].strip() if row["부정원인"] != "없음" else ""
                    rx_map = {
                        "절차 복잡·불편":  {"do": "원스톱 처리 체계 구축 — 한 번에 끝내기", "dont": "고객을 여러 부서로 돌려보내기"},
                        "처리 지연·느림":  {"do": "접수 후 24시간 내 진행 상황 문자 발송", "dont": "처리 완료 전까지 연락 없이 방치"},
                        "처리 오류·부정확": {"do": "처리 전 체크리스트 확인 — 1회 완결 처리", "dont": "확인 없이 신속 처리에만 집중"},
                        "직원 태도·불친절": {"do": "경청→공감→해결 3단계 응대 훈련 실시", "dont": "감정 대응 없이 업무만 처리"},
                        "요금·제도 불만":  {"do": "요금 구조를 고객 눈높이 비유·시각자료로 설명", "dont": "매뉴얼 그대로 읽어주는 기계적 응대"},
                        "정전·안전 관련":  {"do": "예정 정전 72시간 전 사전 알림 체계 구축", "dont": "예고 없는 정전 및 복구 후 미통보"},
                    }
                    rx = rx_map.get(main_cause, {"do": "고객 맞춤 응대 프로세스 정비", "dont": "일괄적·기계적 대응"})
                    card_color = "card-red" if row["부정비율"] >= 10 else "card-gold"
                    weak_str = f'<br>📉 <b>약점항목:</b> {row["약점항목"]}' if row["약점항목"] != "-" else ""
                    st.markdown(
                        f'<div class="{card_color}" style="min-height:320px">'
                        f'<b>{row["지사"]}</b> {row["태그"]}<br>'
                        f'평균 {row["평균점수"]}점 · 부정비율 {row["부정비율"]}%<br><br>'
                        f'📊 <b>고객구성:</b> {row["계약종구성"]}<br>'
                        f'📋 <b>주요업무:</b> {row["주요업무"]}<br>'
                        f'🔥 <b>부정원인:</b> {row["부정원인"]}<br>'
                        f'🔑 <b>부정키워드:</b> {row["부정키워드"]}'
                        f'{weak_str}<br><br>'
                        f'<span style="color:green">✅ Do: {rx["do"]}</span><br>'
                        f'<span style="color:red">❌ Don\'t: {rx["dont"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True)

            # 전체 지사 프로파일 테이블
            st.markdown("**📋 전체 지사 프로파일 (OOS 제외)**")
            st.dataframe(
                df_persona[["지사", "건수", "평균점수", "부정비율", "계약종구성", "주요업무",
                            "부정원인", "부정키워드", "약점항목", "태그"]],
                use_container_width=True, height=500, hide_index=True)

            # 포지셔닝 맵
            fig_sc = px.scatter(df_persona, x="평균점수", y="부정비율", size="건수",
                                text="지사", template=PLOTLY_TPL,
                                color="부정비율",
                                color_continuous_scale=["#27AE60", "#F39C12", "#E74C3C"],
                                title="지사별 포지셔닝 맵 — 평균점수 vs 부정VOC 비율 (OOS 제외)")
            fig_sc.update_traces(textposition="top center", textfont_size=10,
                                  hovertemplate="%{text}<br>평균점수: %{x:.1f}점<br>부정비율: %{y:.1f}%<extra></extra>")
            fig_sc.update_layout(height=480, margin=dict(t=60, b=40, l=40, r=40),
                                  title_font=dict(size=14, color=C["navy"]),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("지사, VOC, 종합점수 컬럼이 모두 필요합니다.")

# ─────────────────────────────────────────────────────────────
#  TAB 11  지사 맞춤형 CS 솔루션 (Google Gemini AI 연동)
# ─────────────────────────────────────────────────────────────

# ── Gemini AI 일괄 인사이트 함수 (API 1회 호출) ─────────────

@st.cache_data(show_spinner=False)
def get_gemini_bulk_insight(items_json):
    """여러 지사·업무 데이터를 한 번에 보내 Gemini API 1회로 전체 인사이트를 도출"""
    items = json.loads(items_json)
    n = len(items)
    if n == 0:
        return []

    if not GEMINI_AVAILABLE:
        return [("AI 미연결", ".env 파일에 GEMINI_API_KEY를 설정해주세요.",
                 "https://aistudio.google.com 에서 무료 발급")] * n

    data_lines = []
    for i, it in enumerate(items, 1):
        data_lines.append(
            f"[항목{i}] 지사: {it['office']} | 취약업무: {it['biz']} | "
            f"계약종별: {it['ct']} | VOC: {it['voc']}")

    prompt = f"""당신은 전력산업에 20년 이상 종사한 최고 고객만족도(CS) 분석 전문가입니다.
아래 {n}개 항목 **각각**에 대해, 지사장이 즉시 액션할 수 있는 맞춤형 인사이트를 도출하세요.

[데이터]
{chr(10).join(data_lines)}

[분석 및 추론 규칙]
1. 단순 친절 교육, 매뉴얼 배포 같은 뻔하고 추상적인 액션 아이템은 절대 금지합니다.
2. 전력산업의 도메인 지식(계절적 요인, 산업 특성, 요금제 등)과 제공된 VOC를 유기적으로 연결하여 뾰족한 인사이트를 뽑아내세요.
3. VOC가 "없음"인 항목도 업무 특성과 계약종별을 근거로 추론하세요.
4. 결과는 반드시 아래 JSON **배열** 형식으로만 출력하세요. 항목 수는 정확히 {n}개. (마크다운·설명 등 다른 텍스트 절대 금지)

[{{"painpoint":"[상황/행위]+[결과/감정] 구체적 키워드","context":"원인 추론 2~3문장","action":"구체적 개선 조치 1문장"}}, ...]"""

    try:
        import urllib.request
        _models = ["gemini-2.0-flash", "gemma-3-12b-it", "gemma-3-27b-it"]
        _payload = {"contents": [{"parts": [{"text": prompt}]}],
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
            raise Exception("모든 AI 모델의 일일 한도가 소진되었습니다. 내일 다시 시도해주세요.")

        res_text = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
        if res_text.startswith("```json"):
            res_text = res_text[7:-3].strip()
        elif res_text.startswith("```"):
            res_text = res_text[3:-3].strip()

        parsed = json.loads(res_text)
        results = []
        for d in parsed:
            results.append((d.get("painpoint", "분석 오류"),
                            d.get("context", "분석 오류"),
                            d.get("action", "분석 오류")))
        # 항목 수 불일치 보정
        while len(results) < n:
            results.append(("분석 누락", "AI 응답에서 해당 항목이 누락되었습니다.", "재분석 필요"))
        return results[:n]

    except Exception as e:
        return [("AI 분석 일시 지연",
                 f"API 통신 중 문제 발생. ({str(e)})",
                 "잠시 후 다시 시도해주세요.")] * n


with tab11:
    st.markdown(
        '<p class="sec-head">🏢 지사 맞춤형 CS 솔루션 — AI 딥러닝 다차원 교차분석</p>',
        unsafe_allow_html=True)

    _need_office = M.get("office")
    _need_score  = M.get("score")
    _need_biz    = M.get("business")
    _has_contract = M.get("contract")
    _has_voc      = M.get("voc")

    if _need_office and _need_score and _need_biz:
        st.markdown(
            f'<div class="card-blue">'
            f'<b>🧠 AI 인사이트 도출 프로세스</b><br><br>'
            f'<b>1차</b> 지사별 취약 업무구분 추출 → '
            f'<b>2차</b> 취약 업무 내 타겟 계약종별 특정 → '
            f'<b>3차</b> 실제 고객 VOC를 Google AI(Gemini)로 실시간 전송<br>'
            f'구글 AI가 문맥을 분석하여 <b>기존에 없던 뾰족한 원인과 맞춤형 액션 아이템</b>을 자동 창작합니다.'
            f'</div>', unsafe_allow_html=True)

        st.markdown("")

        offices = _sort_offices(df_f[_need_office].dropna().unique().tolist())
        sel_mode = st.radio(
            "분석 대상", ["🔍 특정 지사 선택 (API 속도를 위해 권장)", "📋 전체 지사"],
            horizontal=True, key="tab11_sel_mode")

        if "특정" in sel_mode:
            sel_offices = st.multiselect("지사 선택", offices, default=offices[:1] if len(offices) >= 1 else offices,
                                         key="tab11_offices")
        else:
            sel_offices = None

        st.markdown("---")

        score_col = "_점수100" if "_점수100" in df_f.columns else _need_score
        grp = df_f.groupby([_need_office, _need_biz])[score_col].agg(["mean", "count"]).reset_index()
        grp.columns = ["지사", "업무구분", "평균점수", "건수"]
        grp = grp[grp["건수"] >= 3]

        office_avg = df_f.groupby(_need_office)[score_col].mean().reset_index()
        office_avg.columns = ["지사", "지사평균"]
        office_avg = _sort_df_by_office(office_avg, "지사")

        if sel_offices:
            target_offices = [o for o in office_avg["지사"] if o in sel_offices]
        else:
            target_offices = office_avg["지사"].tolist()

        if not target_offices:
            st.warning("선택된 지사가 없습니다.")
        else:
            # ── 1단계: pandas로 전체 데이터 수집 (API 호출 없음) ──
            all_items = []      # Gemini에 보낼 항목
            all_meta = []       # 화면 표시용 메타데이터

            for office in target_offices:
                oavg = office_avg[office_avg["지사"] == office]["지사평균"].values
                oavg_val = oavg[0] if len(oavg) > 0 else 0
                office_grp = grp[grp["지사"] == office].sort_values("평균점수").head(3)

                if office_grp.empty:
                    continue

                for _, biz_row in office_grp.iterrows():
                    biz_name = biz_row["업무구분"]
                    biz_avg  = biz_row["평균점수"]
                    biz_cnt  = int(biz_row["건수"])

                    target_ct = "-"
                    target_ct_avg = 0
                    if _has_contract:
                        ct_sub = df_f[(df_f[_need_office] == office) & (df_f[_need_biz] == biz_name)]
                        ct_grp = ct_sub.groupby(_has_contract)[score_col].agg(["mean", "count"]).reset_index()
                        ct_grp.columns = ["계약종별", "평균", "건수"]
                        ct_grp = ct_grp[ct_grp["건수"] >= 1].sort_values("평균")
                        if not ct_grp.empty:
                            target_ct = ct_grp.iloc[0]["계약종별"]
                            target_ct_avg = ct_grp.iloc[0]["평균"]

                    voc_str = "없음"
                    if _has_voc:
                        voc_filter = (df_f[_need_office] == office) & (df_f[_need_biz] == biz_name)
                        if _has_contract and target_ct != "-":
                            voc_sub = df_f[voc_filter & (df_f[_has_contract] == target_ct) & (df_f[score_col] <= 80)]
                        else:
                            voc_sub = df_f[voc_filter & (df_f[score_col] <= 80)]
                        if not voc_sub.empty:
                            raw = voc_sub[_has_voc].dropna().tolist()
                            meaningful = [str(t).strip() for t in raw if len(str(t).strip()) > 5]
                            if meaningful:
                                voc_str = " / ".join(meaningful[:10])

                    all_items.append({"office": office, "biz": biz_name,
                                       "ct": target_ct, "voc": voc_str})
                    all_meta.append({"office": office, "oavg": oavg_val,
                                      "biz_label": f"{biz_name} ({biz_avg:.1f}점, {biz_cnt}건)",
                                      "ct_label": f"{target_ct}" + (f" ({target_ct_avg:.0f}점)" if target_ct != "-" else "")})

            # ── 2단계: API 1회 호출 ──
            if all_items:
                st.markdown("💡 **과도한 API 호출 방지:** 데이터 세팅이 끝나면 아래 버튼을 눌러주세요.")

                if st.button("🚀 AI 지사 맞춤형 솔루션 분석 시작", type="primary"):
                    with st.spinner(f"✨ 구글 AI가 {len(all_items)}개 항목을 일괄 분석 중입니다... (API 1회 호출)"):
                        insights = get_gemini_bulk_insight(json.dumps(all_items, ensure_ascii=False))

                    # ── 3단계: 지사별로 결과 표시 ──
                    idx = 0
                    for office in target_offices:
                        office_meta = [m for m in all_meta if m["office"] == office]
                        if not office_meta:
                            st.info(f"{office}: 분석 가능한 업무구분 데이터가 부족합니다 (3건 이상 필요).")
                            st.markdown("---")
                            continue

                        st.markdown(f'## 🏢 {office} CS 심층 분석 및 맞춤형 솔루션')
                        st.markdown(f'**전체 평균: {office_meta[0]["oavg"]:.1f}점**')

                        result_rows = []
                        for m in office_meta:
                            painpoint, context, action = insights[idx]
                            idx += 1
                            result_rows.append({
                                "취약 업무구분": m["biz_label"],
                                "주요 타겟 (계약종별)": m["ct_label"],
                                "핵심 페인포인트": painpoint,
                                "원인 분석 및 추론": context,
                                "지사 맞춤형 Action Item": action,
                            })
                        df_result = pd.DataFrame(result_rows)
                        st.dataframe(df_result, use_container_width=True, hide_index=True)
                        st.markdown("---")
            else:
                st.info("분석 가능한 데이터가 없습니다.")
    else:
        missing = []
        if not _need_office: missing.append("지사")
        if not _need_score:  missing.append("종합점수")
        if not _need_biz:    missing.append("업무구분")
        st.info(f"이 분석을 수행하려면 다음 컬럼이 필요합니다: {', '.join(missing)}")

