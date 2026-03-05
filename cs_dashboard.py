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
import io, re, os

# ── 선택적 라이브러리 ──────────────────────────────────────────
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
except Exception:
    KONLPY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except Exception:
    KEYBERT_AVAILABLE = False

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
PLOTLY_TPL    = "plotly_white"

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


def _has_positive_context(text, keyword):
    """키워드 주변(앞뒤 15자)에 긍정 문맥 단어가 있는지 확인"""
    s = str(text)
    idx = s.find(keyword)
    if idx == -1:
        return False
    # 키워드 주변 윈도우 추출 (앞 15자 ~ 뒤 15자)
    start = max(0, idx - 15)
    end = min(len(s), idx + len(keyword) + 15)
    window = s[start:end]
    return any(pw in window for pw in POSITIVE_CONTEXT)


def check_negative(text):
    if not text or str(text).strip() in ("", "nan", "응답없음"):
        return False, []
    s = str(text)
    found = [kw for kw in NEGATIVE_KEYWORDS if kw in s]
    # 긍정 문맥에 둘러싸인 키워드는 제거
    real_neg = [kw for kw in found if not _has_positive_context(s, kw)]
    return bool(real_neg), real_neg


def check_rude(text):
    if not text or str(text).strip() in ("", "nan", "응답없음"):
        return False, []
    s = str(text)
    found = [kw for kw in RUDE_KEYWORDS if kw in s]
    # 긍정 문맥에 둘러싸인 키워드는 제거
    real_rude = [kw for kw in found if not _has_positive_context(s, kw)]
    return bool(real_rude), real_rude


def classify_voc_3tier(text):
    """VOC를 [불친절 / 불만 / 긍정]으로 3단 분류 (문맥 인식 버전)"""
    if not text or str(text).strip() in ("", "nan", "응답없음"):
        return "긍정"
    s = str(text)
    is_rude, _ = check_rude(s)
    if is_rude:
        return "불친절"
    is_neg, _ = check_negative(s)
    if is_neg:
        return "불만"
    return "긍정"


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


def make_wordcloud_image(kw_list, max_words=15):
    if not WORDCLOUD_AVAILABLE or not kw_list:
        return None
    freq = {k: v for k, v in kw_list[:max_words]}
    kwargs = dict(
        width=1200, height=500, background_color="white",
        max_words=max_words, colormap="Blues", prefer_horizontal=0.8,
        min_font_size=18, max_font_size=120,
        relative_scaling=0.6, margin=15,
    )
    if FONT_PATH:
        kwargs["font_path"] = FONT_PATH
    try:
        wc = WordCloud(**kwargs)
        wc.generate_from_frequencies(freq)
        return wc.to_image()
    except Exception:
        return None


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
     "💳 요금·청구 → 청구서 사전 안내 강화, 비용 상담 전담 채널 확충"),
    (["설치","공사","계량","계기","시공"],
     "🔧 설치·공사 → 시공 일정 사전 통보, AS 프로세스 개선"),
    (["정전","단전","고장","누전","장애"],
     "⚡ 정전·장애 → 긴급복구 안내 강화, 사전 예방 점검 확대"),
    (["직원","친절","응대","상담","담당자"],
     "👤 직원 응대 → 고객응대 매뉴얼 재정비, CS 친절 교육 강화"),
    (["앱","홈페이지","사이트","온라인","인터넷","모바일"],
     "📱 디지털 채널 → 앱·웹 UX 개선, 디지털 이용 가이드 제공"),
    (["대기","기다림","오래","느림","지연"],
     "⏱ 대기·지연 → 처리 속도 개선, 처리 현황 실시간 안내"),
    (["안전","위험","누전","화재"],
     "🦺 안전 → 즉시 현장 대응, 안전 점검 프로세스 강화"),
    (["불친절","무시","무성의","태도","반말"],
     "😡 불친절 → 직원 CS 교육 즉시 강화, 불친절 재발 방지 모니터링"),
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
        f"{'✅ WordCloud' if WORDCLOUD_AVAILABLE else '🟡 미설치'}  \n"
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

with st.spinner("데이터를 불러오는 중…"):
    df_raw, orig_len = load_data(uploaded_file.read())

# ══════════════════════════════════════════════════════════════
#  9. 컬럼 자동 매핑 (엑셀 컬럼명 기반)
# ══════════════════════════════════════════════════════════════
# 순번 컬럼 제거
if "순번" in df_raw.columns:
    df_raw.drop(columns=["순번"], inplace=True)

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
        office_opts = sorted(df_raw[M["office"]].dropna().astype(str).unique().tolist())
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
tab1, tab2, tab3, tab4, tab5, tab6, tab8, tab9 = st.tabs([
    "📊  구간별 비중 · 종합 현황",
    "🏢  사업소별 벤치마킹",
    "📡  채널별 · 업무별 분석",
    "☁️  VOC AI 분석",
    "🎯  CS 인사이트 & 사전케어",
    "📈  다차원 교차분석",
    "📅  시계열 트렌드",
    "🔬  지사 심층 · 패턴",
])

# ─────────────────────────────────────────────────────────────
#  TAB 1  구간별 비중 · 종합 현황
# ─────────────────────────────────────────────────────────────
with tab1:
    if M["score"] and avg_score_100 is not None and not np.isnan(avg_score_100):
        # ── 게이지 + 히스토그램 ──
        g_col, h_col = st.columns([1, 2])
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
        with h_col:
            st.markdown('<p class="sec-head">📊 만족도 점수 분포 (100점 환산)</p>', unsafe_allow_html=True)
            fig_hist = px.histogram(score_100.dropna(), nbins=20, color_discrete_sequence=[C["sky"]],
                                    labels={"value": "만족도 점수", "count": "응답 수"}, template=PLOTLY_TPL)
            fig_hist.add_vline(x=avg_score_100, line_color=C["gold"], line_width=2.5, line_dash="dash",
                               annotation_text=f"평균 {avg_score_100:.1f}", annotation_font_color=C["gold"])
            fig_hist.update_layout(height=300, margin=dict(t=30, b=30, l=50, r=20), showlegend=False,
                                    xaxis_title="만족도 점수", yaxis_title="응답 수")
            fig_hist.update_traces(marker_line_width=0)
            st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("---")

        # ── 구간별 비중 통계 (리포트 핵심) ──
        st.markdown('<p class="sec-head">📊 만족도 점수 구간별 비중 통계</p>', unsafe_allow_html=True)
        bucket_cnt = df_f["_점수구간"].value_counts()
        bucket_data = pd.DataFrame({
            "구간": BUCKET_ORDER,
            "건수": [bucket_cnt.get(b, 0) for b in BUCKET_ORDER],
        })
        bucket_data["비율(%)"] = (bucket_data["건수"] / max(bucket_data["건수"].sum(), 1) * 100).round(1)

        bp_col, bt_col = st.columns([1, 1])
        with bp_col:
            fig_bp = px.pie(bucket_data, names="구간", values="건수", color="구간",
                            color_discrete_map=BUCKET_COLORS, hole=0.45, template=PLOTLY_TPL,
                            title="점수 구간별 비율")
            fig_bp.update_traces(textposition="outside", textinfo="percent+label", textfont_size=13,
                                  marker=dict(line=dict(color="#ffffff", width=2)))
            fig_bp.update_layout(height=380, margin=dict(t=50, b=20, l=20, r=20), showlegend=True,
                                  title_font=dict(size=15, color=C["navy"]))
            st.plotly_chart(fig_bp, use_container_width=True)
        with bt_col:
            st.markdown("<br>", unsafe_allow_html=True)
            # 구간별 색상 배지
            for b in BUCKET_ORDER:
                cnt = bucket_cnt.get(b, 0)
                pct = cnt / max(len(df_f), 1) * 100
                color = BUCKET_COLORS.get(b, C["gray"])
                st.markdown(
                    f'<div style="display:flex; align-items:center; margin:0.4rem 0;">'
                    f'<div style="background:{color}; color:white; padding:0.5rem 1rem; border-radius:10px; '
                    f'font-weight:700; min-width:120px; text-align:center;">{b}</div>'
                    f'<div style="margin-left:1rem; font-size:1.1rem; font-weight:700; color:{C["navy"]};">'
                    f'{cnt:,}건 ({pct:.1f}%)</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(bucket_data, use_container_width=True, hide_index=True)

        # ── 사업소별 구간 분포 (리포트 양식) ──
        if M["office"]:
            st.markdown("---")
            st.markdown('<p class="sec-head">📊 사업소별 점수 구간 분포</p>', unsafe_allow_html=True)
            cross_bucket = pd.crosstab(df_f[M["office"]], df_f["_점수구간"])
            for b in BUCKET_ORDER:
                if b not in cross_bucket.columns:
                    cross_bucket[b] = 0
            cross_bucket = cross_bucket[BUCKET_ORDER]
            cross_melt = cross_bucket.reset_index().melt(id_vars=M["office"], var_name="점수구간", value_name="건수")
            fig_stack = px.bar(cross_melt, x=M["office"], y="건수", color="점수구간",
                               color_discrete_map=BUCKET_COLORS, barmode="stack", template=PLOTLY_TPL,
                               title="사업소별 점수 구간 분포")
            fig_stack.update_layout(height=400, margin=dict(t=50, b=80, l=60, r=20), xaxis_tickangle=-25,
                                     title_font=dict(size=14, color=C["navy"]))
            st.plotly_chart(fig_stack, use_container_width=True)

            with st.expander("📋 사업소별 구간 집계표"):
                cross_display = cross_bucket.copy()
                cross_display["합계"] = cross_display.sum(axis=1)
                st.dataframe(cross_display, use_container_width=True)

        st.markdown("---")

    # ── 분포 파이 차트 ──
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
            fig_pie = px.pie(names=counts.index, values=counts.values, color_discrete_sequence=PIE_COLORS,
                             hole=0.42, title=f"{titles_map.get(col_nm, col_nm)} 분포", template=PLOTLY_TPL)
            fig_pie.update_traces(textposition="outside", textinfo="percent+label", textfont_size=12,
                                   marker=dict(line=dict(color="#ffffff", width=2)))
            fig_pie.update_layout(height=360, margin=dict(t=50, b=20, l=20, r=20), showlegend=False,
                                   title_font=dict(size=15, color=C["navy"]))
            with pc_list[idx]:
                st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("📄 원본 데이터 미리보기 (상위 30건)"):
        display_df = df_f[[c for c in df_f.columns if not c.startswith("_")]]
        st.dataframe(display_df.head(30), use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  TAB 2  사업소별 벤치마킹
# ─────────────────────────────────────────────────────────────
with tab2:
    if not M["office"]:
        st.warning("사업소 컬럼을 사이드바에서 먼저 선택해주세요.")
    elif not M["score"]:
        st.warning("만족도 점수 컬럼을 선택해야 벤치마킹이 가능합니다.")
    else:
        st.markdown('<p class="sec-head">🏢 사업소별 만족도 벤치마킹</p>', unsafe_allow_html=True)
        office_grp = df_f.groupby(M["office"])["_점수100"].agg(["mean","count"]).reset_index()
        office_grp.columns = ["사업소", "평균만족도", "응답수"]
        office_grp = office_grp.sort_values("평균만족도", ascending=True)
        overall_avg = avg_score_100

        office_grp["그룹"] = office_grp["평균만족도"].apply(
            lambda v: "⬆ 본부평균 이상" if v >= overall_avg else "⬇ 본부평균 미달")
        color_map = {"⬆ 본부평균 이상": C["teal"], "⬇ 본부평균 미달": C["red"]}

        fig_bench = px.bar(office_grp, x="평균만족도", y="사업소", color="그룹",
                           color_discrete_map=color_map, orientation="h", text="평균만족도",
                           template=PLOTLY_TPL, title=f"사업소별 평균 만족도 — 본부 평균: {overall_avg:.1f}점")
        fig_bench.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bench.add_vline(x=overall_avg, line_color=C["navy"], line_dash="dash", line_width=2.5,
                            annotation_text=f"본부 평균 {overall_avg:.1f}",
                            annotation_font_size=12, annotation_font_color=C["navy"])
        fig_bench.update_layout(height=max(350, len(office_grp) * 35 + 80),
                                 margin=dict(t=60, b=20, l=10, r=100), legend_title_text="",
                                 title_font=dict(size=14, color=C["navy"]))
        st.plotly_chart(fig_bench, use_container_width=True)

        with st.expander("📋 사업소별 상세 데이터"):
            detail = office_grp.sort_values("평균만족도", ascending=False).copy()
            detail["평균만족도"] = detail["평균만족도"].round(1)
            detail["본부평균대비"] = (detail["평균만족도"] - overall_avg).round(1)
            st.dataframe(detail[["사업소","평균만족도","응답수","본부평균대비","그룹"]],
                         use_container_width=True, hide_index=True)


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
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
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
        if not pivot.empty:
            fig_hm = px.imshow(pivot, color_continuous_scale="RdYlGn",
                               text_auto=".1f", aspect="auto", template=PLOTLY_TPL,
                               title=f"지사 × {cat_label} 만족도 (초록=높음 / 빨강=낮음)")
            fig_hm.update_layout(
                height=max(350, len(pivot.index) * 30 + 100),
                margin=dict(t=60, b=60, l=120, r=60),
                title_font=dict(size=14, color=C["navy"]))
            st.plotly_chart(fig_hm, use_container_width=True)

        with st.expander(f"📋 지사별 {cat_label} 상세 데이터"):
            st.dataframe(pivot, use_container_width=True)

        # ── 저점수 건 상세 조회 ──
        _uid = cat_label.replace(" ", "")
        with st.expander(f"🔍 저점수 건 상세 조회 (지사 × {cat_label})"):
            _sel_cols = st.columns(2)
            with _sel_cols[0]:
                _offices = sorted(df[office_col].dropna().astype(str).unique().tolist())
                _sel_ofc = st.selectbox("지사 선택", _offices, key=f"lowdet_ofc_{_uid}")
            with _sel_cols[1]:
                _cats = sorted(df[cat_col].dropna().astype(str).unique().tolist())
                _sel_cat = st.selectbox(f"{cat_label} 선택", _cats, key=f"lowdet_cat_{_uid}")
            _filtered = df[(df[office_col].astype(str) == _sel_ofc) &
                           (df[cat_col].astype(str) == _sel_cat)].copy()
            if len(_filtered) > 0:
                _filtered = _filtered.sort_values(score_col)
                _avg = _filtered[score_col].mean()
                _color = C["red"] if _avg < overall_avg - 5 else C["gold"] if _avg < overall_avg else C["teal"]
                st.markdown(
                    f'평균 <b style="color:{_color}">{_avg:.1f}점</b> · {len(_filtered):,}건'
                    f' (전체 대비 {_avg - overall_avg:+.1f}점)',
                    unsafe_allow_html=True)
                _show_cols = []
                if M.get("receipt_no") and M["receipt_no"] in _filtered.columns:
                    _show_cols.append(M["receipt_no"])
                _show_cols.append(office_col)
                _show_cols.append(cat_col)
                if score_col in _filtered.columns:
                    _show_cols.append(score_col)
                if M.get("voc") and M["voc"] in _filtered.columns:
                    _show_cols.append(M["voc"])
                _show_cols = [c for c in _show_cols if c in _filtered.columns]
                st.dataframe(_filtered[_show_cols].reset_index(drop=True),
                             use_container_width=True, height=300, hide_index=True)
            else:
                st.info("해당 조건에 맞는 데이터가 없습니다.")


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


# ─────────────────────────────────────────────────────────────
#  TAB 4  VOC AI 분석
# ─────────────────────────────────────────────────────────────
with tab4:
    if not M["voc"]:
        st.warning("주관식 답변(VOC) 컬럼을 사이드바에서 먼저 선택해주세요.")
    else:
        voc_raw = df_f[M["voc"]].astype(str).str.strip()
        voc_list = [t for t in voc_raw.tolist() if t and t != "nan" and t != "응답없음"]
        n_voc = len(voc_list)
        _analysis_mode = ("KeyBERT + KoNLPy" if (KEYBERT_AVAILABLE and KONLPY_AVAILABLE)
                          else "KoNLPy 고정밀" if KONLPY_AVAILABLE else "정규식 기본")

        # ── VOC 3단 분류 (리포트 양식: 긍정/불만/불친절) ──
        st.markdown(f'<p class="sec-head">🔬 VOC 3단 분류 — {_analysis_mode} ({n_voc:,}건)</p>',
                    unsafe_allow_html=True)

        if "_VOC분류" in df_f.columns:
            voc_cls = df_f["_VOC분류"].value_counts()
            vc_pos = voc_cls.get("긍정", 0)
            vc_neg = voc_cls.get("불만", 0)
            vc_rude = voc_cls.get("불친절", 0)

            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                st.metric("😊 긍정", f"{vc_pos:,}건", delta=f"{vc_pos / max(n_voc, 1) * 100:.1f}%")
            with vc2:
                st.metric("😠 불만", f"{vc_neg:,}건",
                          delta=f"{vc_neg / max(n_voc, 1) * 100:.1f}%", delta_color="inverse")
            with vc3:
                st.metric("😡 불친절", f"{vc_rude:,}건",
                          delta=f"{vc_rude / max(n_voc, 1) * 100:.1f}%", delta_color="inverse")

            fig_3t = px.pie(names=["긍정", "불만", "불친절"], values=[vc_pos, vc_neg, vc_rude],
                            color_discrete_sequence=[C["teal"], C["gold"], C["red"]],
                            hole=0.5, template=PLOTLY_TPL, title="VOC 3단 분류 비율")
            fig_3t.update_traces(textinfo="percent+label", textfont_size=14,
                                  marker=dict(line=dict(color="white", width=3)))
            fig_3t.update_layout(height=300, margin=dict(t=50, b=10, l=20, r=20),
                                  title_font=dict(size=14, color=C["navy"]))
            st.plotly_chart(fig_3t, use_container_width=True)

            # 사업소별 VOC 3단 분류 (리포트 양식)
            if M["office"]:
                st.markdown("---")
                st.markdown('<p class="sec-head">🏢 사업소별 VOC 3단 분류 현황</p>', unsafe_allow_html=True)
                voc_cross = pd.crosstab(df_f[M["office"]], df_f["_VOC분류"])
                for col in ["긍정", "불만", "불친절"]:
                    if col not in voc_cross.columns:
                        voc_cross[col] = 0
                voc_cross = voc_cross[["긍정", "불만", "불친절"]]
                voc_cross["합계"] = voc_cross.sum(axis=1)
                voc_cross["긍정(%)"] = (voc_cross["긍정"] / voc_cross["합계"] * 100).round(1)
                voc_cross["불만(%)"] = (voc_cross["불만"] / voc_cross["합계"] * 100).round(1)
                voc_cross["불친절(%)"] = (voc_cross["불친절"] / voc_cross["합계"] * 100).round(1)
                st.dataframe(voc_cross, use_container_width=True)

            st.markdown("---")

            # ── 불친절 사례 경고 카드 ──
            if vc_rude > 0:
                st.markdown('<p class="sec-head">😡 불친절 사례 리스트</p>', unsafe_allow_html=True)
                df_rude = df_f[df_f["_VOC분류"] == "불친절"].copy()
                for _, row in df_rude.head(10).iterrows():
                    info_parts = []
                    for key in ["office", "channel", "business"]:
                        if M.get(key) and M[key] in row.index:
                            val = str(row[M[key]])
                            if val and val != "nan":
                                info_parts.append(val)
                    info_str = " · ".join(info_parts) if info_parts else ""
                    voc_text = str(row[M["voc"]]) if M["voc"] in row.index else ""
                    for rk in RUDE_KEYWORDS:
                        if rk in voc_text:
                            voc_text = voc_text.replace(
                                rk, f'<b style="color:#c62828; background:#fce4ec; padding:0 3px; border-radius:3px;">{rk}</b>')
                    cust_id = ""
                    if M.get("id") and M["id"] in row.index:
                        cust_id = f' | 고객번호: {row[M["id"]]}'
                    st.markdown(
                        f'<div class="card-rude"><b>😡 불친절 감지{cust_id}</b><br>'
                        f'<small style="color:#888;">{info_str}</small><br><br>{voc_text}</div>',
                        unsafe_allow_html=True)
                if len(df_rude) > 10:
                    st.caption(f"※ 상위 10건만 표시. 전체 {len(df_rude):,}건")
                st.markdown("---")

        # ── 워드클라우드 & 키워드 ──
        if n_voc > 0:
            with st.spinner("AI가 키워드를 추출 중입니다…"):
                all_kws = extract_keywords(voc_list, top_n=60)
            if all_kws:
                st.markdown('<p class="sec-head">☁️ VOC 키워드 분석</p>', unsafe_allow_html=True)
                wc_col, kw_col = st.columns([3, 2])
                with wc_col:
                    st.markdown('<div class="card-blue">', unsafe_allow_html=True)
                    st.markdown("**☁️ 전체 VOC 워드클라우드**")
                    if WORDCLOUD_AVAILABLE:
                        img = make_wordcloud_image(all_kws)
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.error("워드클라우드 생성 실패")
                    else:
                        st.warning("`pip install wordcloud` 후 재시작")
                    st.markdown('</div>', unsafe_allow_html=True)
                with kw_col:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("**🔑 키워드 빈도 Top 10**")
                    kw_df = pd.DataFrame(all_kws[:10], columns=["키워드", "언급 횟수"])
                    kw_df["비율(%)"] = (kw_df["언급 횟수"] / kw_df["언급 횟수"].sum() * 100).round(1)
                    kw_df["유형"] = kw_df["키워드"].apply(
                        lambda x: "😡 불친절" if any(r in x for r in RUDE_KEYWORDS)
                        else "⚠️ 부정" if any(n in x for n in NEGATIVE_KEYWORDS) else "✅ 일반")
                    st.dataframe(kw_df, use_container_width=True, height=440, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")

                # 키워드 막대 차트
                st.markdown('<p class="sec-head">📊 상위 10개 키워드</p>', unsafe_allow_html=True)
                top30 = all_kws[:10]
                kw_names = [k[0] for k in top30]
                kw_vals = [k[1] for k in top30]
                kw_types = []
                for kw in kw_names:
                    if any(r in kw for r in RUDE_KEYWORDS):
                        kw_types.append("불친절")
                    elif any(n in kw for n in NEGATIVE_KEYWORDS):
                        kw_types.append("부정")
                    else:
                        kw_types.append("일반")
                kw_chart_df = pd.DataFrame({"키워드": kw_names, "언급 횟수": kw_vals, "유형": kw_types})
                fig_kw = px.bar(kw_chart_df, x="키워드", y="언급 횟수", color="유형",
                                color_discrete_map={"불친절": "#e91e63", "부정": C["red"], "일반": C["sky"]},
                                text="언급 횟수", template=PLOTLY_TPL, title="상위 10 키워드 · 빨간=부정 · 핑크=불친절")
                fig_kw.update_traces(texttemplate="%{text}", textposition="outside")
                fig_kw.update_layout(height=400, margin=dict(t=50, b=90, l=60, r=20), xaxis_tickangle=-35,
                                      legend_title_text="", title_font=dict(size=14, color=C["navy"]))
                st.plotly_chart(fig_kw, use_container_width=True)

                # 업무별 VOC
                if M["business"]:
                    st.markdown("---")
                    st.markdown('<p class="sec-head">🏢 업무별 VOC 키워드</p>', unsafe_allow_html=True)
                    biz_sel = st.selectbox("분석할 업무 선택",
                                           df_f[M["business"]].dropna().astype(str).unique(), key="voc_biz_sel")
                    biz_voc = [t for t in df_f.loc[df_f[M["business"]].astype(str) == biz_sel, M["voc"]
                               ].astype(str).str.strip().tolist() if t and t != "nan" and t != "응답없음"]
                    if biz_voc:
                        with st.spinner(f"[{biz_sel}] 분석 중…"):
                            biz_kws = extract_keywords(biz_voc, top_n=30)
                        bwc_c, bkw_c = st.columns([3, 2])
                        with bwc_c:
                            st.markdown(f'<div class="card-blue"><b>[{biz_sel}] 워드클라우드</b>', unsafe_allow_html=True)
                            if WORDCLOUD_AVAILABLE and biz_kws:
                                bimg = make_wordcloud_image(biz_kws)
                                if bimg:
                                    st.image(bimg, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with bkw_c:
                            st.markdown(f'<div class="card"><b>[{biz_sel}] Top 10</b>', unsafe_allow_html=True)
                            if biz_kws:
                                bkdf = pd.DataFrame(biz_kws[:10], columns=["키워드","언급 횟수"])
                                st.dataframe(bkdf, use_container_width=True, height=320, hide_index=True)
                            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("VOC 텍스트가 없습니다.")


# ─────────────────────────────────────────────────────────────
#  TAB 5  CS 인사이트 & 사전케어
# ─────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<p class="sec-head">💡 AI CS 활동 방향 인사이트 & 개선대책</p>', unsafe_allow_html=True)

    if not M["voc"]:
        st.warning("VOC 컬럼을 선택해야 인사이트를 생성할 수 있습니다.")
    else:
        # 부진 하위 3개 업무 → AI 개선대책
        if M["business"] and M["score"]:
            biz_grp_ins = df_f.groupby(M["business"])["_점수100"].mean().sort_values()
            bottom3_biz = biz_grp_ins.head(3).index.tolist()
            st.markdown(
                '<div class="card-red"><b>🎯 AI 자동 개선대책 — 부진 하위 3개 업무</b><br>'
                '만족도 하위 업무의 VOC 키워드를 분석하여 AI가 자동 생성한 개선대책입니다.</div>',
                unsafe_allow_html=True)

            for biz_name in bottom3_biz:
                texts_b = [t for t in df_f.loc[df_f[M["business"]].astype(str) == str(biz_name), M["voc"]
                           ].astype(str).str.strip().tolist() if t and t != "nan" and t != "응답없음"]
                if not texts_b:
                    continue
                kws_b = extract_keywords(texts_b, top_n=15)
                biz_score = biz_grp_ins.get(biz_name, 0)
                recs = generate_ai_recommendations(kws_b, str(biz_name))
                kw_str = " ".join([f"`{k}`" for k, _ in kws_b[:7]])
                st.markdown(
                    f'<div class="card-gold"><b>🔴 [{biz_name}] — 만족도 {biz_score:.1f}점</b><br>'
                    f'키워드: {kw_str}<br><br>' + "<br>".join(recs) + '</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── 취약그룹 타겟 분석 ──
    if M["score"] and available_cats:
        st.markdown('<p class="sec-head">🎯 취약그룹 타겟 VOC 분석</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-gold"><b>📌 분석 방법</b> — 범주별 평균 점수 하위 30% 그룹만 선별하여 '
            '해당 그룹의 VOC 텍스트를 집중 분석합니다. 전체 데이터가 아닌 취약그룹에 대해서만 '
            '키워드를 추출하므로, 실질적인 개선 포인트를 정확히 파악할 수 있습니다.</div>',
            unsafe_allow_html=True)

        vg_cat = st.selectbox("취약그룹 분석 기준", available_cats, key="vg_cat_sel")
        weak_groups, threshold, all_grp = find_vulnerable_group(df_f, vg_cat, "_점수100", percentile=30)

        if len(weak_groups) > 0:
            vg1, vg2 = st.columns([1, 1])
            with vg1:
                st.markdown(f"**하위 30% 기준선:** {threshold:.1f}점")
                st.dataframe(weak_groups, use_container_width=True, hide_index=True)
            with vg2:
                fig_vg = px.bar(
                    all_grp.sort_values("평균"), x="평균", y=vg_cat, orientation="h",
                    color=all_grp["평균"].apply(lambda v: "🔴 취약" if v <= threshold else "일반"),
                    color_discrete_map={"🔴 취약": C["red"], "일반": C["sky"]},
                    template=PLOTLY_TPL, text=all_grp.sort_values("평균")["평균"].round(1),
                    title=f"{vg_cat}별 평균 만족도 (빨간색 = 취약그룹)",
                )
                fig_vg.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig_vg.update_layout(
                    height=max(300, len(all_grp) * 30 + 80),
                    margin=dict(t=60, b=20, l=10, r=80),
                    showlegend=False, title_font=dict(size=14, color=C["navy"]),
                )
                st.plotly_chart(fig_vg, use_container_width=True)

            # 취약그룹 VOC 집중 분석
            if M["voc"]:
                weak_names = weak_groups[vg_cat].tolist()
                df_weak = df_f[df_f[vg_cat].astype(str).isin([str(n) for n in weak_names])]
                weak_voc = [t for t in df_weak[M["voc"]].astype(str).str.strip().tolist()
                            if t and t != "nan" and t != "응답없음"]
                if weak_voc:
                    with st.spinner("취약그룹 VOC 키워드 분석 중…"):
                        weak_kws = extract_keywords(weak_voc, top_n=30)
                    if weak_kws:
                        st.markdown(
                            f'<p class="sec-head">🔍 취약그룹({", ".join(str(n) for n in weak_names)}) VOC 키워드</p>',
                            unsafe_allow_html=True)
                        wk_l, wk_r = st.columns([3, 2])
                        with wk_l:
                            if WORDCLOUD_AVAILABLE:
                                wk_img = make_wordcloud_image(weak_kws)
                                if wk_img:
                                    st.image(wk_img, use_container_width=True)
                        with wk_r:
                            wk_df = pd.DataFrame(weak_kws[:10], columns=["키워드", "언급 횟수"])
                            wk_df["유형"] = wk_df["키워드"].apply(
                                lambda x: "😡 불친절" if any(r in x for r in RUDE_KEYWORDS)
                                else "⚠️ 부정" if any(n in x for n in NEGATIVE_KEYWORDS) else "일반")
                            st.dataframe(wk_df, use_container_width=True, height=380, hide_index=True)

                        # 취약그룹 AI 개선대책
                        recs_vg = generate_ai_recommendations(weak_kws, f"취약그룹({', '.join(str(n) for n in weak_names)})")
                        if recs_vg:
                            st.markdown(
                                '<div class="card-red"><b>💡 취약그룹 AI 개선대책</b><br><br>'
                                + "<br>".join(recs_vg)
                                + '</div>', unsafe_allow_html=True)
                else:
                    st.info("취약그룹에 해당하는 VOC 텍스트가 없습니다.")
        else:
            st.success("하위 30% 기준에 해당하는 취약그룹이 없습니다. 전체적으로 양호합니다.")

        st.markdown("---")

    # ── 잠재 민원고객 사전케어 ──
    st.markdown('<p class="sec-head">🚨 잠재적 민원고객 사전케어 리스트 (AI 자동 추출)</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-red"><b>📌 추출 기준</b><br>'
        '① 종합 점수 하위 30% 고객<br>'
        '② 점수와 무관하게 서술 의견에 부정적 키워드가 포함된 고객<br>'
        '두 조건 중 하나라도 해당되면 잠재 민원고객으로 추출합니다.<br>'
        '해당 고객에게 <b>72시간 이내</b> 선제적으로 연락하여 민원 발생을 사전에 차단하세요.</div>',
        unsafe_allow_html=True)

    if not M["voc"]:
        st.warning("VOC 컬럼을 선택해야 리스트를 추출할 수 있습니다.")
    else:
        with st.spinner("잠재 민원고객 추출 중…"):
            neg_res = df_f[M["voc"]].apply(check_negative)
            neg_kw_s = neg_res.apply(lambda x: ", ".join(x[1]) if x[1] else "")
            # 조건1: 하위 30% 점수
            low_score_mask = pd.Series(False, index=df_f.index)
            if M["score"] and "_점수100" in df_f.columns:
                score_threshold = np.percentile(df_f["_점수100"].dropna(), 30)
                low_score_mask = df_f["_점수100"] <= score_threshold
            # 조건2: 부정 키워드 감지 (점수 무관)
            neg_kw_mask = neg_res.apply(lambda x: x[0])
            # 합집합
            neg_mask = low_score_mask | neg_kw_mask

        df_neg = df_f[neg_mask].copy()
        df_neg["감지된_부정키워드"] = neg_kw_s[neg_mask].values
        # 추출 유형 표시
        _reasons = []
        for idx in df_neg.index:
            r = []
            if low_score_mask.get(idx, False):
                r.append("하위점수")
            if neg_kw_mask.get(idx, False):
                r.append("부정키워드")
            _reasons.append(" + ".join(r) if r else "")
        df_neg["추출유형"] = _reasons
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
                    fig_neg.update_traces(texttemplate="%{text}", textposition="outside")
                    fig_neg.update_layout(height=340, margin=dict(t=50, b=70, l=60, r=20), xaxis_tickangle=-25,
                                           title_font=dict(size=14, color=C["navy"]))
                    st.plotly_chart(fig_neg, use_container_width=True)
                with nk_r:
                    fig_don = px.pie(nkw_df.head(10), names="부정키워드", values="감지횟수", hole=0.5,
                                     color_discrete_sequence=px.colors.sequential.Reds[::-1],
                                     title="부정 키워드 비중", template=PLOTLY_TPL)
                    fig_don.update_traces(textinfo="percent+label", textfont_size=11,
                                           marker=dict(line=dict(color="white", width=2)))
                    fig_don.update_layout(height=340, margin=dict(t=50, b=20, l=20, r=20), showlegend=False,
                                           title_font=dict(size=14, color=C["navy"]))
                    st.plotly_chart(fig_don, use_container_width=True)

            # 리스트 테이블
            display_cols = []
            for key in ["id","name","contact","age","office","channel","contract","business","score","voc"]:
                if M[key] and M[key] in df_neg.columns:
                    display_cols.append(M[key])
            display_cols.extend(["감지된_부정키워드", "추출유형"])
            df_disp = df_neg[[c for c in display_cols if c in df_neg.columns]].reset_index(drop=True)

            st.markdown(f'<p class="sec-head">📋 잠재 민원고객 — 총 <span style="color:{C["red"]}">{neg_n:,}명</span></p>',
                        unsafe_allow_html=True)
            st.dataframe(df_disp, use_container_width=True, height=440, hide_index=True)

            excel_bytes = df_to_excel_bytes(df_disp)
            st.download_button(label="📥  잠재 민원고객 엑셀 다운로드", data=excel_bytes,
                               file_name="잠재민원고객_사전케어리스트.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

            st.markdown("""<div class="card-red"><b>⚠️ 사전케어 행동 가이드</b><br><br>
✅ <b>72시간 이내</b> — 담당자가 직접 전화·문자로 능동 연락<br>
✅ <b>경청 우선</b> — 고객 불만을 끝까지 듣고 공감 후 해결책 제시<br>
✅ <b>CRM 기록 필수</b> — 접촉 일시, 처리 내용, 결과를 반드시 기록<br>
✅ <b>패턴 분석</b> — 동일 유형 불만이 반복되면 프로세스 자체를 개선</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="card-teal">🎉 <b>잠재 민원고객이 없습니다!</b><br>
현재 고객 만족 수준이 양호합니다.</div>""", unsafe_allow_html=True)

    # ── 긍정/부정 비율 ──
    if M["voc"] and voc_texts_all:
        st.markdown("---")
        st.markdown('<p class="sec-head">📊 VOC 긍정 / 부정 비율</p>', unsafe_allow_html=True)
        ratio_l, ratio_r = st.columns([1, 2])
        with ratio_l:
            fig_ratio = go.Figure(go.Indicator(
                mode="gauge+number", value=round(100 - neg_ratio, 1),
                number={"suffix": "%", "font": {"size": 42, "color": C["teal"]}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": C["teal"], "thickness": 0.25},
                       "steps": [{"range": [0, 40], "color": "#ffcdd2"}, {"range": [40, 70], "color": "#fff9c4"},
                                 {"range": [70, 100], "color": "#c8e6c9"}]},
                title={"text": "긍정 VOC 비율", "font": {"size": 15, "color": C["navy"]}}))
            fig_ratio.update_layout(height=260, margin=dict(t=60, b=20, l=30, r=30), paper_bgcolor="white")
            st.plotly_chart(fig_ratio, use_container_width=True)
        with ratio_r:
            fig_pn = px.pie(names=["긍정 VOC", "부정 VOC(잠재 민원)"], values=[pos_cnt, neg_cnt],
                            color_discrete_sequence=[C["teal"], C["red"]], hole=0.55, template=PLOTLY_TPL,
                            title="긍정 / 부정 VOC 비율")
            fig_pn.update_traces(textinfo="percent+label", textfont_size=14,
                                  marker=dict(line=dict(color="white", width=3)))
            fig_pn.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20), showlegend=True,
                                  title_font=dict(size=14, color=C["navy"]))
            st.plotly_chart(fig_pn, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  TAB 6  다차원 교차분석
# ─────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<p class="sec-head">📈 다차원 교차분석</p>', unsafe_allow_html=True)

    if not M["score"]:
        st.warning("만족도 점수 컬럼이 필요합니다.")
    elif not available_cats:
        st.warning("범주형 컬럼(지사, 접수자구분, 업무구분 등)이 필요합니다.")
    else:
        # ── 범주 선택 ──
        cat_sel = st.selectbox("교차분석 기준 범주 선택", available_cats, key="cross_cat_sel")

        # ── Boxplot: 범주별 종합점수 분포 ──
        st.markdown('<p class="sec-head">📦 범주별 점수 분포 (Boxplot)</p>', unsafe_allow_html=True)
        fig_box = px.box(
            df_f, x=cat_sel, y="_점수100", color=cat_sel,
            color_discrete_sequence=MIXED_COLORS, template=PLOTLY_TPL,
            title=f"{cat_sel}별 종합 점수 분포", points="outliers",
        )
        fig_box.add_hline(y=avg_score_100, line_dash="dot", line_color=C["gold"], line_width=2,
                          annotation_text=f"전체 평균 {avg_score_100:.1f}",
                          annotation_font_size=11, annotation_font_color=C["gold"])
        fig_box.update_layout(
            height=450, margin=dict(t=60, b=80, l=60, r=20),
            xaxis_tickangle=-25, showlegend=False,
            yaxis_title="종합 점수 (100점)", title_font=dict(size=14, color=C["navy"]),
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # ── Radar Chart: 범주별 개별 점수 비교 ──
        if individual_scores and len(individual_scores) >= 3:
            st.markdown("---")
            st.markdown('<p class="sec-head">🕸️ 범주별 개별항목 레이더 차트</p>', unsafe_allow_html=True)
            radar_groups = df_f.groupby(cat_sel)[individual_scores].mean()

            # 상위 8개 그룹만 표시 (너무 많으면 가독성 저하)
            if len(radar_groups) > 8:
                radar_groups = radar_groups.head(8)
                st.caption("※ 가독성을 위해 상위 8개 그룹만 표시합니다.")

            fig_radar_cross = go.Figure()
            colors_iter = MIXED_COLORS
            for i, (grp_name, row) in enumerate(radar_groups.iterrows()):
                vals = row.tolist() + [row.tolist()[0]]
                cats = individual_scores + [individual_scores[0]]
                fig_radar_cross.add_trace(go.Scatterpolar(
                    r=vals, theta=cats, fill='toself',
                    name=str(grp_name),
                    fillcolor=f"rgba({int(colors_iter[i % len(colors_iter)][1:3], 16)},{int(colors_iter[i % len(colors_iter)][3:5], 16)},{int(colors_iter[i % len(colors_iter)][5:7], 16)},0.08)",
                    line=dict(color=colors_iter[i % len(colors_iter)], width=2),
                ))
            fig_radar_cross.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9))),
                height=500, template=PLOTLY_TPL,
                title=dict(text=f"{cat_sel}별 개별항목 비교 (레이더)", font=dict(size=14, color=C["navy"])),
                margin=dict(t=80, b=40, l=80, r=80),
            )
            st.plotly_chart(fig_radar_cross, use_container_width=True)

        # ── 범주별 평균 점수 히트맵 (개별항목) ──
        if individual_scores:
            st.markdown("---")
            st.markdown('<p class="sec-head">🌡️ 범주별 개별항목 평균 히트맵</p>', unsafe_allow_html=True)
            pivot_cross = df_f.pivot_table(
                values=individual_scores, index=cat_sel, aggfunc="mean"
            ).round(1)
            if not pivot_cross.empty:
                fig_hm_cross = px.imshow(
                    pivot_cross, color_continuous_scale="RdYlGn",
                    text_auto=".1f", aspect="auto", template=PLOTLY_TPL,
                    title=f"{cat_sel}별 개별항목 평균 점수 (초록=높음 / 빨강=낮음)",
                )
                fig_hm_cross.update_layout(
                    height=max(350, len(pivot_cross) * 30 + 100),
                    margin=dict(t=60, b=60, l=120, r=60),
                    title_font=dict(size=14, color=C["navy"]),
                )
                st.plotly_chart(fig_hm_cross, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  TAB 8  시계열 트렌드 분석
# ─────────────────────────────────────────────────────────────
with tab8:
    st.markdown('<p class="sec-head">📅 시계열 트렌드 분석</p>', unsafe_allow_html=True)

    if not M.get("date"):
        st.info("📌 날짜 컬럼(접수일자, 조사일자 등)이 엑셀에 포함되어 있으면 자동으로 시계열 분석이 활성화됩니다.")
    elif not M["score"]:
        st.warning("만족도 점수 컬럼이 필요합니다.")
    else:
        # ── 8-1. 월별 만족도 추이 ──
        st.markdown('<p class="sec-head">📈 월별 만족도 추이</p>', unsafe_allow_html=True)
        monthly = df_f.groupby("_년월")["_점수100"].agg(["mean","count"]).reset_index()
        monthly.columns = ["년월", "평균만족도", "응답수"]
        monthly = monthly.sort_values("년월")

        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(
            go.Scatter(x=monthly["년월"], y=monthly["평균만족도"], mode="lines+markers+text",
                       name="평균 만족도", line=dict(color=C["blue"], width=3),
                       marker=dict(size=8), text=monthly["평균만족도"].round(1),
                       textposition="top center", textfont=dict(size=10)),
            secondary_y=False)
        fig_trend.add_trace(
            go.Bar(x=monthly["년월"], y=monthly["응답수"], name="응답수",
                   marker_color=C["sky"], opacity=0.35),
            secondary_y=True)
        fig_trend.update_layout(
            height=420, template=PLOTLY_TPL,
            title=dict(text="월별 평균 만족도 추이 & 응답 건수", font=dict(size=14, color=C["navy"])),
            margin=dict(t=60, b=60, l=60, r=60), legend=dict(orientation="h", y=1.12),
        )
        fig_trend.update_yaxes(title_text="만족도 (100점)", secondary_y=False)
        fig_trend.update_yaxes(title_text="응답수", secondary_y=True)
        st.plotly_chart(fig_trend, use_container_width=True)

        # 전월 대비 변화 감지
        if len(monthly) >= 2:
            last = monthly.iloc[-1]
            prev = monthly.iloc[-2]
            diff = last["평균만족도"] - prev["평균만족도"]
            arrow = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
            st.markdown(
                f'<div class="insight-box">{arrow} <b>최근 변화:</b> '
                f'{prev["년월"]} ({prev["평균만족도"]:.1f}점) → {last["년월"]} ({last["평균만족도"]:.1f}점), '
                f'<b>{diff:+.1f}점</b> 변동</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── 8-2. 분기별 비교 ──
        st.markdown('<p class="sec-head">📊 분기별 만족도 비교</p>', unsafe_allow_html=True)
        quarterly = df_f.groupby("_분기")["_점수100"].agg(["mean","count"]).reset_index()
        quarterly.columns = ["분기", "평균만족도", "응답수"]
        quarterly = quarterly.sort_values("분기")

        fig_q = px.bar(quarterly, x="분기", y="평균만족도", text="평균만족도",
                       color_discrete_sequence=[C["blue"]], template=PLOTLY_TPL,
                       title="분기별 평균 만족도")
        fig_q.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_q.update_layout(height=380, margin=dict(t=60, b=60, l=60, r=20),
                             yaxis_title="만족도 (100점)", title_font=dict(size=14, color=C["navy"]))
        st.plotly_chart(fig_q, use_container_width=True)

        st.markdown("---")

        # ── 8-3. 계절별 점수 비교 ──
        st.markdown('<p class="sec-head">🌤️ 계절별 만족도 비교</p>', unsafe_allow_html=True)
        season_order = ["봄", "여름", "가을", "겨울"]
        seasonal = df_f.groupby("_계절")["_점수100"].agg(["mean","count"]).reset_index()
        seasonal.columns = ["계절", "평균만족도", "응답수"]
        seasonal["계절"] = pd.Categorical(seasonal["계절"], categories=season_order, ordered=True)
        seasonal = seasonal.sort_values("계절")
        season_colors = {"봄": "#66bb6a", "여름": "#ef5350", "가을": "#ffa726", "겨울": "#42a5f5"}

        s_l, s_r = st.columns([1, 1])
        with s_l:
            fig_season = px.bar(seasonal, x="계절", y="평균만족도", color="계절",
                                color_discrete_map=season_colors, text="평균만족도",
                                template=PLOTLY_TPL, title="계절별 평균 만족도")
            fig_season.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_season.update_layout(height=380, showlegend=False,
                                      yaxis_title="만족도", title_font=dict(size=14, color=C["navy"]))
            st.plotly_chart(fig_season, use_container_width=True)
        with s_r:
            if individual_scores and len(seasonal) >= 3:
                season_radar = df_f.groupby("_계절")[individual_scores].mean()
                season_radar = season_radar.reindex(season_order).dropna()
                fig_sr = go.Figure()
                for i, (sn, row) in enumerate(season_radar.iterrows()):
                    vals = row.tolist() + [row.tolist()[0]]
                    cats = individual_scores + [individual_scores[0]]
                    fig_sr.add_trace(go.Scatterpolar(
                        r=vals, theta=cats, fill='toself', name=str(sn),
                        line=dict(color=season_colors.get(sn, C["gray"]), width=2),
                    ))
                fig_sr.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9))),
                    height=380, template=PLOTLY_TPL,
                    title=dict(text="계절별 개별항목 비교", font=dict(size=14, color=C["navy"])),
                    margin=dict(t=60, b=20, l=60, r=60))
                st.plotly_chart(fig_sr, use_container_width=True)
            else:
                fig_sp = px.pie(seasonal, names="계절", values="응답수", hole=0.45,
                                color="계절", color_discrete_map=season_colors,
                                template=PLOTLY_TPL, title="계절별 응답 비율")
                fig_sp.update_layout(height=380, title_font=dict(size=14, color=C["navy"]))
                st.plotly_chart(fig_sp, use_container_width=True)

        # 계절 인사이트
        if len(seasonal) >= 2:
            best_s = seasonal.loc[seasonal["평균만족도"].idxmax()]
            worst_s = seasonal.loc[seasonal["평균만족도"].idxmin()]
            gap = best_s["평균만족도"] - worst_s["평균만족도"]
            st.markdown(
                f'<div class="insight-box">🌡️ <b>계절 인사이트:</b> '
                f'<b>{best_s["계절"]}</b>({best_s["평균만족도"]:.1f}점)이 가장 높고, '
                f'<b>{worst_s["계절"]}</b>({worst_s["평균만족도"]:.1f}점)이 가장 낮습니다. '
                f'격차 <b>{gap:.1f}점</b>'
                + (f' — {worst_s["계절"]}철 집중 CS 관리가 필요합니다.' if gap >= 3 else '.')
                + '</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── 8-4. 지사별 월별 추이 ──
        if M["office"]:
            st.markdown('<p class="sec-head">🏢 지사별 월별 만족도 추이</p>', unsafe_allow_html=True)
            office_monthly = df_f.groupby([M["office"], "_년월"])["_점수100"].mean().reset_index()
            office_monthly.columns = ["지사", "년월", "평균만족도"]
            office_monthly = office_monthly.sort_values("년월")

            fig_om = px.line(office_monthly, x="년월", y="평균만족도", color="지사",
                             markers=True, template=PLOTLY_TPL,
                             color_discrete_sequence=MIXED_COLORS,
                             title="지사별 월별 만족도 추이")
            fig_om.update_layout(
                height=450, margin=dict(t=60, b=60, l=60, r=20),
                yaxis_title="만족도 (100점)", title_font=dict(size=14, color=C["navy"]),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_om, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  TAB 9  지사 심층 분석 · 패턴 탐지
# ─────────────────────────────────────────────────────────────
with tab9:
    st.markdown('<p class="sec-head">🔬 지사 심층 분석 · 패턴 탐지</p>', unsafe_allow_html=True)

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

            # 지사별 강점/약점 카드
            offices_sorted = df_f.groupby(M["office"])["_점수100"].mean().sort_values()
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
            fig_gap.update_traces(texttemplate="%{text:.1f}", textposition="outside")
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

        # ── 9-3. 부정 키워드 급증 시점/지사 탐지 ──
        if M["voc"] and M.get("date") and "_년월" in df_f.columns:
            st.markdown('<p class="sec-head">📈 부정 키워드 급증 탐지 (월별)</p>', unsafe_allow_html=True)

            df_f["_부정여부"] = df_f[M["voc"]].apply(lambda x: check_negative(x)[0])
            monthly_neg = df_f.groupby("_년월").agg(
                전체건수=("_부정여부", "count"),
                부정건수=("_부정여부", "sum"),
            ).reset_index()
            monthly_neg["부정비율"] = (monthly_neg["부정건수"] / monthly_neg["전체건수"] * 100).round(1)
            monthly_neg = monthly_neg.sort_values("_년월")

            fig_neg_trend = make_subplots(specs=[[{"secondary_y": True}]])
            fig_neg_trend.add_trace(
                go.Bar(x=monthly_neg["_년월"], y=monthly_neg["부정건수"],
                       name="부정 VOC 건수", marker_color=C["red"], opacity=0.6),
                secondary_y=False)
            fig_neg_trend.add_trace(
                go.Scatter(x=monthly_neg["_년월"], y=monthly_neg["부정비율"],
                           name="부정 비율(%)", mode="lines+markers+text",
                           line=dict(color=C["gold"], width=3), marker=dict(size=7),
                           text=monthly_neg["부정비율"].apply(lambda v: f"{v:.1f}%"),
                           textposition="top center", textfont=dict(size=10)),
                secondary_y=True)
            fig_neg_trend.update_layout(
                height=400, template=PLOTLY_TPL,
                title=dict(text="월별 부정 VOC 건수 & 비율 추이", font=dict(size=14, color=C["navy"])),
                margin=dict(t=60, b=60, l=60, r=60), legend=dict(orientation="h", y=1.12))
            fig_neg_trend.update_yaxes(title_text="부정 건수", secondary_y=False)
            fig_neg_trend.update_yaxes(title_text="부정 비율(%)", secondary_y=True)
            st.plotly_chart(fig_neg_trend, use_container_width=True)

            # 급증 감지
            if len(monthly_neg) >= 2:
                monthly_neg["전월대비"] = monthly_neg["부정비율"].diff()
                surge = monthly_neg[monthly_neg["전월대비"] >= 5].sort_values("전월대비", ascending=False)
                if len(surge) > 0:
                    for _, row in surge.head(3).iterrows():
                        st.markdown(
                            f'<div class="card-red">🚨 <b>{row["_년월"]}</b> — '
                            f'부정 비율 {row["부정비율"]:.1f}% '
                            f'(전월 대비 <b>+{row["전월대비"]:.1f}%p</b> 급증, {int(row["부정건수"])}건)'
                            f'</div>', unsafe_allow_html=True)
                else:
                    st.success("전월 대비 5%p 이상 급증한 월이 없습니다.")

            # 지사별 부정 비율
            if M["office"]:
                st.markdown("---")
                st.markdown('<p class="sec-head">🏢 지사별 부정 VOC 비율</p>', unsafe_allow_html=True)
                office_neg = df_f.groupby(M["office"]).agg(
                    전체건수=("_부정여부", "count"),
                    부정건수=("_부정여부", "sum"),
                ).reset_index()
                office_neg["부정비율"] = (office_neg["부정건수"] / office_neg["전체건수"] * 100).round(1)
                office_neg = office_neg.sort_values("부정비율", ascending=True)

                fig_on = px.bar(office_neg, x="부정비율", y=M["office"], orientation="h",
                                text="부정비율", template=PLOTLY_TPL,
                                color=office_neg["부정비율"].apply(
                                    lambda v: "🔴 위험" if v >= 30 else "🟡 주의" if v >= 15 else "일반"),
                                color_discrete_map={"🔴 위험": C["red"], "🟡 주의": C["gold"], "일반": C["sky"]},
                                title="지사별 부정 VOC 비율 (%)")
                fig_on.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_on.update_layout(
                    height=max(300, len(office_neg) * 30 + 80),
                    margin=dict(t=60, b=20, l=10, r=80),
                    showlegend=False, title_font=dict(size=14, color=C["navy"]))
                st.plotly_chart(fig_on, use_container_width=True)

        elif M["voc"] and not M.get("date"):
            st.info("📌 날짜 컬럼이 있으면 부정 키워드 급증 시점을 자동 탐지할 수 있습니다.")
