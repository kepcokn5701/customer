# ==============================================================
#  고객경험관리 CS 분석 대시보드 v4.0
#  유기적 다차원 교차분석 · 취약그룹 타겟 텍스트분석 · 인사이트 자동생성
#  HuggingFace Spaces (Streamlit SDK, CPU, 16GB RAM) 최적화
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import io, re, os

# ── 선택적 라이브러리 ──────────────────────────────────────────
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
#  0. 한글 폰트 (워드클라우드용)
# ══════════════════════════════════════════════════════════════
FONT_PATH = None
for _c in [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    "/System/Library/Fonts/AppleGothic.ttf",
]:
    if os.path.exists(_c):
        FONT_PATH = _c
        break

# ══════════════════════════════════════════════════════════════
#  1. 디자인 상수
# ══════════════════════════════════════════════════════════════
C = dict(
    navy="#1a3a6c", blue="#0055a5", sky="#2196f3", light="#e3f2fd",
    gold="#f0a500", teal="#00897b", red="#c62828", orange="#e65100",
    green="#2e7d32", gray="#546e7a", white="#ffffff", bg="#f4f7fc",
)
PIE_COLORS = ["#1a3a6c","#0055a5","#2196f3","#42a5f5","#90caf9",
              "#64b5f6","#1565c0","#0288d1","#0097a7","#00897b"]
MIXED_COLORS = ["#1a3a6c","#f0a500","#2196f3","#00897b","#c62828",
                "#7b1fa2","#e65100","#558b2f","#0288d1","#ad1457"]
PLOTLY_TPL = "plotly_white"

# 개별 점수 컬럼명
SCORE_COLS = [
    "이용 편리성", "직원 친절도", "전반적 만족", "사회적 책임",
    "처리 신속도", "처리 정확도", "업무 개선도", "사용 추천도",
]
TOTAL_SCORE_COL = "종합 점수"
VOC_COL = "서술 의견"
CATEGORY_COLS = ["지사", "계약종별", "접수종류", "업무구분", "신청방법", "접수자구분"]

# 부정 키워드
NEGATIVE_KEYWORDS = [
    "불만","불편","민원","항의","화남","짜증","느림","느려","오류","오작동",
    "불량","고장","재방문","지연","오래","기다림","실망","최악","별로",
    "안됨","문제","취소","환불","비싸다","비싸","과다","과금",
    "불친절","무시","황당","어이없","답답","이해불가","부당",
    "잘못","실수","착오","불합리","불공정","반복","여전히",
    "힘듭니다","어렵습니다","불쾌","무성의","소홀","방치",
    "정전","단전","누전","위험","안전","사고",
]
RUDE_KEYWORDS = [
    "불친절","무시","무성의","불쾌","반말","태도","무례","소홀","건방",
    "고압적","짜증","화남","면박","냉담","퉁명","성의없","막말",
]
_STOP = {
    "이","가","은","는","을","를","의","에","에서","으로","로","와","과","하",
    "것","수","있","없","않","못","더","또","그","저","있다","없다","됩니다",
    "합니다","했다","하는","하여","해서","입니다","이다","이고","같은",
    "너무","매우","정말","아주","조금","많이","항상","계속","때문",
    "경우","부분","관련","대한","위한","통해","에게","주세요","바랍니다",
    "감사","드립니다","했습니다","있습니다","없습니다",
    "해주세요","요청","하겠습니다",
    "관리","시스템","문의","확인","처리","접수","신청","등록","변경","조회",
    "이용","서비스","고객","전화","상담","안내","답변","진행","완료","내용",
    "회사","전력","전기","사용","사용량","검침",
    "홈페이지","인터넷","방문","센터","지사","지점","담당","담당자","직원",
    "설문","조사","응답","결과","평가","만족","점수","항목","기타","해당",
}

# ══════════════════════════════════════════════════════════════
#  2. 유틸리티 함수
# ══════════════════════════════════════════════════════════════
def normalize_score(series):
    """점수를 100점 만점으로 자동 변환"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.to_numeric(series, errors="coerce")
    mx = s.max()
    if mx <= 5:
        return pd.to_numeric(series, errors="coerce") * 20
    elif mx <= 10:
        return pd.to_numeric(series, errors="coerce") * 10
    return pd.to_numeric(series, errors="coerce")


def extract_keywords_regex(texts, top_n=30):
    """정규식 기반 한국어 키워드 추출 (빠름)"""
    words = []
    for t in texts:
        if not t or str(t).strip() in ("", "nan"):
            continue
        found = re.findall(r"[가-힣]{2,}", str(t))
        words.extend([w for w in found if w not in _STOP])
    return Counter(words).most_common(top_n)


def extract_keywords_keybert(texts, top_n=30):
    """KeyBERT 기반 키워드 추출 (정밀, 느림)"""
    if not KEYBERT_AVAILABLE or not texts:
        return extract_keywords_regex(texts, top_n)
    try:
        joined = " ".join([str(t) for t in texts if t and str(t).strip() not in ("", "nan")])
        if not joined.strip():
            return []
        kw_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")
        kw_results = kw_model.extract_keywords(
            joined, keyphrase_ngram_range=(1, 2),
            stop_words=None, top_n=top_n, use_mmr=True, diversity=0.5,
        )
        # KeyBERT 결과 + 빈도수 결합
        freq = Counter()
        for t in texts:
            if not t or str(t).strip() in ("", "nan"):
                continue
            for kw, _ in kw_results:
                if kw in str(t):
                    freq[kw] += 1
        return [(kw, freq.get(kw, 1)) for kw, _ in kw_results if freq.get(kw, 0) > 0][:top_n]
    except Exception:
        return extract_keywords_regex(texts, top_n)


def classify_sentiment_keyword(text):
    """키워드 기반 감성 분류 (빠름, 모델 불필요)"""
    if not text or str(text).strip() in ("", "nan"):
        return "중립"
    s = str(text)
    rude = [kw for kw in RUDE_KEYWORDS if kw in s]
    if rude:
        return "불친절"
    neg = [kw for kw in NEGATIVE_KEYWORDS if kw in s]
    if neg:
        return "불만"
    return "긍정"


def make_wordcloud(kw_list):
    if not WORDCLOUD_AVAILABLE or not kw_list:
        return None
    freq = {k: v for k, v in kw_list}
    kwargs = dict(width=900, height=400, background_color="white",
                  max_words=80, colormap="Blues", prefer_horizontal=0.7)
    if FONT_PATH:
        kwargs["font_path"] = FONT_PATH
    try:
        wc = WordCloud(**kwargs)
        wc.generate_from_frequencies(freq)
        return wc.to_image()
    except Exception:
        return None


def df_to_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="분석결과")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
#  3. 페이지 설정 & CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CS 분석 대시보드", page_icon="📊",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  .stApp {{ background-color: {C['bg']}; }}

  .dash-header {{
    background: linear-gradient(120deg, {C['navy']} 0%, {C['blue']} 55%, {C['sky']} 100%);
    padding: 2rem 2.5rem; border-radius: 16px; color: white;
    margin-bottom: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.18);
  }}
  .dash-header h1 {{ font-size:1.9rem; font-weight:800; margin:0; }}
  .dash-header p  {{ font-size:0.95rem; margin:0.4rem 0 0 0; opacity:0.85; }}
  .dash-badge {{
    display:inline-block; background:rgba(255,255,255,0.18);
    border:1px solid rgba(255,255,255,0.35); border-radius:20px;
    padding:0.2rem 0.8rem; font-size:0.78rem; margin-top:0.7rem; margin-right:0.4rem;
  }}

  .insight-box {{
    background: linear-gradient(135deg, #fff8e1 0%, #fff3c4 100%);
    border-left: 6px solid {C['gold']}; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;
    font-size: 1.05rem; color: {C['navy']}; font-weight: 600;
    box-shadow: 0 2px 8px rgba(240,165,0,0.15);
  }}

  [data-testid="stMetric"] {{
    background: {C['white']}; border-radius: 14px;
    padding: 1.1rem 1.2rem; box-shadow: 0 2px 12px rgba(0,85,165,0.10);
    border-top: 4px solid {C['blue']};
  }}
  [data-testid="stMetricLabel"]  {{ font-size:0.82rem !important; color:{C['gray']} !important; font-weight:600; }}
  [data-testid="stMetricValue"]  {{ font-size:1.7rem !important; font-weight:800 !important; color:{C['navy']} !important; }}

  .stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:2px solid #dde4ef; }}
  .stTabs [data-baseweb="tab"] {{ font-size:0.95rem; font-weight:700; padding:0.6rem 1.4rem; border-radius:8px 8px 0 0; color:{C['gray']}; }}
  .stTabs [aria-selected="true"] {{ color:{C['navy']} !important; background:{C['white']} !important; border-bottom:3px solid {C['blue']} !important; }}

  .sec-head {{ font-size:1.1rem; font-weight:800; color:{C['navy']}; border-left:5px solid {C['blue']}; padding:0.2rem 0 0.2rem 0.8rem; margin:1.4rem 0 1rem 0; }}

  .card      {{ background:{C['white']}; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 12px rgba(0,85,165,0.09); margin-bottom:1rem; color:#333; }}
  .card-red  {{ background:#fff5f5; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 12px rgba(198,40,40,0.10); border-left:5px solid {C['red']}; margin-bottom:1rem; color:#333; }}
  .card-gold {{ background:#fffbf0; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 12px rgba(240,165,0,0.12); border-left:5px solid {C['gold']}; margin-bottom:1rem; color:#333; }}
  .card-teal {{ background:#e8f5e9; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 12px rgba(0,137,123,0.10); border-left:5px solid {C['teal']}; margin-bottom:1rem; color:#333; }}
  .card-blue {{ background:{C['light']}; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 12px rgba(0,85,165,0.09); border-left:5px solid {C['sky']}; margin-bottom:1rem; color:#333; }}

  /* 사이드바 */
  [data-testid="stSidebar"] {{ background: linear-gradient(180deg, {C['navy']} 0%, #1e4d8c 100%) !important; }}
  [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] .stMarkdown *, [data-testid="stSidebar"] .stCaption,
  [data-testid="stSidebar"] small {{ color: white !important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] > div {{ background:rgba(255,255,255,0.15) !important; border:1px solid rgba(255,255,255,0.35) !important; border-radius:8px !important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] > div * {{ color:white !important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill:white !important; }}
  [data-testid="stSidebar"] [data-baseweb="tag"] {{ background:rgba(255,255,255,0.22) !important; border:1px solid rgba(255,255,255,0.4) !important; }}
  [data-testid="stSidebar"] [data-baseweb="tag"] * {{ color:white !important; }}
  [data-baseweb="popover"] li, [data-baseweb="popover"] ul, [data-baseweb="popover"] div {{ color:{C['navy']} !important; }}
  [data-testid="stSidebar"] .stExpander {{ background:rgba(255,255,255,0.08) !important; border:1px solid rgba(255,255,255,0.25) !important; border-radius:10px !important; }}
  [data-testid="stSidebar"] .stExpander summary span {{ color:white !important; }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] {{ background:rgba(255,255,255,0.15) !important; border:2px dashed rgba(255,255,255,0.6) !important; border-radius:12px !important; padding:1rem !important; }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] * {{ color:white !important; }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] section {{ background:rgba(255,255,255,0.12) !important; border-radius:8px !important; }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"] button {{ background:{C['gold']} !important; color:{C['navy']} !important; font-weight:700 !important; border:none !important; border-radius:8px !important; }}
  [data-testid="stSidebar"] hr {{ border-color:rgba(255,255,255,0.2) !important; }}
  [data-testid="stSidebar"] .stMultiSelect label {{ font-size:0.82rem !important; }}
  [data-testid="stSidebar"] .stSlider label {{ font-size:0.82rem !important; }}
  [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {{ color:white !important; }}

  .stDownloadButton > button {{ background:linear-gradient(90deg,{C['navy']},{C['blue']}) !important; color:white !important; font-weight:700 !important; border-radius:10px !important; border:none !important; }}
  [data-testid="stDataFrame"] th {{ background:{C['navy']} !important; color:white !important; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  4. 사이드바
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 CS 분석 대시보드")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 📂 데이터 업로드")
    uploaded = st.file_uploader(
        "엑셀 파일(.xlsx)을 업로드하세요", type=["xlsx", "xls"], label_visibility="collapsed"
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("© 2025 CS 분석 시스템 v4.0")

# ══════════════════════════════════════════════════════════════
#  5. 헤더 & 업로드 전 안내
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="dash-header">
  <h1>📊 고객경험관리 CS 분석 대시보드</h1>
  <p>다차원 교차분석 · 상관관계 · 취약그룹 타겟 텍스트분석 · 인사이트 자동생성</p>
  <span class="dash-badge">📈 교차분석</span>
  <span class="dash-badge">🔗 상관관계</span>
  <span class="dash-badge">🎯 취약그룹</span>
  <span class="dash-badge">☁️ VOC 분석</span>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown('<div class="card-blue">', unsafe_allow_html=True)
        st.markdown("### 📋 사용 방법")
        st.markdown("""
1. 왼쪽 사이드바에서 **엑셀 파일(.xlsx)**을 업로드
2. 사이드바 필터로 원하는 데이터 범위 선택
3. 각 **탭**을 클릭하며 분석 결과 확인
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with c_r:
        st.markdown('<div class="card-teal">', unsafe_allow_html=True)
        st.markdown("### 📌 필요 컬럼")
        st.markdown("""
| 컬럼 | 설명 |
|---|---|
| 지사 / 계약종별 / 업무구분 등 | 범주형 분류 |
| 이용 편리성 ~ 사용 추천도 | 개별 점수 (수치) |
| 종합 점수 | 전체 만족도 (수치) |
| 서술 의견 | VOC 텍스트 |
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════
#  6. 데이터 로드
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(raw_bytes):
    try:
        df = pd.read_excel(io.BytesIO(raw_bytes), header=2, engine="openpyxl")
    except Exception:
        df = pd.read_excel(io.BytesIO(raw_bytes), header=2)
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    if "순번" in df.columns:
        df.drop(columns=["순번"], inplace=True)
    df.drop_duplicates(inplace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df

with st.spinner("데이터를 불러오는 중…"):
    df_raw = load_data(uploaded.read())
    total_rows = len(df_raw)

# ── 컬럼 존재 여부 확인 ──
available_scores = [c for c in SCORE_COLS if c in df_raw.columns]
has_total = TOTAL_SCORE_COL in df_raw.columns
has_voc = VOC_COL in df_raw.columns
available_cats = [c for c in CATEGORY_COLS if c in df_raw.columns]

# ── 점수 정규화 ──
for col in available_scores:
    df_raw[col] = normalize_score(df_raw[col])
if has_total:
    df_raw[TOTAL_SCORE_COL] = normalize_score(df_raw[TOTAL_SCORE_COL])

# ══════════════════════════════════════════════════════════════
#  7. 사이드바 필터
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 🔍 데이터 필터")
    df_f = df_raw.copy()

    for cat in available_cats:
        opts = sorted(df_raw[cat].dropna().astype(str).unique().tolist())
        if opts:
            sel = st.multiselect(f"📌 {cat}", opts, default=opts, key=f"f_{cat}")
            if sel:
                df_f = df_f[df_f[cat].astype(str).isin(sel)]

    if has_total:
        sc_s = pd.to_numeric(df_raw[TOTAL_SCORE_COL], errors="coerce").dropna()
        if not sc_s.empty:
            s_min, s_max = float(sc_s.min()), float(sc_s.max())
            sc_rng = st.slider("종합 점수 범위", s_min, s_max, (s_min, s_max), 0.1, key="f_sc")
            df_f = df_f[pd.to_numeric(df_f[TOTAL_SCORE_COL], errors="coerce").between(sc_rng[0], sc_rng[1])]

    st.caption(f"필터 적용: **{len(df_f):,}건** / 전체 {total_rows:,}건")

n_filtered = len(df_f)

# ══════════════════════════════════════════════════════════════
#  8. 핵심 분석 (캐싱)
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_correlation(df, score_cols, total_col):
    """종합 점수와 개별 점수 간 상관계수"""
    cols = [c for c in score_cols if c in df.columns] + [total_col]
    num_df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if num_df.empty or len(num_df) < 3:
        return pd.DataFrame()
    corr = num_df.corr()
    return corr


@st.cache_data(show_spinner=False)
def find_vulnerable_group(df, cat_col, total_col, percentile=30):
    """특정 범주형 컬럼에서 종합 점수 하위 percentile% 그룹 탐색"""
    grp = df.groupby(cat_col)[total_col].agg(["mean", "count"]).reset_index()
    grp.columns = [cat_col, "평균", "건수"]
    grp = grp.sort_values("평균")
    threshold = np.percentile(df[total_col].dropna(), percentile)
    weak = grp[grp["평균"] <= threshold]
    return weak, threshold, grp


@st.cache_data(show_spinner=False)
def generate_insight(df, score_cols, total_col, cat_cols):
    """데이터 기반 한 줄 인사이트 자동 생성"""
    insights = []

    # 1) 상관관계 → 가장 영향 큰 개별 점수
    if total_col in df.columns and score_cols:
        num = df[score_cols + [total_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(num) > 3:
            corr_total = num.corr()[total_col].drop(total_col).sort_values(ascending=False)
            top_corr_item = corr_total.index[0]
            top_corr_val = corr_total.iloc[0]
            insights.append(
                f"종합 점수에 가장 큰 영향을 미치는 항목은 **'{top_corr_item}'** (상관계수 {top_corr_val:.3f})입니다."
            )

    # 2) 범주별 가장 취약한 그룹
    for cat in cat_cols:
        if cat not in df.columns:
            continue
        grp = df.groupby(cat)[total_col].mean()
        if grp.empty:
            continue
        worst_cat = grp.idxmin()
        worst_val = grp.min()
        best_cat = grp.idxmax()
        best_val = grp.max()
        gap = best_val - worst_val
        if gap >= 3:
            insights.append(
                f"'{cat}' 기준으로 **'{worst_cat}'**의 만족도({worst_val:.1f}점)가 가장 낮으며, "
                f"**'{best_cat}'**({best_val:.1f}점)과 {gap:.1f}점 차이가 납니다."
            )
            break  # 가장 차이 큰 하나만

    # 3) VOC 불만 비율
    if VOC_COL in df.columns:
        voc_valid = df[VOC_COL].dropna().astype(str)
        voc_valid = voc_valid[voc_valid.str.strip() != ""]
        if len(voc_valid) > 0:
            neg_cnt = sum(1 for t in voc_valid if any(kw in str(t) for kw in NEGATIVE_KEYWORDS))
            neg_pct = neg_cnt / len(voc_valid) * 100
            if neg_pct >= 10:
                insights.append(
                    f"서술 의견 중 **{neg_pct:.1f}%**에서 부정적 키워드가 감지되었습니다. 집중 관리가 필요합니다."
                )

    if not insights:
        return "데이터가 업로드되었습니다. 각 탭에서 상세 분석을 확인하세요."
    return " ".join(insights)


# ── 인사이트 생성 ──
progress = st.progress(0, text="데이터 분석 중…")
insight_text = generate_insight(df_f, available_scores, TOTAL_SCORE_COL, available_cats)
progress.progress(30, text="상관관계 분석 중…")

corr_df = pd.DataFrame()
if has_total and available_scores:
    corr_df = compute_correlation(df_f, available_scores, TOTAL_SCORE_COL)
progress.progress(60, text="취약 그룹 탐색 중…")

# 취약 그룹 사전 계산
vulnerable_info = {}
if has_total:
    for cat in available_cats:
        weak, thresh, grp = find_vulnerable_group(df_f, cat, TOTAL_SCORE_COL, 30)
        vulnerable_info[cat] = {"weak": weak, "threshold": thresh, "grp": grp}
progress.progress(100, text="분석 완료!")
progress.empty()

# ── 인사이트 박스 ──
st.markdown(f'<div class="insight-box">💡 {insight_text}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  9. KPI
# ══════════════════════════════════════════════════════════════
avg_total = df_f[TOTAL_SCORE_COL].mean() if has_total else None
m_cols = st.columns(5)
with m_cols[0]:
    if avg_total is not None and not np.isnan(avg_total):
        st.metric("⭐ 종합 만족도", f"{avg_total:.1f}점")
    else:
        st.metric("⭐ 종합 만족도", "N/A")
with m_cols[1]:
    st.metric("📋 분석 건수", f"{n_filtered:,}건")
with m_cols[2]:
    if has_voc:
        voc_valid = df_f[VOC_COL].dropna().astype(str)
        voc_cnt = len(voc_valid[voc_valid.str.strip() != ""])
        st.metric("💬 VOC 응답", f"{voc_cnt:,}건")
    else:
        st.metric("💬 VOC 응답", "N/A")
with m_cols[3]:
    if available_scores:
        weakest = df_f[available_scores].mean().idxmin()
        weakest_val = df_f[available_scores].mean().min()
        st.metric("📉 최저 항목", weakest, delta=f"{weakest_val:.1f}점", delta_color="inverse")
    else:
        st.metric("📉 최저 항목", "N/A")
with m_cols[4]:
    if available_scores:
        strongest = df_f[available_scores].mean().idxmax()
        strongest_val = df_f[available_scores].mean().max()
        st.metric("📈 최고 항목", strongest, delta=f"{strongest_val:.1f}점")
    else:
        st.metric("📈 최고 항목", "N/A")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  10. 탭 구성
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 다차원 교차분석",
    "🔗 상관관계 · 영향도",
    "🎯 취약그룹 타겟 분석",
    "☁️ VOC 텍스트 분석",
])

# ─────────────────────────────────────────────────────────────
#  TAB 1: 다차원 교차분석
# ─────────────────────────────────────────────────────────────
with tab1:
    if not has_total:
        st.warning("종합 점수 컬럼이 없습니다.")
    else:
        # ── 범주별 종합 점수 Boxplot ──
        cat_for_box = st.selectbox(
            "분석 기준 범주 선택", available_cats,
            index=available_cats.index("계약종별") if "계약종별" in available_cats else 0,
            key="box_cat"
        )
        st.markdown(f'<p class="sec-head">📊 {cat_for_box}별 종합 점수 분포 (Boxplot)</p>', unsafe_allow_html=True)

        fig_box = px.box(
            df_f, x=cat_for_box, y=TOTAL_SCORE_COL, color=cat_for_box,
            color_discrete_sequence=MIXED_COLORS, template=PLOTLY_TPL,
            title=f"{cat_for_box}별 종합 점수 분포",
            points="outliers",
        )
        fig_box.update_layout(
            height=450, margin=dict(t=60, b=80, l=60, r=20),
            xaxis_tickangle=-20, showlegend=False,
            title_font=dict(size=15, color=C["navy"]),
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # ── 범주별 개별 점수 Radar Chart ──
        if available_scores and len(df_f[cat_for_box].dropna().unique()) >= 2:
            st.markdown(f'<p class="sec-head">🕸️ {cat_for_box}별 개별 점수 레이더 차트</p>', unsafe_allow_html=True)
            radar_grp = df_f.groupby(cat_for_box)[available_scores].mean()

            fig_radar = go.Figure()
            for i, (name, row) in enumerate(radar_grp.iterrows()):
                vals = row.tolist() + [row.tolist()[0]]
                cats = available_scores + [available_scores[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=cats, fill='toself', name=str(name),
                    line=dict(color=MIXED_COLORS[i % len(MIXED_COLORS)]),
                    opacity=0.7,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9))),
                height=500, template=PLOTLY_TPL,
                title=dict(text=f"{cat_for_box}별 개별 항목 비교", font=dict(size=15, color=C["navy"])),
                margin=dict(t=80, b=40, l=80, r=80),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── 범주별 평균 비교 테이블 ──
        st.markdown(f'<p class="sec-head">📋 {cat_for_box}별 항목별 평균 점수</p>', unsafe_allow_html=True)
        score_cols_present = [c for c in available_scores + [TOTAL_SCORE_COL] if c in df_f.columns]
        grp_table = df_f.groupby(cat_for_box)[score_cols_present].mean().round(1)
        grp_table["응답수"] = df_f.groupby(cat_for_box).size()
        grp_table = grp_table.sort_values(TOTAL_SCORE_COL, ascending=False)
        st.dataframe(grp_table, use_container_width=True)

        # ── 히트맵: 범주 x 개별 점수 ──
        if available_scores and len(radar_grp) >= 2:
            st.markdown(f'<p class="sec-head">🌡️ {cat_for_box} × 개별 항목 히트맵</p>', unsafe_allow_html=True)
            fig_hm = px.imshow(
                radar_grp.round(1), color_continuous_scale="RdYlGn",
                text_auto=".1f", aspect="auto", template=PLOTLY_TPL,
                title=f"{cat_for_box} × 개별 항목 평균 (초록=높음, 빨강=낮음)",
            )
            fig_hm.update_layout(
                height=max(350, len(radar_grp) * 30 + 100),
                margin=dict(t=60, b=40, l=150, r=40),
                title_font=dict(size=14, color=C["navy"]),
            )
            st.plotly_chart(fig_hm, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  TAB 2: 상관관계 · 영향도
# ─────────────────────────────────────────────────────────────
with tab2:
    if corr_df.empty:
        st.warning("상관관계를 계산할 수 있는 수치 데이터가 부족합니다.")
    else:
        # ── 상관관계 히트맵 ──
        st.markdown('<p class="sec-head">🔗 종합 점수 × 개별 점수 상관관계 히트맵</p>', unsafe_allow_html=True)
        fig_corr = px.imshow(
            corr_df.round(3), color_continuous_scale="RdBu_r",
            text_auto=".3f", aspect="auto", template=PLOTLY_TPL,
            title="상관계수 행렬 (1에 가까울수록 강한 양의 상관)",
            zmin=-1, zmax=1,
        )
        fig_corr.update_layout(
            height=500, margin=dict(t=60, b=40, l=150, r=40),
            title_font=dict(size=14, color=C["navy"]),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # ── 종합 점수와의 상관 랭킹 ──
        st.markdown('<p class="sec-head">📊 종합 점수와의 상관관계 순위</p>', unsafe_allow_html=True)
        corr_rank = corr_df[TOTAL_SCORE_COL].drop(TOTAL_SCORE_COL).sort_values(ascending=False)
        rank_df = pd.DataFrame({
            "개별 항목": corr_rank.index,
            "상관계수": corr_rank.values,
            "영향도": ["🔴 매우 높음" if v >= 0.7 else "🟠 높음" if v >= 0.5 else "🟡 보통" if v >= 0.3 else "🟢 낮음"
                      for v in corr_rank.values],
        }).reset_index(drop=True)

        r_l, r_r = st.columns([3, 2])
        with r_l:
            fig_rank = px.bar(
                rank_df, x="상관계수", y="개별 항목", orientation="h",
                color="상관계수", color_continuous_scale="Blues",
                text="상관계수", template=PLOTLY_TPL,
                title="종합 점수에 대한 개별 항목 영향도",
            )
            fig_rank.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_rank.update_layout(
                height=400, margin=dict(t=50, b=20, l=10, r=80),
                title_font=dict(size=14, color=C["navy"]),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_rank, use_container_width=True)
        with r_r:
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(rank_df, use_container_width=True, hide_index=True, height=380)

        # ── 산점도: 가장 영향 큰 항목 vs 종합 점수 ──
        top_item = corr_rank.index[0]
        st.markdown(f'<p class="sec-head">🔍 핵심 항목 상세: {top_item} vs 종합 점수</p>', unsafe_allow_html=True)
        scatter_cat = st.selectbox("색상 기준 범주", available_cats, key="scatter_cat") if available_cats else None

        fig_sc = px.scatter(
            df_f, x=top_item, y=TOTAL_SCORE_COL,
            color=scatter_cat if scatter_cat else None,
            color_discrete_sequence=MIXED_COLORS,
            trendline="ols", template=PLOTLY_TPL,
            title=f"'{top_item}' ↔ 종합 점수 (추세선 포함)",
            opacity=0.6,
        )
        fig_sc.update_layout(
            height=450, margin=dict(t=60, b=60, l=60, r=40),
            title_font=dict(size=14, color=C["navy"]),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown(
            f'<div class="card-gold">'
            f'<b>💡 분석 해석:</b> <code>{top_item}</code>의 상관계수가 <b>{corr_rank.iloc[0]:.3f}</b>으로 '
            f'종합 만족도에 가장 큰 영향을 미칩니다. '
            f'이 항목의 점수를 개선하면 전체 만족도 향상에 가장 효과적입니다.'
            f'</div>', unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────
#  TAB 3: 취약그룹 타겟 분석
# ─────────────────────────────────────────────────────────────
with tab3:
    if not has_total:
        st.warning("종합 점수 컬럼이 필요합니다.")
    elif not available_cats:
        st.warning("범주형 컬럼이 없습니다.")
    else:
        vuln_cat = st.selectbox(
            "취약 그룹 탐색 기준", available_cats,
            index=available_cats.index("계약종별") if "계약종별" in available_cats else 0,
            key="vuln_cat"
        )
        vuln_pct = st.slider("하위 기준 (%)", 10, 50, 30, 5, key="vuln_pct")

        with st.spinner(f"'{vuln_cat}' 기준 취약 그룹 분석 중…"):
            weak, thresh, all_grp = find_vulnerable_group(df_f, vuln_cat, TOTAL_SCORE_COL, vuln_pct)

        st.markdown(f'<p class="sec-head">🎯 {vuln_cat}별 만족도 — 하위 {vuln_pct}% 취약 그룹</p>',
                    unsafe_allow_html=True)

        # ── 전체 vs 취약 바 차트 ──
        all_grp["구분"] = all_grp["평균"].apply(
            lambda v: "🔴 취약" if v <= thresh else "일반")
        color_map = {"🔴 취약": C["red"], "일반": C["sky"]}

        fig_vuln = px.bar(
            all_grp.sort_values("평균"), x="평균", y=vuln_cat, color="구분",
            color_discrete_map=color_map, orientation="h", text="평균",
            template=PLOTLY_TPL,
            title=f"{vuln_cat}별 평균 만족도 (빨강 = 하위 {vuln_pct}%)",
        )
        fig_vuln.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_vuln.add_vline(
            x=thresh, line_dash="dash", line_color=C["gray"],
            annotation_text=f"하위 {vuln_pct}% 기준: {thresh:.1f}",
            annotation_font_size=11,
        )
        fig_vuln.update_layout(
            height=max(350, len(all_grp) * 35 + 80),
            margin=dict(t=60, b=20, l=10, r=120),
            title_font=dict(size=14, color=C["navy"]), legend_title_text="",
        )
        st.plotly_chart(fig_vuln, use_container_width=True)

        # ── 취약 그룹 상세 분석 ──
        if weak.empty:
            st.success("취약 그룹이 없습니다. 전반적으로 양호한 상태입니다.")
        else:
            weak_names = weak[vuln_cat].tolist()
            st.markdown(f'<p class="sec-head">🔍 취약 그룹 [{", ".join(str(n) for n in weak_names)}] 상세 분석</p>',
                        unsafe_allow_html=True)

            df_weak = df_f[df_f[vuln_cat].astype(str).isin([str(n) for n in weak_names])]

            # 취약 그룹의 개별 항목 평균 vs 전체 평균
            if available_scores:
                overall_avg = df_f[available_scores].mean()
                weak_avg = df_weak[available_scores].mean()
                compare = pd.DataFrame({
                    "항목": available_scores,
                    "전체 평균": overall_avg.values,
                    "취약그룹 평균": weak_avg.values,
                    "차이": (weak_avg - overall_avg).values,
                })
                compare = compare.sort_values("차이")

                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(
                    x=compare["항목"], y=compare["전체 평균"], name="전체 평균",
                    marker_color=C["sky"], opacity=0.6,
                ))
                fig_compare.add_trace(go.Bar(
                    x=compare["항목"], y=compare["취약그룹 평균"], name="취약그룹 평균",
                    marker_color=C["red"], opacity=0.8,
                ))
                fig_compare.update_layout(
                    barmode="group", template=PLOTLY_TPL,
                    title=dict(text="취약그룹 vs 전체: 어떤 항목에서 차이가 나는가?",
                               font=dict(size=14, color=C["navy"])),
                    height=400, margin=dict(t=60, b=80, l=60, r=20),
                    xaxis_tickangle=-15, legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig_compare, use_container_width=True)

                # 가장 차이 큰 항목 하이라이트
                worst_gap_item = compare.iloc[0]["항목"]
                worst_gap_val = compare.iloc[0]["차이"]
                st.markdown(
                    f'<div class="card-red">'
                    f'<b>🚨 핵심 발견:</b> 취약 그룹은 <b>"{worst_gap_item}"</b> 항목에서 '
                    f'전체 평균 대비 <b>{worst_gap_val:+.1f}점</b> 낮습니다. '
                    f'이 항목의 개선이 최우선 과제입니다.'
                    f'</div>', unsafe_allow_html=True
                )

            # ── 취약 그룹 VOC 키워드 분석 (타겟팅) ──
            if has_voc:
                st.markdown("---")
                st.markdown(f'<p class="sec-head">💬 취약 그룹 VOC 키워드 (타겟 분석: {len(df_weak):,}건만)</p>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="card-blue">'
                    f'<b>🎯 성능 최적화:</b> 전체 {n_filtered:,}건이 아닌, 취약 그룹 <b>{len(df_weak):,}건</b>의 '
                    f'VOC만 집중 분석합니다. "왜 이 그룹의 점수가 낮은가?"에 대한 답을 찾습니다.'
                    f'</div>', unsafe_allow_html=True
                )

                weak_voc = df_weak[VOC_COL].dropna().astype(str)
                weak_voc = weak_voc[weak_voc.str.strip() != ""].tolist()

                if weak_voc:
                    with st.spinner(f"취약 그룹 VOC {len(weak_voc):,}건 키워드 추출 중…"):
                        if len(weak_voc) <= 500 and KEYBERT_AVAILABLE:
                            weak_kws = extract_keywords_keybert(weak_voc, top_n=30)
                        else:
                            weak_kws = extract_keywords_regex(weak_voc, top_n=30)

                    if weak_kws:
                        wc_col, kw_col = st.columns([3, 2])
                        with wc_col:
                            img = make_wordcloud(weak_kws)
                            if img:
                                st.image(img, use_container_width=True,
                                         caption="취약 그룹 VOC 워드클라우드")
                            else:
                                kw_names = [k for k, _ in weak_kws[:15]]
                                kw_vals = [v for _, v in weak_kws[:15]]
                                fig_kbar = px.bar(x=kw_vals, y=kw_names, orientation="h",
                                                  color_discrete_sequence=[C["red"]],
                                                  template=PLOTLY_TPL, title="취약 그룹 키워드 Top 15")
                                fig_kbar.update_layout(height=400, yaxis=dict(autorange="reversed"))
                                st.plotly_chart(fig_kbar, use_container_width=True)

                        with kw_col:
                            kw_df = pd.DataFrame(weak_kws[:20], columns=["키워드", "빈도"])
                            kw_df["유형"] = kw_df["키워드"].apply(
                                lambda x: "😡 불친절" if any(r in x for r in RUDE_KEYWORDS)
                                else "⚠️ 부정" if any(n in x for n in NEGATIVE_KEYWORDS)
                                else "일반")
                            st.dataframe(kw_df, use_container_width=True, height=400, hide_index=True)

                        # 감성 분류
                        sentiments = [classify_sentiment_keyword(t) for t in weak_voc]
                        sent_cnt = Counter(sentiments)
                        st.markdown("---")
                        sc1, sc2, sc3 = st.columns(3)
                        with sc1:
                            st.metric("😊 긍정", f"{sent_cnt.get('긍정', 0):,}건")
                        with sc2:
                            st.metric("😠 불만", f"{sent_cnt.get('불만', 0):,}건",
                                      delta=f"{sent_cnt.get('불만', 0)/max(len(weak_voc),1)*100:.1f}%",
                                      delta_color="inverse")
                        with sc3:
                            st.metric("😡 불친절", f"{sent_cnt.get('불친절', 0):,}건",
                                      delta=f"{sent_cnt.get('불친절', 0)/max(len(weak_voc),1)*100:.1f}%",
                                      delta_color="inverse")
                else:
                    st.info("취약 그룹에 서술 의견이 없습니다.")


# ─────────────────────────────────────────────────────────────
#  TAB 4: VOC 텍스트 분석
# ─────────────────────────────────────────────────────────────
with tab4:
    if not has_voc:
        st.warning("서술 의견(VOC) 컬럼이 없습니다.")
    else:
        all_voc = df_f[VOC_COL].dropna().astype(str)
        all_voc = all_voc[all_voc.str.strip() != ""].tolist()
        n_voc = len(all_voc)

        if n_voc == 0:
            st.info("서술 의견 데이터가 없습니다.")
        else:
            st.markdown(f'<p class="sec-head">📊 전체 VOC 개요 ({n_voc:,}건)</p>', unsafe_allow_html=True)

            # ── 전체 감성 분류 (키워드 기반 — 빠름) ──
            with st.spinner("VOC 감성 분류 중…"):
                all_sentiments = [classify_sentiment_keyword(t) for t in all_voc]
            sent_total = Counter(all_sentiments)

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("💬 전체 VOC", f"{n_voc:,}건")
            with s2:
                st.metric("😊 긍정", f"{sent_total.get('긍정',0):,}건",
                          delta=f"{sent_total.get('긍정',0)/n_voc*100:.1f}%")
            with s3:
                st.metric("😠 불만", f"{sent_total.get('불만',0):,}건",
                          delta=f"{sent_total.get('불만',0)/n_voc*100:.1f}%", delta_color="inverse")
            with s4:
                st.metric("😡 불친절", f"{sent_total.get('불친절',0):,}건",
                          delta=f"{sent_total.get('불친절',0)/n_voc*100:.1f}%", delta_color="inverse")

            # ── 감성 비율 파이 ──
            fig_sent = px.pie(
                names=list(sent_total.keys()), values=list(sent_total.values()),
                color_discrete_sequence=[C["teal"], C["gold"], C["red"]],
                hole=0.5, template=PLOTLY_TPL, title="VOC 감성 분포",
            )
            fig_sent.update_traces(textinfo="percent+label", textfont_size=13)
            fig_sent.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20),
                                    title_font=dict(size=14, color=C["navy"]))
            st.plotly_chart(fig_sent, use_container_width=True)

            st.markdown("---")

            # ── 전체 키워드 (정규식 — 빠름) ──
            st.markdown(f'<p class="sec-head">☁️ 전체 VOC 키워드 (빈도 기반)</p>', unsafe_allow_html=True)
            with st.spinner("키워드 추출 중…"):
                all_kws = extract_keywords_regex(all_voc, top_n=40)

            if all_kws:
                wc_c, kw_c = st.columns([3, 2])
                with wc_c:
                    img = make_wordcloud(all_kws)
                    if img:
                        st.image(img, use_container_width=True, caption="전체 VOC 워드클라우드")
                    else:
                        st.info("워드클라우드 생성에 한글 폰트가 필요합니다.")
                with kw_c:
                    kw_all_df = pd.DataFrame(all_kws[:25], columns=["키워드", "빈도"])
                    kw_all_df["유형"] = kw_all_df["키워드"].apply(
                        lambda x: "😡 불친절" if any(r in x for r in RUDE_KEYWORDS)
                        else "⚠️ 부정" if any(n in x for n in NEGATIVE_KEYWORDS)
                        else "✅ 일반")
                    st.dataframe(kw_all_df, use_container_width=True, height=440, hide_index=True)

            st.markdown("---")

            # ── 범주별 VOC 비교 ──
            if available_cats:
                st.markdown(f'<p class="sec-head">📊 범주별 불만 VOC 비율</p>', unsafe_allow_html=True)
                voc_cat = st.selectbox("기준 범주", available_cats, key="voc_cat")

                df_voc_analysis = df_f[[voc_cat, VOC_COL]].dropna()
                df_voc_analysis = df_voc_analysis[df_voc_analysis[VOC_COL].astype(str).str.strip() != ""]
                df_voc_analysis["_감성"] = df_voc_analysis[VOC_COL].apply(classify_sentiment_keyword)

                voc_cross = pd.crosstab(df_voc_analysis[voc_cat], df_voc_analysis["_감성"])
                for col in ["긍정", "불만", "불친절"]:
                    if col not in voc_cross.columns:
                        voc_cross[col] = 0
                voc_cross = voc_cross[["긍정", "불만", "불친절"]]
                voc_cross["합계"] = voc_cross.sum(axis=1)
                voc_cross["불만률(%)"] = ((voc_cross["불만"] + voc_cross["불친절"]) / voc_cross["합계"] * 100).round(1)
                voc_cross = voc_cross.sort_values("불만률(%)", ascending=False)

                vc_l, vc_r = st.columns([3, 2])
                with vc_l:
                    fig_neg_rate = px.bar(
                        voc_cross.reset_index(), x=voc_cat, y="불만률(%)",
                        color="불만률(%)", color_continuous_scale="Reds",
                        text="불만률(%)", template=PLOTLY_TPL,
                        title=f"{voc_cat}별 부정 VOC 비율",
                    )
                    fig_neg_rate.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_neg_rate.update_layout(
                        height=400, margin=dict(t=50, b=80, l=60, r=20),
                        xaxis_tickangle=-20, title_font=dict(size=14, color=C["navy"]),
                    )
                    st.plotly_chart(fig_neg_rate, use_container_width=True)
                with vc_r:
                    st.dataframe(voc_cross, use_container_width=True, height=380)

            # ── 불만 VOC 상세 리스트 ──
            st.markdown("---")
            st.markdown(f'<p class="sec-head">🚨 불만/불친절 VOC 상세 (최근 20건)</p>', unsafe_allow_html=True)
            neg_voc = [t for t in all_voc if classify_sentiment_keyword(t) in ("불만", "불친절")]
            if neg_voc:
                for t in neg_voc[:20]:
                    display_text = str(t)
                    for kw in RUDE_KEYWORDS + NEGATIVE_KEYWORDS:
                        if kw in display_text:
                            display_text = display_text.replace(
                                kw, f'<b style="color:{C["red"]}; background:#fce4ec; '
                                f'padding:0 3px; border-radius:3px;">{kw}</b>')
                    card_cls = "card-red"
                    st.markdown(f'<div class="{card_cls}">{display_text}</div>', unsafe_allow_html=True)
                if len(neg_voc) > 20:
                    st.caption(f"※ 전체 {len(neg_voc):,}건 중 20건만 표시")
            else:
                st.success("부정 VOC가 없습니다.")

            # ── 다운로드 ──
            if neg_voc:
                neg_df = df_f[df_f[VOC_COL].apply(
                    lambda x: classify_sentiment_keyword(x) in ("불만", "불친절")
                )].copy()
                display_cols = [c for c in available_cats + [TOTAL_SCORE_COL, VOC_COL] if c in neg_df.columns]
                st.download_button(
                    "📥 부정 VOC 엑셀 다운로드", df_to_excel(neg_df[display_cols]),
                    file_name="부정VOC_리스트.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

# ── 원본 데이터 미리보기 ──
with st.expander("📄 원본 데이터 미리보기 (상위 30건)"):
    st.dataframe(df_f.head(30), use_container_width=True)
