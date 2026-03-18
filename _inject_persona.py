# -*- coding: utf-8 -*-
"""
1. Remove 지사별 페르소나 section from tab10
2. Insert single-office persona card into tab_sol (below header card)
"""
import ast, io, sys, textwrap

src = "cs_dashboard.py"
with io.open(src, encoding="utf-8") as f:
    lines = f.readlines()

# ── 1. Find persona block in tab10 to DELETE ─────────────────
del_start = None
del_end   = None
for i, ln in enumerate(lines):
    if "# 3. 지사별 페르소나 (OOS 제외)" in ln and del_start is None:
        for j in range(i - 1, max(i - 5, 0), -1):
            if 'st.markdown("---")' in lines[j]:
                del_start = j
                break
for i, ln in enumerate(lines):
    if del_start and i > del_start and "지사, VOC, 종합점수 컬럼이 모두 필요합니다." in ln:
        del_end = i + 1
        break

if del_start is None or del_end is None:
    print(f"ERROR: persona block not found ({del_start}, {del_end})"); sys.exit(1)
print(f"Delete persona from tab10: lines {del_start+1} to {del_end+1}")

# ── 2. Build single-office persona block ─────────────────────
# Use list of lines to avoid nested quote hell
block_lines = [
    "            # ══════════════════════════════════════════════════\n",
    "            # 1b. 지사 페르소나 카드 (선택 지사 단독)\n",
    "            # ══════════════════════════════════════════════════\n",
    "            _ps_office_col = M.get('office')\n",
    "            if _ps_office_col and M.get('voc') and '_점수100' in df_pure.columns:\n",
    "                _ps_sub = df_pure[df_pure[_ps_office_col] == _sel_off]\n",
    "                if len(_ps_sub) >= 3:\n",
    "                    _ps_n   = len(_ps_sub)\n",
    "                    _ps_avg = _ps_sub['_점수100'].mean()\n",
    "\n",
    "                    # 부정 VOC 분석\n",
    "                    _ps_voc_vals  = _ps_sub[M['voc']].dropna()\n",
    "                    _ps_voc_vals  = _ps_voc_vals[~_ps_voc_vals.isin(['응답없음', 'nan', ''])]\n",
    "                    _ps_neg_cnt   = 0\n",
    "                    _ps_neg_kws   = Counter()\n",
    "                    _ps_cause_cnt = Counter()\n",
    "                    for _psv in _ps_voc_vals:\n",
    "                        _pss = str(_psv)\n",
    "                        _ps_is_neg = False\n",
    "                        for _pskw in NEGATIVE_KEYWORDS:\n",
    "                            if _pskw in _pss:\n",
    "                                _psw = _pss[max(0, _pss.find(_pskw)-15):_pss.find(_pskw)+len(_pskw)+15]\n",
    "                                if not any(_pspw in _psw for _pspw in POSITIVE_CONTEXT):\n",
    "                                    _ps_neg_kws[_pskw] += 1\n",
    "                                    _ps_is_neg = True\n",
    "                        if _ps_is_neg:\n",
    "                            _ps_neg_cnt += 1\n",
    "                            for _ps_cause, _ps_ckws in _CAUSE_TAGS.items():\n",
    "                                if any(_pskw in _pss and not any(\n",
    "                                    _pspw in _pss[max(0,_pss.find(_pskw)-15):_pss.find(_pskw)+len(_pskw)+15]\n",
    "                                    for _pspw in POSITIVE_CONTEXT) for _pskw in _ps_ckws if _pskw in _pss):\n",
    "                                    _ps_cause_cnt[_ps_cause] += 1\n",
    "                                    break\n",
    "                    _ps_neg_ratio = _ps_neg_cnt / max(_ps_n, 1) * 100\n",
    "\n",
    "                    # 계약종 구성\n",
    "                    _ps_ct_dist = ''\n",
    "                    if '_계약종별' in _ps_sub.columns:\n",
    "                        _ps_ct = _ps_sub['_계약종별'].value_counts(normalize=True).head(2)\n",
    "                        _ps_ct_dist = ', '.join(f'{k} {v*100:.0f}%' for k, v in _ps_ct.items())\n",
    "\n",
    "                    # 업무 구성\n",
    "                    _ps_biz_dist = ''\n",
    "                    if M.get('business') and M['business'] in _ps_sub.columns:\n",
    "                        _ps_biz = _ps_sub[M['business']].value_counts(normalize=True).head(2)\n",
    "                        _ps_biz_dist = ', '.join(f'{k} {v*100:.0f}%' for k, v in _ps_biz.items())\n",
    "\n",
    "                    # 원인·키워드\n",
    "                    _ps_cause_str = ', '.join(f'{k}({v})' for k, v in _ps_cause_cnt.most_common(2)) or '없음'\n",
    "                    _ps_neg_str   = ', '.join(f'{k}({v})' for k, v in _ps_neg_kws.most_common(3)) or '없음'\n",
    "\n",
    "                    # 약점 항목\n",
    "                    _ps_weak = []\n",
    "                    for _psc in df_pure.columns:\n",
    "                        if _psc.startswith('_') or _psc == M['score']:\n",
    "                            continue\n",
    "                        _ps_ov = pd.to_numeric(_ps_sub[_psc], errors='coerce').dropna()\n",
    "                        _ps_av = pd.to_numeric(df_pure[_psc], errors='coerce').dropna()\n",
    "                        if len(_ps_ov) > 2 and len(_ps_av) > 5:\n",
    "                            if _ps_ov.mean() - _ps_av.mean() < -3:\n",
    "                                _ps_weak.append(f\"{_psc}({_ps_ov.mean()-_ps_av.mean():+.1f})\")\n",
    "\n",
    "                    # 태그\n",
    "                    _ps_tags = []\n",
    "                    if _ps_neg_ratio >= 10: _ps_tags.append('🔴 부정VOC 다발')\n",
    "                    elif _ps_neg_ratio <= 4: _ps_tags.append('🟢 안정 지사')\n",
    "                    if _ps_avg < 92:   _ps_tags.append('⚠️ 점수 하위권')\n",
    "                    elif _ps_avg >= 95: _ps_tags.append('⭐ 우수 지사')\n",
    "                    _ps_tag_str = ' '.join(_ps_tags) if _ps_tags else '보통'\n",
    "\n",
    "                    # Do/Don't 처방\n",
    "                    _ps_mc = _ps_cause_str.split(',')[0].split('(')[0].strip() if _ps_cause_str != '없음' else ''\n",
    "                    _ps_rxmap = {\n",
    "                        '절차 복잡·불편':   ('원스톱 처리 체계 구축 — 한 번에 끝내기', '고객을 여러 부서로 돌려보내기'),\n",
    "                        '처리 지연·느림':   ('접수 후 24시간 내 진행상황 문자 발송',   '처리 완료 전까지 연락 없이 방치'),\n",
    "                        '처리 오류·부정확': ('처리 전 체크리스트 확인 — 1회 완결 처리', '확인 없이 신속 처리에만 집중'),\n",
    "                        '직원 태도·불친절': ('경청→공감→해결 3단계 응대 훈련 실시',    '감정 대응 없이 업무만 처리'),\n",
    "                        '요금·제도 불만':   ('요금 구조를 고객 눈높이 비유·시각자료로 설명', '매뉴얼 그대로 읽어주는 기계적 응대'),\n",
    "                        '정전·안전 관련':   ('예정 정전 72시간 전 사전 알림 체계 구축', '예고 없는 정전 및 복구 후 미통보'),\n",
    "                    }\n",
    "                    _ps_do, _ps_dont = _ps_rxmap.get(_ps_mc, ('고객 맞춤 응대 프로세스 정비', '일괄적·기계적 대응'))\n",
    "                    _ps_weak_str = '<br>📉 <b>약점항목:</b> ' + ', '.join(_ps_weak) if _ps_weak else ''\n",
    "\n",
    "                    st.markdown(\n",
    "                        '<div style=\"display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;\">'\n",
    "                        f'<div style=\"background:#f3f4f6;border-radius:10px;padding:14px 16px;font-size:0.88em;\">'\n",
    "                        f'<b>🎭 {_sel_off} 페르소나</b> &nbsp; {_ps_tag_str}<br><br>'\n",
    "                        f'📊 <b>고객 구성:</b> {_ps_ct_dist or \"데이터 없음\"}<br>'\n",
    "                        f'📋 <b>주요 업무:</b> {_ps_biz_dist or \"데이터 없음\"}<br>'\n",
    "                        f'🔥 <b>부정 원인:</b> {_ps_cause_str}<br>'\n",
    "                        f'🔑 <b>부정 키워드:</b> {_ps_neg_str}<br>'\n",
    "                        f'📉 <b>부정 비율:</b> {_ps_neg_ratio:.1f}% ({_ps_neg_cnt}/{_ps_n}건)'\n",
    "                        f'{_ps_weak_str}</div>'\n",
    "                        f'<div style=\"background:#f3f4f6;border-radius:10px;padding:14px 16px;font-size:0.88em;\">'\n",
    "                        '<b>💊 CS 처방전</b><br><br>'\n",
    "                        f'<span style=\"color:#1b5e20\">✅ <b>Do:</b> {_ps_do}</span><br><br>'\n",
    "                        f'<span style=\"color:#b71c1c\">❌ <b>Don\\'t:</b> {_ps_dont}</span>'\n",
    "                        '</div></div>',\n",
    "                        unsafe_allow_html=True)\n",
    "\n",
]

PERSONA_BLOCK = "".join(block_lines)

# ── 3. Find insertion point in tab_sol ───────────────────────
insert_idx = None
for i, ln in enumerate(lines):
    if "# 2+3." in ln and "업무 강약점" in ln and "페르소나" in ln:
        insert_idx = i
        break

if insert_idx is None:
    print("ERROR: tab_sol insertion marker '# 2+3.' not found"); sys.exit(1)
print(f"Insert persona block before line {insert_idx+1}")

# ── 4. Apply changes ─────────────────────────────────────────
new_lines = lines[:insert_idx] + block_lines + lines[insert_idx:]

shift = len(block_lines)
new_del_start = del_start + shift
new_del_end   = del_end   + shift
print(f"Delete shifted to: lines {new_del_start+1} to {new_del_end+1}")

del new_lines[new_del_start:new_del_end]

# ── 5. Write & verify ────────────────────────────────────────
result = "".join(new_lines)
try:
    ast.parse(result)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}"); sys.exit(1)

with io.open(src, "w", encoding="utf-8") as f:
    f.write(result)
print("Done.")
