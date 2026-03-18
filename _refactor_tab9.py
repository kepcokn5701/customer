# -*- coding: utf-8 -*-
"""
Remove tab9, move 지사별 강점·약점 section into tab_sol (after 페르소나 미스매치).
"""
import ast, io, sys

src = "cs_dashboard.py"
with io.open(src, encoding="utf-8") as f:
    lines = f.readlines()

# ── 1. Update st.tabs() ──────────────────────────────────────
OLD_TABS = (
    'tab1, tab3, tab_ai, tab5, tab9, tab_sol, tab10, tab11 = st.tabs([\n'
)
NEW_TABS = (
    'tab1, tab3, tab_ai, tab5, tab_sol, tab10, tab11 = st.tabs([\n'
)
OLD_TAB_LABEL = '    "🔬  지사 심층 · 패턴",\n'

text = "".join(lines)
if OLD_TABS not in text:
    print("ERROR: st.tabs() signature not found"); sys.exit(1)
text = text.replace(OLD_TABS, NEW_TABS, 1)
text = text.replace(OLD_TABS.replace("tab9, ", ""), NEW_TABS, 1)  # no-op if already replaced
text = text.replace(OLD_TAB_LABEL, "", 1)

lines = text.splitlines(keepends=True)

# ── 2. Find tab9 block boundaries ────────────────────────────
# start: "# TAB 9" header separator line
# end:   line just before "# TAB SOL" header separator line
tab9_start = None
tab9_end = None
for i, ln in enumerate(lines):
    if "#  TAB 9  지사 심층" in ln:
        tab9_start = i - 1   # include the "# ───" separator line above
        break
for i, ln in enumerate(lines):
    if "#  TAB SOL  지사별 맞춤" in ln:
        tab9_end = i - 1     # stop before the "# ───" separator of TAB SOL
        break

if tab9_start is None or tab9_end is None:
    print(f"ERROR: tab9 boundaries not found ({tab9_start}, {tab9_end})"); sys.exit(1)

print(f"tab9 block: lines {tab9_start+1} to {tab9_end+1}")

# ── 3. Build the 강점·약점 code to insert into tab_sol ────────
# (12-space indent — matches tab_sol else: block level)
STRENGTH_BLOCK = """\
            # ══════════════════════════════════════════════════
            # 3b. 지사별 강점 · 약점 항목 (전체 지사 비교)
            # ══════════════════════════════════════════════════
            if individual_scores:
                st.markdown('<p class="sec-head">💪 지사별 강점 · 약점 항목</p>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="card-blue"><b>📌 분석 방법</b> — 각 지사의 개별항목 점수를 전체 평균과 비교하여 '
                    '가장 잘하는 항목(강점)과 가장 부족한 항목(약점)을 자동 추출합니다.</div>',
                    unsafe_allow_html=True)

                _sw_overall = df_f[individual_scores].mean()
                _sw_detail  = df_f.groupby(M["office"])[individual_scores].mean()
                _sw_detail  = _sw_detail.reindex(_sort_offices(_sw_detail.index.tolist()))

                _sw_ofc_means   = df_f.groupby(M["office"])["_점수100"].mean()
                _sw_offices_srt = _sw_ofc_means.reindex(_sort_offices(_sw_ofc_means.index.tolist()))
                _sw_cols = st.columns(min(3, len(_sw_offices_srt)))

                for _swi, (_swo, _) in enumerate(_sw_offices_srt.items()):
                    if _swo not in _sw_detail.index:
                        continue
                    _swr   = _sw_detail.loc[_swo]
                    _swd   = _swr - _sw_overall
                    _swstr = _swd.idxmax()
                    _swwk  = _swd.idxmin()
                    _swavg = df_f[df_f[M["office"]] == _swo]["_점수100"].mean()
                    _swcc  = ("card-red" if _swavg < avg_score_100 - 3
                              else "card-teal" if _swavg >= avg_score_100
                              else "card")
                    with _sw_cols[_swi % len(_sw_cols)]:
                        st.markdown(
                            f'<div class="{_swcc}">'
                            f'<b>🏢 {_swo}</b> (종합 {_swavg:.1f}점)<br><br>'
                            f'💪 강점: <b>{_swstr}</b> (평균 대비 <span style="color:{C["green"]}">{_swd[_swstr]:+.1f}</span>)<br>'
                            f'⚠️ 약점: <b>{_swwk}</b> (평균 대비 <span style="color:{C["red"]}">{_swd[_swwk]:+.1f}</span>)'
                            f'</div>', unsafe_allow_html=True)

"""

# ── 4. Find insertion point in tab_sol ───────────────────────
# Insert AFTER the st.markdown("---") following scatter plot,
# BEFORE the "# 4+5. 하단" comment block.
# Unique marker: "            # 4+5." comment line
insert_idx = None
for i, ln in enumerate(lines):
    if "# 4+5." in ln and "하단" in ln and "벤치마킹" in ln:
        insert_idx = i   # insert before this line
        break

if insert_idx is None:
    print("ERROR: insertion marker '# 4+5. 하단' not found"); sys.exit(1)

print(f"Inserting strength block before line {insert_idx+1}")

# ── 5. Apply changes ─────────────────────────────────────────
# a) Insert the block
new_lines = lines[:insert_idx] + [STRENGTH_BLOCK] + lines[insert_idx:]

# b) Remove tab9 block (re-find after insertion shifted indices)
# Recalculate: tab9 block is AFTER the tab_sol area now? No—tab9 comes BEFORE tab_sol
# insertion was inside tab_sol which is AFTER tab9, so tab9 indices are unchanged
del new_lines[tab9_start:tab9_end]

# ── 6. Write & verify ────────────────────────────────────────
result = "".join(new_lines)
try:
    ast.parse(result)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}"); sys.exit(1)

with io.open(src, "w", encoding="utf-8") as f:
    f.write(result)
print("Done.")
