# -*- coding: utf-8 -*-
"""
1. st.tabs() 에서 tab_ai, tab11 제거 / tab_sol 레이블 변경
2. TAB AI 블록 삭제 (lines 1866~2171)
3. TAB 11 블록 삭제 (lines 3331~EOF)
"""
import ast, io, sys

src = "cs_dashboard.py"
with io.open(src, encoding="utf-8") as f:
    lines = f.readlines()

def find_line(marker, start=0):
    for i in range(start, len(lines)):
        if marker in lines[i]:
            return i
    return None

# ── 1. st.tabs() 수정 ───────────────────────────────────────
OLD_TABS = (
    'tab1, tab3, tab_ai, tab5, tab_sol, tab10, tab11 = st.tabs([\n'
    '    "\U0001f4ca  \uad6c\uac04\ubcc4 \ube44\uc911 \xb7 \uc885\ud569 \ud604\ud669",\n'
    '    "\U0001f4e1  \uacc4\uc57d\uc885\ubcc4 \xb7 \uc811\uc218\uccb4\ub110\ubcc4 \xb7 \uc5c5\ubb34\uc720\ud615\ubcc4 \ubd84\uc11d",\n'
    '    "\U0001f916  AI \uad50\ucc28\ubd84\uc11d",\n'
    '    "\U0001f3af  \ubbfc\uc6d0 \uc870\uae30 \uacbd\ubcf4 \uc2dc\uc2a4\ud15c",\n'
    '    "\U0001f3e2  \uc9c0\uc0ac\ubcc4 \ub9de\ucda4 \uc194\ub8e8\uc158 \ub3c4\ucd9c \uc5d4\uc9c4",\n'
    '    "\U0001f9e0  CXO \ub51d \uc778\uc0ac\uc774\ud2b8",\n'
    '    "\U0001f3e2  \uc9c0\uc0ac \ub9de\ucda4\ud615 CS \uc194\ub8e8\uc158",\n'
    '])\n'
)
NEW_TABS = (
    'tab1, tab3, tab5, tab_sol, tab10 = st.tabs([\n'
    '    "\U0001f4ca  \uad6c\uac04\ubcc4 \ube44\uc911 \xb7 \uc885\ud569 \ud604\ud669",\n'
    '    "\U0001f4e1  \uacc4\uc57d\uc885\ubcc4 \xb7 \uc811\uc218\uccb4\ub110\ubcc4 \xb7 \uc5c5\ubb34\uc720\ud615\ubcc4 \ubd84\uc11d",\n'
    '    "\U0001f3af  \ubbfc\uc6d0 \uc870\uae30 \uacbd\ubcf4 \uc2dc\uc2a4\ud15c",\n'
    '    "\U0001f3e2  \uc9c0\uc0ac \ub9de\ucda4\ud615 CS \uc194\ub8e8\uc158",\n'
    '    "\U0001f9e0  CXO \ub51d \uc778\uc0ac\uc774\ud2b8",\n'
    '])\n'
)

# 직접 라인 단위로 교체 (유니코드 이슈 방지)
tabs_line = find_line("tab1, tab3, tab_ai, tab5, tab_sol, tab10, tab11 = st.tabs")
if tabs_line is None:
    print("ERROR: st.tabs() not found"); sys.exit(1)

# st.tabs 블록 끝 찾기 ("])" 로 끝나는 줄)
tabs_end = tabs_line
while tabs_end < len(lines) and "])" not in lines[tabs_end]:
    tabs_end += 1
print(f"st.tabs() block: lines {tabs_line+1} to {tabs_end+1}")

new_tabs_lines = [
    'tab1, tab3, tab5, tab_sol, tab10 = st.tabs([\n',
    '    "\U0001f4ca  \uad6c\uac04\ubcc4 \ube44\uc911 \xb7 \uc885\ud569 \ud604\ud669",\n',
    '    "\U0001f4e1  \uacc4\uc57d\uc885\ubcc4 \xb7 \uc811\uc218\uccb4\ub110\ubcc4 \xb7 \uc5c5\ubb34\uc720\ud615\ubcc4 \ubd84\uc11d",\n',
    '    "\U0001f3af  \ubbfc\uc6d0 \uc870\uae30 \uacbd\ubcf4 \uc2dc\uc2a4\ud15c",\n',
    '    "\U0001f3e2  \uc9c0\uc0ac \ub9de\ucda4\ud615 CS \uc194\ub8e8\uc158",\n',
    '    "\U0001f9e0  CXO \ub51d \uc778\uc0ac\uc774\ud2b8",\n',
    '])\n',
]

new_lines = lines[:tabs_line] + new_tabs_lines + lines[tabs_end + 1:]

# ── 2. TAB AI 블록 삭제 ────────────────────────────────────
# 새 lines 기준으로 재탐색
def find_in(lst, marker, start=0):
    for i in range(start, len(lst)):
        if marker in lst[i]:
            return i
    return None

ai_sep   = find_in(new_lines, "#  TAB AI  AI \uad50\ucc28\ubd84\uc11d")
tab5_hdr = find_in(new_lines, "#  TAB 5  \ubbfc\uc6d0 \uc870\uae30 \uacbd\ubcf4")
if ai_sep is None or tab5_hdr is None:
    print("ERROR: TAB AI / TAB 5 boundaries not found"); sys.exit(1)
ai_start = ai_sep - 1  # "# ────" separator
ai_end   = tab5_hdr - 1  # TAB 5 "# ────" separator (exclusive boundary)
print(f"TAB AI block   : lines {ai_start+1} to {ai_end}")
del new_lines[ai_start:ai_end]

# ── 3. TAB 11 블록 삭제 ────────────────────────────────────
tab11_sep = find_in(new_lines, "#  TAB 11  \uc9c0\uc0ac \ub9de\ucda4\ud615 CS \uc194\ub8e8\uc158")
if tab11_sep is None:
    print("ERROR: TAB 11 boundary not found"); sys.exit(1)
tab11_start = tab11_sep - 1  # "# ────" separator
print(f"TAB 11 block   : lines {tab11_start+1} to {len(new_lines)}")
del new_lines[tab11_start:]

# ── 4. 검증 & 저장 ────────────────────────────────────────
result = "".join(new_lines)
try:
    ast.parse(result)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR line {e.lineno}: {e.msg}")
    err_lines = result.splitlines()
    for ln in err_lines[max(0, e.lineno-4):e.lineno+3]:
        print(repr(ln))
    sys.exit(1)

with io.open(src, "w", encoding="utf-8") as f:
    f.write(result)
print("Done.")
