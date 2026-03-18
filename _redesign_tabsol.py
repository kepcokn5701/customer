# -*- coding: utf-8 -*-
"""
3-Level UX 통합 리디자인:
  - tab_sol 전체 교체 (Level 1/2/3)
  - tab_ai 섹션 ⑦ (트리맵) + 섹션 ⑤ (페르소나별 서비스 경로) 삭제
"""
import ast, io, sys
sys.path.insert(0, ".")
from _new_tabsol import TAB_SOL_CODE

src = "cs_dashboard.py"
with io.open(src, encoding="utf-8") as f:
    lines = f.readlines()

def find_line(marker, start=0):
    for i in range(start, len(lines)):
        if marker in lines[i]:
            return i
    return None

# ─── 경계 탐지 ─────────────────────────────────────────────
tabsol_sep  = find_line("#  TAB SOL  지사별 맞춤")   # "# TAB SOL" 설명 줄
tabsol_end  = find_line("#  TAB 10  CXO 딥 인사이트") # 다음 탭 헤더 줄
tab10_sep   = tabsol_end - 1                           # "# ────" separator

tabsol_start = tabsol_sep - 1  # "# ────" separator 앞 줄 포함

sec7_start  = find_line("# ── ⑦ 리스크 집중 구역 도출")
tab5_start  = find_line("with tab5:")

if None in (tabsol_sep, tabsol_end, sec7_start, tab5_start):
    print("ERROR: marker not found"); sys.exit(1)

# tab5 바로 앞의 "# ────" 줄 (TAB 5 블록 separator 시작)
# "# ──...TAB 5..." 줄을 찾아 그 직전까지 삭제
tab5_hdr = find_line("#  TAB 5  민원 조기 경보")
sec5_end  = tab5_hdr - 1  # "# ────" separator 줄 — 이 줄까지 삭제(exclusive: tab5_hdr-1 포함)

print(f"tab_sol block  : lines {tabsol_start+1} to {tabsol_end-1+1}")
print(f"sec7+sec5 del  : lines {sec7_start+1} to {sec5_end+1}")

# ─── 변경 적용 ─────────────────────────────────────────────
# 1) tab_sol 블록 교체 (tabsol_start ~ tabsol_end-1 를 새 코드로)
new_tabsol_lines = [TAB_SOL_CODE + "\n\n"]
new_lines = lines[:tabsol_start] + new_tabsol_lines + lines[tabsol_end:]

# 2) sec7+sec5 삭제 (인덱스는 new_lines 기준, tab_sol 교체 전과 동일 위치)
#    sec7_start < tabsol_start 이므로 인덱스 변동 없음
del new_lines[sec7_start : sec5_end + 1]

# ─── 검증 ─────────────────────────────────────────────────
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
