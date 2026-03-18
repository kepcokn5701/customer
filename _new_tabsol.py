# -*- coding: utf-8 -*-
# 새 tab_sol 블록 내용 — _redesign_tabsol.py 에서 읽어서 사용
TAB_SOL_CODE = """
# ─────────────────────────────────────────────────────────────
#  TAB SOL  지사별 정밀 진단 · AI 심층 분석 (3-Level UX)
# ─────────────────────────────────────────────────────────────
with tab_sol:

    # ══════════════════════════════════════════════════════════
    # LEVEL 1 — 본부 전체 조망 (Discovery)
    # ══════════════════════════════════════════════════════════
    st.markdown('<p class="sec-head">🔭 본부 전체 CS 현황 — 지사별 정밀 진단 엔진</p>', unsafe_allow_html=True)

    if df_f.empty or avg_score_100 is None:
        st.info("데이터를 먼저 업로드하세요.")
    else:
        # ── KPI 상태 바 ────────────────────────────────────────
        _sol_risk_cnt = 0
        if M.get("business"):
            _sol_biz_avg  = df_f.groupby(M["business"])["_점수100"].mean()
            _sol_risk_cnt = int((_sol_biz_avg < avg_score_100).sum())

        _kpi_html = (
            '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:18px;">'
            '<div style="background:#1a237e;color:white;border-radius:10px;padding:14px;text-align:center;">'
            '<div style="font-size:0.78em;opacity:.8">본부 평균 만족도</div>'
            f'<div style="font-size:2em;font-weight:900">{avg_score_100:.1f}<span style="font-size:.45em">점</span></div>'
            '</div>'
            '<div style="background:#1565c0;color:white;border-radius:10px;padding:14px;text-align:center;">'
            '<div style="font-size:0.78em;opacity:.8">전체 응답 건수</div>'
            f'<div style="font-size:2em;font-weight:900">{len(df_f):,}<span style="font-size:.45em">건</span></div>'
            '</div>'
            '<div style="background:#b71c1c;color:white;border-radius:10px;padding:14px;text-align:center;">'
            '<div style="font-size:0.78em;opacity:.8">평균 이하 업무 수</div>'
            f'<div style="font-size:2em;font-weight:900">{_sol_risk_cnt}<span style="font-size:.45em">건</span></div>'
            '</div></div>'
        )
        st.markdown(_kpi_html, unsafe_allow_html=True)

        # ── 트리맵: 계약종별 × 업무유형 ───────────────────────
        if M.get("contract") and M.get("business"):
            st.markdown("##### 🗺️ 계약종별 × 업무유형 리스크 맵 (면적 = 건수, 색상 = 만족도)")
            st.caption("면적이 크고 붉은 영역이 본부 점수를 갉아먹는 **최우선 개선 타겟**입니다.")
            _tm_df = df_f.groupby([M["contract"], M["business"]])["_점수100"].agg(
                ["mean", "count"]).reset_index()
            _tm_df.columns = ["계약종", "업무유형", "평균만족도", "응답수"]
            _tm_df = _tm_df[_tm_df["응답수"] >= 3]
            if not _tm_df.empty:
                _tm_min = float(_tm_df["평균만족도"].min())
                _tm_max = float(_tm_df["평균만족도"].max())
                fig_tm = px.treemap(
                    _tm_df, path=["계약종", "업무유형"],
                    values="응답수", color="평균만족도",
                    color_continuous_scale="RdYlGn",
                    range_color=[max(60, _tm_min - 2), min(100, _tm_max + 2)],
                    hover_data={"평균만족도": ":.1f", "응답수": True},
                    template=PLOTLY_TPL,
                )
                fig_tm.update_traces(
                    texttemplate="<b>%{label}</b><br>%{color:.1f}점<br>%{value}건",
                    hovertemplate="%{label}<br>평균: %{color:.1f}점<br>건수: %{value}건<extra></extra>")
                fig_tm.update_layout(height=440, margin=dict(t=10, b=10, l=10, r=10),
                                     coloraxis_colorbar=dict(title="만족도", len=0.6))
                st.plotly_chart(fig_tm, use_container_width=True)

        st.markdown("---")

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

        # ── Golden Line: 점수 추이 ──────────────────────────────
        if M.get("date") and M["date"] in df_f.columns:
            _trend_df  = df_f.groupby([M["date"], M["office"]])["_점수100"].mean().reset_index()
            _trend_df.columns  = ["날짜", "지사", "만족도"]
            _trend_all = df_f.groupby(M["date"])["_점수100"].mean().reset_index()
            _trend_all.columns = ["날짜", "본부평균"]
            _trend_sel = _trend_df[_trend_df["지사"] == _sel_off].sort_values("날짜")
            if len(_trend_sel) >= 2:
                st.markdown("##### 📈 점수 추이 — Golden Line")
                _trend_all_s = _trend_all.sort_values("날짜")
                _y_min = max(60, min(_trend_sel["만족도"].min(), _trend_all_s["본부평균"].min()) - 3)
                _y_max = min(100, max(_trend_sel["만족도"].max(), _trend_all_s["본부평균"].max()) + 3)
                fig_gl = go.Figure()
                fig_gl.add_trace(go.Scatter(
                    x=_trend_sel["날짜"], y=_trend_sel["만족도"],
                    mode="lines+markers", name=_sel_off,
                    line=dict(color="#FFD700", width=3),
                    marker=dict(size=8, color="#FFD700"),
                    hovertemplate="%{x}<br>%{y:.1f}점<extra></extra>"))
                fig_gl.add_trace(go.Scatter(
                    x=_trend_all_s["날짜"], y=_trend_all_s["본부평균"],
                    mode="lines", name="본부 평균",
                    line=dict(color="#90a4ae", width=1.5, dash="dash"),
                    hovertemplate="%{x}<br>본부 평균: %{y:.1f}점<extra></extra>"))
                fig_gl.update_layout(
                    template=PLOTLY_TPL, height=280,
                    margin=dict(t=10, b=40, l=20, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    yaxis=dict(title="만족도", range=[_y_min, _y_max]))
                st.plotly_chart(fig_gl, use_container_width=True)

        # ── 업무 × 채널 히트맵 (지사 특화) ──────────────────────
        _ch_col = M.get("channel")
        if M.get("business") and _ch_col and _ch_col in df_f.columns:
            st.markdown("##### 🔥 업무 × 채널 리스크 히트맵 — " + _sel_off)
            st.caption("빨간 칸이 **실질적 리스크**입니다. 하단 리스크 카드에서 [상세 원인 분석]으로 범인을 특정하세요.")

            _hm_df = _df_sel.copy()
            _hm_df["_채널그룹_sol"] = _hm_df[_ch_col].apply(_group_channel)
            _hm_df = _hm_df[_hm_df["_채널그룹_sol"].notna()]

            if not _hm_df.empty:
                _hm_pivot = (_hm_df.groupby([M["business"], "_채널그룹_sol"])["_점수100"]
                             .mean().round(1).reset_index()
                             .pivot(index=M["business"], columns="_채널그룹_sol", values="_점수100")
                             .fillna(0))
                _hm_cnt = (_hm_df.groupby([M["business"], "_채널그룹_sol"])["_점수100"]
                           .count().reset_index()
                           .pivot(index=M["business"], columns="_채널그룹_sol", values="_점수100")
                           .fillna(0))
                _hm_vals = _hm_pivot.values
                _hm_nonzero = _hm_vals[_hm_vals > 0]
                _hm_range = [max(60, float(_hm_nonzero.min()) - 3) if len(_hm_nonzero) else 60, 100]

                fig_hm = px.imshow(
                    _hm_pivot, color_continuous_scale="RdYlGn",
                    text_auto=".1f", aspect="auto", template=PLOTLY_TPL,
                    range_color=_hm_range,
                    labels=dict(x="채널", y="업무유형", color="만족도"))
                _hm_hover = []
                for _bi in _hm_pivot.index:
                    _row = []
                    for _ci in _hm_pivot.columns:
                        _sc = _hm_pivot.loc[_bi, _ci]
                        _cn = int(_hm_cnt.loc[_bi, _ci]) if (_bi in _hm_cnt.index and _ci in _hm_cnt.columns) else 0
                        _gp = round(_sc - avg_score_100, 1) if _sc > 0 else 0
                        _row.append(f"{_bi} x {_ci}<br>점수: {_sc:.1f}점<br>건수: {_cn}건<br>본부 대비: {_gp:+.1f}점")
                    _hm_hover.append(_row)
                fig_hm.update_traces(customdata=_hm_hover,
                                     hovertemplate="%{customdata}<extra></extra>")
                fig_hm.update_layout(
                    height=max(320, len(_hm_pivot) * 38 + 100),
                    margin=dict(t=10, b=60, l=120, r=20),
                    coloraxis_colorbar=dict(title="만족도"))
                st.plotly_chart(fig_hm, use_container_width=True)

                # ── 실질적 리스크 TOP 3 카드 ─────────────────────
                _risk_rows = []
                for _bi in _hm_pivot.index:
                    for _ci in _hm_pivot.columns:
                        _sc = _hm_pivot.loc[_bi, _ci]
                        _cn = int(_hm_cnt.loc[_bi, _ci]) if (_bi in _hm_cnt.index and _ci in _hm_cnt.columns) else 0
                        if _sc > 0 and _sc < avg_score_100 and _cn >= 2:
                            _risk_rows.append({
                                "업무": _bi, "채널": _ci,
                                "점수": _sc, "건수": _cn,
                                "impact": round((avg_score_100 - _sc) * _cn, 1)
                            })
                _risk_top3 = sorted(_risk_rows, key=lambda x: x["impact"], reverse=True)[:3]

                if _risk_top3:
                    st.markdown("##### 🚨 실질적 리스크 TOP 3 — 점수 향상 임팩트 기준")
                    _rt_cols = st.columns(len(_risk_top3))
                    for _ri, _rk in enumerate(_risk_top3):
                        _rk_key  = f"sol_cell_{_ri}"
                        _cur_sel = st.session_state.get("sol_cell_sel")
                        _is_sel  = (_cur_sel is not None
                                    and _cur_sel.get("업무") == _rk["업무"]
                                    and _cur_sel.get("채널") == _rk["채널"])
                        _badge = "①②③"[_ri]
                        _bg    = "#fff3e0" if _is_sel else "#ffebee"
                        _bd    = "#ff8f00" if _is_sel else "#ef9a9a"
                        with _rt_cols[_ri]:
                            st.markdown(
                                f'<div style="background:{_bg};border:2px solid {_bd};'
                                'border-radius:10px;padding:12px;margin-bottom:8px;">'
                                f'<div style="font-size:0.78em;color:#c62828;font-weight:700;">리스크 {_badge}</div>'
                                f'<div style="font-size:1em;font-weight:800;margin:4px 0;">{_rk["업무"]} × {_rk["채널"]}</div>'
                                f'<div style="font-size:1.4em;font-weight:900;color:#c62828;">{_rk["점수"]:.1f}점</div>'
                                f'<div style="font-size:0.78em;color:#555;">{_rk["건수"]}건 · 임팩트 {_rk["impact"]:.0f}</div>'
                                '</div>',
                                unsafe_allow_html=True)
                            if st.button(
                                    "▲ 닫기" if _is_sel else "🔍 상세 원인 분석",
                                    key=_rk_key, use_container_width=True,
                                    type="secondary" if _is_sel else "primary"):
                                st.session_state["sol_cell_sel"] = (
                                    None if _is_sel else {"업무": _rk["업무"], "채널": _rk["채널"]})
                                st.rerun()

                    # ══════════════════════════════════════════
                    # LEVEL 3 — AI 심층 진단 (Action)
                    # ══════════════════════════════════════════
                    _cell = st.session_state.get("sol_cell_sel")
                    if _cell:
                        _c3_biz = _cell["업무"]
                        _c3_ch  = _cell["채널"]

                        st.markdown(
                            '<div style="background:linear-gradient(90deg,#4a148c,#6a1b9a);'
                            'border-radius:10px;padding:14px 20px;color:white;margin:16px 0 12px;">'
                            f'<span style="font-size:1.1em;font-weight:800;">'
                            f'🔬 AI 심층 진단 — {_sel_off} · {_c3_biz} × {_c3_ch}</span>'
                            '<span style="font-size:0.82em;opacity:.8;margin-left:10px;">계약종별 세부 원인 규명</span>'
                            '</div>',
                            unsafe_allow_html=True)

                        _c3_df = _hm_df[
                            (_hm_df[M["business"]] == _c3_biz) &
                            (_hm_df["_채널그룹_sol"] == _c3_ch)].copy()

                        _c3_l, _c3_r = st.columns([1, 1])

                        with _c3_l:
                            st.markdown("**📊 계약종별 만족도 — 범인 특정**")
                            if M.get("contract") and M["contract"] in _c3_df.columns and not _c3_df.empty:
                                _c3_ct = (_c3_df.groupby(M["contract"])["_점수100"]
                                          .agg(["mean", "count"]).reset_index())
                                _c3_ct.columns = ["계약종", "만족도", "건수"]
                                _c3_ct = _c3_ct[_c3_ct["건수"] >= 1].sort_values("만족도")
                                if not _c3_ct.empty:
                                    _c3_colors = _c3_ct["만족도"].apply(
                                        lambda x: "#d32f2f" if x < avg_score_100 - 5
                                        else "#f57c00" if x < avg_score_100
                                        else "#388e3c").tolist()
                                    fig_c3 = go.Figure(go.Bar(
                                        y=_c3_ct["계약종"], x=_c3_ct["만족도"],
                                        orientation="h", marker_color=_c3_colors,
                                        text=[f"{v:.1f}" for v in _c3_ct["만족도"]],
                                        textposition="outside",
                                        customdata=_c3_ct["건수"].values,
                                        hovertemplate="%{y}<br>%{x:.1f}점<br>%{customdata}건<extra></extra>"))
                                    fig_c3.add_vline(
                                        x=avg_score_100, line_dash="dash", line_color=C["navy"],
                                        annotation_text=f"본부 {avg_score_100:.1f}",
                                        annotation_position="top right")
                                    fig_c3.update_layout(
                                        template=PLOTLY_TPL, height=280,
                                        margin=dict(t=10, b=10, l=10, r=60),
                                        xaxis=dict(range=[60, 105]))
                                    st.plotly_chart(fig_c3, use_container_width=True)
                                    _culprit = _c3_ct.iloc[0]
                                    _cul_gap = round(_culprit["만족도"] - avg_score_100, 1)
                                    st.markdown(
                                        '<div style="background:#ffebee;border:2px solid #ef9a9a;'
                                        'border-radius:8px;padding:10px 14px;font-size:0.88em;">'
                                        f'🎯 <b>범인 확정:</b> <b>{_culprit["계약종"]}</b> 고객이 '
                                        f'<b>{_c3_ch}</b> 채널로 <b>{_c3_biz}</b> 처리 시 '
                                        f'만족도 <b style="color:#c62828">{_culprit["만족도"]:.1f}점</b>'
                                        f' (본부 대비 {_cul_gap:+.1f}점, {_culprit["건수"]}건)</div>',
                                        unsafe_allow_html=True)
                            else:
                                st.info("계약종 컬럼이 설정되지 않았습니다.")

                        with _c3_r:
                            st.markdown("**📋 실제 VOC 원문**")
                            if M.get("voc") and not _c3_df.empty:
                                _c3_vocs = (
                                    _c3_df[M["voc"]].dropna()
                                    .apply(lambda x: str(x).strip())
                                    .loc[lambda s: (s.str.len() > 2) & (~s.isin(["응답없음", "nan", ""]))]
                                    .head(10))
                                if not _c3_vocs.empty:
                                    with st.expander(f"VOC {len(_c3_vocs)}건 보기", expanded=True):
                                        for _cv in _c3_vocs:
                                            _cv_hl = str(_cv)
                                            for _nkw in NEGATIVE_KEYWORDS:
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
                                    st.info("해당 조합의 VOC가 없습니다.")
                            else:
                                st.info("VOC 컬럼이 설정되지 않았습니다.")

                        # ── AI 처방전 ──────────────────────────
                        st.markdown("---")
                        if st.button("🤖 AI 처방전 생성", key="sol_ai_cell_btn",
                                     type="primary", use_container_width=True):
                            if not GEMINI_AVAILABLE:
                                st.error("Gemini API 키가 설정되지 않았습니다.")
                            else:
                                _c3_ct_lines = ""
                                if M.get("contract") and M["contract"] in _c3_df.columns:
                                    _c3_ct2 = (_c3_df.groupby(M["contract"])["_점수100"]
                                               .agg(["mean", "count"]).reset_index())
                                    _c3_ct_lines = "\\n".join(
                                        f"  - {r.iloc[0]}: {r['mean']:.1f}점 ({r['count']}건)"
                                        for _, r in _c3_ct2.iterrows())
                                _c3_voc_lines = ""
                                if M.get("voc"):
                                    _c3_vl = (_c3_df[M["voc"]].dropna()
                                              .apply(str)
                                              .loc[lambda s: (s.str.len() > 2) &
                                                             (~s.isin(["응답없음", "nan", ""]))]
                                              .head(8))
                                    _c3_voc_lines = "\\n".join(f"  - {v}" for v in _c3_vl)
                                _c3_prompt = (
                                    f"당신은 전력산업 CS 전문가입니다.\\n\\n"
                                    f"[진단 대상]\\n"
                                    f"지사: {_sel_off} | 업무: {_c3_biz} | 채널: {_c3_ch}\\n"
                                    f"지사 평균: {_sel_avg:.1f}점 (본부 {avg_score_100:.1f}점 대비 {_sel_gap:+.1f}점)\\n\\n"
                                    f"[계약종별 만족도]\\n{_c3_ct_lines or '데이터 없음'}\\n\\n"
                                    f"[불만 VOC 원문]\\n{_c3_voc_lines or '없음'}\\n\\n"
                                    "[분석 요청]\\n"
                                    f"1. 핵심 문제 진단: 왜 이 지사의 {_c3_biz} x {_c3_ch} 조합에서 만족도가 낮은가? (2~3문장)\\n"
                                    "2. 즉시 실행 처방: 72시간 내 지사장이 실행 가능한 구체적 조치 1가지\\n"
                                    "3. 근본 원인 처방: 3개월 내 구조적 개선 방안 1가지\\n\\n"
                                    "※ 전력산업 도메인 지식을 활용하고 추상적 제안('친절 교육' 등)은 금지합니다."
                                )
                                with st.spinner("AI가 원인을 분석 중…"):
                                    try:
                                        import urllib.request
                                        _models = ["gemini-2.0-flash", "gemma-3-12b-it", "gemma-3-27b-it"]
                                        _c3_pl  = {
                                            "contents": [{"parts": [{"text": _c3_prompt}]}],
                                            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048}
                                        }
                                        _ctx   = ssl._create_unverified_context()
                                        _body  = None
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
                                            _ai_txt = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
                                            st.markdown(_ai_txt)
                                    except Exception as _c3e:
                                        st.error(f"AI 분석 중 오류: {_c3e}")
            else:
                st.info("채널 또는 업무유형 컬럼이 설정되지 않았습니다.")
"""
