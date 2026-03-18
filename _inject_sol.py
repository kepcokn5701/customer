# -*- coding: utf-8 -*-
import ast, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TAB_SOL_CODE = r'''
# ─────────────────────────────────────────────────────────────
#  TAB SOL  지사별 맞춤 솔루션 도출 엔진
# ─────────────────────────────────────────────────────────────
with tab_sol:
    st.markdown('<p class="sec-head">🏢 지사별 맞춤 솔루션 도출 엔진</p>', unsafe_allow_html=True)
    st.caption("지사 선택 → 종합 컨디션 진단 · 업무별 강약점 · 벤치마킹 처방전 · AI 맞춤 솔루션")

    if not M.get("office") or not M.get("business") or df_f.empty or avg_score_100 is None:
        st.info("지사(사업소) · 업무유형 컬럼과 점수 데이터가 모두 필요합니다.")
    else:
        # ── 지사 선택 ──────────────────────────────────────────
        _sol_offices = sorted(df_f[M["office"]].dropna().unique().tolist())
        _sel_off = st.selectbox("🏢 분석할 지사를 선택하세요", _sol_offices, key="sol_office_sel")
        _df_sel = df_f[df_f[M["office"]] == _sel_off].copy()

        if _df_sel.empty:
            st.warning(f"{_sel_off}의 데이터가 없습니다.")
        else:
            # ── Peer Group (응답 건수 기준 3그룹) ───────────────
            _off_counts = df_f.groupby(M["office"])["_점수100"].count().sort_values()
            _n_off = len(_off_counts)
            _t1 = _off_counts.iloc[_n_off // 3] if _n_off >= 3 else 0
            _t2 = _off_counts.iloc[2 * _n_off // 3] if _n_off >= 3 else float("inf")
            _my_cnt = int(_off_counts.get(_sel_off, 0))
            if _my_cnt <= _t1:
                _peer_label = "소규모"
                _peer_offs = _off_counts[_off_counts <= _t1].index.tolist()
            elif _my_cnt <= _t2:
                _peer_label = "중규모"
                _peer_offs = _off_counts[(_off_counts > _t1) & (_off_counts <= _t2)].index.tolist()
            else:
                _peer_label = "대규모"
                _peer_offs = _off_counts[_off_counts > _t2].index.tolist()
            _peer_avg = df_f[df_f[M["office"]].isin(_peer_offs)]["_점수100"].mean()

            # ── 지사 기초 지표 ───────────────────────────────────
            _sel_avg = _df_sel["_점수100"].mean()
            _sel_cnt = len(_df_sel)
            _sel_gap = _sel_avg - avg_score_100
            _all_rank_s = df_f.groupby(M["office"])["_점수100"].mean().sort_values(ascending=False)
            _rank = list(_all_rank_s.index).index(_sel_off) + 1 if _sel_off in _all_rank_s.index else "?"
            _total_offs = len(_all_rank_s)
            _peer_rank_s = df_f[df_f[M["office"]].isin(_peer_offs)].groupby(M["office"])["_점수100"].mean().sort_values(ascending=False)
            _peer_rank = list(_peer_rank_s.index).index(_sel_off) + 1 if _sel_off in _peer_rank_s.index else "?"
            _reliability = ("✅ 높음" if _sel_cnt >= 50 else ("🟡 보통" if _sel_cnt >= 20 else "⚠️ 낮음"))

            # ── 업무별 집계 (사전 계산) ──────────────────────────
            _biz_sel = _df_sel.groupby(M["business"])["_점수100"].agg(["mean", "count"]).reset_index()
            _biz_sel.columns = ["업무유형", "평균만족도", "건수"]
            _biz_sel["전체대비"] = (_biz_sel["평균만족도"] - avg_score_100).round(1)
            _peer_biz = (df_f[df_f[M["office"]].isin(_peer_offs)]
                         .groupby(M["business"])["_점수100"].mean().reset_index())
            _peer_biz.columns = ["업무유형", "피어평균"]
            _biz_sel = _biz_sel.merge(_peer_biz, on="업무유형", how="left").sort_values("평균만족도")
            _biz_sel["색상"] = _biz_sel["전체대비"].apply(
                lambda d: "#388e3c" if d >= 3 else ("#d32f2f" if d <= -3 else "#1976d2"))
            _weak3 = _biz_sel[_biz_sel["전체대비"] <= -3].sort_values("전체대비").head(3)
            _strong3 = _biz_sel[_biz_sel["전체대비"] >= 3].sort_values("전체대비", ascending=False).head(3)

            # ── 핵심 한 줄 진단 ──────────────────────────────────
            if not _weak3.empty and _weak3.iloc[0]["전체대비"] < -3:
                _worst_biz = _weak3.iloc[0]
                _diag = (f"전반적으로 {'우수' if _sel_gap >= 0 else '개선 필요'}하나, "
                         f"'{_worst_biz['업무유형']}' 업무 만족도가 본부 평균보다 "
                         f"{abs(_worst_biz['전체대비']):.1f}점 낮아 집중 개선이 필요합니다.")
            elif _sel_gap >= 0:
                _diag = (f"본부 평균을 상회하는 우수 지사입니다. "
                         + (f"'{_strong3.iloc[0]['업무유형']}' 업무가 강점입니다." if not _strong3.empty else ""))
            else:
                _diag = f"전반적 만족도가 본부 평균을 {abs(_sel_gap):.1f}점 하회합니다. 전방위 개선이 필요합니다."

            # ══════════════════════════════════════════════════
            # 1. 헤더: 지사 종합 컨디션
            # ══════════════════════════════════════════════════
            st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a237e 0%,#283593 60%,#1565c0 100%);
            border-radius:12px;padding:20px 24px;color:white;margin-bottom:20px;">
  <div style="font-size:1.25em;font-weight:800;margin-bottom:14px;">
    📍 {_sel_off} · 종합 컨디션 리포트
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px;">
    <div style="background:rgba(255,255,255,0.13);border-radius:8px;padding:12px;text-align:center;">
      <div style="font-size:0.72em;opacity:0.8;margin-bottom:4px;">종합 만족도</div>
      <div style="font-size:1.9em;font-weight:900;">{_sel_avg:.1f}<span style="font-size:0.45em;">점</span></div>
      <div style="font-size:0.78em;color:{'#80cbc4' if _sel_gap >= 0 else '#ef9a9a'};">{_sel_gap:+.1f}점 (본부 평균 {avg_score_100:.1f})</div>
    </div>
    <div style="background:rgba(255,255,255,0.13);border-radius:8px;padding:12px;text-align:center;">
      <div style="font-size:0.72em;opacity:0.8;margin-bottom:4px;">본부 랭킹</div>
      <div style="font-size:1.9em;font-weight:900;">{_rank}<span style="font-size:0.45em;">위</span></div>
      <div style="font-size:0.78em;opacity:0.75;">/ {_total_offs}개 지사</div>
    </div>
    <div style="background:rgba(255,255,255,0.13);border-radius:8px;padding:12px;text-align:center;">
      <div style="font-size:0.72em;opacity:0.8;margin-bottom:4px;">{_peer_label} 그룹 랭킹</div>
      <div style="font-size:1.9em;font-weight:900;">{_peer_rank}<span style="font-size:0.45em;">위</span></div>
      <div style="font-size:0.78em;opacity:0.75;">/ {len(_peer_offs)}개 유사 지사 · 피어평균 {_peer_avg:.1f}점</div>
    </div>
    <div style="background:rgba(255,255,255,0.13);border-radius:8px;padding:12px;text-align:center;">
      <div style="font-size:0.72em;opacity:0.8;margin-bottom:4px;">데이터 신뢰도</div>
      <div style="font-size:1.2em;font-weight:700;margin:4px 0;">{_reliability}</div>
      <div style="font-size:0.78em;opacity:0.75;">응답 {_sel_cnt:,}건</div>
    </div>
  </div>
  <div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:10px 14px;
              font-size:0.9em;border-left:4px solid #80cbc4;">
    💡 핵심 진단: "{_diag}"
  </div>
</div>""", unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # 2+3. 중앙: 업무 강약점 + 페르소나 미스매치
            # ══════════════════════════════════════════════════
            _c_left, _c_right = st.columns([1, 1])

            with _c_left:
                st.markdown("##### 📊 업무별 강점/약점 진단 (The Radar)")
                st.caption(f"본부 평균 및 {_peer_label} 그룹 평균과 비교 · ±3점 이상 차이 항목 강조")
                fig_biz = go.Figure()
                fig_biz.add_trace(go.Bar(
                    y=_biz_sel["업무유형"], x=_biz_sel["평균만족도"],
                    orientation="h", name=_sel_off,
                    marker_color=_biz_sel["색상"],
                    text=[f"{v:.1f}" for v in _biz_sel["평균만족도"]],
                    textposition="outside",
                    customdata=_biz_sel[["건수", "전체대비"]].values,
                    hovertemplate="%{y}<br>평균: %{x:.1f}점<br>응답: %{customdata[0]}건<br>본부 대비: %{customdata[1]:+.1f}점<extra></extra>"
                ))
                if "피어평균" in _biz_sel.columns:
                    fig_biz.add_trace(go.Scatter(
                        y=_biz_sel["업무유형"], x=_biz_sel["피어평균"],
                        mode="markers", name=f"{_peer_label} 그룹 평균",
                        marker=dict(symbol="line-ns-open", size=14, color="#f57c00",
                                    line=dict(width=2.5, color="#f57c00")),
                        hovertemplate="%{y}<br>피어 평균: %{x:.1f}점<extra></extra>"
                    ))
                fig_biz.add_vline(x=avg_score_100, line_dash="dash", line_color=C["navy"],
                                  annotation_text=f"본부 {avg_score_100:.1f}", annotation_position="top right")
                fig_biz.update_layout(
                    template=PLOTLY_TPL,
                    height=max(300, len(_biz_sel) * 42 + 100),
                    margin=dict(t=10, b=20, l=10, r=80),
                    legend=dict(orientation="h", y=-0.12),
                    xaxis=dict(range=[max(0, _biz_sel["평균만족도"].min() - 15), 100])
                )
                st.plotly_chart(fig_biz, use_container_width=True)
                # 강점/약점 카드
                if not _strong3.empty:
                    st.markdown("**💪 강점 (본부 평균 +3점 이상)**")
                    for _, r in _strong3.iterrows():
                        st.markdown(
                            f'<div style="background:#e8f5e9;border-left:4px solid #388e3c;'
                            f'padding:6px 12px;border-radius:0 6px 6px 0;margin-bottom:4px;font-size:0.87em;">'
                            f'✅ <b>{r["업무유형"]}</b>: {r["평균만족도"]:.1f}점 '
                            f'<span style="color:#388e3c">({r["전체대비"]:+.1f}점)</span> · {int(r["건수"])}건</div>',
                            unsafe_allow_html=True)
                if not _weak3.empty:
                    st.markdown("**⚠️ 약점 (본부 평균 -3점 이하)**")
                    for _, r in _weak3.iterrows():
                        st.markdown(
                            f'<div style="background:#ffebee;border-left:4px solid #d32f2f;'
                            f'padding:6px 12px;border-radius:0 6px 6px 0;margin-bottom:4px;font-size:0.87em;">'
                            f'🔴 <b>{r["업무유형"]}</b>: {r["평균만족도"]:.1f}점 '
                            f'<span style="color:#d32f2f">({r["전체대비"]:+.1f}점)</span> · {int(r["건수"])}건</div>',
                            unsafe_allow_html=True)

            with _c_right:
                st.markdown("##### 🎯 페르소나별 미스매치 (The Target)")
                st.caption("건수 비중(X) × 만족도(Y) — 우하단이 최우선 개선 타겟")
                _scat_col = M.get("contract") or M.get("business")
                _scat_label = "계약종" if M.get("contract") else "업무유형"
                if _scat_col:
                    _scat = _df_sel.groupby(_scat_col)["_점수100"].agg(["mean", "count"]).reset_index()
                    _scat.columns = [_scat_label, "만족도", "건수"]
                    _scat["비중(%)"] = (_scat["건수"] / len(_df_sel) * 100).round(1)
                    _med_pct = _scat["비중(%)"].median()
                    _scat["색상"] = _scat.apply(
                        lambda r: "#d32f2f" if r["비중(%)"] >= _med_pct and r["만족도"] < avg_score_100
                        else ("#f57c00" if r["비중(%)"] < _med_pct and r["만족도"] < avg_score_100
                              else "#388e3c"), axis=1)
                    fig_scat = go.Figure(go.Scatter(
                        x=_scat["비중(%)"], y=_scat["만족도"],
                        mode="markers+text", text=_scat[_scat_label],
                        textposition="top center",
                        marker=dict(size=_scat["건수"].apply(lambda x: max(12, min(45, x // 3))),
                                    color=_scat["색상"], line=dict(width=1, color="white")),
                        customdata=_scat[["건수", "비중(%)"]].values,
                        hovertemplate=f"%{{text}}<br>만족도: %{{y:.1f}}점<br>비중: %{{x:.1f}}%<br>건수: %{{customdata[0]}}건<extra></extra>"
                    ))
                    fig_scat.add_hline(y=avg_score_100, line_dash="dash", line_color=C["navy"],
                                       annotation_text=f"본부 평균 {avg_score_100:.1f}", annotation_position="right")
                    fig_scat.add_vline(x=_med_pct, line_dash="dot", line_color="gray",
                                       annotation_text="중위 비중", annotation_position="top")
                    fig_scat.update_layout(
                        template=PLOTLY_TPL, height=380,
                        margin=dict(t=20, b=40, l=20, r=20),
                        xaxis=dict(title="건수 비중 (%)"),
                        yaxis=dict(title="평균 만족도")
                    )
                    st.plotly_chart(fig_scat, use_container_width=True)
                    _danger = _scat[(_scat["비중(%)"] >= _med_pct) & (_scat["만족도"] < avg_score_100)].sort_values("비중(%)", ascending=False)
                    _latent = _scat[(_scat["비중(%)"] < _med_pct) & (_scat["만족도"] < avg_score_100)].sort_values("만족도")
                    if not _danger.empty:
                        st.markdown(
                            '<div style="background:#ffebee;border:1px solid #ef9a9a;border-radius:8px;'
                            'padding:8px 12px;margin-bottom:6px;font-size:0.87em;">'
                            '🔴 <b>1순위 개선 (많은 건수+저점수)</b>: '
                            + ", ".join([f'{r[_scat_label]} ({r["만족도"]:.1f}점·{r["비중(%)"]:.1f}%)'
                                         for _, r in _danger.head(2).iterrows()])
                            + '</div>', unsafe_allow_html=True)
                    if not _latent.empty:
                        st.markdown(
                            '<div style="background:#fff8e1;border:1px solid #ffe082;border-radius:8px;'
                            'padding:8px 12px;font-size:0.87em;">'
                            '⚠️ <b>특이 리스크 (소규모+극저점수)</b>: '
                            + ", ".join([f'{r[_scat_label]} ({r["만족도"]:.1f}점)'
                                         for _, r in _latent.head(2).iterrows()])
                            + '</div>', unsafe_allow_html=True)

            st.markdown("---")

            # ══════════════════════════════════════════════════
            # 4+5. 하단: 벤치마킹 솔루션 + 사전케어 대상
            # ══════════════════════════════════════════════════
            _c_bot_l, _c_bot_r = st.columns([1, 1])

            with _c_bot_l:
                st.markdown("##### 🏆 벤치마킹 솔루션 (Learning from Peers)")
                st.caption("약점 업무에서 본부 최고 성과 지사를 찾아 처방을 제안합니다")
                if _weak3.empty:
                    st.success(f"🎉 {_sel_off}는 모든 업무에서 본부 평균 ±3점 이내입니다!")
                else:
                    for _, wrow in _weak3.iterrows():
                        _biz_n = wrow["업무유형"]
                        _biz_all = (df_f[df_f[M["business"]] == _biz_n]
                                    .groupby(M["office"])["_점수100"].agg(["mean", "count"]))
                        _biz_all = _biz_all[_biz_all["count"] >= 3].sort_values("mean", ascending=False)
                        _candidates = [o for o in _biz_all.index if o != _sel_off]
                        if not _candidates:
                            continue
                        _best_off = _candidates[0]
                        _best_score = _biz_all.loc[_best_off, "mean"]
                        st.markdown(
                            f'<div style="background:#e8eaf6;border-left:4px solid #3949ab;'
                            f'border-radius:0 10px 10px 0;padding:12px 14px;margin-bottom:10px;">'
                            f'<b>📌 {_biz_n}</b> 약점 개선 처방<br>'
                            f'<span style="font-size:0.87em">'
                            f'우리 지사 <b style="color:#d32f2f">{wrow["평균만족도"]:.1f}점</b> · '
                            f'<b>{_best_off}</b> <b style="color:#388e3c">{_best_score:.1f}점</b><br>'
                            f'→ <b>{_best_off}</b>의 {_biz_n} 운영 프로세스를 벤치마킹하세요'
                            f'</span></div>',
                            unsafe_allow_html=True)

                # 가성비 업무 (건수 많고 점수 낮은 1순위)
                st.markdown("**🎯 가성비 1순위 개선 과제**")
                _gasungbi_df = _biz_sel[_biz_sel["전체대비"] < 0].copy()
                if not _gasungbi_df.empty:
                    _gasungbi_df["가성비점수"] = _gasungbi_df["건수"] * _gasungbi_df["전체대비"].abs()
                    _top = _gasungbi_df.sort_values("가성비점수", ascending=False).iloc[0]
                    st.markdown(
                        f'<div style="background:#fff3e0;border:2px solid #f57c00;border-radius:10px;'
                        f'padding:14px 16px;text-align:center;">'
                        f'<div style="font-size:0.78em;color:#e65100;font-weight:700">🔥 지금 당장 개선하면 점수 상승 효과가 가장 큰 업무</div>'
                        f'<div style="font-size:1.35em;font-weight:900;color:#bf360c;margin:6px 0">{_top["업무유형"]}</div>'
                        f'<div style="font-size:0.87em;color:#5d4037">'
                        f'{int(_top["건수"])}건 · {_top["평균만족도"]:.1f}점 (본부 대비 {_top["전체대비"]:+.1f}점)<br>'
                        f'<b>→ 건수가 많고 점수가 낮아 개선 즉시 전체 점수 상승 기대</b>'
                        f'</div></div>',
                        unsafe_allow_html=True)
                else:
                    st.success("모든 업무가 본부 평균 이상입니다!")

            with _c_bot_r:
                st.markdown("##### 🚨 미조치 사전케어 대상 (Immediate Action)")
                st.caption("50점 이하 저점수 고객 — 즉각 케어가 필요한 최우선 대상")
                _low_df = _df_sel[_df_sel["_점수100"] <= 50].copy()
                if _low_df.empty:
                    st.success(f"{_sel_off}에 50점 이하 저점수 건이 없습니다. 양호!")
                else:
                    st.markdown(f"**총 {len(_low_df)}건의 저점수 고객** (50점 이하)")
                    for _ci, (_, _row) in enumerate(_low_df.head(5).iterrows()):
                        _sc_v = _row.get("_점수100", 0)
                        _voc_v = str(_row.get(M["voc"], "")) if M.get("voc") else ""
                        _biz_v = str(_row.get(M["business"], "")) if M.get("business") else ""
                        _ct_v = str(_row.get(M["contract"], "")) if M.get("contract") else ""
                        _dt_v = str(_row.get(M["date"], ""))[:10] if M.get("date") else ""
                        with st.expander(f"🔴 #{_ci+1}  {_biz_v} · {_sc_v:.0f}점  [{_dt_v}]"):
                            _ca, _cb = st.columns([2, 1])
                            with _ca:
                                st.markdown(f"**계약종**: {_ct_v}")
                                st.markdown(f"**업무유형**: {_biz_v}")
                                st.markdown(f"**만족도**: {_sc_v:.0f}점")
                                if _voc_v.strip() not in ("", "nan", "응답없음", "None"):
                                    st.markdown(
                                        f'<div style="background:#fff8f8;border-left:3px solid #e53935;'
                                        f'padding:8px 10px;border-radius:0 6px 6px 0;font-size:0.88em;">'
                                        f'{_voc_v}</div>', unsafe_allow_html=True)
                            with _cb:
                                st.button("📞 해피콜 대상 등록", key=f"sol_happycall_{_ci}", use_container_width=True)
                                st.button("✅ 조치 완료 입력", key=f"sol_action_{_ci}", use_container_width=True)

            st.markdown("---")

            # ══════════════════════════════════════════════════
            # 6. AI 처방전
            # ══════════════════════════════════════════════════
            st.markdown("##### 🤖 AI 맞춤형 처방전")
            st.caption("Gemini AI가 해당 지사의 데이터를 심층 분석하여 맞춤형 솔루션을 제안합니다")
            if st.button("🤖 Gemini AI 지사 맞춤 처방전 생성", key="sol_ai_btn", type="primary", use_container_width=True):
                if not GEMINI_AVAILABLE:
                    st.error("Gemini API 키가 설정되지 않았습니다. `.env` 파일에 `GEMINI_API_KEY`를 설정해주세요.")
                else:
                    _ai_biz_lines = "\n".join([
                        f"  - {r['업무유형']}: {r['평균만족도']:.1f}점 ({r['전체대비']:+.1f}점, {int(r['건수'])}건)"
                        for _, r in _biz_sel.sort_values("전체대비").iterrows()])
                    _ai_bench = ""
                    for _, _wr in _weak3.iterrows():
                        _ba = (df_f[df_f[M["business"]] == _wr["업무유형"]]
                               .groupby(M["office"])["_점수100"].agg(["mean", "count"]))
                        _ba = _ba[_ba["count"] >= 3].sort_values("mean", ascending=False)
                        _ba_cands = [o for o in _ba.index if o != _sel_off]
                        if _ba_cands:
                            _ai_bench += f"  - {_wr['업무유형']} 약점: 본부 1위={_ba_cands[0]}({_ba.loc[_ba_cands[0],'mean']:.1f}점)\n"
                    _ai_voc = ""
                    if M.get("voc"):
                        _vl = _df_sel[_df_sel["_점수100"] < 70][M["voc"]].dropna().astype(str).tolist()
                        _vl = [v for v in _vl if v.strip() not in ("", "nan", "응답없음")][:8]
                        _ai_voc = "\n".join([f"  - {v}" for v in _vl])
                    _gasungbi_info = ""
                    if not _gasungbi_df.empty:
                        _gst = _gasungbi_df.sort_values("가성비점수", ascending=False).iloc[0]
                        _gasungbi_info = f"{_gst['업무유형']} ({int(_gst['건수'])}건, {_gst['평균만족도']:.1f}점)"
                    _sol_prompt = f"""당신은 전력산업 고객만족(CS) 전문 컨설턴트입니다.

[분석 대상 지사] {_sel_off}
[종합 만족도] {_sel_avg:.1f}점 (본부 평균 {avg_score_100:.1f}점 대비 {_sel_gap:+.1f}점)
[본부 랭킹] {_rank}위 / {_total_offs}개 지사
[{_peer_label} 그룹 랭킹] {_peer_rank}위 / {len(_peer_offs)}개 유사 지사

[업무유형별 만족도 (본부 대비 편차)]
{_ai_biz_lines}

[가성비 1순위 개선 과제]
{_gasungbi_info if _gasungbi_info else '없음(모든 업무 평균 이상)'}

[벤치마킹 정보 (약점 업무의 본부 1위 지사)]
{_ai_bench if _ai_bench else '없음(심각한 약점 업무 없음)'}

[불만 VOC 원문 (70점 미만)]
{_ai_voc if _ai_voc else '없음'}

[분석 요청]
1. 핵심 문제 진단: {_sel_off}의 가장 시급한 CS 문제를 3줄 이내로 정리하세요
2. 가성비 1순위 처방: 건수 많고 점수 낮은 업무를 타겟으로 72시간 내 실행 가능한 구체적 액션을 제시하세요
3. 벤치마킹 처방: 약점 업무에서 우수 지사의 성공 요인을 분석하고 {_sel_off}에 맞는 도입 방안을 제시하세요
4. 3개월 로드맵: 본부 평균 이상으로 올리기 위한 단계별 계획을 제시하세요

※ 반드시 지사명·업무명·VOC 근거를 명시하고 추상적 제안("교육 강화" 등)은 금지합니다."""
                    with st.spinner(f"Gemini AI가 {_sel_off} 맞춤 처방전 생성 중…"):
                        try:
                            import urllib.request
                            _models = ["gemini-2.0-flash", "gemma-3-12b-it", "gemma-3-27b-it"]
                            _sol_pl = {"contents": [{"parts": [{"text": _sol_prompt}]}],
                                       "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4096}}
                            _ctx = ssl._create_unverified_context()
                            _body = None
                            for _model in _models:
                                _url = f"https://generativelanguage.googleapis.com/v1beta/models/{_model}:generateContent?key={_GEMINI_KEY}"
                                _req = urllib.request.Request(_url, data=json.dumps(_sol_pl).encode("utf-8"),
                                                              headers={"Content-Type": "application/json"}, method="POST")
                                try:
                                    with urllib.request.urlopen(_req, context=_ctx, timeout=90) as _resp:
                                        _body = json.loads(_resp.read().decode("utf-8"))
                                    break
                                except urllib.error.HTTPError as _he:
                                    if _he.code == 429:
                                        continue
                                    raise
                            if _body is None:
                                st.error("모든 AI 모델의 일일 한도가 소진되었습니다. 내일 다시 시도해주세요.")
                            else:
                                _ai_text = _body["candidates"][0]["content"]["parts"][0]["text"].strip()
                                st.markdown(_ai_text)
                        except Exception as _e:
                            st.error(f"AI 분석 중 오류: {_e}")

'''

with open('cs_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find insertion point: just before the TAB 10 section header comment
import re
# Look for the TAB 10 comment block
marker = '\n# ─────────────────────────────────────────────────────────────\n#  TAB 10'
idx = content.find(marker)
if idx == -1:
    # Try alternate: just before "with tab10:"
    marker = '\nwith tab10:'
    idx = content.find(marker)
    if idx == -1:
        print("ERROR: Cannot find tab10 insertion point")
        exit(1)

new_content = content[:idx] + '\n' + TAB_SOL_CODE + content[idx:]

with open('cs_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

# Verify syntax
import ast
try:
    ast.parse(open('cs_dashboard.py', encoding='utf-8').read())
    print('Syntax OK')
except SyntaxError as e:
    print(f'SYNTAX ERROR: {e}')
    exit(1)
