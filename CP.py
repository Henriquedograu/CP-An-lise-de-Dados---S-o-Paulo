# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Dependências opcionais (tratamos ausência com fallback)
try:
    from scipy import stats  # para IC e p-valor
except Exception:  # pragma: no cover
    stats = None

try:
    import statsmodels.api as sm  # para trendline="ols" no Plotly
    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False


# ----------------------------- Dados de Portfólio -----------------------------

PORTIFOLIO = {
    "nome": "Henrique Azevedo",
    "formacao": "Aluno da FIAP - Engenharia de Software",
    "experiencia": "Professor de Inglês",
    "ingles": "C1 Level",
    "skills": [
        "Comunicação",
        "Trabalho em equipe",
        "Domínio de línguas: Português / Inglês / Espanhol",
        "Programação",
        "Banco de Dados",
        "Power BI",
        "Algoritmos",
        "Inteligência Artificial (IA)",
        "Fullstack Java",
    ],
}

# ------------------------ Titulares (nome e sobrenome) ------------------------

TITULARES: Dict[str, str] = {
    "Goleiro": "Rafael Pires",
    "Lateral-Direito": "Igor Vinícius",
    "Zagueiro-Direito": "Robert Arboleda",
    "Zagueiro-Esquerdo": "Diego Costa",
    "Lateral-Esquerdo": "Welington Santos",
    "Volante 1": "Pablo Maia",
    "Volante 2": "Alisson Freitas",
    "Meia-Direita": "Lucas Moura",
    "Meia-Esquerda": "Wellington Rato",
    "Meia-Central": "Rodrigo Nestor",
    "Centroavante": "Jonathan Calleri",
}


# ----------------------------- Utilidades Numéricas ---------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    """Retorna um Generator; se seed for None -> aleatório, senão determinístico."""
    return np.random.default_rng(None if seed is None else int(seed))


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def ci_media_t(series: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    if stats is None:
        # Aproximação normal (fallback)
        z = 1.96
        half = z * std / math.sqrt(max(n, 1))
        return (mean - half, mean + half)
    tcrit = stats.t.ppf(1 - alpha / 2.0, df=n - 1)
    half = tcrit * std / math.sqrt(n)
    return (mean - half, mean + half)


def welch_ttest(a: pd.Series, b: pd.Series) -> Tuple[float, float, float, float]:
    """
    Retorna: diferença de médias (a - b), t-stat, df (Welch), p-valor (bicaudal).
    Se scipy não disponível, p-valor será nan.
    """
    x = pd.to_numeric(a, errors="coerce").dropna()
    y = pd.to_numeric(b, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mx, my = float(x.mean()), float(y.mean())
    vx, vy = float(x.var(ddof=1)), float(y.var(ddof=1))
    nx, ny = len(x), len(y)
    num = mx - my
    den = math.sqrt((vx / nx) + (vy / ny))
    if den == 0:
        return num, 0.0, float("nan"), float("nan")
    tstat = num / den
    if stats is None:
        return num, tstat, float("nan"), float("nan")
    # graus de liberdade aproximados (Welch)
    df_num = (vx / nx + vy / ny) ** 2
    df_den = 0.0
    if nx > 1 and ny > 1:
        df_den = (vx * vx) / (nx * nx * (nx - 1)) + (vy * vy) / (ny * ny * (ny - 1))
    df = df_num / df_den if df_den != 0 else float("nan")
    if not (df is None or math.isnan(df)):
        p = 2.0 * stats.t.sf(abs(tstat), df=df)
    else:
        p = float("nan")
    return num, tstat, df, p


# ----------------------------- Geração de Dataset -----------------------------

@dataclass
class MatchConfig:
    n_matches: int
    start_date: date
    seed: Optional[int] = None


def gerar_partidas(cfg: MatchConfig) -> pd.DataFrame:
    rng = _rng(cfg.seed)
    dates = [cfg.start_date + timedelta(days=int(i * 7)) for i in range(cfg.n_matches)]
    # Converte para datetime64 para evitar erros em .dt e export JSON
    dates = pd.to_datetime(dates)

    home = rng.choice([True, False], size=cfg.n_matches)

    xg_for = _clip(rng.normal(1.45, 0.55, cfg.n_matches), 0.05, 4.0)
    xg_against = _clip(rng.normal(1.20, 0.50, cfg.n_matches), 0.05, 4.0)

    goals_for = rng.poisson(xg_for)
    goals_against = rng.poisson(xg_against)

    shots = _clip((xg_for * 6.5 + rng.normal(0, 2.5, cfg.n_matches)), 3, 30).round()
    sot_ratio = _clip(rng.normal(0.37, 0.08, cfg.n_matches), 0.10, 0.70)
    sot = _clip(shots * sot_ratio, 1, 15).round()
    # Garante que SOT <= Shots
    sot = np.minimum(sot, shots).astype(int)

    poss = _clip(rng.normal(0.56, 0.10, cfg.n_matches) + (home.astype(int) * 0.03), 0.30, 0.75)
    passes = _clip((poss * 700 + rng.normal(0, 40, cfg.n_matches)), 250, 900).round()
    acc = _clip(rng.normal(0.86, 0.03, cfg.n_matches), 0.70, 0.95)
    passes_acc = (passes * acc).round()

    duels_w = _clip((rng.normal(0.52, 0.08, cfg.n_matches) * 100).round(), 20, 80).astype(int)
    corners = _clip((rng.normal(5.2, 2.0, cfg.n_matches)), 0, 14).round()
    fouls = _clip((rng.normal(12, 4, cfg.n_matches)), 4, 26).round()
    yellow = _clip((rng.normal(2.0, 1.0, cfg.n_matches)), 0, 6).round()
    red = rng.binomial(1, 0.05, cfg.n_matches)

    opp_names = [
        "Palmeiras", "Corinthians", "Santos", "Flamengo", "Fluminense",
        "Grêmio", "Internacional", "Atlético-MG", "Athletico-PR", "Bahia",
        "Fortaleza", "Vasco", "Botafogo", "Criciúma", "Juventude",
        "Red Bull Bragantino", "Cuiabá", "América-MG", "Goiás", "Coritiba",
    ]
    opp = rng.choice(opp_names, size=cfg.n_matches)

    records: List[Dict[str, object]] = []
    for i in range(cfg.n_matches):
        gf, ga = int(goals_for[i]), int(goals_against[i])
        if gf > ga:
            result, points = "W", 3
        elif gf == ga:
            result, points = "D", 1
        else:
            result, points = "L", 0
        records.append(
            {
                "team": "São Paulo FC",
                "date": dates[i],
                "home": bool(home[i]),
                "opponent": str(opp[i]),
                "xg_for": float(xg_for[i]),
                "xg_against": float(xg_against[i]),
                "goals_for": gf,
                "goals_against": ga,
                "result": result,
                "points": points,
                "shots": int(shots[i]),
                "shots_on_target": int(sot[i]),
                "possession": float(poss[i]),
                "passes": int(passes[i]),
                "passes_accurate": int(passes_acc[i]),
                "pass_accuracy": float(acc[i]),
                "duels_won_pct": int(duels_w[i]),
                "corners": int(corners[i]),
                "fouls": int(fouls[i]),
                "yellow_cards": int(yellow[i]),
                "red_cards": int(red[i]),
            }
        )
    df = pd.DataFrame.from_records(records)
    df["goal_diff"] = df["goals_for"] - df["goals_against"]
    df["win"] = (df["result"] == "W").astype(int)
    df["draw"] = (df["result"] == "D").astype(int)
    df["loss"] = (df["result"] == "L").astype(int)
    df["season"] = df["date"].dt.year  # agora funciona
    return df


# ------------------------------ Abas do Dashboard -----------------------------

def tab_portfolio():
    st.header("Portifólio")
    st.subheader(PORTIFOLIO["nome"])
    st.write(PORTIFOLIO["formacao"])
    st.write(PORTIFOLIO["experiencia"])
    st.write(f"Inglês: {PORTIFOLIO['ingles']}")
    st.markdown("**Skills**")
    for s in PORTIFOLIO["skills"]:
        st.write(f"- {s}")


def tab_titulares():
    st.header("Titulares do São Paulo FC")
    df_xi = pd.DataFrame([{"Posição": k, "Jogador": v} for k, v in TITULARES.items()])
    st.dataframe(df_xi, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Linhas", len(df_xi))
    with col2:
        st.metric("Posições únicas", df_xi["Posição"].nunique())


def _resumo_basico(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "xg_for", "xg_against", "goals_for", "goals_against", "shots",
        "shots_on_target", "possession", "passes", "passes_accurate",
        "pass_accuracy", "duels_won_pct", "corners", "fouls",
        "yellow_cards", "red_cards", "goal_diff", "points",
    ]
    stats_list = []
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        mean = float(s.mean())
        median = float(s.median())
        std = float(s.std(ddof=1))
        lo, hi = ci_media_t(s)
        stats_list.append(
            {
                "métrica": c,
                "média": round(mean, 3),
                "mediana": round(median, 3),
                "desvio_padrão": round(std, 3),
                "IC95%_inf": round(lo, 3),
                "IC95%_sup": round(hi, 3),
            }
        )
    return pd.DataFrame(stats_list)


def tab_analise(df: pd.DataFrame):
    st.header("Análise de Dados - São Paulo FC")

    if df.empty:
        st.warning("Nenhum jogo encontrado com os filtros atuais. Ajuste os filtros na barra lateral.")
        return

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Jogos", len(df))
    with col_b:
        st.metric("Pontos", int(df["points"].sum()))
    with col_c:
        st.metric("Vitórias", int(df["win"].sum()))
    with col_d:
        st.metric("Gols Pró (média)", round(float(df["goals_for"].mean()), 2))

    st.subheader("Distribuições e Dispersão")
    c1, c2 = st.columns(2)
    with c1:
        fig_g = px.box(
            df,
            y="goals_for",
            points="all",
            title="Gols Pró - Boxplot",
            template="plotly_white",
        )
        st.plotly_chart(fig_g, use_container_width=True)
    with c2:
        fig_xg = px.box(
            df,
            y="xg_for",
            points="all",
            title="xG Pró - Boxplot",
            template="plotly_white",
        )
        st.plotly_chart(fig_xg, use_container_width=True)

    st.subheader("Correlação entre Métricas")
    corr_cols = [
        "xg_for", "xg_against", "goals_for", "goals_against", "shots",
        "shots_on_target", "possession", "passes", "pass_accuracy",
        "duels_won_pct", "corners", "fouls", "goal_diff", "points",
    ]
    cm = df[corr_cols].corr(numeric_only=True)
    fig_corr = px.imshow(
        cm,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Matriz de Correlação",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Relação entre xG e Gols")
    if _HAS_SM:
        fig_sc = px.scatter(
            df,
            x="xg_for",
            y="goals_for",
            trendline="ols",
            title="xG Pró vs Gols Pró (OLS)",
            template="plotly_white",
        )
    else:
        fig_sc = px.scatter(
            df,
            x="xg_for",
            y="goals_for",
            title="xG Pró vs Gols Pró",
            template="plotly_white",
        )
        st.info("Para linha de tendência OLS, instale `statsmodels`.")
    st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("Resumo Estatístico com IC 95%")
    st.dataframe(_resumo_basico(df), use_container_width=True)

    st.subheader("Teste de Hipótese: Casa vs Fora (Gols Pró)")
    home = df.loc[df["home"], "goals_for"]
    away = df.loc[~df["home"], "goals_for"]
    diff, tstat, dfree, pval = welch_ttest(home, away)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Diferença de médias", round(diff, 3) if not math.isnan(diff) else "nan")
    col2.metric("t estatística", round(tstat, 3) if not math.isnan(tstat) else "nan")
    col3.metric("gl (aprox.)", round(dfree, 1) if not math.isnan(dfree) else "nan")
    if stats is None or math.isnan(pval):
        col4.metric("p-valor", "nan")
        st.info(
            "SciPy não disponível ou gl indefinido. p-valor omitido; use a diferença de médias como referência. "
            "Para p-valor, instale `scipy`."
        )
    else:
        col4.metric("p-valor", f"{pval:.4f}")
    st.caption(
        "H0: média de gols pró em casa = média de gols pró fora. "
        "H1: são diferentes (teste t de Welch, bicaudal)."
    )


def tab_relatorio(df: pd.DataFrame):
    st.header("Relatório Automático")

    if df.empty:
        st.warning("Sem dados para o relatório com os filtros atuais.")
        return

    win_rate = df["win"].mean()
    avg_xg = df["xg_for"].mean()
    avg_ga = df["goals_against"].mean()
    avg_poss = df["possession"].mean()
    acc = df["pass_accuracy"].mean()
    top_opp = (
        df.groupby("opponent")["points"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    texto = (
        f"- Taxa de vitórias: {win_rate:.1%}\n"
        f"- xG pró médio: {avg_xg:.2f}\n"
        f"- Gols contra médio: {avg_ga:.2f}\n"
        f"- Posse de bola média: {avg_poss:.1%}\n"
        f"- Precisão de passe média: {acc:.1%}\n"
        f"- Adversários com mais pontos somados: {', '.join(top_opp)}"
    )
    st.text(texto)

    fig_pts = px.line(
        df.assign(jogo=np.arange(1, len(df) + 1)),
        x="jogo",
        y="points",
        markers=True,
        title="Pontos por jogo",
        template="plotly_white",
    )
    st.plotly_chart(fig_pts, use_container_width=True)

    st.subheader("Exportar Base")
    col_a, col_b = st.columns(2)
    with col_a:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar CSV",
            data=csv,
            file_name="spfc_partidas.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_b:
        # date já é datetime64, então exporta bonito no ISO
        j = df.to_json(orient="records", force_ascii=False, date_format="iso")
        st.download_button(
            "Baixar JSON",
            data=j.encode("utf-8"),
            file_name="spfc_partidas.json",
            mime="application/json",
            use_container_width=True,
        )


# --------------------------------- Aplicativo ---------------------------------

def main():
    st.set_page_config(
        page_title="São Paulo FC - Análise de Dados | Portifólio",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Configuração da Simulação")
    seed = st.sidebar.number_input("Seed (opcional)", value=42, step=1)
    n_matches = st.sidebar.slider("Quantidade de jogos", 10, 60, 38, 1)
    start_year = st.sidebar.selectbox("Ano inicial", [2023, 2024, 2025], index=1)
    start_dt = date(int(start_year), 1, 10)

    cfg = MatchConfig(n_matches=int(n_matches), start_date=start_dt, seed=int(seed) if seed is not None else None)
    df = gerar_partidas(cfg)

    st.sidebar.markdown("### Filtros")
    casa_fora = st.sidebar.multiselect(
        "Local",
        options=["Casa", "Fora"],
        default=["Casa", "Fora"],
    )
    opp_sel = st.sidebar.multiselect(
        "Adversários",
        options=sorted(df["opponent"].unique().tolist()),
        default=[],
    )

    # Filtros
    mask = pd.Series(True, index=df.index)
    if casa_fora and len(casa_fora) < 2:
        # "Casa" -> True, "Fora" -> False
        mask &= df["home"].eq("Casa" in casa_fora)
    if opp_sel:
        mask &= df["opponent"].isin(opp_sel)
    df_f = df.loc[mask].reset_index(drop=True)

    tabs = st.tabs(["Portfólio", "Titulares", "Análise de Dados", "Relatório"])
    with tabs[0]:
        tab_portfolio()
    with tabs[1]:
        tab_titulares()
    with tabs[2]:
        tab_analise(df_f)
    with tabs[3]:
        tab_relatorio(df_f)


if __name__ == "__main__":
    main()
