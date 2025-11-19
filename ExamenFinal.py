# ==========================================================
# finance app — Análisis Financiero con IA (CORREGIDO)
# Autor: Eduardo Vázquez Figueroa (UP)
# ==========================================================

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ====== CONFIGURACIÓN BASE ======
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # carga .env

# ====== GEMINI CONFIG ======
# ====== GEMINI CONFIG ======
import google.generativeai as genai

# Intentaremos varios modelos de Gemini.
# Nos quedamos con el primero que funcione para tu cuenta.
GEMINI_MODELS = [
    "models/gemini-2.5-pro-preview-03-25",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro-preview-05-06",
    "models/gemini-2.5-pro-preview-06-05",
    "models/gemini-2.5-pro",
    "models/gemini-2.0-flash-exp",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
    "models/gemini-2.0-flash-exp-image-generation",
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash-lite-preview-02-05",
    "models/gemini-2.0-flash-lite-preview",
    "models/gemini-2.0-pro-exp",
    "models/gemini-2.0-pro-exp-02-05",
    "models/gemini-exp-1206",
    "models/gemini-2.0-flash-thinking-exp-01-21",
    "models/gemini-2.0-flash-thinking-exp",
    "models/gemini-2.0-flash-thinking-exp-1219",
    "models/gemini-2.5-flash-preview-tts",
    "models/gemini-2.5-pro-preview-tts",
    "models/learnlm-2.0-flash-experimental",
    "models/gemma-3-1b-it",
    "models/gemma-3-4b-it",
    "models/gemma-3-12b-it",
    "models/gemma-3-27b-it",
    "models/gemma-3n-e4b-it",
    "models/gemma-3n-e2b-it",
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
    "models/gemini-pro-latest",
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.5-flash-image-preview",
    "models/gemini-2.5-flash-image",
    "models/gemini-2.5-flash-preview-09-2025",
    "models/gemini-2.5-flash-lite-preview-09-2025",
    "models/gemini-3-pro-preview",
    "models/gemini-robotics-er-1.5-preview",
    "models/gemini-2.5-computer-use-preview-10-2025"
]


def get_gemini_api_key() -> str:
    """Obtiene la API key desde .env / secrets.toml / variables de entorno."""
    key = None
    # 1) primero intenta leer de secrets.toml (si lo usas en el futuro)
    try:
        key = st.secrets.get("gemini", {}).get("api_key")
    except Exception:
        key = None

    # 2) variables de entorno (.env ya está cargado con load_dotenv)
    if not key:
        key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GENAI_API_KEY")
        )

    return key.strip() if key else ""


# ==========================================================
#  IA: FUNCIÓN GENERAL DE Q&A
# ==========================================================

def _extract_gemini_text(resp) -> str:
    """
    Extrae texto de la respuesta de Gemini de forma robusta.
    Si resp.text viene vacío, intenta leer candidates/parts.
    """
    # Caso directo: resp.text
    try:
        if getattr(resp, "text", None):
            t = resp.text.strip()
            if t:
                return t
    except Exception:
        pass

    # Caso avanzado: candidates[0].content.parts[*].text
    try:
        cand = resp.candidates[0]
        parts = getattr(cand, "content", cand).parts
        texts = [getattr(p, "text", "") for p in parts]
        t = "\n".join([x for x in texts if x]).strip()
        if t:
            return t
    except Exception:
        pass

    return ""

def ai_general_qna(context_text: str, question: str) -> str:
    """
    Hace una pregunta a Gemini usando el contexto de la app.
    Intenta varios modelos de GEMINI_MODELS y devuelve siempre texto legible.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return "La IA no está disponible porque no se encontró la API KEY."

    genai.configure(api_key=api_key)

    errores = []

    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)

            prompt = f"""
Contexto financiero:

\"\"\"{context_text}\"\"\"

Pregunta del usuario:
\"\"\"{question}\"\"\"

Responde en español con claridad y concisión.
"""
            resp = model.generate_content(prompt)
            texto = _extract_gemini_text(resp)

            if texto:
                return texto

            # Si respondió pero sin texto, lo consideramos error suave
            errores.append(f"{model_name}: respuesta vacía")

        except Exception as e:
            errores.append(f"{model_name}: {type(e).__name__}: {e}")

    # Si llegamos aquí, ningún modelo funcionó
    return (
        "No fue posible usar la IA para responder en este momento.\n"
        "Se intentaron varios modelos de Gemini pero ninguno respondió correctamente.\n\n"
        "Modelos probados (modelo: tipo de error):\n  - "
        + "\n  - ".join(errores)
        + "\n\nRevisa que tu cuenta tenga acceso a al menos uno de estos modelos "
          "y que la librería 'google-generativeai' esté actualizada en tu entorno."
    )

# ==========================================================
# DESCARGA DE DATOS
# ==========================================================

def get_ticker_data(ticker: str, years=5):
    """
    Descarga precios históricos con yfinance y asegura que las columnas
    sean simples: 'Open', 'High', 'Low', 'Close', 'Volume'.

    Esto soluciona el problema de columnas tipo ('Close', 'AAPL').
    """
    try:
        data = yf.download(ticker, period=f"{years}y")

        if data is None or data.empty:
            return None

        # Si las columnas vienen como MultiIndex (ej. ('Close','AAPL')),
        # las convertimos a una sola capa usando solo el primer nivel.
        if isinstance(data.columns, pd.MultiIndex):
            # nos quedamos con el primer elemento de cada tupla
            data.columns = [col[0] for col in data.columns]

        # limpiamos filas con datos faltantes
        data = data.dropna()

        return data

    except Exception:
        return None



# ==========================================================
# MÉTRICAS BÁSICAS
# ==========================================================

def compute_basic_metrics(df: pd.DataFrame):
    """Rendimiento 1Y, precio actual, cambio diario (todo como números float)."""
    # Precio último
    price_now = float(df["Close"].iloc[-1])

    # Rendimiento 1 año (usamos 252 días como aproximación)
    ret_series_1y = df["Close"].pct_change(252)
    ret_1y = float(ret_series_1y.iloc[-1] * 100)

    # Cambio diario más reciente
    ret_daily = df["Close"].pct_change()
    daily_chg = float(ret_daily.iloc[-1] * 100)

    return price_now, ret_1y, daily_chg


# ==========================================================
# MONTE CARLO CORREGIDO (HASTA 4 TICKERS)
# ==========================================================

def monte_carlo_multi(price_dict: dict, n_paths=500, n_days=252):
    """
    Monte Carlo multiactivo (hasta 4 tickers) usando rendimientos logarítmicos
    y matriz de correlaciones histórica.

    Devuelve:
        {
            "tickers": [..],
            "sims": sims (paths, days, assets),
            "start_prices": array de precios iniciales,
            "dates": índice de fechas simuladas (días hábiles)
        }
    """
    if not price_dict:
        return None

    # Nos quedamos con máximo 4 tickers
    tickers = list(price_dict.keys())[:4]

    # Precios iniciales en el mismo orden
    start_prices = np.array([
        float(price_dict[t]["Close"].iloc[-1]) for t in tickers
    ])

    # Rendimientos logarítmicos diarios
    log_returns = pd.DataFrame({
        t: np.log(price_dict[t]["Close"]).diff().dropna()
        for t in tickers
    })

    if log_returns.empty:
        return None

    mu = log_returns.mean().values      # media diaria
    sigma = log_returns.std().values    # volatilidad diaria
    corr = log_returns.corr().values

    # Descomposición de Cholesky (si falla, ajustamos un poco la matriz)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky((corr + np.eye(len(tickers))) / 2.0)

    sims = np.zeros((n_paths, n_days, len(tickers)))

    for p in range(n_paths):
        # ruido normal correlacionado
        z = np.random.normal(size=(n_days, len(tickers)))
        z_corr = z @ L

        prices = np.zeros((n_days, len(tickers)))
        prices[0] = start_prices

        for d in range(1, n_days):
            drift = mu - 0.5 * sigma**2
            shock = sigma * z_corr[d]
            prices[d] = prices[d - 1] * np.exp(drift + shock)

        sims[p] = prices

    # Fechas simuladas (días hábiles a partir del último dato histórico)
    last_date = list(price_dict.values())[0].index[-1]
    dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    return {
        "tickers": tickers,
        "sims": sims,
        "start_prices": start_prices,
        "dates": dates,
    }



# ==========================================================
# STREAMLIT — INTERFAZ
# ==========================================================

st.set_page_config(
    page_title="finance app — Análisis Financiero con IA",
    layout="wide"
)

st.title("💼 finance app — Análisis Financiero con IA")
st.caption("Ingeniería Financiera · Universidad Panamericana · Eduardo Vázquez Figueroa")

# SIDEBAR
st.sidebar.header("⚙️ Parámetros")

ticker_main = st.sidebar.text_input("Ticker principal", "AAPL")
benchmark = st.sidebar.text_input("Índice de referencia", "SPY")

st.sidebar.markdown("---")
st.sidebar.header("Comparación múltiple (máx. 3)")

t1 = st.sidebar.text_input("Ticker extra 1")
t2 = st.sidebar.text_input("Ticker extra 2")
t3 = st.sidebar.text_input("Ticker extra 3")

# ==========================================================
# DATOS DEL TICKER PRINCIPAL
# ==========================================================

df_main = get_ticker_data(ticker_main)

if df_main is None:
    st.error("No se pudo obtener información del ticker principal.")
    st.stop()
# ==========================================================
# CONTINÚA: MÉTRICAS, GRÁFICAS Y TABS
# ==========================================================

# Datos del benchmark
df_bench = get_ticker_data(benchmark)

# Tickers extra válidos
others = [t.strip().upper() for t in [t1, t2, t3] if t.strip()]
price_dict = {ticker_main.upper(): df_main}
for o in others:
    df_o = get_ticker_data(o)
    if df_o is not None:
        price_dict[o] = df_o

# ====== FUNCIONES DE APOYO PARA GRÁFICAS ======

def plot_price_series(df: pd.DataFrame, ticker: str, title_suffix="(5 años)"):
    """Gráfica simple de precio de cierre."""
    fig = px.line(
        df,
        x=df.index,
        y="Close",
        title=f"{ticker} — Precio de cierre {title_suffix}",
        labels={"x": "Fecha", "Close": "Precio de cierre"},
    )
    fig.update_layout(height=400)
    return fig


def plot_candles(df: pd.DataFrame, ticker: str):
    """Velas japonesas con medias móviles."""
    d = df.tail(252)  # ~1 año
    ma50 = d["Close"].rolling(50).mean()
    ma200 = d["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index,
        open=d["Open"],
        high=d["High"],
        low=d["Low"],
        close=d["Close"],
        name=ticker
    ))
    fig.add_trace(go.Scatter(x=d.index, y=ma50, name="MA50"))
    fig.add_trace(go.Scatter(x=d.index, y=ma200, name="MA200"))
    fig.update_layout(
        title=f"{ticker} — Velas japonesas (1 año) con MA50/MA200",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        height=500
    )
    return fig


def plot_base_zero(df_s: pd.DataFrame, df_i: pd.DataFrame, t_s: str, t_i: str):
    """Comparativo base-cero vs índice."""
    ret_s = df_s["Close"].pct_change().fillna(0)
    ret_i = df_i["Close"].pct_change().fillna(0)

    # acumulado
    cum_s = (1 + ret_s).cumprod() - 1
    cum_i = (1 + ret_i).cumprod() - 1

    df_plot = pd.DataFrame({
        t_s: cum_s * 100,
        t_i: cum_i * 100
    }).dropna()

    fig = px.line(
        df_plot,
        x=df_plot.index,
        y=[t_s, t_i],
        labels={"value": "Rendimiento acumulado (%)", "variable": "Activo"},
        title=f"Comparación base cero: {t_s} vs {t_i}"
    )
    fig.update_layout(height=400)
    return fig


def build_multi_table(price_dict: dict, bench_df: pd.DataFrame | None):
    """
    Tabla resumen de rendimiento 1Y, YTD y beta simple vs índice.
    Corrige el problema de longitudes distintas entre el índice de df y la serie de rendimientos.
    """
    rows = []

    # Rendimientos del índice (sin dropna todavía)
    bench_ret = None
    if bench_df is not None and not bench_df.empty:
        bench_ret = bench_df["Close"].pct_change()

    for t, df in price_dict.items():
        # Rendimientos diarios del activo (sin dropna)
        r = df["Close"].pct_change()

        # Precio actual
        price_now = float(df["Close"].iloc[-1])

        # --- Rendimiento 1Y ---
        r_1y = r.tail(252).dropna()
        if not r_1y.empty:
            ret_1y = (r_1y.add(1).prod() - 1) * 100
        else:
            ret_1y = np.nan

        # --- Rendimiento YTD ---
        current_year = df.index[-1].year
        mask_ytd = r.index.year == current_year  # misma longitud que r
        r_ytd = r[mask_ytd].dropna()
        if not r_ytd.empty:
            ret_ytd = (r_ytd.add(1).prod() - 1) * 100
        else:
            ret_ytd = np.nan

        # --- Beta vs índice ---
        beta = np.nan
        if bench_ret is not None:
            # alineamos por fecha y luego dropna
            joined = pd.concat([r, bench_ret], axis=1, join="inner").dropna()
            if len(joined) > 50:
                cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1]
                var = np.var(joined.iloc[:, 1])
                if var > 0:
                    beta = cov / var

        rows.append({
            "Ticker": t,
            "Precio actual": round(price_now, 2),
            "Rendimiento 1Y (%)": round(ret_1y, 2) if pd.notna(ret_1y) else None,
            "Rendimiento YTD (%)": round(ret_ytd, 2) if pd.notna(ret_ytd) else None,
            "Beta vs índice": round(beta, 2) if not np.isnan(beta) else None,
        })

    return pd.DataFrame(rows)



def plot_corr_heatmap(price_dict: dict):
    """Matriz de correlaciones de rendimientos diarios."""
    rets = {}
    for t, df in price_dict.items():
        r = df["Close"].pct_change().dropna()
        rets[t] = r

    if len(rets) < 2:
        return None

    rets_df = pd.DataFrame(rets).dropna()
    corr = rets_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        zmin=-1,
        zmax=1,
        title="Matriz de correlaciones (rendimientos diarios)",
        labels=dict(color="Correlación")
    )
    fig.update_layout(height=500)
    return fig


def rolling_vol_and_sharpe(df: pd.DataFrame, window=60):
    """Volatilidad y Sharpe móviles (diarios a anualizados)."""
    r = df["Close"].pct_change().dropna()
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std()

    vol_ann = roll_std * np.sqrt(252)
    ret_ann = roll_mean * 252
    sharpe = ret_ann / vol_ann.replace(0, np.nan)

    out = pd.DataFrame({
        "Volatilidad anualizada": vol_ann,
        "Sharpe móvil": sharpe
    }).dropna()
    return out


def ai_explain_chart(
    ticker: str,
    chart_name: str,
    df: pd.DataFrame | None = None,
    extra_text: str = "",
    key: str = ""
):
    """
    Muestra un botón que, al pulsarlo, pide a la IA una explicación sencilla
    del gráfico correspondiente.
    """
    if st.button("💡 Explicación del gráfico con IA", key=key):
        lineas = [
            f"Ticker analizado: {ticker.upper()}",
            f"Tipo de gráfico: {chart_name}.",
        ]

        if extra_text:
            lineas.append(extra_text)

        # Si hay serie de precios, añadimos contexto numérico simple
        if df is not None and not df.empty and "Close" in df.columns:
            try:
                price_start = float(df["Close"].iloc[0])
                price_now = float(df["Close"].iloc[-1])
                ret_total = (price_now / price_start - 1) * 100
                lineas.append(f"Precio inicial de la serie: {price_start:,.2f} USD")
                lineas.append(f"Precio final de la serie: {price_now:,.2f} USD")
                lineas.append(f"Rendimiento total aproximado: {ret_total:,.2f}%")
            except Exception:
                pass

        contexto = "\n".join(lineas)

        pregunta = (
            "Explica de forma MUY sencilla qué muestra este gráfico y cómo debe "
            "interpretarlo un estudiante de licenciatura en finanzas. "
            "Máximo 6 líneas, en español."
        )

        respuesta = ai_general_qna(contexto, pregunta)
        st.markdown("**Comentario de la IA sobre este gráfico:**")
        st.write(respuesta)


def translate_summary_to_spanish(ticker: str, english_text: str) -> str:
    """
    Usa la IA para traducir al español la descripción larga de la empresa.
    Si algo falla, regresa el texto original en inglés.
    """
    if not english_text:
        return ""

    contexto = (
        f"Descripción original en inglés de la empresa {ticker.upper()}:\n"
        f"\"\"\"{english_text}\"\"\""
    )

    pregunta = (
        "Traduce este texto al español neutro. Mantén el tono formal y financiero. "
        "No resumas, no agregues comentarios ni explicaciones, "
        "sólo devuelve el texto traducido."
    )

    traducido = ai_general_qna(contexto, pregunta)

    # Si por alguna razón la IA falla, usamos el texto original
    if not traducido:
        return english_text

    texto_lower = traducido.lower()
    if "no fue posible usar la ia" in texto_lower or "la ia no está disponible" in texto_lower:
        return english_text

    return traducido

# ==========================================================
# TABS PRINCIPALES
# ==========================================================

tab_overview, tab_desc, tab_tech, tab_comp, tab_multi, tab_advanced, tab_ai = st.tabs([
    "🏠 Overview",
    "🏢 Descripción",
    "📈 Análisis técnico",
    "📊 Comparativo",
    "🔍 Comparación múltiple",
    "⚙️ Análisis avanzado",
    "🧠 IA financiera",
])

# ----------------------------------------------------------
# TAB 1 — OVERVIEW
# ----------------------------------------------------------
with tab_overview:
    st.subheader(f"🏠 Overview — {ticker_main.upper()}")

    price_now, ret_1y, daily_chg = compute_basic_metrics(df_main)

    c1, c2, c3 = st.columns(3)
    c1.metric("Precio actual", f"${price_now:,.2f}")
    c2.metric("Rendimiento 1 año", f"{ret_1y:,.2f}%")
    c3.metric("Cambio diario", f"{daily_chg:,.2f}%")

    # Gráfico principal de precio
    st.plotly_chart(
        plot_price_series(df_main, ticker_main.upper()),
        use_container_width=True
    )

    # Botón de explicación con IA para este gráfico
    ai_explain_chart(
        ticker=ticker_main,
        chart_name="Gráfico de precio histórico (línea)",
        df=df_main,
        extra_text="Es una serie temporal del precio de cierre diario de la acción.",
        key="ai_overview_price",
    )


# ----------------------------------------------------------
# TAB 2 — DESCRIPCIÓN (INFO SIMPLE CON YFINANCE)
# ----------------------------------------------------------
def get_company_info(ticker: str):
    try:
        t = yf.Ticker(ticker)
        info = t.info  # puede ser lento pero suficiente
    except Exception:
        return None

    if not info:
        return None

    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector", "N/D"),
        "industry": info.get("industry", "N/D"),
        "country": info.get("country", "N/D"),
        "summary": info.get("longBusinessSummary", "Sin descripción disponible.")
    }

with tab_desc:
    st.subheader("🏢 Descripción de la empresa")

    info = get_company_info(ticker_main)
    if info is None:
        st.info("No se pudo obtener la descripción de la empresa con yfinance.")
    else:
        st.markdown(f"**{info['name']}**")
        st.write(
            f"_Sector:_ {info['sector']} · "
            f"_Industria:_ {info['industry']} · "
            f"_País:_ {info['country']}"
        )

        # Texto original de yfinance (en inglés)
        summary_en = info.get("summary", "")

        # Versión en español usando IA (si falla, regresa el inglés)
        summary_es = translate_summary_to_spanish(ticker_main, summary_en)

        # Mostramos siempre el texto que devuelva la función
        st.write(summary_es)

        # Si realmente fue traducido, mostramos una nota
        if summary_es != summary_en:
            st.caption("Traducción automática al español realizada con IA (Gemini).")


# ----------------------------------------------------------
# TAB 3 — ANÁLISIS TÉCNICO
# ----------------------------------------------------------
with tab_tech:
    st.subheader("📈 Análisis técnico")

    # Gráfico de velas japonesas
    st.plotly_chart(
        plot_candles(df_main, ticker_main.upper()),
        use_container_width=True
    )

    # Botón IA para explicar velas japonesas
    ai_explain_chart(
        ticker=ticker_main,
        chart_name="Gráfico de velas japonesas con medias móviles",
        df=df_main.tail(252),
        extra_text=(
            "Cada vela muestra la variación diaria del precio (apertura, máximo, "
            "mínimo y cierre) junto con medias móviles de 50 y 200 días."
        ),
        key="ai_tech_candles",
    )

# ----------------------------------------------------------
# TAB 4 — COMPARATIVO VS ÍNDICE
# ----------------------------------------------------------
with tab_comp:
    st.subheader("📊 Comparativo vs índice")

    if df_bench is None or df_bench.empty:
        st.info("No se pudo descargar información del índice de referencia.")
    else:
        st.plotly_chart(
            plot_base_zero(df_main, df_bench, ticker_main.upper(), benchmark.upper()),
            use_container_width=True
        )

        # Botón IA para explicar comparación vs índice
        ai_explain_chart(
            ticker=ticker_main,
            chart_name=f"Comparación base cero frente al índice {benchmark.upper()}",
            df=df_main,
            extra_text=(
                "El gráfico muestra el rendimiento acumulado del ticker y del índice, "
                "ambos empezando en 0% el mismo día para comparar fácilmente su desempeño."
            ),
            key="ai_comp_basezero",
        )


# ----------------------------------------------------------
# TAB 5 — COMPARACIÓN MÚLTIPLE
# ----------------------------------------------------------
with tab_multi:
    st.subheader("🔍 Comparación múltiple")

    df_table = build_multi_table(price_dict, df_bench)
    st.dataframe(df_table, use_container_width=True)

    if len(price_dict) >= 2:
        fig_corr = plot_corr_heatmap(price_dict)
        if fig_corr is not None:
            st.plotly_chart(fig_corr, use_container_width=True)

            # Botón IA para explicar la matriz de correlaciones
            ai_explain_chart(
                ticker=ticker_main,
                chart_name="Matriz de correlaciones entre los activos seleccionados",
                df=None,
                extra_text=(
                    "Cada celda indica qué tan relacionados están los movimientos "
                    "de los rendimientos diarios de dos activos (de -1 a 1)."
                ),
                key="ai_multi_corr",
            )
    else:
        st.info("Agrega más tickers en la barra lateral para comparar y ver la correlación.")


# ----------------------------------------------------------
# TAB 6 — ANÁLISIS AVANZADO (MONTE CARLO + FRONTERA)
# ----------------------------------------------------------
with tab_advanced:
    st.subheader("⚙️ Análisis avanzado")

    st.markdown("### 🎲 Simulación Monte Carlo (1 año)")
    if len(price_dict) == 0:
        st.info("No hay suficientes datos para la simulación.")
    else:
        mc_result = monte_carlo_multi(price_dict, n_paths=500, n_days=252)
        if mc_result is None:
            st.info("No fue posible ejecutar la simulación (pocos datos históricos).")
        else:
            tickers_mc = mc_result["tickers"]
            sims = mc_result["sims"]          # (paths, days, assets)
            start_prices = mc_result["start_prices"]
            dates = mc_result["dates"]

            # --- Gráfico: varias trayectorias del ticker principal ---
            try:
                main_index = tickers_mc.index(ticker_main.upper())
            except ValueError:
                main_index = 0  # por si el nombre no coincide exacto

            n_show = min(50, sims.shape[0])  # número de caminos que mostraremos
            paths_main = sims[:n_show, :, main_index]  # (paths, days)

            # DataFrame con caminos como columnas
            df_paths = pd.DataFrame(paths_main.T, index=dates)
            df_paths.columns = [f"Camino {i+1}" for i in range(n_show)]

            df_plot = df_paths.reset_index().melt(
                id_vars="index",
                var_name="Camino",
                value_name="Precio"
            )

            fig_mc = px.line(
                df_plot,
                x="index",
                y="Precio",
                color="Camino",
                labels={"index": "Fecha"},
                title=f"Trayectorias simuladas de precio (Monte Carlo, {tickers_mc[main_index]})"
            )
            fig_mc.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_mc, use_container_width=True)

            # Botón IA para explicar la simulación Monte Carlo
            ai_explain_chart(
                ticker=ticker_main,
                chart_name="Simulación Monte Carlo de precio a 1 año",
                df=None,
                extra_text=(
                    "Cada línea representa un posible camino futuro del precio, "
                    "simulado usando rendimientos históricos, volatilidad y correlaciones."
                ),
                key="ai_mc_main",
            )

            # --- Tabla resumen por ticker ---
            final_prices = sims[:, -1, :]  # (paths, assets)
            rows = []
            for i, t in enumerate(tickers_mc):
                p0 = start_prices[i]
                pf = final_prices[:, i]
                exp_price = pf.mean()
                exp_ret = (exp_price / p0 - 1) * 100
                p5 = np.percentile(pf, 5)
                p95 = np.percentile(pf, 95)
                r5 = (p5 / p0 - 1) * 100
                r95 = (p95 / p0 - 1) * 100

                rows.append({
                    "Ticker": t,
                    "Precio inicial": round(p0, 2),
                    "Precio esp. 1 año": round(exp_price, 2),
                    "Retorno esp. 1 año (%)": round(exp_ret, 2),
                    "Rango 5%-95% retorno (%)": f"{round(r5, 2)}% a {round(r95, 2)}%",
                })

            st.markdown("#### Resumen de Monte Carlo (1 año)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📉 Volatilidad y Sharpe móviles (ticker principal)")

    roll_df = rolling_vol_and_sharpe(df_main)
    col1, col2 = st.columns(2)

    with col1:
        fig_vol = px.line(
            roll_df,
            y="Volatilidad anualizada",
            title="Volatilidad anualizada (ventana móvil)",
            labels={"index": "Fecha", "Volatilidad anualizada": "Volatilidad"}
        )
        fig_vol.update_layout(height=350)
        st.plotly_chart(fig_vol, use_container_width=True)

        # Botón IA para explicar la volatilidad móvil
        ai_explain_chart(
            ticker=ticker_main,
            chart_name="Volatilidad anualizada en ventana móvil",
            df=None,
            extra_text=(
                "La línea muestra cómo ha ido cambiando el nivel de volatilidad del "
                "activo a lo largo del tiempo, usando una ventana móvil de rendimientos."
            ),
            key="ai_vol_movil",
        )

    with col2:
        fig_sh = px.line(
            roll_df,
            y="Sharpe móvil",
            title="Sharpe móvil (ventana móvil)",
            labels={"index": "Fecha", "Sharpe móvil": "Sharpe"}
        )
        fig_sh.update_layout(height=350)
        st.plotly_chart(fig_sh, use_container_width=True)

        # Botón IA para explicar el Sharpe móvil
        ai_explain_chart(
            ticker=ticker_main,
            chart_name="Ratio de Sharpe en ventana móvil",
            df=None,
            extra_text=(
                "El ratio de Sharpe móvil indica si el rendimiento obtenido compensa "
                "el riesgo asumido en cada periodo de la ventana de análisis."
            ),
            key="ai_sharpe_movil",
        )


# ----------------------------------------------------------
# TAB 7 — IA FINANCIERA
# ----------------------------------------------------------
with tab_ai:
    st.subheader("🧠 IA financiera (Q&A)")

    api_preview = get_gemini_api_key()
    if api_preview:
        st.caption("Gemini: API configurada ✅")
    else:
        st.caption("Gemini: API no configurada ❌ (revisa tu archivo .env o variables de entorno).")

    # Construimos un contexto sencillo con métricas básicas
    price_now, ret_1y, daily_chg = compute_basic_metrics(df_main)
    context_lines = [
        f"Ticker: {ticker_main.upper()}",
        f"Precio actual: {price_now:,.2f} USD",
        f"Rendimiento 1 año: {ret_1y:,.2f}%",
        f"Cambio diario reciente: {daily_chg:,.2f}%",
    ]
    context_text = "\n".join(context_lines)

    question = st.text_area(
        "Escribe tu pregunta sobre esta acción (en español):",
        value="¿Qué opinas del rendimiento reciente de esta empresa?",
        height=120
    )

    if st.button("Preguntar a la IA"):
        if not question.strip():
            st.warning("Escribe una pregunta primero.")
        else:
            answer = ai_general_qna(context_text, question)
            st.markdown("#### Respuesta de la IA")
            st.write(answer)
