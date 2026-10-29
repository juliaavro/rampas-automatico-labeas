
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.signal import savgol_filter
from scipy.optimize import minimize, LinearConstraint
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO DA INTERFACE
# ============================================================

st.set_page_config(
    page_title="Rampas Automáticas | LABEAS",
    layout="wide"
)

from pathlib import Path

logo = Path(__file__).parent / "5 (2) (1).png"

if logo.exists():
    st.image(str(logo), width=180)

st.title("Rampas Automáticas de Estradas Rurais")
st.caption(
    "LABEAS | Análise do perfil longitudinal e segmentação em rampas"
)


# ============================================================
# LEITURA DOS ARQUIVOS
# ============================================================

def ler_csv(arquivo):
    """Lê CSVs com tabulação, ponto e vírgula ou vírgula."""

    conteudo = arquivo.getvalue()

    try:
        texto = conteudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = conteudo.decode("cp1252")

    linhas = [
        linha.strip()
        for linha in texto.splitlines()
        if linha.strip()
    ]

    if not linhas:
        raise ValueError("O arquivo está vazio.")

    # Alguns arquivos exportados envolvem cada linha inteira em aspas.
    linhas = [
        linha[1:-1]
        if linha.startswith('"') and linha.endswith('"')
        else linha
        for linha in linhas
    ]

    primeira_linha = linhas[0]

    if "\t" in primeira_linha:
        separador = "\t"
    elif ";" in primeira_linha:
        separador = ";"
    else:
        separador = ","

    dados = pd.DataFrame(
        [linha.split(separador) for linha in linhas]
    )

    if dados.shape[1] < 2:
        raise ValueError(
            "Não foi possível identificar duas colunas. "
            "Verifique se os dados estão separados por tabulação, "
            "ponto e vírgula ou vírgula."
        )

    # Utiliza as duas primeiras colunas.
    dados = dados.iloc[:, :2].copy()
    dados.columns = ["Distancia", "Cota"]

    return dados


# ============================================================
# MATRIZ PARA REPRESENTAR O GREIDE CONTÍNUO
# ============================================================

def matriz_interpolacao(x, indices_nos):
    """Representa um greide linear contínuo entre os nós."""

    x_nos = x[indices_nos]
    A = np.zeros((len(x), len(indices_nos)))

    for i, xi in enumerate(x):

        if xi >= x_nos[-1]:
            A[i, -1] = 1
            continue

        j = np.searchsorted(
            x_nos, xi, side="right"
        ) - 1

        j = max(0, min(j, len(x_nos) - 2))

        dx = x_nos[j + 1] - x_nos[j]
        t = (xi - x_nos[j]) / dx

        A[i, j] = 1 - t
        A[i, j + 1] = t

    return A


# ============================================================
# AJUSTE DAS COTAS DOS NÓS
# ============================================================

def ajustar_nos(
    x,
    y,
    indices_nos,
    limite_inclinacao=None
):
    """Ajusta as cotas dos nós e mantém continuidade entre rampas."""

    A = matriz_interpolacao(x, indices_nos)
    quantidade = len(indices_nos)

    # Diferença entre cotas de nós consecutivos.
    D = np.zeros((quantidade - 1, quantidade))

    for i in range(quantidade - 1):
        D[i, i] = -1
        D[i, i + 1] = 1

    dx_nos = np.diff(x[indices_nos])

    def objetivo(z):
        residuos = A @ z - y
        return float(residuos @ residuos)

    # Ponto inicial.
    z0 = np.interp(
        x[indices_nos],
        x,
        y
    )

    restricoes = []

    if limite_inclinacao is not None:

        limite = limite_inclinacao / 100

        restricoes.append(
            LinearConstraint(
                D,
                -limite * dx_nos,
                limite * dx_nos
            )
        )

    resultado = minimize(
        objetivo,
        z0,
        method="SLSQP",
        constraints=restricoes,
        options={
            "maxiter": 1000,
            "ftol": 1e-9
        }
    )

    if not resultado.success:
        raise ValueError(
            "Não foi possível ajustar o greide com os parâmetros "
            "atuais. Tente aumentar o limite de inclinação, "
            "reduzir o número máximo de rampas ou revisar os dados."
        )

    perfil = A @ resultado.x

    return perfil, resultado.x


# ============================================================
# SEGMENTAÇÃO AUTOMÁTICA EM RAMPAS
# ============================================================

def calcular_greide(
    x,
    y,
    tolerancia,
    max_rampas,
    comprimento_minimo,
    limite_inclinacao=None
):
    """
    Cria rampas contínuas e insere nós onde o desvio do perfil
    é maior que a tolerância definida.
    """

    indices_nos = [0, len(x) - 1]

    perfil, cotas_nos = ajustar_nos(
        x,
        y,
        indices_nos,
        limite_inclinacao
    )

    while len(indices_nos) - 1 < max_rampas:

        residuos = np.abs(y - perfil)
        candidatos = []

        for i in range(1, len(x) - 1):

            if i in indices_nos:
                continue

            pos = np.searchsorted(indices_nos, i)

            esquerda = indices_nos[pos - 1]
            direita = indices_nos[pos]

            if (
                x[i] - x[esquerda] >= comprimento_minimo
                and
                x[direita] - x[i] >= comprimento_minimo
            ):
                candidatos.append(i)

        if not candidatos:
            break

        melhor = max(
            candidatos,
            key=lambda i: residuos[i]
        )

        if residuos[melhor] <= tolerancia:
            break

        indices_nos.append(melhor)
        indices_nos.sort()

        perfil, cotas_nos = ajustar_nos(
            x,
            y,
            indices_nos,
            limite_inclinacao
        )

    return perfil, indices_nos, cotas_nos


# ============================================================
# IMPORTAÇÃO DA PLANILHA
# ============================================================

arquivo = st.file_uploader(
    "Importe os dados do terreno natural (.xlsx ou .csv)",
    type=["xlsx", "csv"]
)

if arquivo is None:

    st.info(
        "Carregue uma planilha com duas colunas: "
        "distância em metros e cota do terreno natural."
    )

    st.markdown(
        """
        **Exemplo de dados**

        | Distância (m) | Cota do terreno (m) |
        |---:|---:|
        | 0.00 | 32.40 |
        | 20.00 | 32.85 |
        | 40.00 | 33.10 |
        | 60.00 | 32.70 |
        | 80.00 | 32.20 |
        """
    )

    st.stop()


# ============================================================
# PROCESSAMENTO DOS DADOS
# ============================================================

try:

    if arquivo.name.lower().endswith(".csv"):
        dados = ler_csv(arquivo)
    else:
        dados = pd.read_excel(arquivo)

    if len(dados.columns) < 2:
        st.error(
            "A planilha precisa ter pelo menos duas colunas."
        )
        st.stop()

    st.subheader("1. Configurar os dados")

    colunas = list(dados.columns)

    c1, c2 = st.columns(2)

    coluna_x = c1.selectbox(
        "Coluna de distância (m)",
        colunas,
        index=0
    )

    coluna_y = c2.selectbox(
        "Coluna de cota do terreno (m)",
        colunas,
        index=1
    )

    perfil_dados = pd.DataFrame({
        "Distância (m)": pd.to_numeric(
            dados[coluna_x].astype(str).str.replace(
                ",", ".", regex=False
            ),
            errors="coerce"
        ),
        "Cota natural (m)": pd.to_numeric(
            dados[coluna_y].astype(str).str.replace(
                ",", ".", regex=False
            ),
            errors="coerce"
        )
    }).dropna()

    perfil_dados = (
        perfil_dados
        .groupby("Distância (m)", as_index=False)
        ["Cota natural (m)"]
        .mean()
        .sort_values("Distância (m)")
        .reset_index(drop=True)
    )

    x = perfil_dados["Distância (m)"].to_numpy(
        dtype=float
    )

    y = perfil_dados["Cota natural (m)"].to_numpy(
        dtype=float
    )

    if len(x) < 3:
        st.error(
            "São necessários pelo menos três pontos válidos. "
            "Confira as colunas selecionadas e os dados."
        )
        st.stop()

    if np.any(np.diff(x) <= 0):
        st.error(
            "As distâncias precisam ser distintas e crescentes."
        )
        st.stop()

    st.success(
        f"{len(x)} pontos válidos | "
        f"Extensão: {x[-1] - x[0]:.2f} m"
    )

    # Exibe uma amostra para conferência.
    with st.expander("Conferir dados importados"):
        st.dataframe(
            perfil_dados.head(20),
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # PARÂMETROS
    # ========================================================

    st.subheader("2. Parâmetros do cálculo")

    c1, c2, c3 = st.columns(3)

    suavizar = c1.checkbox(
        "Suavizar terreno natural",
        value=True
    )

    maior_janela = min(
        101,
        len(y) if len(y) % 2 == 1 else len(y) - 1
    )

    janela = c2.number_input(
        "Janela de suavização (pontos, ímpar)",
        min_value=3,
        max_value=maior_janela,
        value=min(7, maior_janela),
        step=2
    )

    ordem = c3.selectbox(
        "Ordem do filtro",
        [1, 2, 3],
        index=1
    )

    c4, c5, c6 = st.columns(3)

    tolerancia = c4.number_input(
        "Desvio para inserir nova rampa (m)",
        min_value=0.01,
        max_value=10.0,
        value=0.25,
        step=0.05
    )

    max_rampas = c5.number_input(
        "Número máximo de rampas",
        min_value=1,
        max_value=100,
        value=12
    )

    comprimento_minimo = c6.number_input(
        "Comprimento mínimo da rampa (m)",
        min_value=0.1,
        value=20.0,
        step=5.0
    )

    usar_limite = st.checkbox(
        "Aplicar limite máximo de inclinação"
    )

    limite_inclinacao = None

    if usar_limite:

        limite_inclinacao = st.number_input(
            "Inclinação máxima em módulo (%)",
            min_value=0.1,
            max_value=50.0,
            value=10.0,
            step=0.5
        )


    # ========================================================
    # SUAVIZAÇÃO
    # ========================================================

    if suavizar:

        if janela > len(y):
            st.error(
                "A janela de suavização é maior que o "
                "número de pontos disponíveis."
            )
            st.stop()

        if janela <= ordem:
            st.error(
                "A janela precisa ser maior que a ordem do filtro."
            )
            st.stop()

        y_ajustado = savgol_filter(
            y,
            int(janela),
            int(ordem)
        )

    else:
        y_ajustado = y.copy()


    # ========================================================
    # EXECUTAR CÁLCULO
    # ========================================================

    st.subheader("3. Gerar o greide")

    if st.button(
        "Calcular greide automático",
        type="primary"
    ):

        try:

            perfil, indices_nos, cotas_nos = calcular_greide(
                x,
                y_ajustado,
                tolerancia,
                int(max_rampas),
                comprimento_minimo,
                limite_inclinacao
            )

            st.session_state["resultado_greide"] = {
                "x": x.copy(),
                "y": y.copy(),
                "y_ajustado": y_ajustado.copy(),
                "perfil": perfil,
                "indices_nos": indices_nos,
                "cotas_nos": cotas_nos
            }

        except ValueError as erro:
            st.error(str(erro))


    # ========================================================
    # EXIBIÇÃO DOS RESULTADOS
    # ========================================================

    if "resultado_greide" not in st.session_state:
        st.stop()

    resultado = st.session_state["resultado_greide"]

    if (
        not np.array_equal(resultado["x"], x)
        or not np.array_equal(resultado["y"], y)
        or not np.array_equal(
            resultado["y_ajustado"], y_ajustado
        )
    ):

        st.warning(
            "Os dados ou a suavização mudaram. "
            "Clique novamente em Calcular greide automático."
        )
        st.stop()

    perfil = resultado["perfil"]
    indices_nos = resultado["indices_nos"]
    cotas_nos = resultado["cotas_nos"]


    # ========================================================
    # TABELA DE RAMPAS
    # ========================================================

    rampas = []

    for i in range(len(indices_nos) - 1):

        a = indices_nos[i]
        b = indices_nos[i + 1]

        distancia = x[b] - x[a]

        diferenca_cota = (
            cotas_nos[i + 1] - cotas_nos[i]
        )

        inclinacao = (
            100 * diferenca_cota / distancia
        )

        rampas.append({
            "Rampa": i + 1,
            "Início (m)": x[a],
            "Fim (m)": x[b],
            "Comprimento (m)": distancia,
            "Cota inicial (m)": cotas_nos[i],
            "Cota final (m)": cotas_nos[i + 1],
            "Desnível (m)": diferenca_cota,
            "Inclinação (%)": inclinacao
        })

    tabela_rampas = pd.DataFrame(rampas)

    perfil_saida = pd.DataFrame({
        "Distância (m)": x,
        "Cota natural (m)": y,
        "Cota analisada (m)": y_ajustado,
        "Cota do greide (m)": perfil,
        "Diferença greide - terreno (m)": perfil - y
    })


    # ========================================================
    # GRÁFICO DO PERFIL LONGITUDINAL
    # ========================================================

    st.subheader("4. Perfil longitudinal")

    grafico = go.Figure()

    grafico.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="Terreno natural"
        )
    )

    grafico.add_trace(
        go.Scatter(
            x=x,
            y=y_ajustado,
            mode="lines",
            name="Terreno suavizado"
        )
    )

    grafico.add_trace(
        go.Scatter(
            x=x,
            y=perfil,
            mode="lines",
            name="Greide proposto",
            line=dict(width=3)
        )
    )

    grafico.update_layout(
        xaxis_title="Distância (m)",
        yaxis_title="Cota (m)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.12
        )
    )

    st.plotly_chart(
        grafico,
        use_container_width=True
    )


    # ========================================================
    # TABELA DE RESULTADOS
    # ========================================================

    st.subheader("5. Tabela de rampas")

    st.dataframe(
        tabela_rampas.style.format({
            "Início (m)": "{:.2f}",
            "Fim (m)": "{:.2f}",
            "Comprimento (m)": "{:.2f}",
            "Cota inicial (m)": "{:.3f}",
            "Cota final (m)": "{:.3f}",
            "Desnível (m)": "{:.3f}",
            "Inclinação (%)": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # EXPORTAÇÃO PARA EXCEL
    # ========================================================

    st.subheader("6. Exportar resultados")

    memoria = io.BytesIO()

    with pd.ExcelWriter(
        memoria,
        engine="openpyxl"
    ) as escritor:

        perfil_saida.to_excel(
            escritor,
            sheet_name="Perfil longitudinal",
            index=False
        )

        tabela_rampas.to_excel(
            escritor,
            sheet_name="Rampas",
            index=False
        )

    st.download_button(
        "Baixar resultados em Excel",
        data=memoria.getvalue(),
        file_name="resultado_greide.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

    st.warning(
        "Resultado preliminar para análise. Verifique os dados, "
        "as inclinações, os pontos de transição e os critérios "
        "de projeto antes de utilizar o greide em um projeto executivo."
    )


except Exception as erro:

    st.error(
        f"Não foi possível ler ou processar a planilha: {erro}"
    )
