import streamlit as st
import pandas as pd
from fpdf import FPDF
import io
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import unicodedata
import tempfile
import os
from datetime import date

# ─── CONSTANTES ────────────────────────────────────────────────────────────────
TETO_INSS   = 7786.02
PISO_INSS   = 1518.00
DATA_REFORMA = date(2019, 11, 13)

AZUL       = (65, 105, 225)
VERDE      = (46, 125, 50)
VERMELHO   = (183, 28, 28)
CINZA      = (60, 60, 60)
CINZA_CL   = (245, 245, 245)

plt.rcParams.update({
    "axes.facecolor": "#FFFFFF", "axes.edgecolor": "#CCCCCC",
    "grid.color": "#EEEEEE", "font.family": "sans-serif", "font.size": 10,
})

# ─── UTILITÁRIOS ───────────────────────────────────────────────────────────────
def fmt_moeda(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def norm(texto: str) -> str:
    if not texto:
        return ""
    s = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in s if not unicodedata.combining(c)).encode("ascii", "ignore").decode("ascii")

def idade_decimal(nasc: date, ref: date = None) -> float:
    r = ref or date.today()
    anos = r.year - nasc.year - ((r.month, r.day) < (nasc.month, nasc.day))
    meses = (r.month - nasc.month) % 12
    return round(anos + meses / 12, 2)

# ─── PARSER CSV DO INSS (CNIS) ─────────────────────────────────────────────────
def parse_cnis(uploaded) -> dict:
    """
    Tenta extrair remunerações e competências do extrato CNIS exportado pelo Meu INSS.
    O formato pode variar conforme a versão do portal, então usamos heurísticas.
    """
    out = {"remuneracoes": [], "tempo_meses": 0, "media": 0.0, "erro": None}
    try:
        raw = uploaded.read()
        texto = None
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                texto = raw.decode(enc)
                break
            except Exception:
                continue
        if texto is None:
            out["erro"] = "Não foi possível decodificar o arquivo."
            return out

        linhas = texto.splitlines()
        remuneracoes: list[float] = []
        competencias: set[tuple] = set()

        for linha in linhas:
            linha = linha.strip()
            if not linha:
                continue

            # Competências no formato MM/AAAA
            for mes, ano in re.findall(r'\b(\d{2})/(\d{4})\b', linha):
                try:
                    m, a = int(mes), int(ano)
                    if 1 <= m <= 12 and 1994 <= a <= date.today().year:
                        competencias.add((a, m))
                except ValueError:
                    pass

            # Valores monetários — ignoramos tokens que parecem datas ou CPF/CNPJ
            for tok in re.findall(r'\b\d{1,3}(?:\.\d{3})*,\d{2}\b', linha):
                try:
                    v = float(tok.replace(".", "").replace(",", "."))
                    if PISO_INSS * 0.3 <= v <= TETO_INSS * 1.5:
                        remuneracoes.append(v)
                except ValueError:
                    pass

        if remuneracoes:
            rem_ord = sorted(remuneracoes, reverse=True)
            n = max(1, int(len(rem_ord) * 0.8))   # usa os 80% maiores (estimativa)
            out["media"] = min(sum(rem_ord[:n]) / n, TETO_INSS)
            out["remuneracoes"] = rem_ord

        out["tempo_meses"] = len(competencias)

    except Exception as e:
        out["erro"] = str(e)
    return out

# ─── MOTOR DE CÁLCULO PREVIDENCIÁRIO (EC 103/2019) ────────────────────────────
class Calculadora:

    # ── Regras progressivas ──────────────────────────────────────────────────
    @staticmethod
    def pontos_minimos(sexo: str, ano: int) -> int:
        """Regra de Pontos — Art. 15, I  (progressão anual)."""
        delta = max(0, ano - 2019)
        return min(86 + delta, 100) if sexo == "F" else min(96 + delta, 105)

    @staticmethod
    def idade_min_progressiva(sexo: str, ano: int) -> float:
        """Idade Mínima Progressiva — Art. 15, II (progressão semestral)."""
        delta = max(0, ano - 2019) * 0.5
        return min(57.0 + delta, 62.0) if sexo == "F" else min(61.0 + delta, 65.0)

    # ── Cálculo do benefício ─────────────────────────────────────────────────
    @staticmethod
    def aliquota(sexo: str, tempo: float) -> float:
        """60 % + 2 % por ano acima de 15 (F) ou 20 (M) anos — EC 103/2019."""
        base_min = 15.0 if sexo == "F" else 20.0
        return min(0.60 + max(0.0, tempo - base_min) * 0.02, 1.0)

    @staticmethod
    def fator_prev(idade: float, tempo: float) -> float:
        """Fator previdenciário — Lei 9.876/1999 (simplificado)."""
        expectativa = max(15.0, 24.5 - max(0.0, idade - 60) * 0.3)
        f = 0.31
        fp = (tempo * f / expectativa) * (1 + (idade + tempo * f) / 100)
        return round(min(max(fp, 0.40), 1.50), 4)

    # ── Cenários completos ───────────────────────────────────────────────────
    def cenarios(
        self,
        sexo: str,
        data_nasc: date,
        tempo_atual: float,          # anos de contribuição hoje
        media_sal: float,
        categoria: str,
        ano_ref: int,
        tempo_em_2019: float,        # anos de contribuição em 13/11/2019
    ) -> dict:
        """
        Calcula elegibilidade e benefício de cada regra para o ano de referência.
        O tempo de contribuição projetado é recalculado internamente.
        """
        anos_a_mais   = max(0, ano_ref - date.today().year)
        tempo_proj    = tempo_atual + anos_a_mais
        data_ref      = date(ano_ref, 7, 1)
        idade         = idade_decimal(data_nasc, data_ref)

        # Redução para professores (5 anos em idade e tempo)
        red_prof = 5.0 if categoria == "Professor" else 0.0
        tempo_min = (30.0 if sexo == "F" else 35.0) - red_prof

        # Quanto faltava para o tempo mínimo NA DATA DA REFORMA
        faltava_2019  = max(0.0, (30.0 if sexo == "F" else 35.0) - tempo_em_2019)
        era_segurado  = tempo_em_2019 > 0          # tinha vínculo antes da reforma

        res: dict[str, dict] = {}

        # ── 1. Regra de Pontos (Art. 15, I) ──────────────────────────────────
        pts_nec  = self.pontos_minimos(sexo, ano_ref)
        pts_at   = idade + tempo_proj
        ali_pt   = self.aliquota(sexo, tempo_proj)
        pode_pt  = pts_at >= pts_nec and tempo_proj >= tempo_min
        res["Regra de Pontos"] = {
            "pode":       pode_pt,
            "requisito":  f"Idade + Contribuição ≥ {pts_nec} pts  |  Contribuição ≥ {tempo_min:.0f} anos",
            "situacao":   f"{pts_at:.1f}/{pts_nec} pts  |  {tempo_proj:.1f}/{tempo_min:.0f} anos",
            "falta":      max(0.0, pts_nec - pts_at) if not pode_pt else 0.0,
            "beneficio":  min(media_sal * ali_pt, TETO_INSS) if pode_pt else 0.0,
            "aliquota":   ali_pt,
            "fator":      None,
            "descricao":  (
                "Regra de transição principal da EC 103/2019. Sem fator previdenciário. "
                "O benefício é de 60% da média + 2% por ano acima de "
                + ("15" if sexo == "F" else "20") + " anos de contribuição."
            ),
        }

        # ── 2. Idade Mínima Progressiva (Art. 15, II) ────────────────────────
        id_prog  = self.idade_min_progressiva(sexo, ano_ref)
        ali_id   = self.aliquota(sexo, tempo_proj)
        pode_id  = idade >= id_prog and tempo_proj >= tempo_min
        res["Idade Mínima Progressiva"] = {
            "pode":       pode_id,
            "requisito":  f"Idade ≥ {id_prog:.1f} anos  |  Contribuição ≥ {tempo_min:.0f} anos",
            "situacao":   f"Idade {idade:.1f}/{id_prog:.1f}  |  {tempo_proj:.1f}/{tempo_min:.0f} anos",
            "falta":      max(0.0, id_prog - idade) if not pode_id else 0.0,
            "beneficio":  min(media_sal * ali_id, TETO_INSS) if pode_id else 0.0,
            "aliquota":   ali_id,
            "fator":      None,
            "descricao":  (
                f"Transição progressiva até {'62' if sexo == 'F' else '65'} anos. "
                "Sem fator previdenciário. Mesma fórmula de benefício da Regra de Pontos."
            ),
        }

        # ── 3. Pedágio 50% (Art. 16) ─────────────────────────────────────────
        # Só para quem, em 13/11/2019, faltava MENOS de 2 anos para o tempo mínimo
        elegivel_50  = era_segurado and 0 < faltava_2019 < 2.0
        tempo_nec_50 = (30.0 if sexo == "F" else 35.0) + faltava_2019 * 0.50
        fp_50        = self.fator_prev(idade, tempo_proj)
        pode_50      = elegivel_50 and tempo_proj >= tempo_nec_50
        if not elegivel_50:
            sit_50 = "Não elegível (faltavam ≥ 2 anos em 13/11/2019)"
        else:
            sit_50 = f"{tempo_proj:.1f}/{tempo_nec_50:.1f} anos  |  Pedágio: {faltava_2019*0.5:.1f} anos"
        res["Pedágio 50%"] = {
            "pode":       pode_50,
            "requisito":  f"Faltavam < 2 anos em 13/11/2019  |  {tempo_nec_50:.1f} anos contribuição",
            "situacao":   sit_50,
            "falta":      max(0.0, tempo_nec_50 - tempo_proj) if elegivel_50 and not pode_50 else 0.0,
            "beneficio":  min(media_sal * fp_50, TETO_INSS) if pode_50 else 0.0,
            "aliquota":   fp_50,
            "fator":      fp_50,
            "descricao":  (
                "Disponível apenas para quem em 13/11/2019 faltava MENOS de 2 anos para o tempo mínimo. "
                "Aplica fator previdenciário. O pedágio é 50% do tempo que faltava naquela data."
            ),
        }

        # ── 4. Pedágio 100% (Art. 15, III) ───────────────────────────────────
        id_min_100   = (57.0 if sexo == "F" else 60.0) - red_prof
        tempo_nec_100 = (30.0 if sexo == "F" else 35.0) + faltava_2019
        pode_100     = era_segurado and idade >= id_min_100 and tempo_proj >= tempo_nec_100
        if not era_segurado:
            sit_100 = "Não elegível (não era segurado em 13/11/2019)"
        else:
            sit_100 = (
                f"Idade {idade:.1f}/{id_min_100:.0f}  |  "
                f"{tempo_proj:.1f}/{tempo_nec_100:.1f} anos  |  Pedágio: {faltava_2019:.1f} anos"
            )
        res["Pedágio 100%"] = {
            "pode":       pode_100,
            "requisito":  f"Idade ≥ {id_min_100:.0f}  |  {tempo_nec_100:.1f} anos contribuição (pedágio de {faltava_2019:.1f} anos)",
            "situacao":   sit_100,
            "falta":      max(0.0, id_min_100 - idade) if era_segurado and not pode_100 else 0.0,
            "beneficio":  min(media_sal, TETO_INSS) if pode_100 else 0.0,
            "aliquota":   1.0,
            "fator":      None,
            "descricao":  (
                "Sem fator previdenciário. Benefício = 100% da média salarial (até o teto). "
                "Exige que o segurado já contribuísse antes de 13/11/2019."
            ),
        }

        # ── 5. Aposentadoria por Idade — regra definitiva ─────────────────────
        id_def   = (62.0 if sexo == "F" else 65.0) - red_prof
        ali_def  = self.aliquota(sexo, tempo_proj)
        pode_def = idade >= id_def and tempo_proj >= 15.0
        res["Aposentadoria por Idade"] = {
            "pode":       pode_def,
            "requisito":  f"Idade ≥ {id_def:.0f} anos  |  15 anos de carência",
            "situacao":   f"Idade {idade:.1f}/{id_def:.0f}  |  {tempo_proj:.1f}/15 anos",
            "falta":      max(0.0, id_def - idade) if not pode_def else 0.0,
            "beneficio":  min(media_sal * ali_def, TETO_INSS) if pode_def else 0.0,
            "aliquota":   ali_def,
            "fator":      None,
            "descricao":  (
                "Regra definitiva da EC 103/2019 (não é de transição). "
                "Exige menor carência (15 anos), mas requer idade mais alta. "
                "Sem fator previdenciário."
            ),
        }

        return res, round(idade, 1), round(tempo_proj, 1)

    def melhor(self, cen: dict) -> tuple:
        disp = {k: v for k, v in cen.items() if v["pode"]}
        if not disp:
            return None, None
        k = max(disp, key=lambda x: disp[x]["beneficio"])
        return k, disp[k]

    def projecao(
        self,
        sexo: str, data_nasc: date, tempo_atual: float, media_sal: float,
        categoria: str, tempo_em_2019: float, anos_range,
    ) -> pd.DataFrame:
        rows = []
        for ano in anos_range:
            cen, idade, tempo = self.cenarios(
                sexo, data_nasc, tempo_atual, media_sal, categoria, ano, tempo_em_2019
            )
            row = {"Ano": ano, "Idade": idade}
            for nome, d in cen.items():
                row[nome] = d["beneficio"] if d["pode"] else None
            rows.append(row)
        return pd.DataFrame(rows)


# ─── GRÁFICOS ──────────────────────────────────────────────────────────────────
CORES_REGRAS = {
    "Regra de Pontos":            "#4169E1",
    "Idade Mínima Progressiva":   "#2E7D32",
    "Pedágio 50%":                "#F57C00",
    "Pedágio 100%":               "#7B1FA2",
    "Aposentadoria por Idade":    "#00838F",
}

def fig_comparativo(cen: dict, nome: str) -> plt.Figure:
    labels = list(cen.keys())
    valores = [c["beneficio"] for c in cen.values()]
    cores = ["#2E7D32" if c["pode"] else "#E0E0E0" for c in cen.values()]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(labels)), valores, color=cores, width=0.55, zorder=3)
    ax.grid(axis="y", zorder=0, alpha=0.5)
    ax.set_title(f"Comparativo de Benefícios — {nome}", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Valor Mensal (R$)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x:,.0f}"))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=9)

    for bar, val, c in zip(bars, valores, cen.values()):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                fmt_moeda(val), ha="center", va="bottom", fontsize=8, fontweight="bold"
            )
        if not c["pode"]:
            ax.text(
                bar.get_x() + bar.get_width() / 2, 200,
                "Não\nelegível", ha="center", va="bottom", fontsize=8, color="#999"
            )

    ax.set_ylim(0, max(valores + [2000]) * 1.25)
    plt.tight_layout()
    return fig


def fig_projecao(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in df.columns:
        if col in ("Ano", "Idade"):
            continue
        serie = df[col].dropna()
        if serie.empty:
            continue
        cor = CORES_REGRAS.get(col, "#888")
        ax.plot(df.loc[serie.index, "Ano"], serie, label=col, color=cor, linewidth=2.2,
                marker="o", markersize=3, markerfacecolor="white", markeredgewidth=1.5)

    ax.set_title("Projeção Temporal dos Benefícios por Regra", fontsize=13, fontweight="bold")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Valor Mensal (R$)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x:,.0f}"))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    return fig


def fig_breakeven(v_a: float, i_a: float, v_b: float, i_b: float, label_a: str, label_b: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.linspace(min(i_a, i_b), 90, 300)

    acc_a = np.array([(v_a * (xi - i_a) * 12) if xi >= i_a else 0 for xi in x])
    acc_b = np.array([(v_b * (xi - i_b) * 12) if xi >= i_b else 0 for xi in x])

    ax.fill_between(x, acc_a, acc_b, where=acc_b > acc_a, alpha=0.08, color="#4169E1")
    ax.plot(x, acc_a, color="#B0BEC5", linewidth=2,   label=f"{label_a} — {fmt_moeda(v_a)}/mês (início {i_a:.1f} anos)")
    ax.plot(x, acc_b, color="#4169E1", linewidth=2.5, label=f"{label_b} — {fmt_moeda(v_b)}/mês (início {i_b:.1f} anos)")

    t_bk = None
    if v_b > v_a and i_b > i_a:
        diff = v_b - v_a
        t_bk = (v_b * i_b - v_a * i_a) / diff
        if i_a < t_bk < 90:
            val_bk = v_a * (t_bk - i_a) * 12
            ax.scatter(t_bk, val_bk, color="red", s=110, zorder=6,
                       label=f"Ponto de equilíbrio: {t_bk:.1f} anos")
            ax.axvline(t_bk, color="red", linestyle="--", alpha=0.45, linewidth=1.2)

    ax.set_title("Análise de Breakeven — Quando compensa esperar?", fontsize=13, fontweight="bold")
    ax.set_xlabel("Idade (anos)")
    ax.set_ylabel("Total Acumulado")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x/1e6:.1f}M"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    return fig, t_bk


# ─── GERADOR DE PDF ────────────────────────────────────────────────────────────
def _salvar_fig(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(buf.getvalue())
    tmp.close()
    return tmp.name


def gerar_pdf(
    nome: str, sexo: str, categoria: str, data_nasc: date,
    media_sal: float, tempo_contrib: float,
    cen: dict, melhor_nome: str,
    f_comp: plt.Figure, f_proj: plt.Figure,
    conclusao: str, ano_ref: int,
) -> bytes:

    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)

    def header(titulo: str):
        pdf.set_fill_color(*AZUL)
        pdf.rect(0, 0, 210, 28, "F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 15)
        pdf.set_y(6)
        pdf.cell(0, 9, norm(titulo), ln=True, align="C")
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 7, norm(f"Cliente: {nome.upper()}  |  Categoria: {categoria}  |  Data-base: {ano_ref}"), ln=True, align="C")
        pdf.set_text_color(*CINZA)
        pdf.set_y(35)

    def linha_tabela(campo: str, valor: str, fill: bool = False):
        if fill:
            pdf.set_fill_color(*CINZA_CL)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(58, 7, norm(campo + ":"), 1, 0, "L", fill)
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 7, norm(valor), 1, 1, "L", fill)

    # ── PÁGINA 1: CAPA + COMPARATIVO ──────────────────────────────────────────
    pdf.add_page()
    header("ESTUDO TECNICO PREVIDENCIARIO — EC 103/2019")

    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(*AZUL)
    pdf.cell(0, 7, "Perfil do Cliente", ln=True)
    pdf.set_text_color(*CINZA)

    idade_hj = idade_decimal(data_nasc)
    linha_tabela("Nome",                 nome,                      False)
    linha_tabela("Data de Nascimento",   data_nasc.strftime("%d/%m/%Y"), True)
    linha_tabela("Idade Atual",          f"{idade_hj:.1f} anos",    False)
    linha_tabela("Sexo",                 "Feminino" if sexo == "F" else "Masculino", True)
    linha_tabela("Categoria",            categoria,                 False)
    linha_tabela("Tempo de Contribuicao", f"{tempo_contrib:.1f} anos", True)
    linha_tabela("Media Salarial",       fmt_moeda(media_sal),      False)

    pdf.ln(4)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(90, 90, 90)
    intro = (
        "Este estudo apresenta as possibilidades de aposentadoria conforme a Reforma da Previdencia "
        "(EC 103/2019) e suas regras de transicao. Os valores sao estimativas baseadas nos dados fornecidos "
        "e nao substituem assessoria juridica especializada."
    )
    pdf.multi_cell(0, 4, norm(intro))

    pdf.ln(3)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(*AZUL)
    pdf.cell(0, 7, "Comparativo de Beneficios", ln=True)
    tmp1 = _salvar_fig(f_comp)
    pdf.image(tmp1, x=10, w=190)
    os.unlink(tmp1)

    # ── PÁGINA 2: ANÁLISE DETALHADA DAS REGRAS ─────────────────────────────────
    pdf.add_page()
    header("ANALISE DAS REGRAS DE TRANSICAO")

    for regra, d in cen.items():
        pdf.ln(2)
        ok = d["pode"]
        pdf.set_fill_color(*(VERDE if ok else VERMELHO))
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 8, norm(f"  {regra} — {'ELEGIVEL' if ok else 'NAO ELEGIVEL'}"), ln=True, fill=True)

        pdf.set_text_color(*CINZA)
        pdf.set_font("Arial", "", 8)
        pdf.cell(0, 5, norm(f"Requisitos: {d['requisito']}"), ln=True)
        pdf.cell(0, 5, norm(f"Situacao: {d['situacao']}"), ln=True)
        if ok:
            pdf.set_font("Arial", "B", 9)
            alq_str = (f"Fator Prev. {d['fator']:.4f}" if d["fator"] else f"Aliquota {d['aliquota']*100:.0f}%")
            pdf.cell(0, 5, norm(f"Beneficio Estimado: {fmt_moeda(d['beneficio'])}  ({alq_str})"), ln=True)
            pdf.set_font("Arial", "", 8)
        pdf.set_font("Arial", "I", 8)
        pdf.set_text_color(90, 90, 90)
        pdf.multi_cell(0, 4, norm(d["descricao"]))
        pdf.set_text_color(*CINZA)

    # ── PÁGINA 3: PROJEÇÃO TEMPORAL ────────────────────────────────────────────
    pdf.add_page()
    header("PROJECAO TEMPORAL DOS BENEFICIOS")

    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(*CINZA)
    pdf.multi_cell(0, 5, norm(
        "O grafico abaixo mostra a evolucao dos beneficios de cada regra ao longo do tempo. "
        "As linhas iniciam quando o segurado atinge os requisitos daquela regra especifica. "
        "Regras sem fator previdenciario tendem a ser mais vantajosas."
    ))
    tmp2 = _salvar_fig(f_proj)
    pdf.image(tmp2, x=5, w=200)
    os.unlink(tmp2)

    # ── PÁGINA 4: PARECER FINAL E BASE LEGAL ──────────────────────────────────
    pdf.add_page()
    header("PARECER TECNICO E RECOMENDACAO")

    if melhor_nome:
        melhor_d = cen[melhor_nome]
        pdf.set_fill_color(*VERDE)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, norm(f"  MELHOR CENARIO: {melhor_nome}"), ln=True, fill=True)

        pdf.set_text_color(*CINZA)
        pdf.set_font("Arial", "B", 10)
        pdf.ln(2)
        pdf.cell(0, 7, norm(f"Beneficio estimado: {fmt_moeda(melhor_d['beneficio'])}"), ln=True)
        alq_txt = (
            f"Fator previdenciario: {melhor_d['fator']:.4f}"
            if melhor_d["fator"] else
            f"Aliquota aplicada: {melhor_d['aliquota']*100:.0f}%"
        )
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 7, norm(alq_txt), ln=True)
    else:
        pdf.set_fill_color(*VERMELHO)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 10, norm("  NENHUMA REGRA ELEGIVEL NO ANO SELECIONADO"), ln=True, fill=True)

    pdf.ln(4)
    pdf.set_text_color(*AZUL)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "Parecer Tecnico", ln=True)
    pdf.set_text_color(*CINZA)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, norm(conclusao))

    pdf.ln(5)
    pdf.set_text_color(*AZUL)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 7, "Base Legal", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(*CINZA)
    for item in [
        "EC 103/2019 — Emenda Constitucional da Reforma da Previdencia",
        "Art. 15, I — Regra de Pontos (Transicao)",
        "Art. 15, II — Regra de Idade Minima Progressiva (Transicao)",
        "Art. 15, III — Pedagio 100% (Transicao)",
        "Art. 16 — Pedagio 50% para segurados a menos de 2 anos do tempo minimo",
        "Lei 9.876/1999 — Fator Previdenciario (aplicavel ao Pedagio 50%)",
        "IN PRES/INSS 128/2022 — Instrucao Normativa sobre procedimentos de concessao",
    ]:
        pdf.cell(0, 5, norm(f"  • {item}"), ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 4, norm(
        "AVISO LEGAL: Este estudo e de carater informativo e nao constitui assessoria juridica. "
        "Os valores sao estimativas e podem divergir do calculo oficial do INSS. "
        "Recomenda-se consultar um advogado previdenciario para analise definitiva."
    ))

    return pdf.output(dest="S").encode("latin-1")


# ─── INTERFACE STREAMLIT ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Planejamento Previdenciário",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.8rem; }
    div[data-testid="stSidebar"] { background: #F8F9FA; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    .card-elegivel   { border: 2px solid #2E7D32; border-radius: 8px; padding: 14px; background: #F1F8E9; margin-bottom: 6px; }
    .card-inelegivel { border: 2px solid #CFD8DC; border-radius: 8px; padding: 14px; background: #FAFAFA;  margin-bottom: 6px; }
    .card-melhor     { border: 2px solid #4169E1; border-radius: 8px; padding: 14px; background: #E8EEFF; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# Estado global
for k, v in [("pagina", "home"), ("dados", {}), ("cen_cache", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

CALC = Calculadora()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.pagina == "home":
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem'>
        <h1 style='color:#4169E1; margin-bottom:4px;'>🛡️ Planejamento Previdenciário</h1>
        <p style='color:#555; font-size:1.05rem;'>
            Análise completa das possibilidades de aposentadoria com base na Reforma de 2019 (EC 103/2019)
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    col_form, col_info = st.columns([3, 2], gap="large")

    with col_form:
        st.subheader("📋 Dados do Cliente")

        nome = st.text_input("Nome completo *", placeholder="Nome do cliente")

        c1, c2 = st.columns(2)
        with c1:
            data_nasc = st.date_input(
                "Data de Nascimento *",
                value=date(1975, 6, 15),
                min_value=date(1930, 1, 1),
                max_value=date.today(),
            )
            sexo = st.selectbox(
                "Sexo *", ["F", "M"],
                format_func=lambda x: "Feminino" if x == "F" else "Masculino",
            )
        with c2:
            categoria = st.selectbox("Categoria *", [
                "Trabalhador Urbano",
                "Trabalhador Rural",
                "Professor",
                "Trabalhador Doméstico",
                "Autônomo / Contribuinte Individual",
            ])
            tempo_contrib = st.number_input(
                "Anos de Contribuição Atual *", 0.0, 50.0, 15.0, 0.5,
                help="Total de anos de contribuição até hoje.",
            )

        media_sal = st.number_input(
            "Média Salarial de Contribuição (R$) *",
            PISO_INSS, TETO_INSS, 3500.0, 100.0,
            help="Média dos salários de contribuição. O INSS usa os 100% dos salários pós-2019.",
        )

        st.divider()
        st.subheader("📂 Importar Extrato CNIS do INSS (opcional)")
        st.caption(
            "Faça upload do CSV exportado em: **Meu INSS → Extrato Previdenciário → CNIS**. "
            "Os dados serão usados para estimar a média salarial e o tempo de contribuição."
        )

        uploaded = st.file_uploader(
            "Arquivo CSV ou TXT do CNIS", type=["csv", "txt", "xls", "xlsx"],
            help="Extrato de vínculos e remunerações exportado do portal Meu INSS.",
        )

        if uploaded:
            with st.spinner("Processando extrato..."):
                dados_csv = parse_cnis(uploaded)

            if dados_csv["erro"]:
                st.warning(f"⚠️ Erro ao processar: {dados_csv['erro']}")
            else:
                n_rem = len(dados_csv["remuneracoes"])
                n_mes = dados_csv["tempo_meses"]
                st.success(f"✅ Extrato processado: {n_rem} remunerações e {n_mes} competências encontradas.")

                col_csv1, col_csv2 = st.columns(2)
                with col_csv1:
                    if dados_csv["media"] > 0:
                        st.metric("Média calculada pelo extrato", fmt_moeda(dados_csv["media"]))
                        if st.checkbox("Usar esta média salarial"):
                            media_sal = dados_csv["media"]
                with col_csv2:
                    if n_mes > 0:
                        tempo_csv = n_mes / 12
                        st.metric("Tempo estimado pelo extrato", f"{tempo_csv:.1f} anos")
                        if st.checkbox("Usar este tempo de contribuição"):
                            tempo_contrib = tempo_csv

        st.divider()

        if st.button("🔍 Gerar Análise Previdenciária", type="primary", use_container_width=True):
            if not nome.strip():
                st.error("❌ Informe o nome do cliente.")
            else:
                # Calcula o tempo que o cliente tinha em 13/11/2019
                dias_desde_reforma = (date.today() - DATA_REFORMA).days
                t_2019 = max(0.0, round(tempo_contrib - dias_desde_reforma / 365.25, 2))

                st.session_state.dados = {
                    "nome":         nome.strip(),
                    "data_nasc":    data_nasc,
                    "sexo":         sexo,
                    "categoria":    categoria,
                    "tempo_atual":  tempo_contrib,
                    "tempo_2019":   t_2019,
                    "media_sal":    media_sal,
                }
                st.session_state.cen_cache = None
                st.session_state.pagina = "analise"
                st.rerun()

    with col_info:
        st.subheader("ℹ️ Como funciona")
        st.markdown("""
**1. Preencha os dados** — a data de nascimento garante cálculo preciso de idade.

**2. Importe o extrato do INSS** (opcional) para calcular automaticamente a média salarial real e o tempo de contribuição.

**3. Clique em "Gerar Análise"** e veja todos os cenários:
- Regra de Pontos (Art. 15, I)
- Idade Mínima Progressiva (Art. 15, II)
- Pedágio 100% (Art. 15, III)
- Pedágio 50% (Art. 16)
- Aposentadoria por Idade (regra definitiva)

**4. Explore os gráficos** de comparativo e projeção temporal.

**5. Gere o PDF** com todos os cenários, análise e recomendação técnica.
        """)

        with st.expander("📌 Regras de Transição — Resumo"):
            st.markdown("""
| Regra | Requisito-chave | Fator Prev.? |
|---|---|---|
| Pontos | Idade + Contribuição ≥ progressão | ❌ Não |
| Idade Mínima | Idade progressiva até 62/65 anos | ❌ Não |
| Pedágio 100% | 57/60 anos + tempo + pedágio | ❌ Não |
| Pedágio 50% | Segurado a < 2 anos do mínimo em 2019 | ✅ Sim |
| Por Idade | 62/65 anos + 15 anos carência | ❌ Não |

*F = Feminino / M = Masculino*
            """)

        st.caption(
            "⚠️ Sistema de apoio técnico. Valores são estimativas. "
            "Consulte sempre um advogado previdenciário."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: ANÁLISE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.pagina == "analise":
    d = st.session_state.dados

    with st.sidebar:
        st.markdown("### Navegação")
        if st.button("← Voltar ao Início", use_container_width=True):
            st.session_state.pagina = "home"
            st.rerun()
        st.divider()
        st.markdown(f"**{d['nome']}**")
        st.markdown(f"Categoria: {d['categoria']}")
        st.markdown(f"Sexo: {'Feminino' if d['sexo']=='F' else 'Masculino'}")
        id_hj = idade_decimal(d["data_nasc"])
        st.markdown(f"Idade: {id_hj:.1f} anos")
        st.markdown(f"Contribuição: {d['tempo_atual']:.1f} anos")
        st.markdown(f"Média salarial: {fmt_moeda(d['media_sal'])}")

    st.markdown(f"<h2 style='color:#4169E1;'>📊 Análise — {d['nome']}</h2>", unsafe_allow_html=True)

    col_ano, col_esp = st.columns([2, 6])
    with col_ano:
        ano_alvo = st.slider("Ano de referência", 2024, 2050, max(2024, date.today().year))

    # Calcula cenários para o ano alvo
    cen, idade_alvo, tempo_alvo = CALC.cenarios(
        d["sexo"], d["data_nasc"], d["tempo_atual"],
        d["media_sal"], d["categoria"], ano_alvo, d["tempo_2019"],
    )
    melhor_nome, melhor_d = CALC.melhor(cen)

    # Banner de resultado
    if melhor_nome:
        st.success(
            f"✅ **Melhor cenário em {ano_alvo}: {melhor_nome}** | "
            f"{fmt_moeda(melhor_d['beneficio'])}/mês | "
            f"Idade: {idade_alvo} anos | Contribuição: {tempo_alvo} anos"
        )
    else:
        st.warning(
            f"⚠️ Nenhuma regra elegível em {ano_alvo} "
            f"(Idade: {idade_alvo} anos | Contribuição: {tempo_alvo} anos). "
            "Avance o ano de referência."
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Cenários", "📈 Projeção Temporal", "⚖️ Breakeven", "📄 Relatório PDF"
    ])

    # ── TAB 1: CENÁRIOS ────────────────────────────────────────────────────────
    with tab1:
        cols = st.columns(len(cen))
        for i, (regra, dados) in enumerate(cen.items()):
            with cols[i]:
                estrela = " ★" if regra == melhor_nome else ""
                if regra == melhor_nome:
                    classe = "card-melhor"
                elif dados["pode"]:
                    classe = "card-elegivel"
                else:
                    classe = "card-inelegivel"

                val_txt = fmt_moeda(dados["beneficio"]) if dados["pode"] else "—"
                st.markdown(f"""
                <div class="{classe}">
                  <b style='font-size:0.82rem'>{regra}{estrela}</b><br>
                  <span style='font-size:1.1rem'>{'✅' if dados['pode'] else '❌'}</span><br>
                  <span style='font-size:0.72rem; color:#555'>{dados['situacao']}</span><br>
                  <b style='color:{"#2E7D32" if dados["pode"] else "#999"}'>{val_txt}</b>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        for regra, dados in cen.items():
            icon = "✅" if dados["pode"] else "❌"
            label = f"{icon} **{regra}**"
            if dados["pode"]:
                label += f"  —  {fmt_moeda(dados['beneficio'])}"
            with st.expander(label):
                ca, cb = st.columns(2)
                with ca:
                    st.markdown(f"**Requisitos:** {dados['requisito']}")
                    st.markdown(f"**Situação:** {dados['situacao']}")
                    if dados["pode"]:
                        st.markdown(f"**Benefício:** {fmt_moeda(dados['beneficio'])}")
                        if dados["fator"]:
                            st.markdown(f"**Fator Previdenciário:** {dados['fator']:.4f}")
                        else:
                            st.markdown(f"**Alíquota:** {dados['aliquota']*100:.0f}%")
                with cb:
                    st.info(dados["descricao"])

        st.divider()
        fc = fig_comparativo(cen, d["nome"])
        st.pyplot(fc, use_container_width=True)
        plt.close(fc)

    # ── TAB 2: PROJEÇÃO ────────────────────────────────────────────────────────
    with tab2:
        st.markdown("Evolução dos benefícios de cada regra ao longo dos anos, considerando contribuição contínua.")

        with st.spinner("Calculando projeção..."):
            df_proj = CALC.projecao(
                d["sexo"], d["data_nasc"], d["tempo_atual"],
                d["media_sal"], d["categoria"], d["tempo_2019"],
                range(date.today().year, 2051),
            )

        fp = fig_projecao(df_proj)
        st.pyplot(fp, use_container_width=True)
        plt.close(fp)

        st.markdown("##### Tabela de Projeção")
        df_exib = df_proj.copy()
        for col in df_exib.columns:
            if col not in ("Ano", "Idade"):
                df_exib[col] = df_exib[col].apply(
                    lambda v: fmt_moeda(v) if pd.notna(v) and v > 0 else "—"
                )
        st.dataframe(df_exib.set_index("Ano"), use_container_width=True)

    # ── TAB 3: BREAKEVEN ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("Compare dois cenários e descubra **quando compensa esperar** por um benefício maior.")

        disponiveis = [(k, v) for k, v in cen.items() if v["pode"]]

        if len(disponiveis) < 2:
            st.info(
                "É preciso ao menos 2 regras elegíveis no ano selecionado para o Breakeven. "
                "Avance o ano de referência ou ajuste os dados."
            )
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Opção A — Aposentadoria Precoce**")
                nome_a = st.selectbox("Regra A", [o[0] for o in disponiveis], key="bk_a")
                v_a    = cen[nome_a]["beneficio"]
                i_a    = float(st.number_input("Idade de início (A)", 45.0, 90.0,
                                                round(idade_alvo, 1), 0.5, key="ia"))
                st.metric("Mensal (A)", fmt_moeda(v_a))

            with c2:
                st.markdown("**Opção B — Aposentadoria Planejada**")
                opcoes_b = [o[0] for o in disponiveis if o[0] != nome_a]
                if opcoes_b:
                    nome_b = st.selectbox("Regra B", opcoes_b, key="bk_b")
                    v_b    = cen[nome_b]["beneficio"]
                    i_b    = float(st.number_input("Idade de início (B)", 45.0, 90.0,
                                                    round(idade_alvo + 2, 1), 0.5, key="ib"))
                    st.metric("Mensal (B)", fmt_moeda(v_b))
                else:
                    st.warning("Selecione regras diferentes em A e B.")
                    nome_b, v_b, i_b = None, 0.0, 0.0

            if nome_b and v_b > 0 and i_b != i_a:
                fbk, t_bk = fig_breakeven(v_a, i_a, v_b, i_b, nome_a, nome_b)
                st.pyplot(fbk, use_container_width=True)
                plt.close(fbk)

                if t_bk and i_a < t_bk < 90:
                    if t_bk < 78:
                        st.success(
                            f"📌 Ponto de equilíbrio: **{t_bk:.1f} anos**. "
                            "Como ocorre antes da expectativa de vida média (78 anos), "
                            "**recomenda-se aguardar** pelo benefício maior."
                        )
                    else:
                        st.warning(
                            f"📌 Ponto de equilíbrio: **{t_bk:.1f} anos**. "
                            "Como ocorre após a expectativa de vida média, "
                            "a aposentadoria precoce pode ser financeiramente mais vantajosa."
                        )
                elif t_bk and t_bk >= 90:
                    st.info("O ponto de equilíbrio fica além dos 90 anos — a opção precoce é mais vantajosa na maioria dos casos.")

    # ── TAB 4: RELATÓRIO PDF ───────────────────────────────────────────────────
    with tab4:
        st.subheader("📄 Gerar Relatório Técnico em PDF")
        st.markdown(
            "O relatório (4 páginas) inclui: **perfil do cliente**, **análise de cada regra**, "
            "**projeção temporal**, **gráficos** e **parecer técnico com recomendação**."
        )

        # Construção da conclusão automática
        elegiveis = [k for k, v in cen.items() if v["pode"]]
        if melhor_nome:
            melhor_ben = cen[melhor_nome]
            outras = [k for k in elegiveis if k != melhor_nome]
            outras_txt = (f" As demais regras elegíveis ({', '.join(outras)}) oferecem valores inferiores."
                          if outras else " Esta é a única regra elegível no período.")
            conclusao = (
                f"Com base no perfil de {d['nome']}, o melhor cenário disponível em {ano_alvo} "
                f"é a {melhor_nome}, com benefício estimado de {fmt_moeda(melhor_ben['beneficio'])} mensais "
                f"({'alíquota de ' + str(int(melhor_ben['aliquota']*100)) + '%' if not melhor_ben['fator'] else 'fator previdenciário de ' + str(melhor_ben['fator'])}) "
                f"sobre a média salarial de {fmt_moeda(d['media_sal'])}."
                f"{outras_txt} "
                f"Recomenda-se {'solicitar o benefício imediatamente' if date.today().year >= ano_alvo else f'planejar a solicitação para {ano_alvo}'}, "
                f"pois esta regra não aplica redutor de fator previdenciário, maximizando o valor do benefício."
                if not melhor_ben["fator"] else
                f"Com base no perfil de {d['nome']}, o melhor cenário disponível em {ano_alvo} "
                f"é a {melhor_nome}, com benefício estimado de {fmt_moeda(melhor_ben['beneficio'])} mensais "
                f"(fator previdenciário: {melhor_ben['fator']:.4f}). "
                f"Recomenda-se avaliar se aguardar mais tempo de contribuição não elevaria o fator e, consequentemente, o benefício."
            )
        else:
            falta_info = "; ".join(
                f"{k}: {v['falta']:.1f} {'pts' if 'Pontos' in k else 'anos'}"
                for k, v in cen.items() if v.get("falta", 0) > 0
            )
            conclusao = (
                f"Em {ano_alvo}, {d['nome']} ainda não atingiu os requisitos para nenhuma regra. "
                f"Situação: {falta_info}. "
                "Recomenda-se continuar contribuindo regularmente e revisar o planejamento anualmente."
            )

        conclusao_edit = st.text_area("Parecer técnico (editável):", conclusao, height=140)

        if st.button("🖨️ Gerar PDF", type="primary", use_container_width=True):
            with st.spinner("Gerando relatório..."):
                try:
                    # Regera os gráficos para o PDF (evita estado fechado de figuras)
                    fc_pdf = fig_comparativo(cen, d["nome"])
                    df_pdf = CALC.projecao(
                        d["sexo"], d["data_nasc"], d["tempo_atual"],
                        d["media_sal"], d["categoria"], d["tempo_2019"],
                        range(date.today().year, 2051),
                    )
                    fp_pdf = fig_projecao(df_pdf)

                    pdf_bytes = gerar_pdf(
                        d["nome"], d["sexo"], d["categoria"], d["data_nasc"],
                        d["media_sal"], tempo_alvo,
                        cen, melhor_nome,
                        fc_pdf, fp_pdf,
                        conclusao_edit, ano_alvo,
                    )
                    plt.close(fc_pdf)
                    plt.close(fp_pdf)

                    nome_arq = f"Estudo_Prev_{d['nome'].replace(' ', '_')}_{ano_alvo}.pdf"
                    st.download_button(
                        "📥 Baixar Relatório PDF",
                        data=pdf_bytes,
                        file_name=nome_arq,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.success("✅ Relatório gerado com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao gerar PDF: {e}")

        st.divider()
        st.caption(
            "⚠️ O relatório é uma ferramenta de apoio técnico. "
            "Não substitui assessoria jurídica especializada em Direito Previdenciário."
        )
