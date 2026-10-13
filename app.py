import streamlit as st
import pandas as pd
from fpdf import FPDF
import pdfplumber
import io
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import unicodedata
import tempfile
import os
from datetime import date, datetime

# ─── CONSTANTES ────────────────────────────────────────────────────────────────
TETO_INSS    = 7786.02
PISO_INSS    = 1518.00
DATA_REFORMA = date(2019, 11, 13)
ANO_HOJE     = date.today().year

# Tábua IBGE 2022 — expectativa de sobrevida por idade (anos)
IBGE_ES = {
    45:37.2, 46:36.3, 47:35.4, 48:34.5, 49:33.5,
    50:32.5, 51:31.6, 52:30.7, 53:29.8, 54:28.9,
    55:28.0, 56:27.1, 57:26.3, 58:25.4, 59:24.6,
    60:23.8, 61:23.0, 62:22.2, 63:21.4, 64:20.7,
    65:19.9, 66:19.2, 67:18.5, 68:17.8, 69:17.1,
    70:16.4, 71:15.8, 72:15.2, 73:14.6, 74:14.0,
    75:13.4, 76:12.9, 77:12.3, 78:11.8, 79:11.3,
    80:10.8,
}

MODALIDADES = {
    "Trabalhador Urbano (RGPS)":         {"red_i": 0, "red_t": 0},
    "Professor (RGPS)":                  {"red_i": 5, "red_t": 5},
    "Atividade Especial / Insalubre":    {"red_i": 0, "red_t": 5},
    "Trabalhador Rural (RGPS)":          {"red_i": 0, "red_t": 0, "rural": True},
    "Servidor Público (RPPS)":           {"red_i": 0, "red_t": 0, "rpps": True},
}

AZUL    = (65, 105, 225)
VERDE   = (46, 125, 50)
VERM    = (183, 28, 28)
CINZA   = (60, 60, 60)
CINZA_CL = (245, 245, 245)

plt.rcParams.update({
    "axes.facecolor": "#FFFFFF", "axes.edgecolor": "#CCCCCC",
    "grid.color": "#EEEEEE", "font.family": "sans-serif", "font.size": 10,
})

# ─── UTILITÁRIOS ───────────────────────────────────────────────────────────────
def fmt_moeda(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def norm(t: str) -> str:
    s = unicodedata.normalize("NFKD", str(t or ""))
    return "".join(c for c in s if not unicodedata.combining(c)).encode("ascii", "ignore").decode("ascii")

def idade_dec(nasc: date, ref: date = None) -> float:
    r = ref or date.today()
    a = r.year - nasc.year - ((r.month, r.day) < (nasc.month, nasc.day))
    m = (r.month - nasc.month) % 12
    return round(a + m / 12, 2)

def expectativa_sobrevida(idade: float) -> float:
    k = min(80, max(45, int(idade)))
    return IBGE_ES.get(k, 15.0)

# ─── PARSER PDF CNIS ───────────────────────────────────────────────────────────
def parse_pdf_cnis(uploaded) -> dict:
    """
    Extrai nome, vínculos e salários do Extrato CNIS (PDF do Meu INSS).

    Prioridade das estratégias (apenas UMA é usada):
      A) CNPJs detectados → cada seção entre CNPJs é um vínculo
      B) Tabelas pdfplumber → agrupa competências por gap de data
      C) Texto linha a linha → fallback final
    """
    out = {"nome": "", "cpf": "", "data_nasc": None, "vinculos": [], "erro": None, "_texto_raw": ""}

    _RE_CNPJ = re.compile(r'\b\d{2}\.?\d{3}\.?\d{3}[/\\]?\d{4}-?\d{2}\b')
    _RE_MES  = re.compile(r'\b((?:0[1-9]|1[0-2])/(?:19[89]\d|20[0-2]\d))\b')
    _RE_VAL  = re.compile(r'\b(\d{1,3}(?:\.\d{3})*,\d{2})\b')

    # Palavras que indicam linhas de cabeçalho/rodapé (não são competências)
    _RE_CABEC = re.compile(
        r'\b(CNPJ|CPF|NIT|Empregador|Emprego|Raz.o|Inicio|Fim|Sit|Seq|'
        r'Compet|Remun|Indicad|Total|Data|Periodo|Vinculo|Beneficio|'
        r'Pagina|Extrato|INSS|Emiss)\b',
        re.IGNORECASE,
    )

    def parse_data(s: str):
        for fmt in ("%m/%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(s.strip(), fmt).date()
            except Exception:
                pass
        return None

    def val_sal(s: str):
        s = re.sub(r'[*\sR$]', '', s)
        if ',' in s:
            s = s.replace('.', '').replace(',', '.')
        try:
            v = float(s)
            # Aceita entre 5% do piso (salários antigos) e 2× o teto
            if PISO_INSS * 0.05 <= v <= TETO_INSS * 2:
                return v
        except Exception:
            pass
        return None

    def comps_de_trecho(trecho: str) -> list[tuple]:
        """
        Extrai (date, valor) de cada trecho.
        Suporta layouts verticais (1 mês por linha) e horizontais
        (vários meses lado a lado na mesma linha).
        Para cada data encontrada, o valor associado é o primeiro
        número monetário entre ela e a próxima data (ou fim da linha).
        Deduplica: uma entrada por mês.
        """
        visto: dict[date, float] = {}
        for linha in trecho.splitlines():
            linha = linha.strip()
            if not linha or _RE_CABEC.search(linha):
                continue

            # Encontra TODAS as ocorrências de datas na linha
            ocorrs = list(_RE_MES.finditer(linha))
            if not ocorrs:
                continue

            for idx, m_dt in enumerate(ocorrs):
                dt = parse_data(m_dt.group(1))
                if not dt or dt in visto:
                    continue

                # Segmento entre esta data e a próxima (ou fim da linha)
                ini = m_dt.end()
                fim = ocorrs[idx + 1].start() if idx + 1 < len(ocorrs) else len(linha)
                seg = linha[ini:fim]

                for m_v in _RE_VAL.finditer(seg):
                    v = val_sal(m_v.group(1))
                    if v is not None and v > 0:
                        visto[dt] = v
                        break

        return list(visto.items())   # mantém ordem de descoberta (= ordem do PDF)

    def agrupar_por_gap(comps: list[tuple], gap_max: int = 3) -> list[dict]:
        """Agrupa competências consecutivas (gap ≤ gap_max meses) em vínculos."""
        if not comps:
            return []
        grupos: list[list[tuple]] = [[comps[0]]]
        for dt, v in comps[1:]:
            prev = grupos[-1][-1][0]
            gap  = (dt.year - prev.year) * 12 + (dt.month - prev.month)
            if gap <= gap_max:
                grupos[-1].append((dt, v))
            else:
                grupos.append([(dt, v)])
        result = []
        for g in grupos:
            dts  = [x[0] for x in g]
            vals = [x[1] for x in g]
            result.append({
                "Início": dts[0].strftime("%m/%Y"),
                "Fim":    dts[-1].strftime("%m/%Y"),
                "Meses":  len(g),
                "Média Salarial (R$)": round(sum(vals) / len(vals), 2),
                "Tipo":   "Contribuição",
            })
        return result

    try:
        raw = uploaded.read()
        paginas_texto: list[str] = []
        tabelas_raw:   list      = []

        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                t = page.extract_text(layout=True) or page.extract_text() or ""
                paginas_texto.append(t)
                for tbl in (page.extract_tables() or []):
                    if tbl:
                        tabelas_raw.append(tbl)

        texto = "\n".join(paginas_texto)
        out["_texto_raw"] = texto

        # ── Nome ──────────────────────────────────────────────────────────────
        for pat in [
            r'(?:Nome|NOME|Segurado|SEGURADO)\s*:?\s*([A-ZÁÉÍÓÚÃÕÂÊÔÜ][A-ZÁÉÍÓÚÃÕÂÊÔÜa-záéíóúãõâêôü ]{4,60})',
            r'NIT[^\n]{0,30}\n([A-ZÁÉÍÓÚÃÕÂÊÔÜ][A-ZÁÉÍÓÚÃÕÂÊÔÜa-záéíóúãõâêôü ]{4,60})',
        ]:
            m = re.search(pat, texto, re.I | re.MULTILINE)
            if m:
                cand = m.group(1).strip()
                if (len(cand.split()) >= 2 and
                        not re.search(r'\b(LTDA|S\.?A\.?|ME|EPP|EIRELI|INSS|CNPJ|EXTRATO)\b', cand, re.I)):
                    out["nome"] = cand.title()
                    break

        # ── CPF (ancorado na palavra CPF para não confundir com NIT/CNPJ) ───────
        m_cpf = re.search(
            r'\bCPF\b[:\s/]*(\d{3}\.?\d{3}\.?\d{3}[-.]?\d{2})\b', texto, re.I
        )
        if m_cpf:
            out["cpf"] = m_cpf.group(1)

        # ── Data de Nascimento ─────────────────────────────────────────────────
        for pat_nasc in [
            r'(?:Data\s+de\s+Nasc(?:imento)?|Nascimento|Nasc\.?|DT\.?\s*NASC\.?)'
            r'[:\s]+(\d{2}/\d{2}/\d{4})',
            r'(?:Data\s+Nasc(?:imento)?)[:\s]+(\d{2}/\d{2}/\d{4})',
        ]:
            m_nasc = re.search(pat_nasc, texto, re.I)
            if m_nasc:
                try:
                    out["data_nasc"] = datetime.strptime(m_nasc.group(1), "%d/%m/%Y").date()
                except Exception:
                    pass
                break

        all_comps: dict[date, tuple] = {}   # date → (valor, tipo)

        # ════════════════════════════════════════════════════════════════════
        # ESTRATÉGIA A — Uma seção por CNPJ (mais precisa)
        # ════════════════════════════════════════════════════════════════════
        pos_cnpjs = [(m.start(), m.group()) for m in _RE_CNPJ.finditer(texto)]

        if pos_cnpjs:
            limites = [p for p, _ in pos_cnpjs] + [len(texto)]
            for i in range(len(pos_cnpjs)):
                secao = texto[limites[i]: limites[i + 1]]
                for dt, v in comps_de_trecho(secao):
                    if dt not in all_comps:
                        all_comps[dt] = (v, "Empregado")

        # ════════════════════════════════════════════════════════════════════
        # ESTRATÉGIA B — Tabelas pdfplumber (fallback)
        # ════════════════════════════════════════════════════════════════════
        if not all_comps:
            for tabela in tabelas_raw:
                for linha in tabela:
                    celulas = [str(c or "").strip() for c in linha]
                    for i, cel in enumerate(celulas):
                        if not _RE_MES.fullmatch(cel):
                            continue
                        dt = parse_data(cel)
                        if not dt or dt in all_comps:
                            continue
                        for c2 in celulas[i + 1: i + 5]:
                            v = val_sal(re.sub(r'[*\s]', '', c2))
                            if v is not None and v > 0:
                                all_comps[dt] = (v, "Contribuição")
                                break

        # ════════════════════════════════════════════════════════════════════
        # ESTRATÉGIA C — Texto linha a linha (fallback final)
        # ════════════════════════════════════════════════════════════════════
        if not all_comps:
            for dt, v in comps_de_trecho(texto):
                if dt not in all_comps:
                    all_comps[dt] = (v, "Contribuição")

        # Converte para lista de linhas individuais (1 linha = 1 mês) — ordem do PDF
        out["vinculos"] = [
            {"Competência": dt.strftime("%m/%Y"), "Salário Real (R$)": v, "Tipo": tipo}
            for dt, (v, tipo) in all_comps.items()
        ]

    except Exception as e:
        import traceback
        out["erro"] = str(e)
        out["_texto_raw"] = traceback.format_exc()

    return out

# ─── MOTOR DE CÁLCULO (EC 103/2019) ───────────────────────────────────────────
class Calc:
    @staticmethod
    def pontos_min(sexo: str, ano: int) -> int:
        d = max(0, ano - 2019)
        return min(86 + d, 100) if sexo == "F" else min(96 + d, 105)

    @staticmethod
    def idade_min_prog(sexo: str, ano: int) -> float:
        d = max(0, ano - 2019) * 0.5
        return min(57.0 + d, 62.0) if sexo == "F" else min(61.0 + d, 65.0)

    @staticmethod
    def aliquota(sexo: str, tempo: float) -> float:
        base = 15.0 if sexo == "F" else 20.0
        return min(0.60 + max(0.0, tempo - base) * 0.02, 1.0)

    @staticmethod
    def fator_prev(idade: float, tempo: float) -> float:
        es = expectativa_sobrevida(idade)
        a = 0.31
        fp = (tempo * a / es) * (1 + (idade + tempo * a) / 100)
        return round(min(max(fp, 0.40), 1.50), 4)

    def cenarios(
        self, sexo: str, data_nasc: date, tempo_atual: float,
        media_sal: float, modalidade: str, ano_ref: int, tempo_2019: float,
    ) -> tuple:
        mod = MODALIDADES.get(modalidade, {"red_i": 0, "red_t": 0})
        anos_proj = max(0, ano_ref - ANO_HOJE)
        tempo = tempo_atual + anos_proj
        data_ref = date(ano_ref, 7, 1)
        idade = idade_dec(data_nasc, data_ref)

        red_i = mod["red_i"]
        red_t = mod["red_t"]
        tempo_min_base = 30.0 if sexo == "F" else 35.0
        tempo_min = tempo_min_base - red_t

        faltava_2019 = max(0.0, tempo_min_base - tempo_2019)
        era_seg = tempo_2019 > 0

        def entry(pode, req, sit, ben, ali, fator, desc):
            return {
                "pode": pode, "requisito": req, "situacao": sit,
                "beneficio": min(ben, TETO_INSS) if pode else 0.0,
                "aliquota": ali, "fator": fator, "descricao": desc,
                "falta": 0.0,
            }

        pts_nec = self.pontos_min(sexo, ano_ref)
        pts_at = idade + tempo
        ali_pt = self.aliquota(sexo, tempo)
        pode_pt = pts_at >= pts_nec and tempo >= tempo_min
        c1 = entry(
            pode_pt,
            f"Pontos ≥ {pts_nec}  |  Contrib. ≥ {tempo_min:.0f} anos",
            f"{pts_at:.1f}/{pts_nec} pts  |  {tempo:.1f}/{tempo_min:.0f} anos",
            media_sal * ali_pt, ali_pt, None,
            f"Regra de transição — Art. 15, I. Sem fator previdenciário. "
            f"Benefício: 60% + 2%/ano acima de {'15' if sexo=='F' else '20'} anos de contribuição.",
        )
        if not pode_pt:
            c1["falta"] = max(0.0, pts_nec - pts_at)

        id_prog = self.idade_min_prog(sexo, ano_ref) - red_i
        ali_id = self.aliquota(sexo, tempo)
        pode_id = idade >= id_prog and tempo >= tempo_min
        c2 = entry(
            pode_id,
            f"Idade ≥ {id_prog:.1f}  |  Contrib. ≥ {tempo_min:.0f} anos",
            f"Idade {idade:.1f}/{id_prog:.1f}  |  {tempo:.1f}/{tempo_min:.0f} anos",
            media_sal * ali_id, ali_id, None,
            f"Art. 15, II — progressiva até {'62' if sexo=='F' else '65'} anos. Sem fator previdenciário.",
        )
        if not pode_id:
            c2["falta"] = max(0.0, id_prog - idade)

        eleg_50 = era_seg and 0 < faltava_2019 < 2.0
        tnec_50 = tempo_min_base + faltava_2019 * 0.5
        fp_50 = self.fator_prev(idade, tempo)
        pode_50 = eleg_50 and tempo >= tnec_50
        sit_50 = (
            f"{tempo:.1f}/{tnec_50:.1f} anos  |  Pedágio: {faltava_2019*0.5:.1f} anos"
            if eleg_50 else "Não elegível (faltavam ≥ 2 anos em 2019)"
        )
        c3 = entry(
            pode_50,
            f"Faltavam < 2 anos em 13/11/2019  |  {tnec_50:.1f} anos contrib.",
            sit_50, media_sal * fp_50, fp_50, fp_50,
            "Art. 16 — pedágio de 50% do tempo que faltava. Aplica fator previdenciário (tábua IBGE).",
        )
        if eleg_50 and not pode_50:
            c3["falta"] = max(0.0, tnec_50 - tempo)

        id_min_100 = (57.0 if sexo == "F" else 60.0) - red_i
        tnec_100 = tempo_min_base + faltava_2019
        pode_100 = era_seg and idade >= id_min_100 and tempo >= tnec_100
        sit_100 = (
            f"Idade {idade:.1f}/{id_min_100:.0f}  |  {tempo:.1f}/{tnec_100:.1f} anos"
            if era_seg else "Não elegível (sem vínculo antes de 13/11/2019)"
        )
        c4 = entry(
            pode_100,
            f"Idade ≥ {id_min_100:.0f}  |  {tnec_100:.1f} anos contrib. (pedágio de {faltava_2019:.1f} anos)",
            sit_100, media_sal, 1.0, None,
            "Art. 15, III — sem fator previdenciário. Benefício = 100% da média salarial.",
        )
        if era_seg and not pode_100:
            c4["falta"] = max(0.0, id_min_100 - idade)

        id_def = (62.0 if sexo == "F" else 65.0) - red_i
        ali_def = self.aliquota(sexo, tempo)
        pode_def = idade >= id_def and tempo >= 15.0
        c5 = entry(
            pode_def,
            f"Idade ≥ {id_def:.0f}  |  15 anos de carência",
            f"Idade {idade:.1f}/{id_def:.0f}  |  {tempo:.1f}/15 anos",
            media_sal * ali_def, ali_def, None,
            "Regra definitiva (EC 103/2019). Sem fator previdenciário.",
        )
        if not pode_def:
            c5["falta"] = max(0.0, id_def - idade)

        cen = {
            "Regra de Pontos": c1,
            "Idade Mín. Progressiva": c2,
            "Pedágio 50%": c3,
            "Pedágio 100%": c4,
            "Apos. por Idade": c5,
        }
        return cen, round(idade, 1), round(tempo, 1)

    def melhor(self, cen: dict):
        disp = {k: v for k, v in cen.items() if v["pode"]}
        if not disp:
            return None, None
        k = max(disp, key=lambda x: disp[x]["beneficio"])
        return k, disp[k]

    def matriz(
        self, sexo: str, data_nasc: date, tempo_atual: float,
        media_sal: float, modalidade: str, tempo_2019: float,
        anos=range(ANO_HOJE, 2051),
    ) -> pd.DataFrame:
        rows = []
        for ano in anos:
            cen, idade, tempo = self.cenarios(
                sexo, data_nasc, tempo_atual, media_sal, modalidade, ano, tempo_2019
            )
            row = {"Ano": ano, "Idade": idade, "Contrib.(anos)": tempo}
            for nome, d in cen.items():
                row[nome] = d["beneficio"] if d["pode"] else None
            rows.append(row)
        return pd.DataFrame(rows)


CALC = Calc()

# ─── GRÁFICOS ──────────────────────────────────────────────────────────────────
CORES = {
    "Regra de Pontos":         "#4169E1",
    "Idade Mín. Progressiva":  "#2E7D32",
    "Pedágio 50%":             "#F57C00",
    "Pedágio 100%":            "#7B1FA2",
    "Apos. por Idade":         "#00838F",
}


def fig_barra(cen: dict, nome: str) -> plt.Figure:
    labels = list(cen.keys())
    vals   = [c["beneficio"] for c in cen.values()]
    cores  = ["#2E7D32" if c["pode"] else "#E0E0E0" for c in cen.values()]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(range(len(labels)), vals, color=cores, width=0.5, zorder=3)
    ax.grid(axis="y", zorder=0, alpha=0.4)
    ax.set_title(f"Comparativo de Benefícios — {nome}", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Valor Mensal (R$)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x:,.0f}"))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8.5)

    for bar, val, c in zip(bars, vals, cen.values()):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
                    fmt_moeda(val), ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        if not c["pode"]:
            ax.text(bar.get_x() + bar.get_width() / 2, 150,
                    "Não\nelegível", ha="center", va="bottom", fontsize=7.5, color="#999")

    ax.set_ylim(0, max(vals + [1500]) * 1.25)
    plt.tight_layout()
    return fig


def fig_projecao(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for col in df.columns:
        if col in ("Ano", "Idade", "Contrib.(anos)"):
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        ax.plot(df.loc[s.index, "Ano"], s, label=col, color=CORES.get(col, "#888"),
                linewidth=2, marker="o", markersize=3,
                markerfacecolor="white", markeredgewidth=1.5)

    ax.set_title("Projeção Temporal dos Benefícios por Regra", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Valor Mensal (R$)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x:,.0f}"))
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def fig_breakeven(va: float, ia: float, vb: float, ib: float, la: str, lb: str):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.linspace(min(ia, ib), 90, 300)
    acc_a = np.array([(va * (xi - ia) * 12) if xi >= ia else 0 for xi in x])
    acc_b = np.array([(vb * (xi - ib) * 12) if xi >= ib else 0 for xi in x])

    ax.fill_between(x, acc_a, acc_b, where=acc_b > acc_a, alpha=0.07, color="#4169E1")
    ax.plot(x, acc_a, color="#B0BEC5", linewidth=2,   label=f"{la} — {fmt_moeda(va)}/mês")
    ax.plot(x, acc_b, color="#4169E1", linewidth=2.5, label=f"{lb} — {fmt_moeda(vb)}/mês")

    tbk = None
    if vb > va and ib > ia:
        tbk = (vb * ib - va * ia) / (vb - va)
        if ia < tbk < 90:
            vbk = va * (tbk - ia) * 12
            ax.scatter(tbk, vbk, color="red", s=100, zorder=6, label=f"Breakeven: {tbk:.1f} anos")
            ax.axvline(tbk, color="red", linestyle="--", alpha=0.45, linewidth=1.2)

    ax.set_title("Análise de Breakeven — Quando compensa esperar?", fontsize=12, fontweight="bold")
    ax.set_xlabel("Idade")
    ax.set_ylabel("Total Acumulado")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x/1e6:.2f}M"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, tbk


# ─── GERADOR DE PDF ────────────────────────────────────────────────────────────
def _fig_png(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(buf.getvalue())
    tmp.close()
    return tmp.name


def gerar_pdf(
    nome: str, sexo: str, modalidade: str, data_nasc: date,
    media_sal: float, tempo_contrib: float,
    cen: dict, melhor_nome: str,
    f_barra: plt.Figure, f_proj: plt.Figure,
    parecer: str, ano_ref: int,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(True, 14)

    # ── helpers ───────────────────────────────────────────────────────────────
    def cabec(titulo: str):
        pdf.set_fill_color(*AZUL)
        pdf.rect(0, 0, 210, 26, "F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 14)
        pdf.set_y(5)
        pdf.cell(0, 8, norm(titulo), ln=True, align="C")
        pdf.set_font("Arial", "", 8.5)
        pdf.cell(0, 7,
                 norm(f"Cliente: {nome.upper()}  |  Modalidade: {modalidade}  |  Ano-base: {ano_ref}"),
                 ln=True, align="C")
        pdf.set_text_color(*CINZA)
        pdf.set_y(32)

    def linha_tabela(campo: str, valor: str, fill: bool = False):
        if fill:
            pdf.set_fill_color(*CINZA_CL)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(55, 7, norm(campo + ":"), 1, 0, "L", fill)
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 7, norm(str(valor)), 1, 1, "L", fill)

    def secao(titulo: str):
        pdf.ln(3)
        pdf.set_fill_color(*AZUL)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 7, norm(f"  {titulo}"), ln=True, fill=True)
        pdf.set_text_color(*CINZA)
        pdf.ln(1)

    def legenda(texto: str):
        pdf.set_font("Arial", "I", 8)
        pdf.set_text_color(80, 80, 80)
        pdf.set_fill_color(248, 249, 250)
        pdf.multi_cell(0, 4.5, norm(texto), fill=True)
        pdf.set_text_color(*CINZA)
        pdf.ln(2)

    # ════════════════════════════════════════════════════════════════════════
    # PÁG 1 — Capa + perfil + gráfico comparativo
    # ════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    cabec("ESTUDO TECNICO PREVIDENCIARIO — EC 103/2019")

    secao("Perfil do Segurado")
    idade_atual = idade_dec(data_nasc)
    linha_tabela("Nome",               nome)
    linha_tabela("Data de Nascimento", data_nasc.strftime("%d/%m/%Y"),              True)
    linha_tabela("Idade Atual",        f"{idade_atual:.1f} anos")
    linha_tabela("Sexo",               "Feminino" if sexo == "F" else "Masculino",  True)
    linha_tabela("Modalidade",         modalidade)
    linha_tabela("Contribuicao",       f"{tempo_contrib:.1f} anos",                 True)
    linha_tabela("Media Salarial",     fmt_moeda(media_sal))

    pdf.ln(2)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4.5, norm(
        "Este estudo analisa as possibilidades de aposentadoria conforme a Reforma da Previdencia "
        "(EC 103/2019) e suas regras de transicao. Os valores sao estimativas baseadas nos dados "
        "fornecidos e nao substituem assessoria juridica especializada."
    ))

    secao("Grafico 1 — Comparativo de Beneficios por Regra (Ano-base: " + str(ano_ref) + ")")
    t1 = _fig_png(f_barra)
    pdf.image(t1, x=10, w=190)
    os.unlink(t1)
    legenda(
        "Como ler este grafico: cada barra representa o valor mensal estimado de aposentadoria "
        "para uma regra diferente no ano-base selecionado. Barras em verde indicam regras para as "
        "quais o segurado JA e elegivel; barras cinzas indicam regras ainda nao atingidas. "
        "O maior valor entre as barras verdes representa o melhor cenario disponivel."
    )

    # ════════════════════════════════════════════════════════════════════════
    # PÁG 2 — Análise individual de cada regra
    # ════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    cabec("ANALISE DAS REGRAS DE TRANSICAO — EC 103/2019")

    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 4.5, norm(
        "A EC 103/2019 criou regras de transicao para proteger quem ja contribuia antes de "
        "13/11/2019. Abaixo, cada regra e avaliada individualmente com sua situacao atual."
    ))
    pdf.ln(2)

    for regra, d in cen.items():
        pdf.ln(1)
        pdf.set_fill_color(*(VERDE if d["pode"] else VERM))
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 9)
        label = "ELEGIVEL" if d["pode"] else "NAO ELEGIVEL"
        pdf.cell(0, 7, norm(f"  {regra} — {label}"), ln=True, fill=True)

        pdf.set_text_color(*CINZA)
        pdf.set_font("Arial", "", 8)
        pdf.cell(0, 4.5, norm(f"Requisitos: {d['requisito']}"), ln=True)
        pdf.cell(0, 4.5, norm(f"Situacao atual: {d['situacao']}"), ln=True)
        if d["pode"]:
            alq_s = (f"FP {d['fator']:.4f} (Tabua IBGE 2022)" if d["fator"]
                     else f"Aliquota {d['aliquota']*100:.0f}%")
            pdf.set_font("Arial", "B", 8.5)
            pdf.cell(0, 5, norm(f"Beneficio estimado: {fmt_moeda(d['beneficio'])}  |  {alq_s}"), ln=True)
            pdf.set_font("Arial", "", 8)
        else:
            if d.get("falta", 0) > 0:
                unidade = "pts" if "Pontos" in regra else "anos"
                pdf.set_font("Arial", "B", 8)
                pdf.set_text_color(150, 50, 50)
                pdf.cell(0, 4.5, norm(f"Faltam: {d['falta']:.1f} {unidade}"), ln=True)
                pdf.set_text_color(*CINZA)
                pdf.set_font("Arial", "", 8)
        pdf.set_font("Arial", "I", 7.5)
        pdf.set_text_color(90, 90, 90)
        pdf.multi_cell(0, 3.8, norm(d["descricao"]))
        pdf.set_text_color(*CINZA)

    # ════════════════════════════════════════════════════════════════════════
    # PÁG 3 — Projeção temporal com gráfico e explicação
    # ════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    cabec("PROJECAO TEMPORAL DOS BENEFICIOS")

    secao("Grafico 2 — Evolucao dos Beneficios ao Longo do Tempo")
    t2 = _fig_png(f_proj)
    pdf.image(t2, x=5, w=200)
    os.unlink(t2)
    legenda(
        "Como ler este grafico: cada linha colorida representa uma regra de aposentadoria. "
        "A linha so aparece a partir do ano em que o segurado atinge os requisitos daquela regra. "
        "O crescimento das linhas ao longo do tempo reflete o aumento da aliquota (60% + 2% por "
        "ano adicional de contribuicao). Regras sem fator previdenciario (linhas continuas) tendem "
        "a crescer de forma mais previsivel do que o Pedagio 50%, que depende do fator calculado "
        "pela Tabua IBGE."
    )

    # Tabela resumo: primeiro ano elegível por regra
    secao("Resumo — Primeiro Ano de Elegibilidade por Regra")
    pdf.set_fill_color(*AZUL)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 8)
    pdf.cell(65, 7, "Regra",          1, 0, "C", True)
    pdf.cell(28, 7, "1o Ano",         1, 0, "C", True)
    pdf.cell(25, 7, "Idade",          1, 0, "C", True)
    pdf.cell(35, 7, "Beneficio",      1, 0, "C", True)
    pdf.cell(0,  7, "Aliquota / FP",  1, 1, "C", True)
    pdf.set_text_color(*CINZA)
    pdf.set_font("Arial", "", 8)
    alt = False
    for regra, d in cen.items():
        fill = alt
        alt = not alt
        if fill:
            pdf.set_fill_color(*CINZA_CL)
        status = "Elegivel" if d["pode"] else f"-{d.get('falta',0):.1f}"
        alq_txt = (f"FP {d['fator']:.4f}" if d["fator"] else f"{d['aliquota']*100:.0f}%")
        ben_txt = fmt_moeda(d["beneficio"]) if d["pode"] else "—"
        pdf.cell(65, 6, norm(regra),           1, 0, "L", fill)
        pdf.cell(28, 6, norm(str(ano_ref)),    1, 0, "C", fill)
        pdf.cell(25, 6, norm(status),          1, 0, "C", fill)
        pdf.cell(35, 6, norm(ben_txt),         1, 0, "C", fill)
        pdf.cell(0,  6, norm(alq_txt),         1, 1, "C", fill)

    # ════════════════════════════════════════════════════════════════════════
    # PÁG 4 — Parecer técnico e base legal
    # ════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    cabec("PARECER TECNICO E RECOMENDACAO FINAL")

    if melhor_nome:
        md = cen[melhor_nome]
        pdf.set_fill_color(*VERDE)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, norm(f"  MELHOR CENARIO: {melhor_nome}"), ln=True, fill=True)
        pdf.set_text_color(*CINZA)
        pdf.ln(2)
        pdf.set_font("Arial", "B", 9)
        alq_s = (f"Fator Prev.: {md['fator']:.4f}" if md["fator"]
                 else f"Aliquota: {md['aliquota']*100:.0f}%")
        pdf.cell(0, 6, norm(f"Beneficio mensal estimado: {fmt_moeda(md['beneficio'])}  |  {alq_s}"), ln=True)
        pdf.set_font("Arial", "", 8.5)
        pdf.cell(0, 5, norm(f"Situacao: {md['situacao']}"), ln=True)
    else:
        pdf.set_fill_color(*VERM)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 9, norm("  NENHUMA REGRA ELEGIVEL NO ANO SELECIONADO"), ln=True, fill=True)

    pdf.ln(4)
    pdf.set_text_color(*AZUL)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 6, "Parecer Tecnico", ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_text_color(*CINZA)
    pdf.set_font("Arial", "", 9.5)
    pdf.set_fill_color(252, 252, 252)
    pdf.multi_cell(0, 5.5, norm(parecer), border=1, fill=True)

    pdf.ln(5)
    pdf.set_text_color(*AZUL)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(0, 6, "Base Legal", ln=True)
    pdf.set_font("Arial", "", 8)
    pdf.set_text_color(*CINZA)
    for item in [
        "EC 103/2019 — Reforma da Previdencia Social",
        "Art. 15, I — Regra de Pontos (Transicao progressiva de pontos)",
        "Art. 15, II — Idade Minima Progressiva (Transicao por idade)",
        "Art. 15, III — Pedagio 100% (Transicao com pedagio integral)",
        "Art. 16 — Pedagio 50% (para segurados a menos de 2 anos do minimo em 2019)",
        "Lei 9.876/1999 — Fator Previdenciario",
        "Tabua de Mortalidade IBGE 2022 — Expectativa de Sobrevida por Idade",
        "IN PRES/INSS 128/2022 — Instrucao Normativa sobre concessao de beneficios",
    ]:
        pdf.cell(0, 5, norm(f"  • {item}"), ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "I", 7.5)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 4, norm(
        "AVISO LEGAL: Este documento e de carater estritamente informativo e nao constitui "
        "assessoria juridica ou previdenciaria. Os valores apresentados sao estimativas baseadas "
        "nos dados fornecidos e podem divergir dos calculos oficiais do INSS. Recomenda-se "
        "consultar um advogado especializado em Direito Previdenciario para analise definitiva."
    ))

    return bytes(pdf.output())


# ─── EXPORTAÇÃO / IMPORTAÇÃO XLS ──────────────────────────────────────────────
def gerar_xls(dados: dict, df_vinc: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Aba 1 — Dados do segurado
        df_info = pd.DataFrame([
            {"Campo": "Nome",               "Valor": dados.get("nome", "")},
            {"Campo": "Data de Nascimento", "Valor": dados.get("data_nasc", "")},
            {"Campo": "Sexo",               "Valor": "Feminino" if dados.get("sexo") == "F" else "Masculino"},
            {"Campo": "Modalidade",         "Valor": dados.get("modalidade", "")},
            {"Campo": "CPF",                "Valor": dados.get("cpf", "")},
        ])
        df_info.to_excel(writer, sheet_name="Dados", index=False)

        # Aba 2 — Competências (tabela editável)
        df_exp = df_vinc.copy() if not df_vinc.empty else pd.DataFrame(
            columns=["Competência", "Salário Real (R$)", "Tipo"]
        )
        df_exp.to_excel(writer, sheet_name="Vínculos", index=False)

        # Formata largura das colunas
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_w = max(len(str(cell.value or "")) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_w + 4, 40)

    return output.getvalue()


def ler_xls(uploaded_xls) -> pd.DataFrame:
    """Lê a aba 'Vínculos' de um XLS exportado por esta aplicação."""
    try:
        df = pd.read_excel(uploaded_xls, sheet_name="Vínculos", engine="openpyxl")
        colunas_esperadas = ["Competência", "Salário Real (R$)", "Tipo"]
        for col in colunas_esperadas:
            if col not in df.columns:
                df[col] = None
        return df[colunas_esperadas]
    except Exception as e:
        return None, str(e)


# ─── INTERFACE ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Planejamento Previdenciário",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.main .block-container{padding-top:1.6rem}
[data-testid="stSidebar"]{background:#F8F9FA}
.stTabs [data-baseweb="tab"]{font-weight:600;font-size:0.93rem}
.card-ok   {border:2px solid #2E7D32;border-radius:8px;padding:12px;background:#F1F8E9;margin-bottom:4px}
.card-no   {border:2px solid #CFD8DC;border-radius:8px;padding:12px;background:#FAFAFA;margin-bottom:4px}
.card-best {border:2px solid #4169E1;border-radius:8px;padding:12px;background:#EEF2FF;margin-bottom:4px}
.chip-ok   {display:inline-block;background:#E8F5E9;color:#2E7D32;border-radius:4px;padding:1px 7px;font-size:0.78rem;font-weight:700}
.chip-no   {display:inline-block;background:#FFEBEE;color:#B71C1C;border-radius:4px;padding:1px 7px;font-size:0.78rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

# Estado global
_defaults = {
    "pagina":       "home",
    "modo":         None,       # "manual" | "pdf"
    "dados":        {},
    "vinculos_df":  None,
    "parecer":      "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# HOME — seleção de modo + dados manuais
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.pagina == "home":

    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 0.4rem'>
    <h1 style='color:#4169E1;margin-bottom:4px'>🛡️ Planejamento Previdenciário</h1>
    <p style='color:#555;font-size:1.05rem'>
        Análise técnica de aposentadoria — EC 103/2019 · Fator Previdenciário (Tábua IBGE 2022)
    </p></div>""", unsafe_allow_html=True)
    st.divider()

    modo = st.radio(
        "**Modo de entrada de dados:**",
        ["✏️  Preenchimento Manual", "📄  Upload do Extrato CNIS (PDF do Meu INSS)"],
        horizontal=True,
    )
    st.session_state.modo = "manual" if "Manual" in modo else "pdf"

    st.markdown("---")
    col_form, col_info = st.columns([3, 2], gap="large")

    with col_form:
        nome = st.text_input("Nome completo *", placeholder="Nome do cliente")
        c1, c2 = st.columns(2)
        with c1:
            data_nasc = st.date_input("Data de Nascimento *",
                                      value=date(1975, 6, 15),
                                      min_value=date(1930, 1, 1),
                                      max_value=date.today())
            sexo = st.selectbox("Sexo *", ["F", "M"],
                                format_func=lambda x: "Feminino" if x == "F" else "Masculino")
        with c2:
            modalidade = st.selectbox("Modalidade *", list(MODALIDADES.keys()))
            tempo_contrib = st.number_input("Anos de Contribuição *",
                                            0.0, 55.0, 15.0, 0.5,
                                            help="Total de anos contribuídos até hoje.")

        media_sal = st.number_input(
            "Média Salarial de Contribuição (R$) *",
            PISO_INSS, TETO_INSS, 3500.0, 50.0,
            help="Média dos salários de contribuição. Será calculada automaticamente se fizer upload do CNIS.",
        )

        uploaded = None
        if st.session_state.modo == "pdf":
            st.markdown("**Upload do Extrato CNIS (PDF)**")
            st.caption("Meu INSS → Extrato Previdenciário → Extrato do CNIS → Exportar PDF")
            uploaded = st.file_uploader("Arquivo PDF do CNIS", type=["pdf"])

        st.markdown("")
        if st.button("▶️  Prosseguir", type="primary", use_container_width=True):
            if not nome.strip():
                st.error("Informe o nome do cliente.")
            elif st.session_state.modo == "pdf" and not uploaded:
                st.error("Faça o upload do PDF do CNIS.")
            else:
                dias = (date.today() - DATA_REFORMA).days
                t2019 = max(0.0, round(tempo_contrib - dias / 365.25, 2))

                st.session_state.dados = {
                    "nome":       nome.strip(),
                    "cpf":        "",
                    "data_nasc":  data_nasc,
                    "sexo":       sexo,
                    "modalidade": modalidade,
                    "tempo":      tempo_contrib,
                    "t2019":      t2019,
                    "media":      media_sal,
                }

                if st.session_state.modo == "pdf":
                    with st.spinner("Lendo PDF..."):
                        res = parse_pdf_cnis(uploaded)
                    if res["erro"]:
                        st.error(f"Erro ao ler PDF: {res['erro']}")
                        with st.expander("🔍 Texto bruto extraído (debug)"):
                            st.text(res.get("_texto_raw", ""))
                    else:
                        if res["nome"]:
                            st.session_state.dados["nome"] = res["nome"]
                        if res.get("cpf"):
                            st.session_state.dados["cpf"] = res["cpf"]
                        if res.get("data_nasc"):
                            st.session_state.dados["data_nasc"] = res["data_nasc"]
                        st.session_state.vinculos_df = pd.DataFrame(
                            res["vinculos"] if res["vinculos"] else [],
                            columns=["Competência", "Salário Real (R$)", "Tipo"]
                        )
                        st.session_state["_texto_raw"] = res.get("_texto_raw", "")
                        st.session_state.pagina = "conferencia"
                        st.rerun()
                else:
                    st.session_state.vinculos_df = None
                    st.session_state.pagina = "analise"
                    st.rerun()

    with col_info:
        st.subheader("ℹ️ Como funciona")
        st.markdown("""
**Manual** — preencha os campos ao lado e clique em Prosseguir.

**PDF CNIS** — faça o upload do extrato. O sistema extrai os vínculos automaticamente e exibe uma tela de conferência antes de calcular.

**Modalidades:**
- **Trabalhador Urbano** — regras padrão
- **Professor** — redução de 5 anos em idade e tempo
- **Atividade Especial** — redução de 5 anos no tempo mínimo
- **Rural** — regras específicas RGPS rural
- **RPPS** — Servidor Público (regras próprias)

**Regras analisadas (EC 103/2019):**
| Regra | Artigo | FP? |
|---|---|---|
| Pontos | Art. 15, I | ❌ |
| Idade Progressiva | Art. 15, II | ❌ |
| Pedágio 100% | Art. 15, III | ❌ |
| Pedágio 50% | Art. 16 | ✅ |
| Apos. por Idade | Definitiva | ❌ |
        """)
        st.caption("⚠️ Estimativas. Consulte advogado previdenciário.")

# ══════════════════════════════════════════════════════════════════════════════
# CONFERÊNCIA — revisar dados extraídos do PDF
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.pagina == "conferencia":
    d = st.session_state.dados

    with st.sidebar:
        if st.button("← Voltar", use_container_width=True):
            st.session_state.pagina = "home"
            st.rerun()

    st.markdown(f"<h2 style='color:#4169E1'>📋 Conferência de Dados — {d['nome']}</h2>",
                unsafe_allow_html=True)
    st.info(
        "Revise os dados extraídos do PDF. Corrija valores incorretos, "
        "adicione ou remova vínculos antes de calcular."
    )

    # Campos básicos editáveis
    ca, cb, cc = st.columns(3)
    with ca:
        d["nome"] = st.text_input("Nome", d.get("nome", ""))
        d["cpf"]  = st.text_input("CPF", d.get("cpf", ""))
    with cb:
        d["data_nasc"] = st.date_input("Data de Nascimento", d.get("data_nasc", date(1975, 1, 1)),
                                        min_value=date(1930, 1, 1), max_value=date.today())
        d["sexo"] = st.selectbox("Sexo", ["F", "M"],
                                  index=0 if d.get("sexo") == "F" else 1,
                                  format_func=lambda x: "Feminino" if x == "F" else "Masculino")
    with cc:
        d["modalidade"] = st.selectbox("Modalidade", list(MODALIDADES.keys()),
                                        index=list(MODALIDADES.keys()).index(d.get("modalidade", list(MODALIDADES.keys())[0])))
        _media_val = max(PISO_INSS, min(TETO_INSS, float(d.get("media", PISO_INSS))))
        d["media"] = st.number_input("Média Salarial (R$)", PISO_INSS, TETO_INSS,
                                      _media_val, 50.0)

    _COLS_BASE   = ["Competência", "Salário Real (R$)", "Tipo"]
    _LINHA_VAZIA = {"Competência": "", "Salário Real (R$)": 0.0, "Tipo": "Empregado"}

    df_vinc = st.session_state.vinculos_df
    if df_vinc is None or df_vinc.empty:
        df_vinc = pd.DataFrame(columns=_COLS_BASE)

    # ── Seção: Download XLS + Upload XLS ─────────────────────────────────────
    st.divider()
    st.subheader("📥 Exportar / Importar Planilha")

    col_dl, col_ul = st.columns(2, gap="large")

    with col_dl:
        st.markdown("**1. Baixe a planilha com os dados extraídos**")
        st.caption("Edite no Excel e reimporte abaixo, ou prossiga diretamente com os dados na tela.")
        xls_bytes = gerar_xls(d, df_vinc)
        nome_xls  = f"CNIS_{d['nome'].replace(' ', '_')}.xlsx"
        st.download_button(
            "⬇️ Baixar planilha Excel (.xlsx)",
            data=xls_bytes,
            file_name=nome_xls,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with col_ul:
        st.markdown("**2. (Opcional) Reimporte uma planilha corrigida**")
        st.caption("A planilha deve seguir o formato exportado acima (aba 'Vínculos').")
        xls_up = st.file_uploader("Upload do Excel corrigido", type=["xlsx", "xls"], key="xls_upload")
        if xls_up is not None:
            df_xls = ler_xls(xls_up)
            if df_xls is None:
                st.error("Não foi possível ler a planilha. Verifique se ela contém a aba 'Vínculos'.")
            else:
                st.session_state.vinculos_df = df_xls
                df_vinc = df_xls
                st.success(f"{len(df_xls)} vínculos importados da planilha.")

    # ── Seção: Tabela editável ────────────────────────────────────────────────
    st.divider()
    st.subheader("📑 Conferência dos Vínculos")
    st.caption("Edite diretamente na tabela ou use os controles abaixo para inserir linhas em qualquer posição.")

    # Controles de inserção / limpeza
    n_linhas = len(df_vinc)
    ci1, ci2, ci3 = st.columns([2, 1, 1])
    with ci1:
        pos_insert = st.number_input(
            "Inserir linha na posição (0 = início)",
            min_value=0, max_value=max(n_linhas, 0),
            value=max(n_linhas, 0), step=1, key="pos_insert",
        )
    with ci2:
        st.markdown("<div style='padding-top:28px'>", unsafe_allow_html=True)
        if st.button("➕ Inserir linha", use_container_width=True):
            nova = pd.DataFrame([_LINHA_VAZIA])
            if df_vinc.empty:
                df_vinc = nova.copy()
            else:
                df_vinc = pd.concat(
                    [df_vinc.iloc[:pos_insert], nova, df_vinc.iloc[pos_insert:]],
                    ignore_index=True,
                )
            st.session_state.vinculos_df = df_vinc
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with ci3:
        st.markdown("<div style='padding-top:28px'>", unsafe_allow_html=True)
        if st.button("🗑️ Limpar tudo", use_container_width=True):
            st.session_state.vinculos_df = pd.DataFrame(columns=_COLS_BASE)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Coluna de média acumulada (somente para exibição)
    df_vinc_display = df_vinc.copy()
    if not df_vinc_display.empty and "Salário Real (R$)" in df_vinc_display.columns:
        df_vinc_display["Média dos Salários (R$)"] = (
            df_vinc_display["Salário Real (R$)"].fillna(0).expanding().mean().round(2)
        )
    else:
        df_vinc_display["Média dos Salários (R$)"] = pd.Series(dtype=float)

    df_editado = st.data_editor(
        df_vinc_display,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Competência":             st.column_config.TextColumn("Competência (MM/AAAA)"),
            "Salário Real (R$)":       st.column_config.NumberColumn(
                                           "Salário Real (R$)",
                                           min_value=0.0, max_value=TETO_INSS * 2, format="R$ %.2f"),
            "Média dos Salários (R$)": st.column_config.NumberColumn(
                                           "Média dos Salários (R$)",
                                           disabled=True, format="R$ %.2f"),
            "Tipo":                    st.column_config.SelectboxColumn("Tipo",
                                           options=["Empregado", "Autônomo",
                                                    "Contribuinte Fac.", "Avulso"]),
        },
        key="editor_vinculos",
    )
    st.session_state.vinculos_df = df_editado[[c for c in _COLS_BASE if c in df_editado.columns]]

    # Debug: texto bruto do PDF
    texto_raw = st.session_state.get("_texto_raw", "")
    if texto_raw:
        with st.expander("🔍 Texto bruto extraído do PDF"):
            st.text_area("Conteúdo extraído:", texto_raw, height=300, disabled=True)

    # ── Agregados ─────────────────────────────────────────────────────────────
    if not df_editado.empty and "Salário Real (R$)" in df_editado.columns:
        total_meses = int(df_editado["Salário Real (R$)"].notna().sum())
        total_anos  = total_meses / 12
        medias      = df_editado["Salário Real (R$)"].dropna()
        media_calc  = medias.mean() if not medias.empty else d["media"]

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total de meses contrib.", f"{int(total_meses)} meses")
        m2.metric("Total em anos", f"{total_anos:.1f} anos")
        m3.metric("Média salarial calculada", fmt_moeda(media_calc))

        usar_calc = st.checkbox("Usar esses valores calculados (substituir campos acima)", value=True)
        if usar_calc:
            d["tempo"] = round(total_anos, 2)
            d["media"] = round(media_calc, 2)

    # Recalcula t2019
    dias = (date.today() - DATA_REFORMA).days
    d["t2019"] = max(0.0, round(d["tempo"] - dias / 365.25, 2))
    st.session_state.dados = d

    st.markdown("")
    if st.button("✅ Confirmar e Calcular", type="primary", use_container_width=True):
        st.session_state.pagina = "analise"
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISE — resultados
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.pagina == "analise":
    d = st.session_state.dados

    with st.sidebar:
        st.markdown("### Navegação")
        if st.button("← Início", use_container_width=True):
            st.session_state.pagina = "home"
            st.rerun()
        if st.button("← Conferência", use_container_width=True,
                     disabled=(st.session_state.modo == "manual")):
            st.session_state.pagina = "conferencia"
            st.rerun()
        st.divider()
        st.markdown(f"**{d['nome']}**")
        st.markdown(f"{d['modalidade']}")
        st.markdown(f"Sexo: {'Feminino' if d['sexo']=='F' else 'Masculino'}")
        st.markdown(f"Idade: {idade_dec(d['data_nasc']):.1f} anos")
        st.markdown(f"Contrib.: {d['tempo']:.1f} anos")
        st.markdown(f"Média: {fmt_moeda(d['media'])}")

    st.markdown(f"<h2 style='color:#4169E1'>📊 Análise — {d['nome']}</h2>",
                unsafe_allow_html=True)

    col_sl, _ = st.columns([2, 6])
    with col_sl:
        ano_alvo = st.slider("Ano de referência", ANO_HOJE, 2050, ANO_HOJE)

    cen, idade_alvo, tempo_alvo = CALC.cenarios(
        d["sexo"], d["data_nasc"], d["tempo"],
        d["media"], d["modalidade"], ano_alvo, d["t2019"],
    )
    melhor_nome, melhor_d = CALC.melhor(cen)

    if melhor_nome:
        st.success(
            f"✅ **{ano_alvo}: {melhor_nome}** | "
            f"{fmt_moeda(melhor_d['beneficio'])}/mês | "
            f"Idade {idade_alvo} anos | {tempo_alvo} anos contrib."
        )
    else:
        st.warning(f"⚠️ Nenhuma regra elegível em {ano_alvo}. Avance o slider.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Cenários", "📋 Matriz de Elegibilidade", "📈 Projeção", "⚖️ Breakeven", "📄 Relatório PDF"
    ])

    # ── TAB 1: CENÁRIOS ──────────────────────────────────────────────────────
    with tab1:
        cols = st.columns(len(cen))
        for i, (regra, dados) in enumerate(cen.items()):
            with cols[i]:
                classe = "card-best" if regra == melhor_nome else ("card-ok" if dados["pode"] else "card-no")
                badge = "★ MELHOR" if regra == melhor_nome else ("✅" if dados["pode"] else "❌")
                val_txt = fmt_moeda(dados["beneficio"]) if dados["pode"] else "—"
                cor_val = "#2E7D32" if dados["pode"] else "#999"
                st.markdown(f"""
                <div class="{classe}">
                  <b style="font-size:0.8rem">{regra}</b><br>
                  <span style="font-size:0.85rem">{badge}</span><br>
                  <span style="font-size:0.7rem;color:#555">{dados['situacao']}</span><br>
                  <b style="color:{cor_val}">{val_txt}</b>
                </div>""", unsafe_allow_html=True)

        st.divider()
        for regra, dados in cen.items():
            icon = "✅" if dados["pode"] else "❌"
            lbl = f"{icon} **{regra}**"
            if dados["pode"]:
                lbl += f"  —  {fmt_moeda(dados['beneficio'])}"
            with st.expander(lbl):
                ea, eb = st.columns(2)
                with ea:
                    st.markdown(f"**Requisitos:** {dados['requisito']}")
                    st.markdown(f"**Situação:** {dados['situacao']}")
                    if dados["pode"]:
                        st.markdown(f"**Benefício:** {fmt_moeda(dados['beneficio'])}")
                        if dados["fator"]:
                            st.markdown(f"**Fator Prev. (IBGE 2022):** {dados['fator']:.4f}")
                        else:
                            st.markdown(f"**Alíquota:** {dados['aliquota']*100:.0f}%")
                with eb:
                    st.info(dados["descricao"])

        st.divider()
        fb = fig_barra(cen, d["nome"])
        st.pyplot(fb)
        plt.close(fb)

    # ── TAB 2: MATRIZ DE ELEGIBILIDADE ──────────────────────────────────────
    with tab2:
        st.markdown(
            "Checklist de elegibilidade e valor estimado do benefício **por ano e por regra**. "
            "Verde = elegível. Em branco = ainda não elegível."
        )

        anos_mat = list(range(ANO_HOJE, min(ANO_HOJE + 16, 2051)))
        with st.spinner("Calculando matriz..."):
            df_mat = CALC.matriz(
                d["sexo"], d["data_nasc"], d["tempo"],
                d["media"], d["modalidade"], d["t2019"], anos_mat,
            )

        # Monta pivot para exibição
        regras_cols = [c for c in df_mat.columns if c not in ("Ano", "Idade", "Contrib.(anos)")]
        df_exib = df_mat[["Ano", "Idade", "Contrib.(anos)"] + regras_cols].copy()

        # Destaca melhor valor de cada linha
        def estilo_linha(row):
            styles = [""] * len(row)
            vals = {c: row[c] for c in regras_cols if pd.notna(row.get(c)) and row.get(c, 0) > 0}
            if vals:
                best = max(vals, key=vals.get)
                for i, col in enumerate(row.index):
                    if col == best:
                        styles[i] = "background-color: #E8F5E9; font-weight: bold"
                    elif col in vals:
                        styles[i] = "background-color: #F1F8E9"
            return styles

        fmt = {c: lambda v: fmt_moeda(v) if pd.notna(v) and v > 0 else "—"
               for c in regras_cols}
        fmt["Idade"] = "{:.1f}"
        fmt["Contrib.(anos)"] = "{:.1f}"

        st.dataframe(
            df_exib.style.apply(estilo_linha, axis=1).format(fmt),
            use_container_width=True,
            height=420,
        )

        # Análise do melhor ano
        st.markdown("##### Análise do Melhor Ano por Regra")
        for col in regras_cols:
            primeiros = df_mat[df_mat[col].notna() & (df_mat[col] > 0)]
            if primeiros.empty:
                st.markdown(f"- **{col}**: Não elegível no período analisado.")
            else:
                primeiro = primeiros.iloc[0]
                maior    = primeiros.loc[primeiros[col].idxmax()]
                st.markdown(
                    f"- **{col}**: Elegível a partir de **{int(primeiro['Ano'])}** "
                    f"(idade {primeiro['Idade']:.1f}) — "
                    f"{fmt_moeda(primeiro[col])}/mês. "
                    f"Valor máximo em {int(maior['Ano'])}: {fmt_moeda(maior[col])}/mês."
                )

    # ── TAB 3: PROJEÇÃO ──────────────────────────────────────────────────────
    with tab3:
        with st.spinner("Calculando projeção..."):
            df_proj = CALC.matriz(
                d["sexo"], d["data_nasc"], d["tempo"],
                d["media"], d["modalidade"], d["t2019"],
            )
        fp = fig_projecao(df_proj)
        st.pyplot(fp)
        plt.close(fp)

        with st.expander("Tabela completa"):
            regras_p = [c for c in df_proj.columns if c not in ("Ano", "Idade", "Contrib.(anos)")]
            df_p = df_proj[["Ano", "Idade"] + regras_p].copy()
            for c in regras_p:
                df_p[c] = df_p[c].apply(lambda v: fmt_moeda(v) if pd.notna(v) and v > 0 else "—")
            st.dataframe(df_p.set_index("Ano"), use_container_width=True)

    # ── TAB 4: BREAKEVEN ─────────────────────────────────────────────────────
    with tab4:
        st.markdown(
            "**Breakeven dinâmico:** se eu esperar mais tempo para ganhar um benefício maior, "
            "em quantos anos recupero o que deixei de receber?"
        )
        disp = [(k, v) for k, v in cen.items() if v["pode"]]

        if len(disp) < 2:
            st.info("São necessárias ao menos 2 regras elegíveis. Avance o ano de referência.")
        else:
            ba, bb = st.columns(2)
            with ba:
                st.markdown("**Opção A — Mais cedo**")
                na = st.selectbox("Regra A", [o[0] for o in disp], key="bka")
                va = cen[na]["beneficio"]
                ia = st.number_input("Idade início (A)", 45.0, 85.0,
                                     round(float(idade_alvo), 1), 0.5, key="ia")
                st.metric("Mensal A", fmt_moeda(va))

            with bb:
                st.markdown("**Opção B — Mais tarde**")
                opts_b = [o[0] for o in disp if o[0] != na]
                if opts_b:
                    nb = st.selectbox("Regra B", opts_b, key="bkb")
                    vb = cen[nb]["beneficio"]
                    ib = st.number_input("Idade início (B)", 45.0, 85.0,
                                         min(85.0, round(float(idade_alvo) + 2, 1)), 0.5, key="ib")
                    st.metric("Mensal B", fmt_moeda(vb))

                    if vb > va and ib > ia:
                        fbk, tbk = fig_breakeven(va, ia, vb, ib, na, nb)
                        st.pyplot(fbk)
                        plt.close(fbk)

                        if tbk and ia < tbk < 90:
                            exp_vida = expectativa_sobrevida(ia) + ia
                            if tbk < exp_vida:
                                st.success(
                                    f"📌 Breakeven: **{tbk:.1f} anos**. "
                                    f"Antes da expectativa de vida estimada ({exp_vida:.0f} anos). "
                                    "**Recomenda-se aguardar** o benefício maior."
                                )
                            else:
                                st.warning(
                                    f"📌 Breakeven: **{tbk:.1f} anos**. "
                                    f"Após a expectativa de vida ({exp_vida:.0f} anos). "
                                    "A opção precoce pode ser mais vantajosa."
                                )
                        elif tbk and tbk >= 90:
                            st.info("Breakeven após os 90 anos — a opção precoce é provavelmente mais vantajosa.")
                    else:
                        st.info("Ajuste as idades para que B seja posterior a A, com benefício maior.")

    # ── TAB 5: RELATÓRIO PDF ─────────────────────────────────────────────────
    with tab5:
        st.subheader("📄 Relatório Técnico em PDF (4 páginas)")
        st.caption("Perfil · Análise de regras · Projeção temporal · Parecer técnico · Base legal")

        # ── Gera o parecer automático (muda quando ano ou melhor cenário mudam) ──
        chave_parecer = f"parecer__{ano_alvo}__{melhor_nome}"

        if chave_parecer not in st.session_state:
            if melhor_nome:
                md_b  = cen[melhor_nome]
                alq_s = (f"fator previdenciário de {md_b['fator']:.4f}" if md_b["fator"]
                         else f"alíquota de {md_b['aliquota']*100:.0f}%")
                elegiveis = [k for k, v in cen.items() if v["pode"]]
                outras    = [k for k in elegiveis if k != melhor_nome]
                outras_txt = (
                    f" As demais regras elegíveis ({', '.join(outras)}) apresentam valores inferiores."
                    if outras else " Esta é a única regra elegível no período analisado."
                )
                recomendacao = (
                    "Recomenda-se solicitar o benefício imediatamente"
                    if ano_alvo <= ANO_HOJE
                    else f"Recomenda-se planejar a solicitação para o ano de {ano_alvo}"
                )
                sem_fp = "não aplica fator previdenciário, maximizando o valor do benefício"
                com_fp = (f"considera o fator previdenciário de {md_b['fator']:.4f} (Tábua IBGE 2022)"
                          if md_b["fator"] else "aplica alíquota progressiva")
                st.session_state[chave_parecer] = (
                    f"Com base no perfil de {d['nome']}, o melhor cenário disponível em {ano_alvo} "
                    f"é a {melhor_nome}, com benefício estimado de {fmt_moeda(md_b['beneficio'])} mensais "
                    f"({alq_s}).{outras_txt} {recomendacao}, pois esta regra "
                    f"{sem_fp if not md_b['fator'] else com_fp}. "
                    f"O segurado terá {idade_alvo} anos e {tempo_alvo} anos de contribuição nessa data."
                )
            else:
                partes = [
                    f"{k}: {v['falta']:.1f} {'pts' if 'Pontos' in k else 'anos'}"
                    for k, v in cen.items() if v.get("falta", 0) > 0
                ]
                st.session_state[chave_parecer] = (
                    f"Em {ano_alvo}, {d['nome']} ainda não atingiu os requisitos para nenhuma regra de "
                    f"aposentadoria. Déficits identificados: {'; '.join(partes)}. "
                    "Recomenda-se manter as contribuições em dia e revisar o planejamento anualmente, "
                    "avançando o ano de referência para identificar o primeiro momento de elegibilidade."
                )

        # ── Editor do parecer ────────────────────────────────────────────────
        col_btn, _ = st.columns([2, 6])
        with col_btn:
            if st.button("🔄 Regenerar texto automático"):
                del st.session_state[chave_parecer]
                st.rerun()

        parecer_edit = st.text_area(
            "✏️ Parecer técnico (editável — o texto abaixo irá para o PDF):",
            value=st.session_state[chave_parecer],
            height=180,
            key=f"textarea_{chave_parecer}",
        )
        # Persiste edições do usuário na mesma chave
        st.session_state[chave_parecer] = parecer_edit

        st.divider()

        # ── Gera PDF automaticamente e exibe botão de download ───────────────
        chave_pdf = f"pdf__{ano_alvo}__{melhor_nome}__{hash(parecer_edit)}"

        if chave_pdf not in st.session_state:
            with st.spinner("Gerando relatório..."):
                try:
                    fb_pdf = fig_barra(cen, d["nome"])
                    df_pdf = CALC.matriz(
                        d["sexo"], d["data_nasc"], d["tempo"],
                        d["media"], d["modalidade"], d["t2019"],
                    )
                    fp_pdf = fig_projecao(df_pdf)
                    pdf_bytes = gerar_pdf(
                        d["nome"], d["sexo"], d["modalidade"], d["data_nasc"],
                        d["media"], tempo_alvo,
                        cen, melhor_nome,
                        fb_pdf, fp_pdf,
                        parecer_edit, ano_alvo,
                    )
                    plt.close(fb_pdf)
                    plt.close(fp_pdf)
                    st.session_state[chave_pdf] = pdf_bytes
                except Exception as e:
                    st.error(f"Erro ao gerar PDF: {e}")
                    st.session_state[chave_pdf] = None

        pdf_bytes = st.session_state.get(chave_pdf)
        if pdf_bytes:
            nome_arq = f"Prev_{d['nome'].replace(' ', '_')}_{ano_alvo}.pdf"
            st.download_button(
                "📥 Baixar Relatório PDF",
                data=pdf_bytes,
                file_name=nome_arq,
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
            st.success("✅ Relatório pronto para download.")

        st.divider()
        st.caption("⚠️ Documento de apoio técnico. Não substitui assessoria jurídica especializada.")
