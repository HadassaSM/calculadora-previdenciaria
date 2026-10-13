# 🛡️ Calculadora Previdenciária

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://calculadora-previdenciaria.streamlit.app/)
[![CI](https://github.com/HadassaSM/calculadora-previdenciaria/actions/workflows/ci.yml/badge.svg)](https://github.com/HadassaSM/calculadora-previdenciaria/actions/workflows/ci.yml)

**▶ Acesse online: https://calculadora-previdenciaria.streamlit.app/**

Aplicação web em **Python + Streamlit** para planejamento de aposentadoria no **RGPS (INSS)**, considerando as regras de transição da **Reforma da Previdência (EC 103/2019)**.

A partir dos dados do segurado, digitados à mão ou extraídos automaticamente do **extrato do CNIS em PDF**, o app compara todas as regras de aposentadoria, mostra em que ano cada uma passa a valer e gera um **relatório técnico em PDF**.

## Funcionalidades

- **Leitura do CNIS em PDF** (`pdfplumber`): extrai vínculos, remunerações e tempo de contribuição.
- **Cenários por regra**: pontos, idade mínima progressiva, pedágio de 50% e de 100%, regra permanente etc., com destaque para a mais vantajosa.
- **Matriz de elegibilidade**: em que ano cada regra é atingida.
- **Projeção** do tempo de contribuição e do benefício.
- **Breakeven**: compara duas regras e calcula em quanto tempo a que paga mais compensa a espera.
- **Fator previdenciário** calculado com a expectativa de sobrevida do IBGE.
- **Modalidades**: trabalhador urbano, professor, atividade especial, rural e servidor público (RPPS).
- **Exportação e importação** dos vínculos em Excel e **relatório em PDF** (`fpdf2`) com gráficos.

## Tecnologias

| Camada | Ferramentas |
|---|---|
| Interface | Streamlit |
| Dados | pandas, NumPy, openpyxl |
| Gráficos | Matplotlib |
| PDF | pdfplumber (leitura), fpdf2 (geração) |

## Como executar

```bash
git clone https://github.com/HadassaSM/calculadora-previdenciaria.git
cd calculadora-previdenciaria
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

O app abre em `http://localhost:8501`.

## Aviso

Ferramenta educacional, desenvolvida durante a graduação em Ciências Atuariais (UFMG). Os resultados são estimativas e **não substituem** a análise de um profissional nem a simulação oficial do INSS (Meu INSS).

## Licença

[MIT](LICENSE)
