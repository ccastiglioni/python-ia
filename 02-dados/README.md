# 02 · Dados

Datasets brutos usados nos estudos. Nenhuma transformação aqui — limpeza/engenharia de
features fica no código de cada etapa (ex: `03-predicao-classificacao/implementacao-manual/`).

## dados_predicao_modelos.csv

Base sintética de classificação binária, 250 linhas.

- **Features**: `Idade` (18–65), `Renda_Anual_K`, `Score_Credito` (300–850),
  `Pontuacao_Engajamento` (1–10).
- **Target**: `Compro_Produto` (0 = não comprou, 1 = comprou). Distribuição ~60/40.
- Gerada a partir de uma combinação logística das features com ruído estatístico (overfitting
  perfeito não deve ocorrer por design).

## dados_saude_predicao.csv

Base sintética de classificação binária, 250 linhas.

- **Features**: `Idade` (25–80), `Pressao_Arterial` (100–170 mmHg), `Colesterol_Total`
  (150–310 mg/dL), `Frequencia_Cardiaca_Max` (60–110 bpm).
- **Target**: `Risco_Internacao` (0 = baixo risco, 1 = alto risco/necessidade de internação).
  Distribuição 50/50.

Fonte original: [tias/3_predicao_previsao_codigos_exemplos](https://github.com/alexandrezamberlan/tias/tree/main/3_predicao_previsao_codigos_exemplos)
(disciplina TIAS).
