# 03 · Predição / Classificação

Estudo comparativo de algoritmos de classificação binária, a partir do exercício da disciplina
TIAS (Tecnologia Aplicada à Saúde). Objetivo: comparar 7 modelos de classificação em dois
problemas, diagnosticar overfitting via treino vs. teste, e depois reproduzir o mesmo estudo
com AutoML (PyCaret) para comparar as duas abordagens.

## Estrutura

- **`teoria/`** — conceitos usados para interpretar os resultados: treino vs. teste,
  overfitting/underfitting, matriz de confusão, acurácia, precisão, recall, F1-Score. Ver
  [`teoria/metricas-e-overfitting.md`](teoria/metricas-e-overfitting.md).
- **`implementacao-manual/`** — os 7 modelos (Logistic Regression, Decision Tree, Random
  Forest, KNN, Naive Bayes, SVM, Gradient Boosting) implementados diretamente com scikit-learn.
- **`automl-pycaret/`** — reprodução do mesmo estudo usando PyCaret (`compare_models`), para
  comparar contra a implementação manual.
- **`resultados/`** — CSVs e gráficos exportados por cada etapa, sufixados por origem
  (`_manual`, `_pycaret`) para facilitar comparação lado a lado.

## Os dois problemas (dados em [`../02-dados/`](../02-dados/))

1. **Predição de compra** (`dados_predicao_modelos.csv`) — target `Compro_Produto`, ~60/40.
2. **Risco de internação** (`dados_saude_predicao.csv`) — target `Risco_Internacao`, 50/50.

## Achados principais (implementação manual)

- **Problema 1**: nenhum modelo generaliza com folga — melhor gap honesto foi Logistic
  Regression (+0.07), enquanto Random Forest/Decision Tree memorizaram o treino (F1 = 1.00) e
  caíram para ~0.63–0.65 no teste. Gradient Boosting foi o pior no teste apesar de quase decorar
  o treino — overfitting severo.
- **Problema 2**: Naive Bayes e Logistic Regression generalizam bem (gap ≈ 0, F1 teste
  0.76–0.81) — candidatos reais a produção. Árvores/ensembles repetem o padrão de overfitting
  do Problema 1, ainda que menos severo (base mais "limpa").

Detalhes e números completos em [`resultados/resultados_problema1_manual.csv`](resultados/resultados_problema1_manual.csv)
e [`resultados/resultados_problema2_manual.csv`](resultados/resultados_problema2_manual.csv).
