# Métricas de classificação e diagnóstico de overfitting

Notas consolidadas a partir do estudo do exercício de comparação de modelos de predição
(disciplina TIAS). Cobre os conceitos necessários para justificar, com propriedade, se um
modelo está bem treinado e pronto para produção.

## 1. Treino vs. Teste

Ao treinar um modelo, dividimos os dados em dois blocos:

- **Treino** (`X_train`, `y_train`): usado em `modelo.fit(X_train, y_train)`. O algoritmo vê as
  respostas certas e ajusta seus parâmetros internos (pesos, cortes de árvore, etc.) para
  minimizar erro nesses dados.
- **Teste** (`X_test`, `y_test`): usado em `modelo.predict(X_test)`. O modelo nunca viu esses
  dados durante o treino — é a simulação mais próxima de "dado novo do mundo real" que temos
  antes de liberar o modelo em produção.

Analogia: treino é o material de estudo com gabarito; teste é a prova, com questões que o aluno
nunca viu. Avaliar um modelo só com dados de treino é como corrigir a prova com as mesmas
questões que ele estudou em casa — não prova que ele aprendeu o padrão geral.

Em produção, a função que roda é a mesma (`modelo.predict(...)`), só que sobre dado real —
nunca sobre o `X_test` em si, que é reservado só para a fase de avaliação/decisão.

## 2. Overfitting e underfitting

**Overfitting**: o modelo decorou particularidades do conjunto de treino (inclusive ruído) em
vez de aprender o padrão geral. Sintoma: desempenho ótimo no treino, desempenho bem pior no
teste.

**Underfitting**: o modelo é simples/fraco demais e não aprendeu nem o padrão do treino.
Sintoma: desempenho ruim nos dois conjuntos.

| Cenário | Treino | Teste | Diagnóstico |
|---|---|---|---|
| Overfitting | Muito alto (ex: 1.00) | Bem mais baixo (ex: 0.65) | Decorou, não generaliza |
| Underfitting | Baixo | Baixo, parecido com o treino | Não aprendeu o padrão |
| Bom ajuste | Alto | Próximo do treino | Generaliza — candidato a produção |

**Regra prática usada nos scripts deste repositório**: calcular `Gap = métrica_treino -
métrica_teste` para cada modelo. Gap grande (ex: > 0.15–0.20) é sinal de alerta de overfitting;
gap perto de zero (ou negativo) é sinal de boa generalização.

**Importante**: a matriz de confusão do teste, isolada, **não é evidência de overfitting** — ela
só mostra a qualidade atual das previsões, não a diferença entre treino e teste. Só a
comparação entre os dois conjuntos permite esse diagnóstico.

Exemplo real observado (Problema 1 — compra, ver `../resultados/resultados_problema1_manual.csv`):
Random Forest e Decision Tree chegam a **F1 = 1.0000 no treino** (decoraram 100% dos casos) mas
caem para ~0.63–0.65 no teste — gap de +0.35, overfitting clássico de árvores sem limite de
profundidade.

## 3. Matriz de confusão (base de tudo)

Para classificação binária:

| | Predito Positivo | Predito Negativo |
|---|---|---|
| **Real: Positivo** | Verdadeiro Positivo (VP) | Falso Negativo (FN) |
| **Real: Negativo** | Falso Positivo (FP) | Verdadeiro Negativo (VN) |

"Positivo/Negativo" são apenas os rótulos das classes do `y` (ex: `Risco_Internacao = 1` é
"Positivo"), vindos dos dados reais — não é uma métrica calculada.

## 4. As métricas

**Acurácia** — acertos totais / total de casos. Usa os 4 quadrantes juntos.

```
Acurácia = (VP + VN) / (VP + VN + FP + FN)
```

Confiável quando as classes estão balanceadas. Perigo: em base desbalanceada, um modelo que só
chuta a classe majoritária pode ter acurácia alta e ainda assim ser inútil (nunca detecta a
classe rara).

**Precisão** — de tudo que o modelo *previu* como positivo, quanto estava certo.

```
Precisão = VP / (VP + FP)
```

Importa quando o custo de um Falso Positivo é alto (ex: filtro de spam marcando e-mail
importante).

**Recall (Sensibilidade)** — de tudo que *era* positivo de verdade, quanto o modelo encontrou.

```
Recall = VP / (VP + FN)
```

Importa quando o custo de um Falso Negativo é alto — ex: diagnóstico de saúde (Problema 2,
`Risco_Internacao`): deixar passar um paciente que precisava de internação é mais grave que
gerar uma triagem extra desnecessária.

**F1-Score** — média harmônica entre Precisão e Recall.

```
F1 = 2 * (Precisão * Recall) / (Precisão + Recall)
```

Usa média *harmônica* (não simples) porque ela pune desequilíbrio: um modelo com precisão alta e
recall baixo (ou vice-versa) não consegue "mascarar" a fraqueza tirando uma média alta — a
harmônica puxa o resultado para baixo. Por isso é a métrica preferida em bases desbalanceadas ou
quando se quer checar equilíbrio entre as duas pontas.

## 5. Qual métrica usar para comparar modelos?

1. Olhe a acurácia só para um panorama geral inicial.
2. Olhe a matriz de confusão (treino e teste) para ver onde o modelo erra — gera mais alarmes
   falsos (FP) ou deixa passar mais casos reais (FN)?
3. Use F1-Score como critério de desempate/ranking principal, especialmente se as classes não
   forem perfeitamente balanceadas — que é o caso do Problema 1 (60/40). Mesmo em bases
   balanceadas (Problema 2, 50/50), F1 ainda revela desequilíbrio entre precisão e recall que a
   acurácia sozinha esconde.

## 6. Checklist para defender um modelo como "pronto para produção"

- [ ] Gap entre métrica de treino e teste é pequeno (não há overfitting evidente).
- [ ] F1-Score no teste é consistentemente bom (não só a acurácia).
- [ ] O resultado é reproduzível (`random_state` fixo, ambiente documentado em
      `requirements.txt`).
- [ ] A matriz de confusão do teste não mostra uma classe sendo sistematicamente ignorada.
