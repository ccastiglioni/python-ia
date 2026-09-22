import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Modelos de classificação
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ============================================================
# Problema 1 - Predição de Compra (Compro_Produto)
# Base sintética de classificação binária, 250 linhas.
# Features: Idade, Renda_Anual_K, Score_Credito, Pontuacao_Engajamento
# Target:   Compro_Produto (0 = não comprou, 1 = comprou) — ~60/40
# Dataset:  ../../02-dados/dados_predicao_modelos.csv
# ============================================================

pasta_script = os.path.dirname(os.path.abspath(__file__))
pasta_dados = os.path.join(pasta_script, '..', '..', '02-dados')
pasta_resultados = os.path.join(pasta_script, '..', 'resultados')
os.makedirs(pasta_resultados, exist_ok=True)

# 1. Carregar os dados
df = pd.read_csv(os.path.join(pasta_dados, 'dados_predicao_modelos.csv'))

# 2. Features e variável alvo
features = ['Idade', 'Renda_Anual_K', 'Score_Credito', 'Pontuacao_Engajamento']
target = 'Compro_Produto'

X = df[features]
y = df[target]

# 3. Divisão treino/teste com estratificação (mesma metodologia do exemplo base da glicose)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Padronizar (necessário para SVM e KNN; não atrapalha os demais)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Modelos de classificação (random_state fixo para resultados reprodutíveis)
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
}

# 6. Avaliar cada modelo em TREINO e TESTE (necessário para diagnosticar overfitting)
resultados = []

print("=" * 70)
print("Problema 1 - Predição de Compra (Compro_Produto)")
print("=" * 70)

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_train = f1_score(y_train, y_pred_train, average='macro', zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

    gap_acc = acc_train - acc_test
    gap_f1 = f1_train - f1_test

    resultados.append((nome, acc_train, acc_test, f1_train, f1_test, gap_f1))

    print(f"\nModelo: {nome}")
    print(f"  Treino -> Acurácia: {acc_train:.4f} | F1 (Macro): {f1_train:.4f}")
    print(f"  Teste  -> Acurácia: {acc_test:.4f} | F1 (Macro): {f1_test:.4f}")
    print(f"  Gap (Treino - Teste) -> Acurácia: {gap_acc:+.4f} | F1: {gap_f1:+.4f}")

    print("  Matriz de Confusão (Treino):")
    print(confusion_matrix(y_train, y_pred_train))
    print("  Matriz de Confusão (Teste):")
    print(confusion_matrix(y_test, y_pred_test))

    print("  Relatório de Classificação (Teste):")
    print(classification_report(y_test, y_pred_test, zero_division=0))
    print("-" * 70)

# 7. Ranking final por F1-Score (Macro) no TESTE
resultados.sort(key=lambda x: x[4], reverse=True)

print("\nRanking Final dos Modelos (ordenado por F1-Teste):")
print(f"{'Modelo':<22}{'Acc Treino':<12}{'Acc Teste':<12}{'F1 Treino':<12}{'F1 Teste':<12}{'Gap F1':<10}")
print("-" * 80)
for nome, acc_tr, acc_te, f1_tr, f1_te, gap in resultados:
    print(f"{nome:<22}{acc_tr:<12.4f}{acc_te:<12.4f}{f1_tr:<12.4f}{f1_te:<12.4f}{gap:<+10.4f}")

# 8. Exportar resultados e gerar gráfico comparativo (treino vs teste, para a apresentação)
# 8. Exportar resultados e gerar gráfico comparativo
df_resultados = pd.DataFrame(
    resultados,
    columns=[
        'Modelo',
        'Acc_Treino',
        'Acc_Teste',
        'F1_Treino',
        'F1_Teste',
        'Gap_F1'
    ]
)

df_resultados.to_csv(
    os.path.join(
        pasta_resultados,
        'resultados_problema1_manual.csv'
    ),
    index=False
)

matplotlib.use('Agg')

plt.figure(figsize=(12, 7))

bar_width = 0.35
index = np.arange(len(df_resultados))

barras_treino = plt.bar(
    index,
    df_resultados['F1_Treino'],
    bar_width,
    label='F1 Treino',
    color='skyblue'
)

barras_teste = plt.bar(
    index + bar_width,
    df_resultados['F1_Teste'],
    bar_width,
    label='F1 Teste',
    color='orange'
)

plt.xlabel('Modelo')
plt.ylabel('F1-Score (Macro)')
plt.title(
    'Problema 1 - F1 Treino vs Teste '
    '(diagnóstico de overfitting)'
)

plt.xticks(
    index + bar_width / 2,
    df_resultados['Modelo'],
    rotation=45,
    ha='right'
)

plt.legend()

# ============================================================
# Mostrar ranking + valor do F1 Teste nas barras
# ============================================================

ranking_teste = {
    'Random Forest': ('1º', 0.6514),
    'Decision Tree': ('2º', 0.6346),
    'KNN': ('3º', 0.6238),
    'SVM': ('4º', 0.6237),
}

for i, row in df_resultados.iterrows():

    nome = row['Modelo']

    if nome in ranking_teste:

        posicao, valor = ranking_teste[nome]

        plt.text(
            i + bar_width,
            valor + 0.015,
            f'{posicao}\n{valor:.4f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

plt.ylim(0, 1.12)

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.7
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        pasta_resultados,
        'grafico_problema_1_manual.png'
    ),
    dpi=150
)

plt.close()

print(
    "\nResultados exportados para "
    "'resultados/resultados_problema1_manual.csv'"
)

print(
    "Gráfico salvo em "
    "'resultados/grafico_problema1_manual.png'"
)
