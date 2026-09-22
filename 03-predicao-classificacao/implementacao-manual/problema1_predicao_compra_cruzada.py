from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib
matplotlib.use('Agg')


# ============================================================
# Problema 1 - Predição de Compra com VALIDAÇÃO CRUZADA
#
# Base sintética de classificação binária, 250 linhas.
#
# Features:
#   - Idade
#   - Renda_Anual_K
#   - Score_Credito
#   - Pontuacao_Engajamento
#
# Target:
#   - Compro_Produto
#       0 = não comprou
#       1 = comprou
#
# Estratégia:
#   Stratified K-Fold com 5 folds.
#
# Em cada rodada:
#   - 80% dos dados são usados no treino
#   - 20% dos dados são usados na validação
#
# Ao final das 5 rodadas:
#   - TODOS os 250 registros participaram do treino em 4 folds
#   - TODOS os 250 registros participaram da validação exatamente 1 vez
# ============================================================


# 1. Caminhos
pasta_script = os.path.dirname(os.path.abspath(__file__))
pasta_dados = os.path.join(pasta_script, '..', '..', '02-dados')
pasta_resultados = os.path.join(pasta_script, '..', 'resultados')
os.makedirs(pasta_resultados, exist_ok=True)


# 2. Carregar os dados
df = pd.read_csv(
    os.path.join(pasta_dados, 'dados_predicao_modelos.csv')
)


# 3. Features e variável alvo
features = [
    'Idade',
    'Renda_Anual_K',
    'Score_Credito',
    'Pontuacao_Engajamento',
]

target = 'Compro_Produto'

X = df[features]
y = df[target]


# 4. Configuração da validação cruzada
#
# n_splits=5     -> 5 folds
# shuffle=True   -> embaralha antes de formar os folds
# random_state   -> torna a divisão reproduzível
# StratifiedKFold preserva aproximadamente a proporção das classes.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# 5. Modelos
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
}


# 6. Métricas
metricas = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
}


# 7. Avaliação dos modelos
#
# O StandardScaler fica DENTRO do Pipeline para evitar data leakage.
# Em cada fold ele aprende média/desvio apenas com o treino daquele fold.
resultados = []

print('=' * 82)
print('Problema 1 - Predição de Compra com Validação Cruzada Estratificada')
print('=' * 82)
print(f'Quantidade de registros: {len(df)}')
print('Estratégia: StratifiedKFold com 5 folds')
print('Em cada rodada: aproximadamente 80% treino / 20% validação')
print('=' * 82)


for nome, modelo in modelos.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', modelo),
    ])

    # Executa as 5 rodadas e mede treino e validação em cada uma.
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=metricas,
        return_train_score=True,
        n_jobs=-1,
    )

    acc_train = scores['train_accuracy']
    acc_valid = scores['test_accuracy']
    f1_train = scores['train_f1_macro']
    f1_valid = scores['test_f1_macro']

    acc_train_media = acc_train.mean()
    acc_valid_media = acc_valid.mean()
    f1_train_media = f1_train.mean()
    f1_valid_media = f1_valid.mean()

    acc_valid_std = acc_valid.std()
    f1_valid_std = f1_valid.std()

    gap_acc = acc_train_media - acc_valid_media
    gap_f1 = f1_train_media - f1_valid_media

    # Predição out-of-fold (OOF): cada registro é predito por um modelo
    # que NÃO foi treinado usando aquele próprio registro.
    y_pred_cv = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        n_jobs=-1,
    )

    acc_oof = accuracy_score(y, y_pred_cv)
    f1_oof = f1_score(
        y,
        y_pred_cv,
        average='macro',
        zero_division=0,
    )

    resultados.append({
        'Modelo': nome,
        'Acc_Treino_Media': acc_train_media,
        'Acc_CV_Media': acc_valid_media,
        'Acc_CV_Desvio': acc_valid_std,
        'F1_Treino_Media': f1_train_media,
        'F1_CV_Media': f1_valid_media,
        'F1_CV_Desvio': f1_valid_std,
        'Gap_Acc': gap_acc,
        'Gap_F1': gap_f1,
        'Acc_OOF': acc_oof,
        'F1_OOF': f1_oof,
    })

    print(f'\nModelo: {nome}')

    print('  Accuracy por fold:')
    print(f'    Treino    : {np.round(acc_train, 4)}')
    print(f'    Validação : {np.round(acc_valid, 4)}')

    print('  F1 Macro por fold:')
    print(f'    Treino    : {np.round(f1_train, 4)}')
    print(f'    Validação : {np.round(f1_valid, 4)}')

    print(
        f'  Média Treino    -> Acurácia: {acc_train_media:.4f} | '
        f'F1 (Macro): {f1_train_media:.4f}'
    )

    print(
        f'  Média Validação -> Acurácia: {acc_valid_media:.4f} ± {acc_valid_std:.4f} | '
        f'F1 (Macro): {f1_valid_media:.4f} ± {f1_valid_std:.4f}'
    )

    print(
        f'  Gap médio (Treino - Validação) -> '
        f'Acurácia: {gap_acc:+.4f} | F1: {gap_f1:+.4f}'
    )

    print(
        f'  Resultado OOF nos 250 registros -> '
        f'Acurácia: {acc_oof:.4f} | F1 (Macro): {f1_oof:.4f}'
    )

    print('  Matriz de Confusão (predições Out-of-Fold):')
    print(confusion_matrix(y, y_pred_cv))

    print('  Relatório de Classificação (predições Out-of-Fold):')
    print(classification_report(y, y_pred_cv, zero_division=0))

    print('-' * 82)


# 8. Ranking final por F1 médio da validação cruzada
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values(
    by='F1_CV_Media',
    ascending=False,
).reset_index(drop=True)

print('\nRanking Final dos Modelos')
print('(ordenado pelo F1-Score Macro médio da validação cruzada)')
print()

print(
    f"{'Modelo':<22}"
    f"{'Acc Treino':<12}"
    f"{'Acc CV':<12}"
    f"{'F1 Treino':<12}"
    f"{'F1 CV':<12}"
    f"{'± F1':<10}"
    f"{'Gap F1':<10}"
)

print('-' * 90)

for _, row in df_resultados.iterrows():
    print(
        f"{row['Modelo']:<22}"
        f"{row['Acc_Treino_Media']:<12.4f}"
        f"{row['Acc_CV_Media']:<12.4f}"
        f"{row['F1_Treino_Media']:<12.4f}"
        f"{row['F1_CV_Media']:<12.4f}"
        f"{row['F1_CV_Desvio']:<10.4f}"
        f"{row['Gap_F1']:<+10.4f}"
    )


# 9. Exportar resultados
arquivo_csv = os.path.join(
    pasta_resultados,
    'resultados_problema1_validacao_cruzada.csv',
)

df_resultados.to_csv(
    arquivo_csv,
    index=False,
)

# ------------------------------------------------------------
# 10. Gráfico - F1 Treino vs F1 Validação Cruzada
# ------------------------------------------------------------

plt.figure(figsize=(13, 7))

bar_width = 0.35
index = np.arange(len(df_resultados))

barras_treino = plt.bar(
    index,
    df_resultados['F1_Treino_Media'],
    bar_width,
    label='F1 Treino (média)'
)

barras_cv = plt.bar(
    index + bar_width,
    df_resultados['F1_CV_Media'],
    bar_width,
    label='F1 Validação Cruzada (média)'
)

plt.xlabel('Modelo')
plt.ylabel('F1-Score (Macro)')

plt.title(
    'Problema 1 - F1 Treino vs Validação Cruzada '
    '(5-Fold StratifiedKFold)'
)

plt.xticks(
    index + bar_width / 2,
    df_resultados['Modelo'],
    rotation=45,
    ha='right'
)

plt.legend()

# ============================================================
# Mostrar posição no ranking + F1 CV diretamente nas barras
# ============================================================

ranking_cv = {
    'SVM': ('1º', 0.6785),
    'Naive Bayes': ('2º', 0.6621),
    'Logistic Regression': ('3º', 0.6608),
    'KNN': ('4º', 0.6458),
    'Random Forest': ('5º', 0.6344),
    'Gradient Boosting': ('6º', 0.6314),
    'Decision Tree': ('7º', 0.6176),
}

for i, row in df_resultados.iterrows():

    nome = row['Modelo']

    posicao, valor = ranking_cv[nome]

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

arquivo_grafico = os.path.join(
    pasta_resultados,
    'grafico_problema1_validacao_cruzada.png'
)

plt.savefig(
    arquivo_grafico,
    dpi=150
)

plt.close()

print()
print(
    "Resultados exportados para "
    "'resultados/resultados_problema1_validacao_cruzada.csv'"
)

print(
    "Gráfico salvo em "
    "'resultados/grafico_problema1_validacao_cruzada.png'"
)
