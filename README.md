# python-ia

Repositório pessoal de estudos de Inteligência Artificial / Machine Learning, organizado por
**estágio conceitual do raciocínio**, não por disciplina ou data — para que o conteúdo continue
fazendo sentido conforme novos tópicos forem estudados em disciplinas diferentes.

## Estrutura

```
01-fundamentos/              conceitos base: o que é IA/ML, tipos de aprendizado
02-dados/                    datasets brutos usados nos estudos
03-predicao-classificacao/   comparação de algoritmos de classificação
├── teoria/                  conceitos: overfitting, treino vs teste, métricas
├── implementacao-manual/    scripts scikit-learn
├── automl-pycaret/          reprodução com PyCaret
└── resultados/              csvs e gráficos exportados
04-outros-paradigmas/        regressão, clustering, etc. (futuro)
apresentacoes/                material entregue em aula, por disciplina/data
```

Cada pasta de módulo (ex: `03-predicao-classificacao/`) segue o mesmo padrão interno: teoria →
implementação → resultados, para que o próximo módulo (`04-...`) possa repetir a lógica.

## Ambiente

Cada subpasta com código tem seu próprio `requirements.txt` e seu próprio `venv/` local
(ignorado no git — ver `.gitignore`). Para rodar algo:

```bash
cd 03-predicao-classificacao/implementacao-manual
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python problema1_predicao_compra.py
```
