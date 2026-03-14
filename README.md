# 🎓 Análise de Risco de Seguro com TensorFlow.js

**Exercício Prático - Pós-Graduação em Inteligência Artificial**

Este projeto demonstra a aplicação de **Machine Learning
supervisionado** para análise de risco de seguro utilizando
**TensorFlow.js no Node.js**.

O sistema simula um **motor de decisão de risco semelhante aos
utilizados por seguradoras**, classificando motoristas em categorias de
risco com base em características demográficas e comportamentais.

------------------------------------------------------------------------

# 📋 Estrutura do Projeto

    exemplo-00/
    ├── aula/
    │   ├── aula.js
    │   └── aulav2.js
    │
    ├── myVersion/
    │   └── insuranceCompany/
    │       ├── riskAnalysis.js
    │       ├── generateDataset.js
    │       └── trainingData.json
    │
    ├── README.md
    └── package.json

### Estrutura de Aprendizado

  Pasta                  Descrição
  ---------------------- ----------------------------------------------
  `aula/`                Exemplos didáticos utilizados na aula
  `myVersion/`           Implementação prática desenvolvida
  `riskAnalysis.js`      Script principal da rede neural
  `generateDataset.js`   Gerador automático de dataset
  `trainingData.json`    Dataset persistente utilizado no treinamento

------------------------------------------------------------------------

# 🎯 Objetivo

Criar um modelo capaz de classificar motoristas em **3 categorias de
risco**:

  Categoria      Significado
  -------------- ------------------------------------
  🟢 **BAIXO**   Motoristas experientes
  🟡 **MÉDIO**   Motoristas intermediários
  🔴 **ALTO**    Motoristas jovens ou inexperientes

------------------------------------------------------------------------

# 🧠 Modelo de Machine Learning

### Arquitetura da Rede Neural

    Entrada (36 features)
            ↓
    Camada Oculta (Dense + ReLU)
            ↓
    Camada de Saída (Softmax)
            ↓
    Probabilidades [baixo, médio, alto]

### Configuração

  Parâmetro      Valor
  -------------- --------------------------
  Input Shape    36
  Hidden Units   50
  Output Units   3
  Epochs         150
  Otimizador     Adam
  Loss           Categorical Crossentropy

Framework utilizado:

-   TensorFlow.js
-   Node.js

------------------------------------------------------------------------

# 📊 Features Utilizadas

Cada motorista é representado por **36 características**.

## 1️⃣ Dados Demográficos

  Feature                  Tipo
  ------------------------ ---------
  Idade Normalizada        Float
  Sexo                     Binário
  Faixa 18--24             One-Hot
  Faixa 25--30             One-Hot
  Faixa 31--45             One-Hot
  Faixa \>45               One-Hot
  Primeiro Carro           Binário
  Habilitação Provisória   Binário
  Habilitação Definitiva   Binário

------------------------------------------------------------------------

## 2️⃣ Localização

One-hot encoding para **27 estados brasileiros**:

    AC, AL, AP, AM, BA, CE, DF, ES, GO,
    MA, MT, MS, MG, PA, PB, PR, PE, PI,
    RJ, RN, RS, RO, RR, SC, SP, SE, TO

------------------------------------------------------------------------

# 📚 Dataset Persistente

O projeto utiliza **aprendizado incremental**.

Cada nova análise salva um registro no dataset.

Arquivo:

    trainingData.json

### Estrutura do Dataset

``` json
{
  "dataset": [
    {
      "input": {
        "nome": "Pessoa_1",
        "idade": 22,
        "sexo": "M",
        "primeiroCarro": true,
        "habilitacao": "PROVISORIA",
        "estado": "SP"
      },
      "features": [...],
      "label": [0,0,1],
      "risk": "alto",
      "createdAt": "2026-03-14T22:05:00.000Z"
    }
  ]
}
```

Cada registro contém:

  Campo       Descrição
  ----------- ---------------------------------
  input       dados brutos do motorista
  features    dados transformados para a rede
  label       vetor one-hot da classe
  risk        classificação textual
  createdAt   data da análise

------------------------------------------------------------------------

# 🤖 Aprendizado Contínuo

Fluxo do sistema:

    Usuário fornece dados
            ↓
    Sistema normaliza as features
            ↓
    Rede neural faz previsão
            ↓
    Resultado exibido
            ↓
    Dados salvos no JSON
            ↓
    Próximo treinamento usa mais dados

------------------------------------------------------------------------

# ⚙️ Gerador Automático de Dataset

Script criado para gerar um dataset inicial maior.

Arquivo:

    generateDataset.js

### Gerar Dataset

``` bash
node generateDataset.js
```

Saída esperada:

    trainingData.json criado com 200 registros

------------------------------------------------------------------------

# 🚀 Executar o Projeto

## 1️⃣ Instalar dependências

``` bash
npm install @tensorflow/tfjs-node
```

## 2️⃣ Gerar dataset inicial

``` bash
node generateDataset.js
```

## 3️⃣ Executar análise de risco

``` bash
node riskAnalysis.js
```

------------------------------------------------------------------------

# 🖥️ Exemplo de Execução

    🧠 Treinando modelo...

    📊 Dataset usado: 200 amostras

    Nome: Filipe
    Idade: 23
    Sexo (M/F): M
    Primeiro carro? (S/N): S
    Habilitação provisória? (S/N): S
    Estado (ex SP): SP

    🎯 RESULTADO

    1. 🔴 ALTO 92.45%
    2. 🟡 MEDIO 5.23%
    3. 🟢 BAIXO 2.32%

    ⚠️ Risco principal: ALTO

    📦 Registro salvo no dataset

------------------------------------------------------------------------

# 💡 Conceitos de IA Aplicados

-   Normalização de dados
-   One-Hot Encoding
-   Redes neurais densas
-   Classificação multiclasse
-   Softmax
-   Crossentropy
-   Treinamento supervisionado
-   Dataset incremental
-   Feature engineering

------------------------------------------------------------------------

# 📈 Possíveis Melhorias Futuras

-   Aumentar dataset para **1000+ registros**
-   Implementar **score numérico de risco**
-   Criar **API REST**
-   Salvar e carregar **modelo treinado**
-   Criar **dashboard web**
-   Implementar **validação do modelo**

------------------------------------------------------------------------

# 👨‍🎓 Autor

**Filipe Santana Cordeiro**\
Pós‑Graduação em Inteligência Artificial

------------------------------------------------------------------------

Ano: **2026**\
Status: **Projeto Educacional**
