# 🎓 Análise de Risco de Seguro com TensorFlow.js

**Exercício Prático - Pós-Graduação em Inteligência Artificial**

Este é um projeto educacional desenvolvido como exercício prático da pós-graduação em Inteligência Artificial. O projeto demonstra a aplicação de **aprendizado de máquina supervisionado** para classificação multiclasse usando TensorFlow.js.

---

## 📋 Estrutura do Projeto

```
exemplo-00/
├── aula/                          # Aulas teóricas e exemplos base
│   ├── aula.js                   # Exemplo inicial
│   └── aulav2.js                 # Modelo base com 3 features (cor, localização)
│
├── myVersion/                     # 🎯 PRÁTICA - Implementação do aprendizado
│   └── insuranceCompany/
│       └── riskAnalysis.js        # Análise de risco de seguro (36 features)
│
├── README.md                      # Este arquivo
└── package.json                   # Dependências do projeto
```

### Estrutura de Aprendizado
- **`aula/`**: Contém exemplos teóricos e o modelo original simples (`aulav2.js`)
- **`myVersion/`**: Pasta onde os conceitos aprendidos foram praticados e expandidos, implementando uma solução mais complexa e realista de análise de risco de seguro

---

## 🎯 Objetivo

Desenvolver um modelo de rede neural que classifique motoristas em **3 categorias de risco de seguro**:
- 🟢 **BAIXO**: Motoristas com baixo risco (experiência, habilitação definitiva, idade avançada)
- 🟡 **MÉDIO**: Motoristas com risco intermediário
- 🔴 **ALTO**: Motoristas com alto risco (jovens, inexperientes, habilitação provisória)

---

## 🧠 Modelo de Machine Learning

### Arquitetura da Rede Neural
```
Entrada (36 features)
    ↓
Camada Densa (ReLU, 100 neurônios)
    ↓
Camada Saída (Softmax, 3 neurônios)
    ↓
Probabilidades [baixo, médio, alto]
```

### Configuração
- **Input Shape**: 36 features
- **Hidden Units**: 100 neurônios
- **Output Units**: 3 classes
- **Epochs**: 200 iterações
- **Otimizador**: Adam
- **Loss Function**: Categorical Crossentropy
- **Framework**: TensorFlow.js (Node.js)

---

## 📊 Features (Características de Entrada)

Cada motorista é representado por **36 features**:

### 1️⃣ Características Demográficas e Comportamentais (0-8)
| Posição | Feature | Tipo | Valores |
|---------|---------|------|---------|
| 0 | Idade Normalizada | Float | [0.0 - 1.0] |
| 1 | Sexo | Binário | 1=Masculino, 0=Feminino |
| 2-5 | Faixas Etárias | One-Hot | [18-24], [25-30], [30-45], [>46] |
| 6 | Primeiro Carro | Binário | 1=Sim, 0=Não |
| 7 | Habilitação Provisória | Binário | 1=Sim, 0=Não |
| 8 | Habilitação Definitiva | Binário | 1=Sim, 0=Não |

### 2️⃣ Localização Geográfica (9-35)
**One-hot encoding para 27 UFs brasileiras** (em ordem alfabética):
```
AC, AL, AP, AM, BA, CE, DF, ES, GO, MA, MT, MS, MG, PA, PB, PR, PE, PI, RJ, RN, RS, RO, RR, SC, SP, SE, TO
```

---

## 📚 Dados de Treinamento

O modelo foi treinado com **9 amostras** de motoristas com padrões variados:

| Nome | Idade | Sexo | Experiência | Risco |
|------|-------|------|-------------|-------|
| FILIPE | 24 | M | Primeiro carro, prov. | 🔴 ALTO |
| OTAVIO | 18 | M | Primeiro carro, prov. | 🔴 ALTO |
| PEDRO | 22 | M | Primeiro carro, prov. | 🔴 ALTO |
| MARIA | 28 | F | Experiente, def. | 🟡 MÉDIO |
| ANA | 35 | F | Experiente, def. | 🟡 MÉDIO |
| SARA | 38 | F | Experiente, def. | 🟡 MÉDIO |
| JOÃO | 42 | M | Experiente, def. | 🟢 BAIXO |
| CARLOS | 50 | M | Experiente, def. | 🟢 BAIXO |
| LUCIA | 55 | F | Experiente, def. | 🟢 BAIXO |

---

## 🚀 Como Executar

### 1. Instalação de Dependências
```bash
npm install @tensorflow/tfjs-node
```

### 2. Executar o Modelo
```bash
cd myVersion/insuranceCompany
node riskAnalysis.js
```

### 3. Saída Esperada
**Nota**: Esta é a saída esperada baseada na execução real do programa. Os valores de probabilidade podem variar ligeiramente devido à natureza estocástica do treinamento neural.

```
🧠 Iniciando treinamento do modelo com amostras de análise...

⏳ Modelo analisando 9 amostras de motoristas...
✅ Análise das amostras concluída!

📋 Analisando novo motorista:

👤 Nome: FILIPE TESTE
📅 Idade: 18 anos
🏠 Localização: SP
👨 Sexo: Masculino
🚗 Primeiro Carro: Sim
📜 Habilitação: Provisória

🎯 ANÁLISE DE RISCO:

1. 🔴 ALTO: 95.61%
2. 🟢 BAIXO: 3.71%
3. 🟡 MEDIO: 0.68%

⚠️  Risco Principal: ALTO

✅ Análise finalizada!
```

---

## 🔍 Detalhamento Técnico: Normalização de Features

### Age (Idade)
$$\text{idadeNormalizada} = \frac{\text{idade} - 18}{60 - 18}$$

**Exemplo**: 24 anos → (24-18)/42 ≈ 0.143

### Faixas Etárias (One-Hot)
Para 24 anos:
```javascript
[1, 0, 0, 0]  // [18-24, 25-30, 30-45, >46]
```

### Localização (One-Hot)
Para São Paulo (SP, índice 24):
```javascript
[0, 0, 0, ..., 1, 0, 0]  // 27 elementos, SP na posição 24
```

---

## 📁 Arquivos Principais

### `aula/aulav2.js`
- **Propósito**: Modelo base educacional
- **Features**: 7 (idade, 3 cores, 3 localizações)
- **Categoria**: premium, medium, basic
- **Amostras**: 3 pessoas

### `myVersion/insuranceCompany/riskAnalysis.js`
- **Propósito**: Modelo prático de análise de risco de seguro
- **Features**: 36 (idade, sexo, faixas, habilitação, 27 estados)
- **Categoria**: baixo, médio, alto
- **Amostras**: 9 motoristas reais

---

## 💡 Conceitos Aprendidos e Aplicados

✅ **Normalização de dados** - Min-Max scaling para idade  
✅ **One-hot encoding** - Categóricos nominais (sexo, faixas, localização)  
✅ **Redes neurais sequenciais** - Camadas densas e funções de ativação  
✅ **Funções de ativação** - ReLU (camada oculta), Softmax (saída)  
✅ **Compilação e treinamento** - Adam, Categorical crossentropy  
✅ **Previsão e probabilidades** - Classificação multiclasse  
✅ **Gestão de memória** - Disposal de tensores  
✅ **Escalabilidade** - De 3 para 9 amostras, 7 para 36 features  

---

## 🎓 Resumo do Aprendizado

| Aspecto | Aula (aulav2.js) | Prática (myVersion) |
|---------|---|---|
| **Estrutura** | Exemplo simples | Aplicação realista |
| **Features** | 7 | 36 |
| **Amostras** | 3 | 9 |
| **Problema** | Categorização genérica | Análise de risco específica |
| **Domínio** | Teórico | Prático (Seguros) |

---

## 🚀 Melhorias Futuras

- [ ] Aumentar dataset para 50+ amostras reais
- [ ] Validação cruzada (K-Fold cross-validation)
- [ ] Teste em conjunto de dados separado
- [ ] Ajuste de hiperparâmetros (epochs, hidden units)
- [ ] Visualização de métricas (loss, accuracy)
- [ ] Salvar/carregar modelo treinado
- [ ] API REST para previsões online
- [ ] Interface gráfica web

---

## 📖 Tecnologias Utilizadas

- **TensorFlow.js** v3+
- **Node.js** (runtime)
- **JavaScript ES6+**
- **Terminal/CLI**

---

## 👨‍🎓 Autor
Filipe - Pós-Graduação em Inteligência Artificial

---

**Data**: 10 de março de 2026  
**Status**: ✅ Funcional | 🎓 Exercício Educacional | 📊 Testado com 95.61% confiança no risco alto</content>
<filePath">/home/filipe/Área de trabalho/Pos/exemplo-00/README.md