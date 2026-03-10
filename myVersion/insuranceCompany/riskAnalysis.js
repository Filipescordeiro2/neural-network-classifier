import tf from '@tensorflow/tfjs-node';

/**
 * ============================================
 * CONFIGURAÇÃO DO MODELO
 * ============================================
 */
const CONFIG = {
    inputShape: 36,          // 36 features: idade + sexo + 4 faixas etárias + 3 binários (carro/hab) + 27 estados (26 estados + DF)
    hiddenUnits: 100,        // Aumentei para 100 neurônios (mais complexo que o exemplo original)
    outputUnits: 3,          // 3 categorias de risco
    epochs: 200,             // Aumentei para 200 (dataset maior pode precisar de mais iterações)
    categories: ['baixo', 'medio', 'alto']  // Categorias de risco (alto = alto risco, etc.)
};

/**
 * ============================================
 * FUNÇÃO PARA NORMALIZAR IDADE
 * ============================================
 */
function normalizeIdade(idade, min = 18, max = 60) {
    return (idade - min) / (max - min);
}


/**
 * ============================================
 * DADOS DE TREINAMENTO
 * ============================================
 */
const trainingData = {
    pessoas: [
        { nome: 'FILIPE', idade: 24, sexo: 1, faixa18_24: 1, faixa25_30: 0, faixa30_45: 0, faixa46mais: 0, primeiroCarro: 1, habProvisoria: 1, habDefinitiva: 0, localizacao: 'SP' },
        { nome: 'ANA', idade: 35, sexo: 0, faixa18_24: 0, faixa25_30: 0, faixa30_45: 1, faixa46mais: 0, primeiroCarro: 0, habProvisoria: 0, habDefinitiva: 1, localizacao: 'SP' },
        { nome: 'CARLOS', idade: 50, sexo: 1, faixa18_24: 0, faixa25_30: 0, faixa30_45: 0, faixa46mais: 1, primeiroCarro: 1, habProvisoria: 0, habDefinitiva: 1, localizacao: 'SP' },
        { nome: 'OTAVIO', idade: 18, sexo: 1, faixa18_24: 1, faixa25_30: 0, faixa30_45: 0, faixa46mais: 0, primeiroCarro: 1, habProvisoria: 1, habDefinitiva: 0, localizacao: 'SP' },
        { nome: 'MARIA', idade: 28, sexo: 0, faixa18_24: 0, faixa25_30: 1, faixa30_45: 0, faixa46mais: 0, primeiroCarro: 0, habProvisoria: 0, habDefinitiva: 1, localizacao: 'SP' },
        { nome: 'JOÃO', idade: 42, sexo: 1, faixa18_24: 0, faixa25_30: 0, faixa30_45: 1, faixa46mais: 0, primeiroCarro: 1, habProvisoria: 0, habDefinitiva: 1, localizacao: 'SP' },
        { nome: 'LUCIA', idade: 55, sexo: 0, faixa18_24: 0, faixa25_30: 0, faixa30_45: 0, faixa46mais: 1, primeiroCarro: 0, habProvisoria: 0, habDefinitiva: 1, localizacao: 'SP' },
        { nome: 'PEDRO', idade: 22, sexo: 1, faixa18_24: 1, faixa25_30: 0, faixa30_45: 0, faixa46mais: 0, primeiroCarro: 1, habProvisoria: 1, habDefinitiva: 0, localizacao: 'SP' },
        { nome: 'SARA', idade: 38, sexo: 0, faixa18_24: 0, faixa25_30: 0, faixa30_45: 1, faixa46mais: 0, primeiroCarro: 1, habProvisoria: 0, habDefinitiva: 1, localizacao: 'SP' }
    ],
    
    // Features normalizadas: [idadeNorm, sexo, faixa18_24, ..., habDef, locAC, locAL, ..., locTO]
    features: [                                      // Posicoes para estados (27 elementos)
        [normalizeIdade(24), 1, 1, 0, 0, 0, 1, 1, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], // FILIPE (SP=35º)
        [normalizeIdade(35), 0, 0, 0, 1, 0, 0, 0, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], // ANA (RJ=19º)
        [normalizeIdade(50), 1, 0, 0, 0, 1, 1, 0, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],  // CARLOS (MG=13º)
        [normalizeIdade(18), 1, 1, 0, 0, 0, 1, 1, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], // OTAVIO (SP=25º)
        [normalizeIdade(28), 0, 0, 1, 0, 0, 0, 0, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], // MARIA (RS=21º)
        [normalizeIdade(42), 1, 0, 0, 1, 0, 1, 0, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], // JOÃO (CE=6º)
        [normalizeIdade(55), 0, 0, 0, 0, 1, 0, 0, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], // LUCIA (BA=5º)
        [normalizeIdade(22), 1, 1, 0, 0, 0, 1, 1, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], // PEDRO (PR=16º)
        [normalizeIdade(38), 0, 0, 0, 1, 0, 1, 0, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]  // SARA (PE=17º)
    ],
    
    // Labels: [baixo, medio, alto] (one-hot)

    labels: [
        [0, 0, 1], // FILIPE → alto (jovem e inexperiente)
        [0, 1, 0], // ANA → medio
        [1, 0, 0], // CARLOS → baixo (mais velho e experiente)
        [0, 0, 1], // OTAVIO → alto (jovem e inexperiente)
        [0, 1, 0], // MARIA → medio (mulher experiente)
        [1, 0, 0], // JOÃO → baixo (homem experiente)
        [1, 0, 0], // LUCIA → baixo (mais velha e experiente)
        [0, 0, 1], // PEDRO → alto (jovem e inexperiente)
        [0, 1, 0]  // SARA → medio (mulher de meia idade)
    ]
};


/**
 * ============================================
 * CONSTRUIR E TREINAR O MODELO
 * ============================================
 */

async function buildAndTrainModel(inputXs, outputYs) {
    const model = tf.sequential({
        layers: [
            // Camada 1: Input → Hidden
            // Função de ativação ReLU filtra valores negativos
            tf.layers.dense({
                inputShape: [CONFIG.inputShape],
                units: CONFIG.hiddenUnits,
                activation: 'relu'
            }),
            
            // Camada 2: Hidden → Output
            // Softmax normaliza saída em probabilidades (somam 1)
            tf.layers.dense({
                units: CONFIG.outputUnits,
                activation: 'softmax'
            })
        ]
    });

    // Compilar o modelo
    model.compile({
        optimizer: 'adam',                      // Otimizador adaptativo
        loss: 'categoricalCrossentropy',        // Perda para classificação multi-classe
        metrics: ['accuracy']                   // Métrica de desempenho
    });

    // Treinar o modelo
    await model.fit(inputXs, outputYs, {
        epochs: CONFIG.epochs,
        verbose: 0,
        shuffle: true
    });

    return model;
}


/**
 * ============================================
 * MAPA DE ESTADOS BRASILEIROS
 * ============================================
 */
const ESTADOS = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'];

/**
 * ============================================
 * FUNÇÃO PARA NORMALIZAR PESSOA (Adaptada)
 * ============================================
 */
function normalizarPessoa(pessoa) {
    // Normalizar idade: (valor - min) / (max - min)
    // Min=18, Max=60 (conforme CONFIG)
    const idadeNormalizada = normalizeIdade(pessoa.idade);

    // Sexo: 1 = masculino, 0 = feminino (já vem assim no input)
    const sexo = pessoa.sexo;

    // One-hot encoding para faixas etárias: [18-24, 25-30, 30-45, >46]
    const faixasOneHot = [
        pessoa.faixa18_24 || 0,
        pessoa.faixa25_30 || 0,
        pessoa.faixa30_45 || 0,
        pessoa.faixa46mais || 0
    ];

    // Binários: [primeiroCarro, habProvisoria, habDefinitiva]
    const binarios = [
        pessoa.primeiroCarro || 0,
        pessoa.habProvisoria || 0,
        pessoa.habDefinitiva || 0
    ];

    // One-hot encoding para localização: 27 estados (AC, AL, ..., TO)
    const localizacaoOneHot = ESTADOS.map(uf => (pessoa.localizacao === uf ? 1 : 0));

    return [idadeNormalizada, sexo, ...faixasOneHot, ...binarios, ...localizacaoOneHot];
}

/**
 * ============================================
 * FAZER PREVISÃO
 * ============================================
 */
async function predict(model, pessoaFeatures) {
    // Converter array JavaScript em tensor
    const tfInput = tf.tensor2d(pessoaFeatures);
    
    // Obter predição (vetor de 3 probabilidades)
    const pred = model.predict(tfInput);
    const predArray = await pred.array();
    
    // Converter em objeto com probabilidade e categoria
    return predArray[0].map((prob, index) => ({
        categoria: CONFIG.categories[index],
        probabilidade: prob,
        percentual: (prob * 100).toFixed(2)
    }));
}

/**
 * ============================================
 * TREINAR E ANALISAR RISCO DE NOVA PESSOA
 * ============================================
 */
async function trainAndSayRisk() {
    console.log(`🧠 Iniciando treinamento do modelo com amostras de análise...\n`);

    // 1. Preparar dados de treinamento (análise)
    const inputXs = tf.tensor2d(trainingData.features);
    const outputYs = tf.tensor2d(trainingData.labels);

    // 2. Treinar o modelo com as amostras
    console.log(`⏳ Modelo analisando ${trainingData.pessoas.length} amostras de motoristas...`);
    const model = await buildAndTrainModel(inputXs, outputYs);
    console.log(`✅ Análise das amostras concluída!\n`);

    // 3. Fazer previsão para uma nova pessoa inserida como input
    console.log(`📋 Analisando novo motorista:\n`);
    
    const novaPessoa = {
        nome: 'FILIPE TESTE',
        idade: 50,
        sexo: 1,
        faixa18_24: 0,
        faixa25_30: 0,
        faixa30_45: 0,
        faixa46mais: 1,
        primeiroCarro: 0,
        habProvisoria: 0,
        habDefinitiva: 1,
        localizacao: 'SP'
    };

    console.log(`👤 Nome: ${novaPessoa.nome}`);
    console.log(`📅 Idade: ${novaPessoa.idade} anos`);
    console.log(`🏠 Localização: ${novaPessoa.localizacao}`);
    console.log(`👨 Sexo: ${novaPessoa.sexo === 1 ? 'Masculino' : 'Feminino'}`);
    console.log(`🚗 Primeiro Carro: ${novaPessoa.primeiroCarro === 1 ? 'Sim' : 'Não'}`);
    console.log(`📜 Habilitação: ${novaPessoa.habDefinitiva === 1 ? 'Definitiva' : 'Provisória'}\n`);

    // Normalizar dados da nova pessoa
    const pessoaNormalizada = normalizarPessoa(novaPessoa);
    
    // Fazer previsão
    const predictions = await predict(model, [pessoaNormalizada]);

    // Ordenar por probabilidade
    const resultadosOrdenados = predictions
        .sort((a, b) => b.probabilidade - a.probabilidade);

    // Exibir resultado
    console.log(`🎯 ANÁLISE DE RISCO:\n`);
    resultadosOrdenados.forEach((p, index) => {
        const riskLevel = p.categoria === 'alto' ? '🔴' : p.categoria === 'medio' ? '🟡' : '🟢';
        console.log(`${index + 1}. ${riskLevel} ${p.categoria.toUpperCase()}: ${p.percentual}%`);
    });

    const mainRisk = resultadosOrdenados[0];
    console.log(`\n⚠️  Risco Principal: ${mainRisk.categoria.toUpperCase()}`);

    // Limpeza de tensores
    inputXs.dispose();
    outputYs.dispose();

    console.log(`\n✅ Análise finalizada!`);
}

/**
 * ============================================
 * EXECUTAR PROGRAMA PRINCIPAL
 * ============================================
 */
async function main() {
    try {
        await trainAndSayRisk();
    } catch (error) {
        console.error('❌ Erro durante a execução:', error);
    }
}

// Executar programa
main().catch(console.error);