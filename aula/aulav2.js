import tf from '@tensorflow/tfjs-node';

/**
 * ============================================
 * CONFIGURAÇÃO DO MODELO
 * ============================================
 */
const CONFIG = {
    inputShape: 7,          // idade normalizada + 3 cores + 3 localizações
    hiddenUnits: 80,        // neurônios da camada oculta
    outputUnits: 3,         // 3 categorias: premium, medium, basic
    epochs: 100,
    categories: ['premium', 'medium', 'basic']
};

/**
 * ============================================
 * DADOS DE TREINAMENTO
 * ============================================
 */
const trainingData = {
    pessoas: [
        { nome: 'Erick', idade: 30, cor: 'azul', localizacao: 'São Paulo' },
        { nome: 'Ana', idade: 25, cor: 'vermelho', localizacao: 'Rio' },
        { nome: 'Carlos', idade: 40, cor: 'verde', localizacao: 'Curitiba' }
    ],
    
    // Features já normalizadas: [idadeNormalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
    features: [
        [0.33, 1, 0, 0, 1, 0, 0], // Erick
        [0, 0, 1, 0, 0, 1, 0],    // Ana
        [1, 0, 0, 1, 0, 0, 1]     // Carlos
    ],
    
    // Labels em one-hot encoding: [premium, medium, basic]
    labels: [
        [1, 0, 0], // Erick → premium
        [0, 1, 0], // Ana → medium
        [0, 0, 1]  // Carlos → basic
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
 * NORMALIZAR DADOS DE NOVA PESSOA
 * ============================================
 */
function normalizarPessoa(pessoa) {
    // Normalizar idade: (valor - min) / (max - min)
    // No treinamento: idade_min=25, idade_max=40
    const idadeMin = 25;
    const idadeMax = 40;
    const idadeNormalizada = (pessoa.idade - idadeMin) / (idadeMax - idadeMin);

    // One-hot encoding para cor: [azul, vermelho, verde]
    const coresOneHot = [
        pessoa.cor === 'azul' ? 1 : 0,
        pessoa.cor === 'vermelho' ? 1 : 0,
        pessoa.cor === 'verde' ? 1 : 0
    ];

    // One-hot encoding para localização: [São Paulo, Rio, Curitiba]
    const localizacaoOneHot = [
        pessoa.localizacao === 'São Paulo' ? 1 : 0,
        pessoa.localizacao === 'Rio' ? 1 : 0,
        pessoa.localizacao === 'Curitiba' ? 1 : 0
    ];

    return [idadeNormalizada, ...coresOneHot, ...localizacaoOneHot];
}

/**
 * ============================================
 * TREINAR MÚLTIPLAS VEZES E FAZER PREVISÃO
 * ============================================
 */
async function trainMultipleTimesAndPredict(numTreinamentos) {
    console.log(`🧠 Iniciando ${numTreinamentos} ciclos de treinamento...\n`);

    // 1. Preparar dados (reutilizados em cada treino)
    const inputXs = tf.tensor2d(trainingData.features);
    const outputYs = tf.tensor2d(trainingData.labels);

    let model = null;

    // 2. Treinar múltiplas vezes
    for (let i = 1; i <= numTreinamentos; i++) {
        console.log(`⏳ Ciclo ${i}/${numTreinamentos}...`);
        model = await buildAndTrainModel(inputXs, outputYs);
        console.log(`✅ Ciclo ${i} concluído\n`);
    }

    console.log(`🎉 Modelo treinado ${numTreinamentos} vezes com sucesso!\n`);

    // 3. Fazer previsão para nova pessoa
    const novaPessoa = {
        nome: 'Zé',
        idade: 25,
        cor: 'vermelho',
        localizacao: 'Rio'
    };

    console.log(`🔍 Predição para: ${novaPessoa.nome} (${novaPessoa.idade} anos, ${novaPessoa.cor}, ${novaPessoa.localizacao})\n`);

    // 4. Normalizar dados da nova pessoa
    const pessoaNormalizada = normalizarPessoa(novaPessoa);
    console.log(`Features normalizadas: [${pessoaNormalizada.join(', ')}]\n`);

    // 5. Obter predições
    const predictions = await predict(model, [pessoaNormalizada]);

    // 6. Exibir resultados ordenados por probabilidade
    const resultadosOrdenados = predictions
        .sort((a, b) => b.probabilidade - a.probabilidade)
        .map(p => `${p.categoria.toUpperCase()} ${p.percentual}%`)
        .join('\n');

    console.log('📊 Resultados:\n');
    console.log(resultadosOrdenados);

    // Limpeza
    inputXs.dispose();
    outputYs.dispose();
}

/**
 * ============================================
 * EXECUTAR PROGRAMA PRINCIPAL
 * ============================================
 */
async function main() {
    // Número de vezes para treinar o modelo
    const NUM_TREINAMENTOS = 5;

    await trainMultipleTimesAndPredict(NUM_TREINAMENTOS);
}

// Executar programa
main().catch(console.error);
