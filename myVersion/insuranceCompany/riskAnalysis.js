import tf from '@tensorflow/tfjs-node';
import readline from 'readline';
import fs from 'fs';

/**
 * ============================================
 * CONFIGURAÇÃO DO MODELO
 * ============================================
 */

const CONFIG = {
    inputShape: 36,
    hiddenUnits: 50,
    outputUnits: 3,
    epochs: 150,
    categories: ['baixo','medio','alto']
};

/**
 * ============================================
 * NORMALIZAÇÃO DE IDADE
 * ============================================
 */

function normalizeIdade(idade,min=18,max=60){
    return (idade-min)/(max-min);
}

/**
 * ============================================
 * ESTADOS BRASILEIROS
 * ============================================
 */

const ESTADOS=[
'AC','AL','AP','AM','BA','CE','DF','ES','GO','MA','MT','MS',
'MG','PA','PB','PR','PE','PI','RJ','RN','RS','RO','RR','SC',
'SP','SE','TO'
];

/**
 * ============================================
 * CARREGAR DATASET JSON
 * ============================================
 */

function loadTrainingData(){

    try{

        const data=fs.readFileSync('./trainingData.json','utf8');
        const json=JSON.parse(data);

        const features=json.dataset.map(d=>d.features);
        const labels=json.dataset.map(d=>d.label);

        return {features,labels};

    }catch{

        return {features:[],labels:[]};

    }

}

/**
 * ============================================
 * SALVAR DATASET JSON
 * ============================================
 */

function saveTrainingData(pessoa,features,label,risk){

    let dataset;

    try{

        const data=fs.readFileSync('./trainingData.json','utf8');
        dataset=JSON.parse(data);

    }catch{

        dataset={dataset:[]};

    }

    const registro={

        input:{
            nome:pessoa.nome,
            idade:pessoa.idade,
            sexo:pessoa.sexo===1?'M':'F',
            primeiroCarro:pessoa.primeiroCarro===1,
            habilitacao:pessoa.habProvisoria===1?'PROVISORIA':'DEFINITIVA',
            estado:pessoa.localizacao
        },

        features:features,

        label:label,

        risk:risk,

        createdAt:new Date().toISOString()

    };

    dataset.dataset.push(registro);

    if(dataset.dataset.length>5000){
        dataset.dataset.shift();
    }

    fs.writeFileSync(
        './trainingData.json',
        JSON.stringify(dataset,null,2)
    );

}

/**
 * ============================================
 * INPUT DO USUÁRIO
 * ============================================
 */

function getUserInput(){

    const rl=readline.createInterface({
        input:process.stdin,
        output:process.stdout
    });

    return new Promise(resolve=>{

        const pessoa={};

        rl.question('Nome: ',nome=>{
            pessoa.nome=nome;

            rl.question('Idade: ',idade=>{
                pessoa.idade=parseInt(idade);

                rl.question('Sexo (M/F): ',sexo=>{
                    pessoa.sexo=sexo.toUpperCase()==='M'?1:0;

                    rl.question('Primeiro carro? (S/N): ',pc=>{
                        pessoa.primeiroCarro=pc.toUpperCase()==='S'?1:0;

                        rl.question('Habilitação provisória? (S/N): ',hab=>{

                            const provisoria=hab.toUpperCase()==='S';

                            pessoa.habProvisoria=provisoria?1:0;
                            pessoa.habDefinitiva=provisoria?0:1;

                            rl.question('Estado (ex SP): ',uf=>{

                                pessoa.localizacao=uf.toUpperCase();

                                rl.close();
                                resolve(pessoa);

                            });

                        });

                    });

                });

            });

        });

    });

}

/**
 * ============================================
 * NORMALIZAR PESSOA
 * ============================================
 */

function normalizarPessoa(pessoa){

    const idadeNorm=normalizeIdade(pessoa.idade);

    const faixa18_24=pessoa.idade>=18&&pessoa.idade<=24?1:0;
    const faixa25_30=pessoa.idade>=25&&pessoa.idade<=30?1:0;
    const faixa30_45=pessoa.idade>=30&&pessoa.idade<=45?1:0;
    const faixa46mais=pessoa.idade>45?1:0;

    const binarios=[
        pessoa.primeiroCarro||0,
        pessoa.habProvisoria||0,
        pessoa.habDefinitiva||0
    ];

    const loc=ESTADOS.map(uf=>pessoa.localizacao===uf?1:0);

    return[
        idadeNorm,
        pessoa.sexo,
        faixa18_24,
        faixa25_30,
        faixa30_45,
        faixa46mais,
        ...binarios,
        ...loc
    ];

}

/**
 * ============================================
 * DADOS INICIAIS DE TREINAMENTO
 * ============================================
 */

const trainingData={

features:[

[normalizeIdade(24),1,1,0,0,0,1,1,0,...new Array(27).fill(0)],
[normalizeIdade(35),0,0,0,1,0,0,0,1,...new Array(27).fill(0)],
[normalizeIdade(50),1,0,0,0,1,1,0,1,...new Array(27).fill(0)]

],

labels:[

[0,0,1],
[0,1,0],
[1,0,0]

]

};

/**
 * ============================================
 * CRIAR E TREINAR MODELO
 * ============================================
 */

async function buildAndTrainModel(xs,ys){

    const model=tf.sequential({

        layers:[

            tf.layers.dense({
                inputShape:[CONFIG.inputShape],
                units:CONFIG.hiddenUnits,
                activation:'relu'
            }),

            tf.layers.dense({
                units:CONFIG.outputUnits,
                activation:'softmax'
            })

        ]

    });

    model.compile({

        optimizer:'adam',
        loss:'categoricalCrossentropy',
        metrics:['accuracy']

    });

    await model.fit(xs,ys,{
        epochs:CONFIG.epochs,
        shuffle:true,
        verbose:0
    });

    return model;

}

/**
 * ============================================
 * PREVISÃO
 * ============================================
 */

async function predict(model,features){

    const tensor=tf.tensor2d(features);

    const pred=model.predict(tensor);

    const arr=await pred.array();

    return arr[0].map((p,i)=>({

        categoria:CONFIG.categories[i],
        probabilidade:p,
        percentual:(p*100).toFixed(2)

    }));

}

/**
 * ============================================
 * PIPELINE PRINCIPAL
 * ============================================
 */

async function trainAndSayRisk(){

    console.log('\n🧠 Treinando modelo...\n');

    const jsonData=loadTrainingData();

    const features=[
        ...trainingData.features,
        ...jsonData.features
    ];

    const labels=[
        ...trainingData.labels,
        ...jsonData.labels
    ];

    console.log(`📊 Dataset usado: ${features.length} amostras\n`);

    const xs=tf.tensor2d(features);
    const ys=tf.tensor2d(labels);

    const model=await buildAndTrainModel(xs,ys);

    const pessoa=await getUserInput();

    const pessoaNorm=normalizarPessoa(pessoa);

    const predictions=await predict(model,[pessoaNorm]);

    const ordenado=predictions.sort((a,b)=>b.probabilidade-a.probabilidade);

    console.log('\n🎯 RESULTADO\n');

    ordenado.forEach((p,i)=>{

        const icon=p.categoria==='alto'?'🔴':p.categoria==='medio'?'🟡':'🟢';

        console.log(`${i+1}. ${icon} ${p.categoria.toUpperCase()} ${p.percentual}%`);

    });

    const principal=ordenado[0];

    console.log(`\n⚠️ Risco principal: ${principal.categoria.toUpperCase()}`);

    const label=CONFIG.categories.map(c=>c===principal.categoria?1:0);

    saveTrainingData(
        pessoa,
        pessoaNorm,
        label,
        principal.categoria
    );

    xs.dispose();
    ys.dispose();

}

/**
 * ============================================
 * EXECUÇÃO
 * ============================================
 */

async function main(){

    try{

        await trainAndSayRisk();

    }catch(e){

        console.error(e);

    }

}

main();
