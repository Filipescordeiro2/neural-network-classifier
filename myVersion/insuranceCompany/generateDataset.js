import fs from "fs";

const TOTAL = 200;

const ESTADOS = [
'AC','AL','AP','AM','BA','CE','DF','ES','GO','MA','MT','MS',
'MG','PA','PB','PR','PE','PI','RJ','RN','RS','RO','RR','SC',
'SP','SE','TO'
];

function normalizeIdade(idade){
  return (idade - 18) / (60 - 18);
}

function faixaEtaria(idade){
  return [
    idade >=18 && idade <=24 ? 1 : 0,
    idade >=25 && idade <=30 ? 1 : 0,
    idade >=31 && idade <=45 ? 1 : 0,
    idade >45 ? 1 : 0
  ];
}

function oneHotEstado(estado){
  return ESTADOS.map(e => e === estado ? 1 : 0);
}

function riscoLabel(idade, primeiroCarro, habProvisoria){

  if(idade <= 24 && primeiroCarro && habProvisoria){
    return {label:[0,0,1],risk:"alto"};
  }

  if(idade <= 40){
    return {label:[0,1,0],risk:"medio"};
  }

  return {label:[1,0,0],risk:"baixo"};
}

const dataset = {
  dataset:[]
};

for(let i=0;i<TOTAL;i++){

  const idade = Math.floor(Math.random()*42)+18;

  const sexo = Math.random() > 0.5 ? 1 : 0;

  const primeiroCarro = Math.random() > 0.5 ? 1 : 0;

  const habProvisoria = idade < 25 ? 1 : 0;

  const habDefinitiva = habProvisoria ? 0 : 1;

  const estado = ESTADOS[Math.floor(Math.random()*ESTADOS.length)];

  const idadeNorm = normalizeIdade(idade);

  const faixas = faixaEtaria(idade);

  const estadoOneHot = oneHotEstado(estado);

  const features = [
    idadeNorm,
    sexo,
    ...faixas,
    primeiroCarro,
    habProvisoria,
    habDefinitiva,
    ...estadoOneHot
  ];

  const risco = riscoLabel(idade,primeiroCarro,habProvisoria);

  const registro = {

    input:{
      nome:`Pessoa_${i+1}`,
      idade:idade,
      sexo:sexo===1?'M':'F',
      primeiroCarro:primeiroCarro===1,
      habilitacao:habProvisoria===1?'PROVISORIA':'DEFINITIVA',
      estado:estado
    },

    features:features,

    label:risco.label,

    risk:risco.risk,

    createdAt:new Date().toISOString()

  };

  dataset.dataset.push(registro);

}

fs.writeFileSync(
  "./trainingData.json",
  JSON.stringify(dataset,null,2)
);

console.log("✅ trainingData.json criado com", TOTAL, "registros");
