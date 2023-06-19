const data =require("./output-failed 2.json")[0]
const sorted = data.sort((a,b)=>a.params.loss-b.params.loss)
console.log(sorted.filter((e,i)=>i<5));
console.log(JSON.stringify(sorted.filter((e,i)=>i<5)))