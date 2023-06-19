const data =require("./output.json")[0]
const best =require("./best.json")
const sorted = data.sort((a,b)=>a.params.loss-b.params.loss)
const best_best=[...sorted.filter((e,i)=>i<5),...best]
const sorted_best = best_best.sort((a,b)=>a.params.loss-b.params.loss)
console.log(JSON.stringify(sorted_best))