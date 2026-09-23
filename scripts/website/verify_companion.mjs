/** Dependency-free application checks using a DOM stand-in, not a browser engine. */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const release=process.argv[2];
assert(release,'Usage: node scripts/website/verify_companion.mjs RELEASE_DIRECTORY');
const script=fs.readFileSync(path.join(release,'app.js'),'utf8');
const seed={window:{}};
vm.runInNewContext(fs.readFileSync(path.join(release,'site-data.js'),'utf8'),seed);
const data=JSON.parse(JSON.stringify(seed.window.PAPER_DATA));
assert.equal(data.samples.length,100);
const decode=s=>s.replaceAll('&amp;','&').replaceAll('&quot;','"').replaceAll('&#39;',"'").replaceAll('&lt;','<').replaceAll('&gt;','>');
function parse(html,doc){
 return [...html.matchAll(/<([a-z][\w-]*)\b([^>]*)>/gi)].map(([,tag,raw])=>{
  const attrs={};for(const [,key,quoted,single,bare] of raw.matchAll(/([\w-]+)(?:="([^"]*)"|='([^']*)'|=([^\s>]+))?/g))attrs[key]=decode(quoted??single??bare??'');
  return new Element(tag,attrs,doc);
 });
}
class Element{
 constructor(tag,attrs,doc){this.tag=tag;this.attrs=attrs;this.doc=doc;this.children=[];this.handlers={};this.dataset={};this.value=attrs.value||'';this.paused=true;this.disabled=false;for(const [k,v]of Object.entries(attrs))if(k.startsWith('data-'))this.dataset[k.slice(5).replace(/-([a-z])/g,(_,x)=>x.toUpperCase())]=v;}
 set innerHTML(value){this.markup=value;this.children=parse(value,this.doc);}
 get innerHTML(){return this.markup||'';}
 addEventListener(type,fn){this.handlers[type]=fn;}
 setAttribute(key,value){this.attrs[key]=value;}
 insertAdjacentHTML(where,html){this.innerHTML=this.innerHTML+html;}
 pause(){this.paused=true;}
 focus(){this.doc.activeElement=this;}
 blur(){this.doc.activeElement=null;}
 closest(){return null;}
}
class Document{
 constructor(html){this.nodes=parse(html,this);this.handlers={};}
 all(){const visit=node=>[node,...node.children.flatMap(visit)];return this.nodes.flatMap(visit);}
 querySelectorAll(selector){return this.all().filter(node=>{
  if(selector.startsWith('#'))return node.attrs.id===selector.slice(1);
  if(selector==='audio')return node.tag==='audio';
  const attrs=[...selector.matchAll(/\[([^=\]]+)(?:="([^"]*)")?\]/g)];
  return attrs.length>0&&attrs.every(([,key,value])=>key in node.attrs&&(value===undefined||node.attrs[key]===value));
 });}
 querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
 addEventListener(type,fn){this.handlers[type]=fn;}
}
function app(page,query='',inputData=data){
 const document=new Document(fs.readFileSync(path.join(release,page),'utf8'));
 const location={search:query};
 const context={document,window:{PAPER_DATA:inputData,addEventListener(){}},URLSearchParams,location,history:{replaceState(_,__,url){location.search=url;}},console};
 vm.runInNewContext(script,context,{filename:'app.js'});
 const get=selector=>{const el=document.querySelector(selector);assert(el,selector);return el;};
 const fire=(selector,type='click',value)=>{const el=get(selector);if(value!==undefined)el.value=value;assert(el.handlers[type],`${selector} handles ${type}`);el.handlers[type]({target:el});};
 return {document,context,get,fire,markup:()=>get('#selected-example').innerHTML};
}
for(const page of ['index.html','dataset.html','geval.html','results.html','model.html']){
 const rendered=app(page);
 assert.equal(rendered.document.querySelectorAll('[data-site-style]').length,0);
}
const gevalPage=app('geval.html');
for(const direction of ['a2s','s2a']){
 const table=gevalPage.get(`#geval-summary-${direction}`).innerHTML;
 assert(table.includes('Ground truth')&&table.includes('Overall'));
 assert.equal((table.match(/<td>/g)||[]).length,(direction==='a2s'?4:3)*6);
}
assert.equal((gevalPage.get('#mismatch-results').innerHTML.match(/class="delta-row"/g)||[]).length,2);
assert.equal((gevalPage.get('#criteria-guide').innerHTML.match(/class="criterion-scope"/g)||[]).length,10);
assert(gevalPage.get('#criteria-guide').innerHTML.includes('Input + output'));
assert(gevalPage.get('#criteria-guide').innerHTML.includes('Output only'));
const demos=gevalPage.get('#criterion-demos').innerHTML;
assert.equal((demos.match(/class="criterion-demo"/g)||[]).length,10);
assert.equal((demos.match(/class="demo-card"/g)||[]).length,20);
assert.equal((demos.match(/<audio /g)||[]).length,40);
assert.equal((demos.match(/HIGHEST IN SUBSET/g)||[]).length,10);
assert.equal((demos.match(/LOWEST IN SUBSET/g)||[]).length,10);
for(const [,src] of demos.matchAll(/<audio [^>]*src="([^"]+)"/g))assert(fs.existsSync(path.join(release,decode(src))),src);
const demoLinks=[...demos.matchAll(/href="examples\.html\?([^"]+)"/g)].map(match=>decode(match[1]));
assert.equal(demoLinks.length,20);
for(const link of demoLinks){const focused=app('examples.html',`?${link}`);assert(focused.markup().includes('id="reviewed-criterion"'));assert.equal((focused.markup().match(/id="reviewed-criterion"/g)||[]).length,1);}
let ui=app('examples.html');
assert.equal(ui.document.querySelectorAll('audio').length,5);
assert.equal((ui.markup().match(/class="rating"/g)||[]).length,15);
assert.equal((ui.markup().match(/class="verdict"/g)||[]).length,15);
assert.equal((ui.markup().match(/class="prompt"/g)||[]).length,15);
assert(!/undefined|NaN/.test(ui.markup()));
assert(!ui.markup().includes('Output transcript')&&!ui.markup().includes('Output audio caption'));
assert(ui.markup().includes('AF-Next (Audio Flamingo 3)'));
assert(ui.markup().includes('Llama 3.3 70B Instruct'));
assert(decode(ui.markup()).includes(`<h2>${data.samples[0].audio_caption}</h2>`));
assert(decode(ui.get('#sample-list').innerHTML).includes(`<strong>${data.samples[0].audio_caption}</strong>`));
const initial=new URLSearchParams(ui.context.location.search).get('sample');
ui.fire('[data-direction="s2a"]');
assert.equal(new URLSearchParams(ui.context.location.search).get('sample'),initial);
assert(decode(ui.markup()).includes(`<h2>${data.samples[0].speech_text}</h2>`));
assert(decode(ui.get('#sample-list').innerHTML).includes(`<strong>${data.samples[0].speech_text}</strong>`));
assert.equal(ui.document.querySelectorAll('audio').length,4);
assert.equal((ui.markup().match(/class="rating"/g)||[]).length,10);
assert(ui.markup().includes('Contextual acoustic fit'));
assert(!ui.markup().includes('Expressive delivery'));
assert(ui.markup().includes('Whisper large-v3'));
ui.fire('[data-view="pair"]');
assert.equal(ui.document.querySelectorAll('audio').length,2);
assert(!ui.markup().includes('class="rating"'));
ui.fire('[data-view="compare"]');
ui.fire('#next-sample');
assert.notEqual(new URLSearchParams(ui.context.location.search).get('sample'),initial);
ui.fire('#previous-sample');
assert.equal(new URLSearchParams(ui.context.location.search).get('sample'),initial);
ui.fire('#sample-search','input',data.samples[70].sample_id);
assert.equal(new URLSearchParams(ui.context.location.search).get('sample'),data.samples[70].sample_id);
assert(ui.get('#previous-sample').disabled&&ui.get('#next-sample').disabled);
ui.fire('#sample-search','input','there-is-no-such-salm-sample');
assert(ui.markup().includes('No matching examples'));
assert.equal(ui.document.querySelectorAll('audio').length,0);
ui.fire('#reset-search');
assert(ui.markup().includes('TEST PAIR'));
ui.fire('#reset-search');
ui.fire('#mobile-sample','change',data.samples[99].sample_id);
assert.equal(new URLSearchParams(ui.context.location.search).get('sample'),data.samples[99].sample_id);
const audios=ui.document.querySelectorAll('audio');audios.forEach(a=>a.paused=false);audios[0].handlers.play();assert(audios.slice(1).every(a=>a.paused));ui.fire('#pause-audio');assert(audios.every(a=>a.paused));
// Exercise every sample, direction and view, checking all output paths and criteria.
let states=0;
for(const sample of data.samples){
 assert(!Object.hasOwn(sample,'outputText'));
 for(const direction of ['a2s','s2a']){
  ui=app('examples.html',`?direction=${direction}&sample=${encodeURIComponent(sample.sample_id)}`);
  assert.equal(new URLSearchParams(ui.context.location.search).get('sample'),sample.sample_id);
  assert.equal((ui.markup().match(/class="rating"/g)||[]).length,direction==='a2s'?15:10);
  assert(!/undefined|NaN/.test(ui.markup()));
  for(const a of ui.document.querySelectorAll('audio'))assert(fs.existsSync(path.join(release,a.attrs.src)),a.attrs.src);
  ui.fire('[data-view="pair"]');assert.equal(ui.document.querySelectorAll('audio').length,2);states+=2;
 }
}
// Hostile reference text is displayed as text, never interpreted as markup.
const hostile=JSON.parse(JSON.stringify(data));hostile.samples[0].audio_caption='<img src=x onerror="alert(1)">';
ui=app('examples.html','',hostile);assert(ui.markup().includes('&lt;img'));assert(!ui.markup().includes('<img'));
// Missing judgments retain five visible unavailable rating slots per output.
const missing=JSON.parse(JSON.stringify(data));missing.samples[0].judgments={};
ui=app('examples.html','',missing);assert.equal((ui.markup().match(/class="rating"/g)||[]).length,15);assert(ui.markup().includes('Feedback unavailable'));
assert(!/\d[\d,]+\s+(?:of|\/)\s+\d[\d,]+/.test(data.benchmarks.a2s.note));
console.log(JSON.stringify({result:'pass',states,samples:data.samples.length,pages:6,checks:['G-Eval criterion tables','matched/mismatched controls','ten high/low criterion demos and deep links','shared-ID direction switch','dataset/compare views','caption, transcript and ID search','previous/next','mobile selector','single-audio playback','all media paths','five score tiles per output','feedback and prompt markup','escaping and missing data'],limitation:'DOM stand-in; no browser layout, keyboard or audio decoding test'},null,2));
