/** Editable, source-checked architecture figures. No models or external assets. */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const dest=path.join(root,'paper_companion/figures');
fs.mkdirSync(dest,{recursive:true});
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const colors={learned:['#f1ebf8','#79559a'],frozen:['#edf5f5','#527f82'],operation:['#fff','#93949f'],data:['#fff','#bbc0cc'],loss:['#fff3ed','#b96b49']};
class Figure{
 constructor(id,title,desc,height){
  this.id=id;this.title=title;this.desc=desc;this.width=1280;this.height=height;this.parts=[];
 }
 text(x,y,lines,size=20,anchor='start',fill='#333344',weight=400){
  const values=Array.isArray(lines)?lines:[lines];
  this.parts.push(`<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" fill="${fill}" font-weight="${weight}">${values.map((line,i)=>`<tspan x="${x}" dy="${i?size*1.35:0}">${esc(line)}</tspan>`).join('')}</text>`);
 }
 box(id,x,y,w,h,title,sub='',kind='operation'){
  const [fill,stroke]=colors[kind];
  this.parts.push(`<g id="${id}" data-kind="${kind}"><title>${esc([title,sub].flat().join('. '))}</title><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${fill}" stroke="${stroke}" stroke-width="1.8"${kind==='frozen'?' stroke-dasharray="6 4"':''}/>`);
  const titles=Array.isArray(title)?title:[title];
  const subs=sub?(Array.isArray(sub)?sub:[sub]):[];
  const total=(titles.length-1)*26+(subs.length?25+(subs.length-1)*22:0);
  const first=y+h/2-total/2+7;
  const titleSize=Math.min(21,(w-22)/(Math.max(...titles.map(t=>t.length))*.61));
  const subSize=subs.length?Math.min(17,(w-20)/(Math.max(...subs.map(t=>t.length))*.56)):17;
  titles.forEach((t,i)=>this.text(x+w/2,first+i*26,t,titleSize.toFixed(2),'middle','#292936',600));
  subs.forEach((t,i)=>this.text(x+w/2,first+(titles.length-1)*26+25+i*22,t,subSize.toFixed(2),'middle','#575867'));
  this.parts.push('</g>');
 }
 arrow(points,{dashed=false,color='#747684'}={}){
  this.parts.push(`<path d="M ${points.map(p=>p.join(' ')).join(' L ')}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"${dashed?' stroke-dasharray="6 5"':''} marker-end="url(#arrow)"/>`);
 }
 join(x,y,label=''){
  this.parts.push(`<circle cx="${x}" cy="${y}" r="18" fill="#fff" stroke="#747684" stroke-width="1.8"/>`);
  this.text(x,y+7,'C',21,'middle');if(label)this.text(x-25,y-12,label,17,'end');
 }
 dot(x,y){this.parts.push(`<circle cx="${x}" cy="${y}" r="3.2" fill="#747684"/>`);}
 panel(y,label,title){this.text(26,y,label,23,'start','#79559a',700);this.text(75,y,title,24,'start','#292936',600);}
 rule(y){this.parts.push(`<path d="M 26 ${y} H 1254" stroke="#e4e3e9" stroke-width="1.2"/>`);}
 legend(){
  this.text(26,31,this.title,24,'start','#292936',600);
  for(const [x,kind,label] of [[618,'learned','Learned / fine-tuned'],[898,'frozen','Frozen'],[1068,'operation','Fixed operation']]){
   const [fill,stroke]=colors[kind];
   this.parts.push(`<rect x="${x}" y="16" width="22" height="18" rx="3" fill="${fill}" stroke="${stroke}" stroke-width="1.6"${kind==='frozen'?' stroke-dasharray="4 3"':''}/>`);
   this.text(x+31,31,label,17);
  }
 }
 svg(){return `<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="${this.height}" viewBox="0 0 1280 ${this.height}" role="img" aria-labelledby="figure-title figure-desc"><title id="figure-title">${esc(this.title)}</title><desc id="figure-desc">${esc(this.desc)}</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#747684"/></marker></defs><style>text{font-family:'DejaVu Sans',sans-serif}</style><rect width="1280" height="${this.height}" fill="#fff"/>${this.parts.join('\n')}</svg>\n`;}
}

const overview=new Figure('generation','Waveform-to-waveform generation','Two separately trained directions. Speech-to-audio uses a frozen Whisper encoder, a learned pooled conditioner, an AudioLDM U-Net, and frozen waveform decoding. Audio-to-speech uses frozen CLAP scene frames both for a global conditioner and an acoustic content LM. Predicted code-duration pairs expand through a frozen centroid codebook; a learned projector supplies frame-aligned channels to the speech diffusion U-Net. Inference uses only the source waveform and random noise.',670);
overview.legend();overview.panel(84,'(a)','Speech → environmental audio');
overview.box('s2a-input',26,142,150,90,['Input','speech'],'waveform','data');
overview.box('whisper',218,142,162,90,'Whisper','large-v3 encoder','frozen');
overview.box('s2a-conditioner',422,142,174,90,'Conditioner',['Transformer + pool','global vector g'],'learned');
overview.box('s2a-unet',650,142,180,90,['AudioLDM','U-Net'],'latent denoising','learned');
overview.box('s2a-decoder',874,142,180,90,['VAE decoder','+ vocoder'],'latent → waveform','frozen');
overview.box('s2a-output',1098,142,156,90,['Generated','audio'],'environmental','data');
for(const [a,b]of [[176,218],[380,422],[596,650],[830,874],[1054,1098]])overview.arrow([[a,187],[b,187]]);
overview.text(741,109,'Gaussian noise z_T',18,'middle');overview.arrow([[740,116],[740,142]]);
overview.text(614,176,'g',18);overview.text(847,176,'z₀',18);
overview.text(26,269,'The source speech is encoded directly; no transcript is generated along this path.',18);
overview.rule(295);overview.panel(336,'(b)','Environmental audio → speech');
overview.box('a2s-input',26,430,150,90,['Input audio'],'environmental','data');
overview.box('clap-frames',218,430,162,90,['CLAP audio','encoder'],'scene frames','frozen');
overview.box('a2s-conditioner',446,365,180,82,['Scene','conditioner'],'global vector g','learned');
overview.box('content-lm',446,510,180,88,'Content LM',['flat or hierarchical','codes + durations'],'learned');
overview.box('expand-lookup',670,510,180,88,['Expand runs','Codebook lookup'],'256 × 512 features','frozen');
overview.box('content-projector',894,510,178,88,['Content','projector'],'8 extra channels','learned');
overview.box('a2s-unet',894,365,178,82,['AudioLDM','U-Net'],'latent denoising','learned');
overview.box('a2s-decoder',1112,365,142,82,['VAE decoder','+ vocoder'],'','frozen');
overview.box('a2s-output',1112,550,142,70,['Generated','speech'],'','data');
overview.arrow([[176,475],[218,475]]);
overview.arrow([[380,475],[409,475],[409,406],[446,406]]);
overview.arrow([[409,475],[409,554],[446,554]]);
overview.dot(409,475);
overview.arrow([[626,406],[894,406]]);overview.text(760,395,'g',18,'middle');
overview.arrow([[626,554],[670,554]]);overview.arrow([[850,554],[894,554]]);
overview.join(983,476,'channel concat');overview.arrow([[983,510],[983,494]]);overview.arrow([[983,458],[983,447]]);
overview.text(1063,482,'latent z_t',18);overview.arrow([[1052,476],[1001,476]]);
overview.arrow([[1072,406],[1112,406]]);overview.text(1092,395,'z₀',18,'middle');
overview.arrow([[1183,447],[1183,550]]);
overview.text(26,647,'Acoustic content tokens are not words. Target speech and text are used during training, not at inference.',18);

const training=new Figure('training','Training the diffusion renderers','Panel a trains the speech-to-audio conditioner and U-Net using target environmental audio encoded by a frozen VAE. Forward noise is predicted with MSE; the source condition is aligned to a frozen CLAP audio embedding of the target audio. Panel b trains the audio-to-speech scene conditioner, content projector and U-Net using continuous oracle features extracted from paired speech by a frozen CTC encoder. Its target speech is also VAE-encoded and noised. The global scene condition is aligned to a frozen CLAP text embedding of the paired transcript. The content LM is trained separately after target preparation; see the content LM figure.',1100);
training.legend();training.panel(84,'(a)','Speech → audio: paired audio supplies the diffusion target');
training.box('train-s2a-speech',26,140,150,78,['Source','speech'],'waveform','data');
training.box('train-whisper',220,140,158,78,'Whisper','encoder','frozen');
training.box('train-s2a-conditioner',422,140,174,78,'Conditioner','global vector g','learned');
training.box('train-s2a-target',26,270,150,78,['Target audio'],'environmental','data');
training.box('train-s2a-vae',220,270,158,78,['VAE encoder'],'target latent z₀','frozen');
training.box('train-s2a-noise',422,270,174,78,'Add noise',['sample ε and t','z₀ → z_t']);
training.box('train-s2a-unet',652,270,174,78,['AudioLDM','U-Net'],'predict ε̂','learned');
training.box('s2a-diffusion-loss',950,270,260,78,'Diffusion loss','MSE(ε̂, ε)','loss');
training.box('target-clap-audio',220,362,158,78,['CLAP audio','encoder'],'target audio e_a','frozen');
training.box('s2a-align-loss',650,362,260,78,'Alignment loss','1 − cos(g, e_a)','loss');
for(const [a,b,y]of [[176,220,179],[378,422,179],[176,220,309],[378,422,309],[596,652,309],[826,950,309],[378,650,401]])training.arrow([[a,y],[b,y]]);
training.arrow([[596,179],[739,179],[739,270]]);training.text(756,242,'g',18);
training.arrow([[195,309],[195,401],[220,401]]);
// Break the vertical alignment edge where the independent noise-prediction edge crosses.
training.parts.push('<path d="M 739 179 H 928 V 299" fill="none" stroke="#747684" stroke-width="2"/><path d="M 928 319 V 401 H 910" fill="none" stroke="#747684" stroke-width="2" marker-end="url(#arrow)"/>');
training.dot(739,179);training.dot(195,309);
training.text(26,475,'Optimize conditioner + U-Net with diffusion MSE + cosine alignment. Whisper, CLAP and VAE stay frozen.',18);
training.rule(500);training.panel(542,'(b)','Audio → speech: train the renderer with oracle speech content');
training.box('train-a2s-input',26,578,150,78,['Source audio'],'environmental','data');
training.box('train-clap-scene',220,578,158,78,['CLAP audio','encoder'],'scene frames','frozen');
training.box('train-a2s-conditioner',422,578,174,78,['Scene','conditioner'],'global vector g','learned');
training.box('train-a2s-speech',26,710,150,78,['Target','speech'],'waveform','data');
training.box('oracle-ctc',220,710,158,78,['CTC encoder'],'oracle features H','frozen');
training.box('train-projector',422,710,174,78,['Content','projector'],'8 content channels','learned');
training.box('train-a2s-unet',950,710,260,90,'AudioLDM U-Net','predict ε̂','learned');
training.box('train-a2s-vae',220,832,158,78,'VAE encoder','target latent z₀','frozen');
training.box('train-a2s-noise',422,832,174,78,'Add noise',['sample ε and t','z₀ → z_t']);
training.box('a2s-diffusion-loss',950,832,260,78,'Diffusion loss','MSE(ε̂, ε)','loss');
training.box('paired-transcript',26,956,150,78,['Speech','transcript'],'paired target','data');
training.box('clap-text-target',220,956,158,78,['CLAP text','encoder'],'transcript e_text','frozen');
training.box('a2s-align-loss',650,956,260,78,'Alignment loss','1 − cos(g, e_text)','loss');
for(const [a,b,y]of [[176,220,617],[378,422,617],[176,220,749],[378,422,749],[378,422,871],[176,220,995],[378,650,995]])training.arrow([[a,y],[b,y]]);
training.arrow([[596,617],[1080,617],[1080,710]]);training.text(877,606,'g',18);
training.arrow([[1080,617],[1232,617],[1232,995],[910,995]]);
training.dot(1080,617);training.dot(195,749);
training.arrow([[195,749],[195,871],[220,871]]);
training.join(830,749,'channel concat');training.arrow([[596,749],[812,749]]);training.arrow([[848,749],[950,749]]);
training.arrow([[596,871],[830,871],[830,767]]);training.text(745,859,'z_t',18);
training.arrow([[1080,800],[1080,832]]);
training.text(26,1072,'Optimize scene conditioner + projector + U-Net. The CTC encoder, VAE and CLAP towers stay frozen.',18);

const content=new Figure('content-lm','Inside the acoustic content LM','Training-only targets come from paired speech encoded by a frozen CTC speech encoder, quantized with a frozen 2048-centroid codebook, and run-length encoded. A six-layer, 512-wide, eight-head causal Transformer decoder cross-attends to projected CLAP scene frames. The input sums previous code, duration and position embeddings. Two heads predict the next code or stop token, and the next duration. Flat sequences predict fine code-duration pairs directly. Hierarchical sequences first predict a strided coarse prefix, then SEP and fine pairs in the same Transformer. Only fine pairs are expanded and rendered. Teacher-forced code cross entropy and fine-duration cross entropy train the content LM independently of the diffusion renderer.',1000);
content.legend();content.panel(84,'(a)','Prepare training targets from paired speech');
content.box('target-speech',26,120,172,90,'Paired speech','mel features','data');
content.box('ctc-target-features',242,120,204,90,'CTC encoder',['256 × 512 features','transcript-trained'],'frozen');
content.box('quantized-targets',490,120,210,90,['Codebook','quantizer'],['K = 2,048 centroids'],'frozen');
content.box('run-length-targets',744,120,220,90,['Run-length','encoding'],['code + duration']);
content.box('lm-targets',1008,120,246,90,'Training targets',['next code / stop token','fine duration'],'data');
for(const [a,b]of [[198,242],[446,490],[700,744],[964,1008]])content.arrow([[a,165],[b,165]],{dashed:true});
content.text(26,245,'Dashed arrows: target preparation only. Fit CTC with transcripts and the codebook with k-means, then freeze both.',18);
content.rule(268);content.panel(310,'(b)','One shared Transformer, two prediction heads');
content.box('scene-memory',26,348,190,74,['CLAP frames'],'source audio','data');
content.box('ambient-projection',250,348,196,74,['Linear projection'],'+ scene positions','learned');
content.box('previous-pair',26,484,190,86,['Previous tokens'],'code + duration','data');
content.box('input-embeddings',250,484,196,86,['Embed inputs'],['code + duration','+ token position'],'learned');
content.parts.push('<g id="decoder-stack" data-kind="learned"><rect x="506" y="448" width="354" height="178" rx="7" fill="#f1ebf8" stroke="#79559a" stroke-width="1.8"/>');
content.text(683,478,'Causal Transformer × 6',21,'middle','#292936',600);
content.text(683,516,'Masked self-attention',20,'middle');
content.text(683,552,'Cross-attention to scene frames',19,'middle');
content.text(683,588,'Feed-forward',20,'middle');content.parts.push('</g>');
content.text(683,646,'width 512 · 8 heads · final LayerNorm',18,'middle');
content.box('code-head',950,448,304,80,'Code head','ID / EOS (+ SEP in hier.)','learned');
content.box('duration-head',950,552,304,80,'Duration head','next fine duration: 1–32','learned');
content.arrow([[216,385],[250,385]]);content.arrow([[446,385],[683,385],[683,448]]);content.text(683,373,'scene memory',18,'middle');
content.arrow([[216,527],[250,527]]);content.arrow([[446,527],[506,527]]);
content.arrow([[860,537],[902,537],[902,488],[950,488]]);content.arrow([[902,537],[902,592],[950,592]]);
content.dot(902,537);
content.text(26,611,'Teacher forcing: use previous target pairs.',18);
content.text(26,640,'Inference: append predictions and repeat.',18);
content.text(26,686,'Training loss: code cross-entropy + fine-duration cross-entropy. The renderer is not optimized in this stage.',18);
content.rule(710);content.panel(752,'(c)','Flat and hierarchical: different sequences, the same mechanism');
content.text(26,805,'Flat',21,'start','#79559a',600);
content.text(235,805,'BOS → (c₁, d₁) → (c₂, d₂) → … → EOS',23);
content.text(26,858,'Hierarchical',21,'start','#79559a',600);
content.text(235,858,'BOS → c₁ → c₁₃ → … → SEP → (c₁, d₁) → (c₂, d₂) → … → EOS',22);
content.text(235,891,'Coarse prefix: every 12th fine token; no durations. Only the fine pairs are rendered.',18);
content.box('duration-expansion',26,927,358,54,'Expand fine durations','');
content.box('centroid-lookup',461,927,358,54,'Frozen centroid lookup','','frozen');
content.box('speech-renderer',896,927,358,54,'Speech diffusion renderer','','frozen');
content.arrow([[384,954],[461,954]]);content.arrow([[819,954],[896,954]]);

const entries=[];
for(const figure of [overview,training,content]){
 const svg=path.join(dest,`${figure.id}.svg`);fs.writeFileSync(svg,figure.svg());
 const files=[];
 for(const format of ['pdf','png']){
  const out=path.join(dest,`${figure.id}.${format}`);
  // A 174 mm vector PDF spans two ICASSP columns; PNG is a 2× preview.
  const args=format==='pdf'?['--format','pdf','--width','174mm','--keep-aspect-ratio','--output',out,svg]:['--format','png','--zoom','2','--output',out,svg];
  const result=spawnSync('rsvg-convert',args,{encoding:'utf8',env:{...process.env,XDG_CACHE_HOME:'/tmp/salm-figure-cache'}});
  if(result.status!==0)throw new Error(`rsvg-convert ${figure.id}: ${result.stderr}`);
 }
 for(const suffix of ['svg','pdf','png']){
  const name=`${figure.id}.${suffix}`,bytes=fs.readFileSync(path.join(dest,name));
  files.push({name,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});
 }
 entries.push({id:figure.id,title:figure.title,width:figure.width,height:figure.height,files});
}
fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify({figures:entries},null,2)+'\n');
console.log(JSON.stringify({figures:entries.map(f=>({id:f.id,formats:f.files.map(x=>x.name)})),output:dest},null,2));
