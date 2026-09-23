(() => {
  'use strict';
  const data = window.PAPER_DATA;
  if (!data) return;
  const $ = selector => document.querySelector(selector);
  const $$ = selector => [...document.querySelectorAll(selector)];
  const escape = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const score = value => Number.isFinite(value) ? value.toFixed(Math.abs(value) >= 10 ? 2 : 3) : '—';
  const metrics = {
    geval_audio_plausibility: ['G-Eval Audio', '↑', 'Listening-based cross-modal plausibility and quality, on a 1–5 scale.'],
    utmos: ['UTMOS', '↑', 'Prediction of speech naturalness without a reference waveform. This is a model score, not a human listening rating.'],
    dnsmos: ['DNSMOS', '↑', 'Prediction of speech perceptual quality without a reference waveform. This is a model score, not a human listening rating.'],
    fad: ['FAD', '↓', 'Fréchet Audio Distance: distance between reference and generated audio feature distributions. Reported for the corpus.'],
    kad: ['KAD', '↓', 'Kernel Audio Distance: kernel-based distance between reference and generated audio feature distributions. Reported for the corpus.'],
    inception_score: ['Inception Score', '↑', 'PANNs classifier confidence and class diversity across the generated corpus.'],
  };
  const metricOrder = {a2s:['geval_audio_plausibility','utmos','dnsmos'],s2a:['geval_audio_plausibility','fad','kad','inception_score']};
  const glossaryOrder = Object.keys(metrics);
  const criterionNames = {
    scenario_plausibility:'Scenario plausibility', controlled_detail:'Controlled detail',
    perceptual_audio_quality:'Perceptual quality', target_modality_success:'Target modality',
    expressive_delivery_fit:'Expressive delivery', contextual_acoustic_fit:'Contextual acoustic fit'
  };
  const criteriaFor = direction => data.geval.criterionOrder?.[direction] || ['scenario_plausibility','controlled_detail','perceptual_audio_quality','target_modality_success',direction==='a2s'?'expressive_delivery_fit':'contextual_acoustic_fit'];
  const contextCriterion = key => !['target_modality_success','perceptual_audio_quality'].includes(key);
  const criterionShort = {target_modality_success:'MS',perceptual_audio_quality:'PQ',scenario_plausibility:'SP',controlled_detail:'CD',expressive_delivery_fit:'EF',contextual_acoustic_fit:'AF'};
  let tipIndex = 0;
  function metricTip(key) {
    const [name, arrow, definition] = metrics[key], id = `metric-tip-${++tipIndex}`;
    return `<span class="metric-tip"><button type="button" aria-describedby="${id}">${escape(name)} <span aria-hidden="true">${arrow}</span></button><span class="tooltip" role="tooltip" id="${id}">${escape(definition)} ${arrow==='↑'?'Higher':'Lower'} is better.</span></span>`;
  }
  function renderResults(direction) {
    const block=data.benchmarks[direction];
    return `<div class="table-scroll"><table class="results-table"><caption class="sr-only">${direction==='a2s'?'Audio to speech':'Speech to audio'} full-test results</caption><thead><tr><th scope="col">System</th>${metricOrder[direction].map(m=>`<th scope="col">${metricTip(m)}</th>`).join('')}</tr></thead><tbody>${block.systems.map(system=>`<tr><th scope="row">${escape(system.name)}<span class="cell-note">${escape(system.detail)}</span></th>${metricOrder[direction].map(m=>`<td><strong class="table-value">${score(system.scores[m])}</strong></td>`).join('')}</tr>`).join('')}</tbody></table></div><p class="table-note">${escape(block.note)} Means use successfully scored examples.</p>`;
  }
  $$('[data-results]').forEach(el=>el.innerHTML=renderResults(el.dataset.results));
  $$('[data-count]').forEach(el=>el.textContent=data.samples.length || '100');
  if ($('#metric-glossary')) $('#metric-glossary').innerHTML=glossaryOrder.map(key=>{const [name,arrow,definition]=metrics[key];return `<div><h3>${escape(name)} ${arrow}</h3><p>${escape(definition)}</p></div>`;}).join('');
  if ($('#criteria-guide')) $('#criteria-guide').innerHTML=['a2s','s2a'].map(direction=>`<section><h3>${direction==='a2s'?'Audio → Speech':'Speech → Audio'}</h3><ol class="criteria-guide">${criteriaFor(direction).map(key=>`<li><strong>${criterionNames[key]}</strong><span class="criterion-scope">${contextCriterion(key)?'Input + output':'Output only'}</span><p>${escape(data.geval.criteria[key])}</p></li>`).join('')}</ol></section>`).join('');
  function gevalSummary(direction){
    const block=data.geval.summary?.[direction];
    if(!block)return '<p class="missing">Build the release to load criterion-level results.</p>';
    const order=criteriaFor(direction);
    return `<div class="table-scroll"><table class="geval-table"><caption class="sr-only">${direction==='a2s'?'Audio to speech':'Speech to audio'} v6 criterion means</caption><thead><tr><th scope="col">System</th>${order.map(key=>`<th scope="col"><abbr title="${escape(criterionNames[key])}">${criterionShort[key]}</abbr></th>`).join('')}<th scope="col">Overall</th></tr></thead><tbody>${block.systems.map(system=>`<tr><th scope="row">${escape(system.name)}</th>${order.map(key=>`<td>${score(system.criteria[key]?.mean)}</td>`).join('')}<td><strong>${score(system.mean)}</strong></td></tr>`).join('')}</tbody></table></div>`;
  }
  for(const direction of ['a2s','s2a'])if($(`#geval-summary-${direction}`))$(`#geval-summary-${direction}`).innerHTML=gevalSummary(direction);
  function mismatchTable(direction){
    const block=data.geval.matchedMismatch?.[direction];
    if(!block)return '<p class="missing">Build the release to load matched/mismatched scores.</p>';
    const order=criteriaFor(direction).filter(contextCriterion);
    const names={matched:'Matched ground truth',mismatched:'Mismatched ground truth'};
    return `<section class="mismatch-block"><h3>${direction==='a2s'?'Audio → Speech':'Speech → Audio'}</h3><div class="table-scroll"><table class="geval-table"><caption class="sr-only">${direction} matched versus mismatched v6 context criterion means</caption><thead><tr><th scope="col">Pairing</th>${order.map(key=>`<th scope="col"><abbr title="${escape(criterionNames[key])}">${criterionShort[key]}</abbr></th>`).join('')}</tr></thead><tbody>${['matched','mismatched'].map(kind=>`<tr><th scope="row">${names[kind]}</th>${order.map(key=>`<td>${score(block.scores[kind][key])}</td>`).join('')}</tr>`).join('')}<tr class="delta-row"><th scope="row">Matched − mismatched</th>${order.map(key=>`<td>+${score(block.scores.matched[key]-block.scores.mismatched[key])}</td>`).join('')}</tr></tbody></table></div></section>`;
  }
  if($('#mismatch-results'))$('#mismatch-results').innerHTML=['a2s','s2a'].map(mismatchTable).join('');
  function judgmentCandidates(direction,key){
    const files={a2s:{ours:'a2s_model_b',flat:'a2s_model_a',cascade:'a2s_cascaded'},s2a:{ours:'s2a_ours',cascade:'s2a_cascaded'}}[direction];
    return data.samples.flatMap(sample=>Object.entries(files).map(([system,file])=>({sample,system,output:sample.files[file],input:sample.files[direction==='a2s'?'reference_audio':'reference_speech'],item:sample.judgments?.[direction]?.[system]?.criteria?.[key]}))).filter(row=>Number.isFinite(row.item?.score)&&row.output&&row.input);
  }
  function demoCard(direction,key,row,label){
    const link=new URLSearchParams({direction,sample:row.sample.sample_id,view:'compare',system:row.system,criterion:key});
    const system=data.benchmarks[direction].systems.find(item=>item.id===row.system);
    return `<article class="demo-card"><div class="demo-card-heading"><span class="small-label">${label}</span><strong>${row.item.score} / 5</strong></div><p>${escape(system?.name||row.system)} · ${escape(row.sample.sample_id)}</p><label>Input</label><audio controls preload="none" aria-label="Input for ${escape(row.sample.sample_id)}" src="${escape(row.input)}"></audio><label>Generated output</label><audio controls preload="none" aria-label="${escape(system?.name||row.system)} output for ${escape(row.sample.sample_id)}" src="${escape(row.output)}"></audio><details><summary>Judge feedback</summary><p>${escape(row.item.feedback||'Feedback unavailable.')}</p></details><a href="examples.html?${link}">Open all five scores and prompt ↗</a></article>`;
  }
  function demo(direction,key){
    const candidates=judgmentCandidates(direction,key).sort((a,b)=>a.item.score-b.item.score||a.sample.sample_id.localeCompare(b.sample.sample_id)||a.system.localeCompare(b.system));
    if(!candidates.length)return '';
    const lowest=candidates[0],highest=candidates[candidates.length-1];
    return `<details class="criterion-demo"><summary><span><strong>${escape(criterionNames[key])}</strong><small>${direction==='a2s'?'Audio → Speech':'Speech → Audio'} · ${contextCriterion(key)?'Input + output':'Output only'}</small></span><span class="demo-range">${lowest.item.score}–${highest.item.score} / 5 <span aria-hidden="true">⌄</span></span></summary><div class="demo-pair">${demoCard(direction,key,highest,'HIGHEST IN SUBSET')}${demoCard(direction,key,lowest,'LOWEST IN SUBSET')}</div></details>`;
  }
  if($('#criterion-demos'))$('#criterion-demos').innerHTML=['a2s','s2a'].map(direction=>`<section class="demo-direction"><h3>${direction==='a2s'?'Audio → Speech':'Speech → Audio'}</h3>${criteriaFor(direction).map(key=>demo(direction,key)).join('')||'<p class="missing">Build the release to load listening examples.</p>'}</section>`).join('');
  if($('#criterion-demos'))$$('#criterion-demos audio').forEach(a=>a.addEventListener('play',()=>$$('#criterion-demos audio').forEach(other=>{if(other!==a)other.pause();})));
  if (!data.release) $('#main')?.insertAdjacentHTML('afterbegin','<p class="notice"><strong>Source template.</strong> Build the release to load real recordings and verified evaluation results.</p>');
  document.addEventListener('keydown',event=>{if(event.key==='Escape' && document.activeElement?.closest('.metric-tip'))document.activeElement.blur();});
  if (!$('#selected-example')) return;

  const params=new URLSearchParams(location.search);
  let direction=params.get('direction')==='s2a'?'s2a':'a2s';
  let view=params.get('view')==='pair'?'pair':'compare';
  let selectedId=params.get('sample') || data.samples[0]?.sample_id;
  let focusSystem=params.get('system'),focusCriterion=params.get('criterion');
  if(!['ours','flat','cascade'].includes(focusSystem)||!Object.hasOwn(criterionNames,focusCriterion)){focusSystem=null;focusCriterion=null;}
  const search=$('#sample-search'), list=$('#sample-list'), panel=$('#selected-example');
  search.value=params.get('q') || '';
  function filtered(){const q=search.value.trim().toLowerCase();return data.samples.filter(s=>!q||[s.sample_id,s.audio_caption,s.speech_text].join(' ').toLowerCase().includes(q));}
  function pauseAudio(){$$('audio').forEach(a=>a.pause());}
  function syncURL(){const query=new URLSearchParams({direction,view});if(selectedId)query.set('sample',selectedId);if(search.value.trim())query.set('q',search.value.trim());if(view==='compare'&&focusSystem&&focusCriterion){query.set('system',focusSystem);query.set('criterion',focusCriterion);}history.replaceState(null,'',`?${query}`);}
  function clearFocus(){focusSystem=null;focusCriterion=null;}
  function audio(label,src){return src?`<audio controls preload="none" aria-label="${escape(label)}" src="${escape(src)}"><a href="${escape(src)}">Listen to WAV</a></audio>`:'<p class="missing">Recording unavailable in this snapshot.</p>';}
  function ratingRows(judgment,system){
    return `<div class="ratings" aria-label="Five G-Eval criterion ratings">${criteriaFor(direction).map(key=>{
      const item=judgment?.criteria?.[key], valid=Number.isFinite(item?.score);
      return `<details class="rating" data-criterion="${key}"${system===focusSystem&&key===focusCriterion?' open id="reviewed-criterion"':''}><summary><span class="rating-name">${criterionNames[key]}</span><span class="rating-value">${valid?`${item.score}<span> / 5</span>`:'—'}</span></summary><div class="verdict"><p class="small-label">JUDGE FEEDBACK</p><p>${escape(item?.feedback || 'Feedback unavailable in this snapshot.')}</p><details class="prompt"><summary>Exact evaluation prompt <span aria-hidden="true">↗</span></summary>${item?.prompt?`<p class="prompt-meta">${escape(item.promptVersion || data.geval.promptVersion)}<br>Judge: ${escape(item.judge || 'Not recorded')}</p><pre tabindex="0" aria-label="Exact prompt for ${escape(criterionNames[key])}">${escape(item.prompt)}</pre>`:'<p class="missing">The saved prompt is unavailable in this snapshot.</p>'}</details></div></details>`;
    }).join('')}</div>`;
  }
  function stage(label,value){return `<li><strong>${label}</strong><p${value?'':' class="missing"'}>${escape(value || 'Awaiting saved stage output.')}</p></li>`;}
  function cascade(sample){const stages=sample.cascadeStages?.[direction]||{},a2s=direction==='a2s';return `<details class="stage-trace"><summary>Follow the cascade <span aria-hidden="true">↗</span></summary><ol>${stage(a2s?'Audio → caption':'Speech → transcript (ASR)',stages.perception)}${stage(a2s?'Caption → speech text (LLM)':'Transcript → audio caption (LLM)',stages.bridge)}<li><strong>${a2s?'Speech text → speech (CosyVoice3)':'Audio caption → audio (Audio-Omni)'}</strong><p>Final recording is playable above.</p></li></ol><p class="quiet">These are generation stages. Dataset reference text is not a substitute for missing stage outputs; TTS input text is not a verified output transcript.</p></details>`;}
  function output(sample,system){
    const a2s=direction==='a2s',spec=data.benchmarks[direction].systems.find(s=>s.id===system),files=sample.files;
    const src=system==='cascade'?(a2s?files.a2s_cascaded:files.s2a_cascaded):system==='flat'?files.a2s_model_a:(a2s?files.a2s_model_b:files.s2a_ours);
    const judgment=sample.judgments?.[direction]?.[system];
    const label=spec?.name || (system==='cascade'?'Text cascade':'SpeechAudioLM');
    const text=sample.outputText?.[direction]?.[system];
    const values=sample.scores?.[direction]?.[system]||{};
    return `<article class="system-output" data-system="${system}"><header><span class="small-label">${system==='cascade'?'CASCADE':'DIRECT MODEL'}</span><h3>${escape(label)}</h3><p class="system-detail">${escape(spec?.detail)}</p></header>${audio(`${label} output for ${sample.sample_id}`,src)}<div class="judgment-heading"><span>G-Eval Audio <small>v6</small></span><strong>${score(judgment?.score)} <small>mean / 5</small></strong></div>${ratingRows(judgment,system)}<p class="rating-instruction">Open a criterion block to read its feedback.</p>${Object.keys(values).length?`<div class="individual-metrics">${Object.entries(values).map(([key,value])=>`<div>${metricTip(key)}<strong>${score(value)}</strong></div>`).join('')}</div>`:''}${system==='cascade'?cascade(sample):''}<details class="generated-text"><summary>${a2s?'Output transcript':'Output audio caption'}</summary><p${text?'':' class="missing"'}>${escape(text || 'Awaiting a transcript/caption of this generated recording.')}</p></details></article>`;
  }
  function pairRecording(sample,isInput){
    const environmental=(direction==='a2s')===isInput;
    const label=isInput?'Input':'Paired reference';
    const kind=environmental?'Environmental audio':'Speech';
    return `<section class="pair-recording"><div class="recording-label"><span class="small-label">${label.toUpperCase()}</span><span>${kind}</span></div>${audio(`${label} ${kind.toLowerCase()} for ${sample.sample_id}`,environmental?sample.files.reference_audio:sample.files.reference_speech)}<p class="text-label">${environmental?'Dataset caption':'Dataset transcript'}</p><p class="reference-text">${escape(environmental?sample.audio_caption:sample.speech_text)}</p></section>`;
  }
  function renderPanel(sample){
    if(!sample){panel.innerHTML='<div class="empty-state"><h2>No matching examples</h2><p>Try another search or reset the filters.</p></div>';return;}
    panel.innerHTML=`<header class="example-title"><p class="small-label">TEST PAIR <span class="sample-id">${escape(sample.sample_id)}</span></p><h2>${escape(sample.audio_caption)}</h2></header><div class="pair-recordings">${pairRecording(sample,true)}${pairRecording(sample,false)}</div><details class="metadata"><summary>Dataset metadata</summary><dl><div><dt>Speaker gender</dt><dd>${escape(sample.speaker_gender)}</dd></div><div><dt>Speaker emotion</dt><dd>${escape(sample.speaker_emotion)}</dd></div><div class="metadata-rationale"><dt>Pairing rationale</dt><dd>${escape(sample.reasoning || 'Rationale unavailable in this snapshot.')}</dd></div></dl><p>Speaker labels describe the paired reference recording.</p></details>${view==='pair'?'<p class="reference-note">The paired reference is one plausible target. Switch to “Compare outputs” to hear how each system responds to this same input.</p>':`<section class="comparison"><div class="comparison-heading"><h2>Same input. Different systems.</h2><span>${direction==='a2s'?'3':'2'} outputs · 5 criterion scores each</span></div><div class="systems-grid ${direction==='a2s'?'three-systems':'two-systems'}">${(direction==='a2s'?['ours','flat','cascade']:['ours','cascade']).map(s=>output(sample,s)).join('')}</div><p class="reference-note">The reference is one plausible target. Corpus-level metrics (FAD, KAD, Inception Score) are described on the <a href="results.html">Results page</a>.</p></section>`}`;
    $$('audio').forEach(a=>{a.addEventListener('play',()=>$$('audio').forEach(other=>{if(other!==a)other.pause();}));a.addEventListener('error',()=>{if(!a.nextElementSibling?.classList.contains('audio-error'))a.insertAdjacentHTML('afterend','<p class="audio-error missing" role="status">Recording could not be loaded. Check the media folder or reload the page.</p>');});});
  }
  function render(){
    pauseAudio();
    const samples=filtered();
    if(!samples.some(s=>s.sample_id===selectedId))selectedId=samples[0]?.sample_id;
    const selected=samples.find(s=>s.sample_id===selectedId),index=samples.indexOf(selected);
    $('#sample-count').textContent=`${samples.length} / ${data.samples.length} pairs`;
    list.innerHTML=`<label class="mobile-sample-label" for="mobile-sample">Selected example</label><select class="mobile-sample" id="mobile-sample" aria-label="Select an example">${samples.map(s=>`<option value="${escape(s.sample_id)}"${s.sample_id===selectedId?' selected':''}>${escape(s.audio_caption)}</option>`).join('')}</select><div class="sample-buttons">${samples.map(s=>`<button class="sample-choice" data-sample="${escape(s.sample_id)}"${s.sample_id===selectedId?' aria-current="true"':''}><span class="sample-index">${String(s.sample_index+1).padStart(2,'0')}</span><span><strong>${escape(s.audio_caption)}</strong><small>${escape(s.sample_id)}</small></span></button>`).join('')}</div>`;
    $('#sample-position').textContent=selected?`Example ${index+1} of ${samples.length}`:'No example selected';
    $('#previous-sample').disabled=index<=0;$('#next-sample').disabled=index<0||index>=samples.length-1;
    $$('[data-direction]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.direction===direction)));
    $$('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===view)));
    renderPanel(selected);
    $$('[data-sample]').forEach(button=>button.addEventListener('click',()=>{clearFocus();selectedId=button.dataset.sample;render();$('[data-sample][aria-current="true"]')?.focus({preventScroll:true});}));
    $('#mobile-sample').addEventListener('change',event=>{clearFocus();selectedId=event.target.value;render();$('#mobile-sample').focus();});
    syncURL();
  }
  search.addEventListener('input',()=>{clearFocus();render();});
  $('#reset-search').addEventListener('click',()=>{clearFocus();search.value='';render();search.focus();});
  $$('[data-direction]').forEach(b=>b.addEventListener('click',()=>{clearFocus();direction=b.dataset.direction;render();}));
  $$('[data-view]').forEach(b=>b.addEventListener('click',()=>{clearFocus();view=b.dataset.view;render();}));
  for(const [id,delta] of [['previous-sample',-1],['next-sample',1]])$('#'+id).addEventListener('click',()=>{const rows=filtered(),index=rows.findIndex(s=>s.sample_id===selectedId);if(rows[index+delta]){clearFocus();selectedId=rows[index+delta].sample_id;render();}});
  $('#pause-audio').addEventListener('click',pauseAudio);
  window.addEventListener('popstate',()=>{const p=new URLSearchParams(location.search);direction=p.get('direction')==='s2a'?'s2a':'a2s';view=p.get('view')==='pair'?'pair':'compare';selectedId=p.get('sample');focusSystem=p.get('system');focusCriterion=p.get('criterion');search.value=p.get('q')||'';render();});
  render();
})();
