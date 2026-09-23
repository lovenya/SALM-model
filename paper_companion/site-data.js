/* Template only. The release builder supplies verified results and real recordings. */
window.PAPER_DATA = {
  release: false,
  benchmarks: {
    s2a: {directionName:'speech-to-audio',testCount:1878,note:'Build the release to load measured results.',systems:[
      {id:'ours',name:'SpeechAudioLM',detail:'direct · epoch 70',scores:{}},
      {id:'cascade',name:'Text cascade',detail:'ASR → LLM → Audio-Omni',scores:{}}
    ]},
    a2s: {directionName:'audio-to-speech',testCount:2197,note:'Build the release to load measured results.',systems:[
      {id:'ours',name:'SpeechAudioLM · hierarchical',detail:'K=2048 · epoch 12',scores:{}},
      {id:'flat',name:'SpeechAudioLM · flat',detail:'K=2048 · epoch 11',scores:{}},
      {id:'cascade',name:'Text cascade',detail:'captioner → LLM → CosyVoice3',scores:{}}
    ]}
  },
  geval:{promptVersion:'geval_audio_plausibility_v6',criteria:{
    scenario_plausibility:'Do the input and generated output form a believable shared situation?',
    controlled_detail:'Is the output specific enough without unsupported or conflicting invention?',
    perceptual_audio_quality:'Is the generated waveform clear, coherent, natural, and artifact-free?',
    target_modality_success:'Did generation produce usable speech or environmental audio in the requested modality?',
    expressive_delivery_fit:'For generated speech, does delivery fit the content and scene?',
    contextual_acoustic_fit:'For environmental audio, does the acoustic character fit the situation implied by the speech?'
  }},
  samples:[]
};
