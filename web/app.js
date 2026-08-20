const $ = (selector) => document.querySelector(selector);
const state = { files: [], sources: [], jobs: [], activeId: null, activeJob: null, polling: null, mediaDuration: null, cuePage: 1 };
const videoExt = /\.(mkv|mp4|mov|avi|webm|m4v|ts|mts|m2ts|mpeg|mpg|wmv|mxf|vob|3gp)$/i;
const audioExt = /\.(wav|flac|mp3|m4a|aac|ogg|opus|wma|aiff|aif)$/i;
const subExt = /\.(srt|ass|ssa|vtt|sub|idx)$/i;
const chunkSize = 16 * 1024 * 1024;

document.addEventListener('DOMContentLoaded', () => {
  bindUI(); loadSystem(); loadJobs();
  state.polling = setInterval(loadJobs, 2500);
});

function bindUI() {
  const drop = $('#dropzone'), input = $('#files');
  $('#chooseFiles').onclick = (event) => { event.stopPropagation(); input.click(); };
  drop.onclick = () => input.click();
  drop.onkeydown = (event) => { if (event.key === 'Enter' || event.key === ' ') input.click(); };
  input.onchange = () => selectFiles([...input.files]);
  for (const name of ['dragenter', 'dragover']) drop.addEventListener(name, event => { event.preventDefault(); drop.classList.add('dragging'); });
  for (const name of ['dragleave', 'drop']) drop.addEventListener(name, event => { event.preventDefault(); drop.classList.remove('dragging'); });
  drop.addEventListener('drop', event => selectFiles([...event.dataTransfer.files]));
  $('#startDub').onclick = uploadAndStart;
  $('#startLocal').onclick = startLocal;
  $('#probeLocal').onclick = async () => { try { await probeLocal(); } catch (error) { toast(error.message); } };
  $('#fullRange').onclick = () => { $('#rangeStart').value = ''; $('#rangeEnd').value = ''; updateRangeCopy(); };
  $('#rangeStart').oninput = updateRangeCopy;
  $('#rangeEnd').oninput = updateRangeCopy;
  $('#newJob').onclick = showCreate;
  $('#backButton').onclick = showCreate;
  $('#mobileJobs').onclick = () => $('#rail').classList.toggle('open');
  $('#pauseButton').onclick = () => control('pause');
  $('#cancelButton').onclick = () => control('cancel');
  $('#resumeButton').onclick = () => control('resume');
  $('#approveButton').onclick = approvePending;
}

async function loadSystem() {
  try {
    const info = await api('/api/system');
    const separationReady = info.separator_ready && info.roformer_ready && info.recovery_ready;
    const intelligenceReady = info.asr_ready && info.asr_escalation_ready && info.aligner_ready
      && info.adapter_ready && info.translation_qc_ready && info.diarization_ready && info.visual_speaker_ready && info.tts_fallback_ready;
    const ready = info.ffmpeg && info.ffprobe && info.cuda && info.model_ready
      && info.whisper_ready && intelligenceReady && separationReady;
    const safety = info.gpu_safety || {};
    const unsafe = ['unsafe', 'canary'].includes(safety.status);
    const label = unsafe ? `${info.gpu} · checking GPU recovery before continuing`
      : safety.status === 'active' ? `${info.gpu} · protected GPU stage running`
      : ready ? `${info.gpu} · ready for a full film` : `${info.gpu} · setup needs attention`;
    $('#systemState').classList.toggle('warning', !ready || unsafe);
    $('#systemState').querySelector('span').textContent = label;
  } catch { $('#systemState').querySelector('span').textContent = 'Local service unavailable'; }
}

function selectFiles(files) {
  const accepted = files.filter(file => videoExt.test(file.name) || audioExt.test(file.name) || subExt.test(file.name));
  const videos = accepted.filter(file => videoExt.test(file.name) || audioExt.test(file.name) || file.type.startsWith('video/') || file.type.startsWith('audio/'));
  if (!videos.length) return toast('Choose at least one video or audio programme.');
  state.files = accepted;
  state.sources = videos;
  const video = videos[0], extras = accepted.filter(file => !videos.includes(file));
  $('#selection').classList.remove('hidden');
  $('#selection').innerHTML = `<div class="file-icon">▶</div><div><strong>${videos.length === 1 ? escapeHtml(video.name) : `${videos.length} media files queued as separate jobs`}</strong><span>${formatBytes(videos.reduce((sum,item)=>sum+item.size,0))} · ${extras.length ? extras.map(x => escapeHtml(x.name)).join(', ') : 'Embedded subtitles will be checked'}</span></div><button aria-label="Clear selection">×</button>`;
  $('#selection button').onclick = () => { state.files = []; state.sources=[]; $('#selection').classList.add('hidden'); $('#startDub').disabled = true; };
  $('#startDub').disabled = false;
  inspectBrowserDuration(video);
}

function options() {
  const glossary = Object.fromEntries($('#glossary').value.split(/\r?\n/).map(line => line.split(/\s*=\s*/,2)).filter(parts => parts.length === 2 && parts[0] && parts[1]));
  return { source_language: 'auto', target_language: $('#targetLanguage').value || 'English', subtitle_mode: 'auto',
    audio_mode: 'separate', engine: 'indextts', emotion_mode: 'auto',
    workflow_mode: $('#approvalWorkflow').checked ? 'approval' : 'automatic', mastering_preset: 'cinema',
    range_start: parseClock($('#rangeStart').value), range_end: parseClock($('#rangeEnd').value),
    audio_stream_index: $('#audioTrack').value === '' ? null : Number($('#audioTrack').value), glossary,
    subtitle_stream_index: $('#subtitleTrack').value === '' ? null : Number($('#subtitleTrack').value),
    voice_rights_confirmed: $('#voiceRights').checked, allow_same_language: $('#allowSameLanguage').checked };
}

function validateRange() {
  if (!$('#voiceRights').checked) throw new Error('Confirm that you have permission to dub the media and reproduce its voices.');
  const start = parseClock($('#rangeStart').value), end = parseClock($('#rangeEnd').value);
  if ($('#rangeStart').value.trim() && start === null) throw new Error('Enter the start as MM:SS or HH:MM:SS.');
  if ($('#rangeEnd').value.trim() && end === null) throw new Error('Enter the end as MM:SS or HH:MM:SS.');
  if (start !== null && end !== null && end <= start) throw new Error('The section end must be after its start.');
  if (state.mediaDuration && start !== null && start >= state.mediaDuration) throw new Error('The section starts after the film ends.');
  if (state.mediaDuration && end !== null && end > state.mediaDuration + .25) throw new Error('The section end is beyond the film length.');
}

async function uploadAndStart() {
  const button = $('#startDub'); button.disabled = true;
  try {
    validateRange();
    let job;
    for (const source of state.sources) {
      const stem=source.name.replace(/\.[^.]+$/,'').toLowerCase();
      const matching=state.files.filter(file=>subExt.test(file.name) && (state.sources.length===1 || file.name.replace(/\.[^.]+$/,'').toLowerCase()===stem));
      const jobFiles=[source,...matching];
      const specs = jobFiles.map(file => ({ name: file.name, size: file.size, kind: kindFor(file) }));
      job = await api('/api/jobs', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({files: specs, options: options()}) });
      state.activeId = job.id; showJob(job);
      for (let index = 0; index < jobFiles.length; index++) {
        const file = jobFiles[index], upload = job.uploads[index]; let offset = upload.received || 0;
        while (offset < file.size) {
          const body = file.slice(offset, Math.min(file.size, offset + chunkSize));
          const response = await fetch(`/api/jobs/${job.id}/files/${upload.id}`, { method:'PUT', headers:{'Upload-Offset':String(offset)}, body });
          const data = await response.json();
          if (!response.ok && response.status !== 409) throw new Error(data.detail || 'Upload interrupted');
          offset = data.offset;
          const sent = jobFiles.slice(0,index).reduce((sum,f)=>sum+f.size,0)+offset;
          renderUpload(sent/jobFiles.reduce((sum,f)=>sum+f.size,0)*100,file.name);
        }
      }
      job = await api(`/api/jobs/${job.id}/finalize`, {method:'POST'});
    }
    renderJob(job); await loadJobs(); toast(`${state.sources.length} job${state.sources.length===1?'':'s'} queued.`);
  } catch (error) { toast(error.message); button.disabled = false; }
}

async function startLocal() {
  const paths = $('#localPath').value.split(/\r?\n/).map(value=>value.trim()).filter(Boolean);
  if (!paths.length) return toast('Enter one or more full local media paths.');
  try {
    await probeLocal(); validateRange();
    let job;
    for (const path of paths) {
      const jobOptions = {...options()};
      // The visible selector describes the first probed file.  Other files in a
      // batch are independently auto-selected so stream indexes are never
      // accidentally copied across unrelated containers.
      if (paths.length > 1) { jobOptions.audio_stream_index = null; jobOptions.subtitle_stream_index = null; }
      job = await api('/api/jobs/local', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path, options:jobOptions})});
    }
    state.activeId = job.id; showJob(job); await loadJobs(); toast(`${paths.length} local job${paths.length === 1 ? '' : 's'} queued.`);
  } catch (error) { toast(error.message); }
}

async function probeLocal() {
  const path = $('#localPath').value.split(/\r?\n/).map(value=>value.trim()).filter(Boolean)[0];
  if (!path) throw new Error('Enter the full path to a local video.');
  const info = await api('/api/media/probe', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path})});
  state.mediaDuration = Number(info.duration);
  const tracks = info.audio_streams || [], select = $('#audioTrack');
  select.innerHTML = tracks.map(track => `<option value="${track.index}">${escapeHtml(`${track.ordinal + 1}. ${track.title} · ${track.language} · ${track.channels}ch ${track.codec}`)}</option>`).join('');
  $('#audioTrackRow').classList.toggle('hidden', tracks.length < 2);
  const preferred = tracks.find(track => track.default && !/commentary|description/i.test(track.title)) || tracks[0];
  if (preferred) select.value = String(preferred.index);
  const subtitles = (info.subtitle_streams || []).filter(track=>track.text), subtitleSelect = $('#subtitleTrack');
  subtitleSelect.innerHTML = `<option value="">Automatic / ASR</option>` + subtitles.map(track => `<option value="${track.index}">${escapeHtml(`${track.title} · ${track.language} · ${track.codec}${track.forced ? ' · forced' : ''}`)}</option>`).join('');
  $('#subtitleTrackLabel').classList.toggle('hidden', subtitles.length < 2);
  updateRangeCopy(); return info;
}

function inspectBrowserDuration(file) {
  state.mediaDuration = null; updateRangeCopy();
  const video = document.createElement('video'), url = URL.createObjectURL(file);
  video.preload = 'metadata';
  video.onloadedmetadata = () => {
    state.mediaDuration = Number.isFinite(video.duration) ? video.duration : null;
    URL.revokeObjectURL(url); updateRangeCopy();
  };
  video.onerror = () => URL.revokeObjectURL(url);
  video.src = url;
}

function updateRangeCopy() {
  const start = parseClock($('#rangeStart').value), end = parseClock($('#rangeEnd').value);
  if (start !== null || end !== null) {
    const finish = end !== null ? formatCueTime(end) : (state.mediaDuration ? formatCueTime(state.mediaDuration) : 'end');
    $('#sourceDuration').textContent = `Only ${formatCueTime(start || 0)} to ${finish} will be dubbed; character registration still scans the whole film.`;
  } else {
    $('#sourceDuration').textContent = state.mediaDuration ? `${formatCueTime(state.mediaDuration)} detected · complete film selected.` : 'Leave both times blank to dub the complete film.';
  }
}

function kindFor(file) {
  if (/\.idx$/i.test(file.name)) return 'subtitle_index';
  if (subExt.test(file.name)) return 'subtitle';
  if (audioExt.test(file.name) || file.type.startsWith('audio/')) return 'audio';
  return 'video';
}

async function loadJobs() {
  try {
    state.jobs = await api('/api/jobs'); renderJobList();
    if (state.activeId) {
      const summary = state.jobs.find(job => job.id === state.activeId);
      if (summary) {
        const stale = !state.activeJob || state.activeJob.id !== summary.id
          || state.activeJob.cue_revision !== summary.cue_revision
          || state.activeJob.log_revision !== summary.log_revision;
        if (stale) state.activeJob = await api(`/api/jobs/${summary.id}`);
        else state.activeJob = {...state.activeJob, ...summary, cues:state.activeJob.cues, logs:state.activeJob.logs};
        renderJob(state.activeJob);
      }
    }
  } catch { /* The next poll will recover. */ }
}

function renderJobList() {
  const list = $('#jobList');
  if (!state.jobs.length) { list.innerHTML = '<div class="list-empty">No projects yet</div>'; return; }
  list.innerHTML = state.jobs.map(job => { const passed = job.status === 'complete' && !Number(job.qc?.flagged_count || 0) && job.qc?.passed !== false; return `<button class="job-item ${job.id === state.activeId ? 'active':''}" data-id="${job.id}"><span class="job-thumb">${passed ? '✓' : (['complete','needs_review'].includes(job.status) ? '!' : '▶')}</span><span><strong>${escapeHtml(job.filename)}</strong><small>${labelStatus(job)} · ${relativeTime(job.updated_at)}</small></span><i style="--p:${Number(job.progress)||0}"></i></button>`; }).join('');
  list.querySelectorAll('.job-item').forEach(button => button.onclick = async () => {
    state.activeId = button.dataset.id;
    try { state.activeJob = await api(`/api/jobs/${state.activeId}`); showJob(state.activeJob); }
    catch (error) { toast(error.message); }
    $('#rail').classList.remove('open');
  });
}

function showCreate() {
  state.activeId = null; state.activeJob = null; $('#jobView').classList.add('hidden'); $('#createView').classList.remove('hidden'); renderJobList();
}

function showJob(job) {
  state.activeJob = job; $('#createView').classList.add('hidden'); $('#jobView').classList.remove('hidden'); renderJob(job); renderJobList();
}

function renderUpload(percent, filename) {
  const job = { id:state.activeId, filename, status:'uploading', stage:`Uploading ${filename}`, progress:percent, cues:[], logs:['Sending the source in restart-safe 16 MB pieces'] };
  renderJob(job);
}

function renderJob(job) {
  if (!job) return;
  const progress = Math.max(0, Math.min(100, Number(job.progress)||0));
  const complete = ['complete','needs_review'].includes(job.status), failed = job.status === 'error';
  const deliveryFailed = complete && job.qc?.passed === false;
  const cueReview = complete && Number(job.qc?.flagged_count || 0) > 0;
  $('#jobTitle').textContent = job.filename;
  $('#jobStatus').textContent = labelStatus(job).toUpperCase();
  $('#jobStatus').className = `kicker status-${job.status}`;
  const detected = job.detected_language;
  $('#jobStage').textContent = (job.stage || 'Queued') + (detected ? ` · ${detected.language} ${Math.round(Number(detected.confidence||0)*100)}%` : '');
  $('#jobPercent').textContent = `${Math.round(progress)}%`;
  if (job.status === 'processing' && job.processing_started_at) {
    const elapsed = Number(job.active_processing_seconds||0) + (job.active_run_started_at ? Date.now()/1000-Number(job.active_run_started_at) : 0);
    const predicted=Number(job.eta?.predicted_seconds||0);
    $('#etaLabel').textContent = predicted > 0
      ? `About ${formatTime(Math.max(0,predicted-elapsed))} remaining · based on ${job.eta.sample_jobs} local film${job.eta.sample_jobs===1?'':'s'}`
      : `${formatTime(elapsed)} elapsed · first completed film establishes this PC's ETA`;
  } else $('#etaLabel').textContent = job.status === 'queued' ? 'Waiting for the single local GPU worker' : 'Timeline protected';
  $('#progressBar').style.width = `${progress}%`;
  $('#progressRing').style.setProperty('--progress', `${progress * 3.6}deg`);
  $('#jobMeta').textContent = job.media ? `${job.media.video_codec?.toUpperCase() || job.media.media_kind?.toUpperCase() || 'MEDIA'} · ${formatTime(job.media.duration)} · processed locally` : 'Preparing source details';
  document.querySelectorAll('.pipeline-steps span').forEach(step => step.classList.toggle('done', progress >= Number(step.dataset.at)));
  $('#cueSource').textContent = job.cue_source || 'Detecting…';
  $('#lineCount').textContent = (job.cues?.length || job.cue_count) ? (job.cues?.length || job.cue_count).toLocaleString() : '—';
  $('#runtime').textContent = job.media ? formatTime(job.media.duration) : '—';
  $('#dimensions').textContent = job.media?.width ? `${job.media.width} × ${job.media.height}` : '—';
  renderCues(job.cues || [], job.status === 'needs_review' || (complete && job.options?.workflow_mode === 'review'), job.status === 'paused');
  $('#logList').innerHTML = (job.logs || []).map(item => `<p>${escapeHtml(item)}</p>`).join('') || '<p>No notes yet.</p>';
  $('#resultBanner').classList.toggle('hidden', !complete);
  $('#resultBanner').classList.toggle('delivery-failed', deliveryFailed || cueReview);
  $('#errorBanner').classList.toggle('hidden', !failed);
  const awaitingApproval = job.status === 'paused' && job.stage === 'Translation ready for approval';
  const awaitingTrack = job.status === 'awaiting_selection';
  $('#approvalBanner').classList.toggle('hidden', !(awaitingApproval || awaitingTrack));
  $('#approvalTitle').textContent = awaitingTrack ? 'Choose the programme tracks' : 'Transcript and translation ready';
  $('#approvalText').textContent = awaitingTrack ? 'Choose dialogue audio—not commentary or audio description—and the full dialogue subtitle track when one is useful.' : 'Review or edit the lines below. Voice generation will not start until you continue.';
  $('#jobTrackPickers').classList.toggle('hidden', !awaitingTrack);
  $('#approveButton').textContent = awaitingTrack ? 'Start automatic dub →' : 'Continue to voices →';
  if (awaitingTrack) {
    const audio=(job.media_selection?.audio_streams || []), subtitles=(job.media_selection?.subtitle_streams || []);
    $('#jobAudioTrack').innerHTML = audio.map(track=>`<option value="${track.index}">${escapeHtml(`${track.ordinal+1}. ${track.title} · ${track.language} · ${track.channels}ch`)}</option>`).join('');
    const preferredAudio=audio.find(track=>track.default&&!/commentary|description|director|isolated|music only|karaoke/i.test(track.title))
      || audio.find(track=>!/commentary|description|director|isolated|music only|karaoke/i.test(track.title)) || audio[0];
    if(preferredAudio)$('#jobAudioTrack').value=String(preferredAudio.index);
    $('#jobSubtitleTrack').innerHTML = `<option value="">Automatic / ASR</option>`+subtitles.map(track=>`<option value="${track.index}">${escapeHtml(`${track.title} · ${track.language}${track.forced?' · forced':''}`)}</option>`).join('');
    const preferredSubtitle=subtitles.find(track=>!track.forced&&!/commentary|director|signs|songs|trivia/i.test(track.title));
    if(preferredSubtitle)$('#jobSubtitleTrack').value=String(preferredSubtitle.index);
  }
  $('#pauseButton').classList.toggle('hidden', !['processing','queued'].includes(job.status));
  $('#cancelButton').classList.toggle('hidden', !['processing','queued','paused','uploading','awaiting_selection'].includes(job.status));
  if (complete) {
    $('#downloadButton').href = `/api/jobs/${job.id}/download`; $('#qcButton').href = `/api/jobs/${job.id}/qc`;
    $('#srtExport').href = `/api/jobs/${job.id}/export/srt`; $('#csvExport').href = `/api/jobs/${job.id}/export/csv`;
    $('#edlExport').href = `/api/jobs/${job.id}/export/edl`; $('#clipsExport').href = `/api/jobs/${job.id}/export/clips`;
    $('#mixExport').href = `/api/jobs/${job.id}/export/mix`; $('#dialogueExport').href = `/api/jobs/${job.id}/export/dialogue`;
    $('#resultIcon').textContent = (deliveryFailed || cueReview) ? '!' : '✓';
    $('#resultLabel').textContent = deliveryFailed ? 'DUB PRODUCED · DELIVERY QC FAILED' : (cueReview ? 'DUB PRODUCED · LINES NEED REVIEW' : 'ENGLISH DUB READY');
    $('#resultTitle').textContent = (deliveryFailed || cueReview) ? 'Review the flagged evidence before release.' : 'Every automatic check passed.';
    $('#outputSize').textContent = `${formatBytes(job.output_size)} Matroska video · ${job.qc?.flagged_count || 0} flagged line(s) · ${(job.qc?.failures || []).length} delivery issue(s)`;
  }
  if (failed) $('#errorText').textContent = job.error || 'An unknown error occurred.';
}

function renderCues(cues, flaggedOnly=false, editable=false) {
  const list = $('#cueList');
  if (flaggedOnly) cues = cues.filter(cue => cue.needs_review);
  if (!cues.length) { list.innerHTML = '<div class="cue-empty">Lines will appear after dialogue analysis.</div>'; return; }
  const visible = cues.slice(0, state.cuePage * 500);
  list.innerHTML = visible.map(cue => { const voice=cue.speaker_id==null?'Identifying…':(cue.speaker||'Uncertain voice'); return `<div class="cue-row ${cue.needs_review ? 'needs-review':''}"><time>${formatCueTime(cue.start)}<small>${formatCueTime(cue.end)}</small></time><div><strong>${escapeHtml(cue.english || '')}</strong><span>${escapeHtml(cue.needs_review ? `Needs review · ${(cue.review_reasons || []).join('; ')}` : (cue.performance_source || cue.emotion || 'analysis pending'))}</span></div><span class="voice-name"><i>${cue.speaker_id==null?'…':initials(voice)}</i>${escapeHtml(voice)}</span>${editable ? `<div class="cue-edit-actions"><button class="cue-fix" data-cue="${cue.id}" data-text="${escapeHtml(cue.english || '')}">Edit</button><button class="cue-split" data-cue="${cue.id}">Split</button><button class="cue-merge" data-cue="${cue.id}">Merge</button></div>` : (cue.needs_review ? `<div class="cue-edit-actions"><button class="cue-fix" data-cue="${cue.id}" data-text="${escapeHtml(cue.english || '')}">Fix</button><button class="cue-takes" data-cue="${cue.id}">Takes</button></div>` : `<span class="cue-state ${cue.status}">${cue.status === 'complete' || cue.status === 'voiced' ? '✓ Passed' : 'Waiting'}</span>`)}</div>`; }).join('') + (visible.length < cues.length ? `<button class="cue-more">Show the next ${Math.min(500,cues.length-visible.length)} lines</button>` : '');
  list.querySelectorAll('.cue-fix').forEach(button => button.onclick = () => editAndRegenerate(button));
  list.querySelectorAll('.cue-split').forEach(button => button.onclick = () => splitCue(button.dataset.cue));
  list.querySelectorAll('.cue-merge').forEach(button => button.onclick = () => mergeCue(button.dataset.cue));
  list.querySelectorAll('.cue-takes').forEach(button => button.onclick = () => restoreTake(button.dataset.cue));
  const more = list.querySelector('.cue-more'); if (more) more.onclick = () => { state.cuePage += 1; renderCues(cues, false, editable); };
}

async function editAndRegenerate(button) {
  const cue = (state.activeJob?.cues || []).find(item=>String(item.id)===String(button.dataset.cue));
  const source = prompt('Correct the source transcript.', cue?.source || '');
  if (source === null) return;
  const revised = prompt('Edit the adapted English dialogue.', button.dataset.text);
  if (!revised) return;
  const speaker = prompt('Character / voice name (renames this voice throughout the project).', cue?.speaker || '');
  if (speaker === null || !speaker.trim()) return;
  const startText = prompt('Line start (MM:SS or HH:MM:SS).', formatCueTime(cue?.start || 0));
  if (startText === null) return;
  const endText = prompt('Line end (MM:SS or HH:MM:SS).', formatCueTime(cue?.end || 0));
  if (endText === null) return;
  const start=parseClock(startText), end=parseClock(endText);
  if(start===null || end===null || end<=start)return toast('Enter a valid start and an end after it.');
  const unchanged = revised.trim() === String(cue?.english || '').trim()
    && source.trim() === String(cue?.source || '').trim()
    && speaker.trim() === String(cue?.speaker || '').trim()
    && Math.abs(start-Number(cue?.start||0))<.001 && Math.abs(end-Number(cue?.end||0))<.001;
  if (unchanged) return;
  try {
    const edited = await api(`/api/jobs/${state.activeId}/cues/${button.dataset.cue}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({source:source.trim(),english:revised.trim(),speaker_name:speaker.trim(),start,end})});
    if (state.activeJob?.stage === 'Translation ready for approval') {
      state.activeJob = edited; renderJob(edited); toast('Translation updated. Continue when the cue sheet is ready.');
    } else {
      const job = await api(`/api/jobs/${state.activeId}/cues/${button.dataset.cue}/regenerate`, {method:'POST'});
      state.activeJob = job; renderJob(job); toast('The corrected line is queued for regeneration.');
    }
  } catch (error) { toast(error.message); }
}

async function splitCue(cueId) {
  const cue=(state.activeJob?.cues||[]).find(item=>String(item.id)===String(cueId)); if(!cue)return;
  const atText=prompt('Split at absolute timeline time (MM:SS or HH:MM:SS).',formatCueTime((Number(cue.start)+Number(cue.end))/2)); if(!atText)return;
  const at=parseClock(atText); if(at===null)return toast('Enter a valid split time.');
  const first=prompt('First English line.',cue.english||''); if(!first)return;
  const second=prompt('Second English line.',''); if(!second)return;
  try{state.activeJob=await api(`/api/jobs/${state.activeId}/cues/${cueId}/split`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({at,first_text:first,second_text:second})});renderJob(state.activeJob);}catch(error){toast(error.message);}
}

async function mergeCue(cueId) {
  if(!confirm('Merge this line with the following line? All voice takes will be regenerated.'))return;
  try{state.activeJob=await api(`/api/jobs/${state.activeId}/cues/${cueId}/merge-next`,{method:'POST'});renderJob(state.activeJob);}catch(error){toast(error.message);}
}

async function restoreTake(cueId) {
  try {
    const takes=await api(`/api/jobs/${state.activeId}/cues/${cueId}/takes`); if(!takes.length)return toast('No earlier takes are stored for this line.');
    const choice=prompt(`Choose a take ID:\n${takes.map(t=>`${t.id} · ${t.files.join(', ')}`).join('\n')}`,takes[0].id); if(!choice)return;
    state.activeJob=await api(`/api/jobs/${state.activeId}/cues/${cueId}/takes/${encodeURIComponent(choice)}/restore`,{method:'POST'});renderJob(state.activeJob);toast('Prior take restored and queued for final QC.');
  } catch(error){toast(error.message);}
}

async function control(action) {
  if (!state.activeId) return;
  try { const job = await api(`/api/jobs/${state.activeId}/control/${action}`, {method:'POST'}); state.activeJob = job; renderJob(job); }
  catch (error) { toast(error.message); }
}

async function approvePending() {
  if (!state.activeId) return;
  if (state.activeJob?.status === 'awaiting_selection') {
    try { const subtitle=$('#jobSubtitleTrack').value; const job=await api(`/api/jobs/${state.activeId}/media-tracks`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({audio_index:Number($('#jobAudioTrack').value),subtitle_index:subtitle===''?null:Number(subtitle)})}); state.activeJob=job;renderJob(job); }
    catch(error){toast(error.message);} return;
  }
  control('resume');
}

async function api(url, options) {
  const response = await fetch(url, options); const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'The local service could not complete that request.');
  return data;
}

function labelStatus(job) {
  if (job.status === 'complete' && (Number(job.qc?.flagged_count || 0) || job.qc?.passed === false)) return 'Needs review';
  return ({uploading:'Uploading',awaiting_selection:'Choose tracks',queued:'Queued',processing:'Dubbing',paused:'Paused',cancelled:'Cancelled',complete:'QC passed',needs_review:'Needs review',error:'Needs attention'})[job.status] || job.status;
}
function formatBytes(bytes=0) { if (!bytes) return '0 B'; const units=['B','KB','MB','GB','TB'], i=Math.min(units.length-1,Math.floor(Math.log(bytes)/Math.log(1024))); return `${(bytes/1024**i).toFixed(i?1:0)} ${units[i]}`; }
function formatTime(seconds=0) { const h=Math.floor(seconds/3600), m=Math.floor(seconds%3600/60), s=Math.floor(seconds%60); return h ? `${h}h ${m}m` : `${m}:${String(s).padStart(2,'0')}`; }
function formatCueTime(seconds=0) { const h=Math.floor(seconds/3600), m=Math.floor(seconds%3600/60), s=Math.floor(seconds%60); return h ? `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}` : `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`; }
function parseClock(value='') { const text=String(value).trim(); if(!text)return null; const parts=text.split(':'); if(parts.length>3||parts.some(x=>!/^\d+(?:\.\d+)?$/.test(x)))return null; let seconds=0; for(const part of parts)seconds=seconds*60+Number(part); return Number.isFinite(seconds)&&seconds>=0?seconds:null; }
function relativeTime(ts) { const delta=Math.max(0,Date.now()/1000-ts); if(delta<60)return 'now'; if(delta<3600)return `${Math.floor(delta/60)}m ago`; if(delta<86400)return `${Math.floor(delta/3600)}h ago`; return `${Math.floor(delta/86400)}d ago`; }
function initials(text='Voice') { return text.split(/\s+/).map(x=>x[0]).join('').slice(0,2).toUpperCase(); }
function escapeHtml(value='') { return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char])); }
function toast(message) { const el=$('#toast'); el.textContent=message; el.classList.remove('hidden'); clearTimeout(toast.timer); toast.timer=setTimeout(()=>el.classList.add('hidden'),4500); }
