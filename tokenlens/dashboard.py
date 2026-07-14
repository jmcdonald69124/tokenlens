"""Self-contained live telemetry dashboard, served by the proxy at /tokenlens/.

Single HTML string with inline CSS/JS. It polls /tokenlens/feed once a second
and re-renders. No external assets, no build step.
"""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TokenLens — live telemetry</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --line:#30363d; --fg:#e6edf3; --muted:#8b949e;
    --accent:#58a6ff; --good:#3fb950; --warn:#d29922; --bad:#f85149; --save:#a371f7;
  }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f6f8fa; --panel:#fff; --line:#d0d7de; --fg:#1f2328; --muted:#636c76;
            --accent:#0969da; --good:#1a7f37; --warn:#9a6700; --bad:#cf222e; --save:#8250df; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  header { display:flex; align-items:center; gap:12px; flex-wrap:wrap;
           padding:16px 20px; border-bottom:1px solid var(--line); }
  header h1 { font-size:16px; margin:0; font-weight:600; letter-spacing:.3px; }
  .mode { color:var(--muted); }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--muted); display:inline-block; }
  .dot.live { background:var(--good); box-shadow:0 0 6px var(--good); }
  .dot.dead { background:var(--bad); }
  .spacer { flex:1; }
  .meta { color:var(--muted); font-size:12px; }
  main { padding:20px; max-width:1100px; margin:0 auto; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; }
  .card .label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.6px; }
  .card .value { font-size:24px; font-weight:600; margin-top:4px; }
  .card .sub { color:var(--muted); font-size:11px; margin-top:2px; }
  .value.save { color:var(--save); }
  .value.cost { color:var(--accent); }
  .value.good { color:var(--good); }
  .value.bad  { color:var(--bad); }
  .value.idle { color:var(--muted); }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px;
           padding:12px 16px; margin-top:12px; }
  .panel .lbl { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.6px; }
  .panel .hint { color:var(--muted); font-size:11.5px; margin-top:6px; }
  .jrow { display:flex; align-items:baseline; gap:10px; padding:6px 0;
          border-bottom:1px solid var(--line); font-size:12.5px; }
  .jrow:last-child { border-bottom:none; }
  .jrow .t { color:var(--muted); }
  .jrow .g { font-variant-numeric:tabular-nums; }
  .jrow .note { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .pill { padding:1px 7px; border-radius:99px; font-size:10.5px; border:1px solid var(--line); }
  .pill.ok { color:var(--good); border-color:var(--good); }
  .pill.bad { color:var(--bad); border-color:var(--bad); }
  .banner { margin:16px 0; padding:10px 14px; border-radius:8px; border:1px solid var(--line);
            display:flex; align-items:center; gap:10px; font-size:13px; }
  .banner.good { border-color:var(--good); color:var(--good); }
  .banner.warn { border-color:var(--warn); color:var(--warn); }
  .tablewrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; margin-top:8px; }
  table { border-collapse:collapse; width:100%; font-size:12.5px; }
  th,td { text-align:right; padding:7px 10px; white-space:nowrap; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; text-transform:uppercase; font-size:10.5px; letter-spacing:.5px;
       position:sticky; top:0; background:var(--panel); }
  td.l,th.l { text-align:left; }
  tr:last-child td { border-bottom:none; }
  .status-2 { color:var(--good); } .status-4 { color:var(--warn); } .status-5 { color:var(--bad); }
  .saved { color:var(--save); }
  .zero { color:var(--muted); }
  .empty { padding:24px; text-align:center; color:var(--muted); }
  button { font:inherit; background:var(--panel); color:var(--fg); border:1px solid var(--line);
           border-radius:6px; padding:5px 12px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  .spark { background:var(--panel); border:1px solid var(--line); border-radius:8px;
           padding:12px 16px; margin-top:12px; }
  .spark .lbl { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.6px; }
  .spark svg { width:100%; height:56px; display:block; margin-top:6px; }
  .spark-line { fill:none; stroke:var(--save); stroke-width:2; }
  .spark-fill { fill:var(--save); opacity:.12; }
</style>
</head>
<body>
<header>
  <span class="dot" id="dot"></span>
  <h1>TokenLens</h1>
  <span class="mode" id="mode">live telemetry</span>
  <span class="spacer"></span>
  <span class="meta" id="meta">connecting…</span>
  <button id="pause">Pause</button>
</header>
<main>
  <div class="cards" id="cards"></div>
  <div class="spark" id="sparkwrap" style="display:none">
    <div class="lbl" id="sparklbl">cumulative tokens saved</div>
    <svg viewBox="0 0 600 56" preserveAspectRatio="none">
      <path class="spark-fill" id="sparkfill"></path>
      <path class="spark-line" id="sparkpath"></path>
    </svg>
  </div>
  <div class="banner" id="cache"></div>
  <div class="panel" id="judgewrap" style="display:none">
    <div class="lbl">model judgement — compressed vs cleartext answers</div>
    <div id="judgements"></div>
    <div class="hint">A judge model grades the answer your compressed prompt produced against
      the answer the original prompt produces, on the same request, without being told which
      is which. 100% means it could not tell them apart.</div>
  </div>
  <div class="panel" id="evalwrap" style="display:none">
    <div class="lbl" id="evallbl">calibration — tokenlens eval</div>
    <div class="tablewrap" style="margin-top:8px">
      <table>
        <thead><tr><th class="l">arm</th><th>reduction</th><th>quality</th>
          <th>worst case</th><th class="l">verdict</th></tr></thead>
        <tbody id="evalrows"></tbody>
      </table>
    </div>
    <div class="hint" id="evalpolicy"></div>
  </div>
  <div class="panel" id="taskswrap" style="display:none">
    <div class="lbl" id="taskslbl">golden set — what the calibration was measured on</div>
    <div class="tablewrap" style="margin-top:8px">
      <table>
        <thead><tr><th class="l">task</th><th class="l">class</th>
          <th class="l">what it asks</th><th>context</th></tr></thead>
        <tbody id="taskrows"></tbody>
      </table>
    </div>
    <div class="hint">These ten synthetic tasks are a smoke-grade starter set, not a
      benchmark. Point <code>--tasks</code> at traffic shaped like yours before you
      trust a policy — a curve from someone else's golden set is a more expensive guess.</div>
  </div>
  <div class="tablewrap">
    <table>
      <thead><tr>
        <th class="l">time</th><th class="l">model</th><th>status</th>
        <th>in</th><th>cache r</th><th>cache w</th><th>out</th>
        <th>saved</th><th>ms</th><th>cost $</th>
      </tr></thead>
      <tbody id="rows"><tr><td class="empty" colspan="10">waiting for requests…</td></tr></tbody>
    </table>
  </div>
</main>
<script>
let paused = false, lastOk = 0;
const $ = id => document.getElementById(id);
$("pause").onclick = () => { paused = !paused; $("pause").textContent = paused ? "Resume" : "Pause"; };

function n(x){ return (x==null?0:x).toLocaleString(); }
function card(label, value, sub, cls){
  return `<div class="card"><div class="label">${label}</div>`+
         `<div class="value ${cls||''}">${value}</div>`+
         `<div class="sub">${sub||''}</div></div>`;
}

async function tick(){
  if (paused) return;
  try {
    const r = await fetch("/tokenlens/feed?limit=50", {cache:"no-store"});
    const d = await r.json();
    render(d);
    lastOk = Date.now();
    $("dot").className = "dot live";
  } catch(e){
    $("dot").className = "dot dead";
    $("meta").textContent = "disconnected — is the proxy running?";
  }
}

function drawSpark(series){
  const W=600,H=56;
  const fill=$("sparkfill"), path=$("sparkpath");
  if(!series || series.length<2){ path.setAttribute("d",""); fill.setAttribute("d",""); return; }
  const max=Math.max(...series,1), step=W/(series.length-1);
  let d=""; series.forEach((v,i)=>{ const x=i*step, y=H-(v/max)*(H-6)-3; d+=(i?"L":"M")+x.toFixed(1)+" "+y.toFixed(1)+" "; });
  path.setAttribute("d", d.trim());
  fill.setAttribute("d", d.trim()+`L${W} ${H} L0 ${H} Z`);
}

// The note comes from a model. Escape it before it touches the DOM.
function esc(s){
  return String(s==null?"":s).replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function renderJudgements(d){
  const js = d.judgements || [];
  if (!js.length){ $("judgewrap").style.display = "none"; return; }
  $("judgewrap").style.display = "block";
  $("judgements").innerHTML = js.map(j => {
    const pct = Math.round(100*j.retention);
    const pill = j.degraded ? `<span class="pill bad">${pct}%</span>`
                            : `<span class="pill ok">${pct}%</span>`;
    return `<div class="jrow"><span class="t">${esc(j.time)}</span>${pill}`+
           `<span class="g">${j.cleartext} → ${j.compressed}</span>`+
           `<span class="note">${esc(j.note)}</span></div>`;
  }).join("");
}

function renderEval(ev){
  if (!ev || !ev.curve || !ev.curve.length){ $("evalwrap").style.display = "none"; return; }
  $("evalwrap").style.display = "block";
  $("evallbl").textContent =
    `calibration — ${ev.tasks} golden tasks on ${ev.model}, judged by ${ev.judge_model}`;
  const floor = ev.noise_floor || 0;
  $("evalrows").innerHTML = ev.curve.map(s => {
    let cls = "status-2", verdict = "✓ within tolerance";
    if (s.is_control){ cls = "idle"; verdict = "— noise floor (nothing compressed)"; }
    else if (!s.passes){ cls = "status-5"; verdict = "✗ degrades quality"; }
    else if (s.resolved === false && floor > 0){
      cls = "status-4"; verdict = "? provisional — loss inside the noise";
    }
    return `<tr><td class="l">${esc(s.arm)}</td>`+
           `<td class="saved">${s.savings_pct.toFixed(1)}%</td>`+
           `<td>${(100*s.mean_retention).toFixed(1)}%</td>`+
           `<td>${(100*s.worst_retention).toFixed(0)}%</td>`+
           `<td class="l ${cls}">${verdict}</td></tr>`;
  }).join("");
  const p = (ev.policy||{}).default;
  const floorNote = floor > 0
    ? ` Noise floor is ${(100*floor).toFixed(1)}% — the control arm lost that much `+
      `compressing nothing, so anything under it is provisional, not measured.`
    : "";
  $("evalpolicy").textContent = (p
    ? (p.method === "none"
        ? `Calibrated policy: none — ${p.reason}.`
        : `Calibrated policy: ${p.arm} — ${p.savings_pct.toFixed(1)}% smaller at `+
          `${(100*p.mean_retention).toFixed(1)}% of cleartext quality`+
          (p.confidence === "provisional" ? " (provisional)" : "") + ".")
    : "") + floorNote;
  renderTasks(ev.task_catalog);
}

function renderTasks(cat){
  if (!cat || !cat.length){ $("taskswrap").style.display = "none"; return; }
  $("taskswrap").style.display = "block";
  $("taskslbl").textContent = `golden set — the ${cat.length} tasks the calibration was measured on`;
  $("taskrows").innerHTML = cat.map(t =>
    `<tr><td class="l">${esc(t.id)}</td>`+
    `<td class="l">${esc(t.task_class)}</td>`+
    `<td class="l">${esc(t.question)}</td>`+
    `<td>${n(t.context_chars)}c</td></tr>`
  ).join("");
}

function render(d){
  const t = d.totals;
  const cacheTotal = t.cache_read_tokens + t.cache_write_tokens;
  const cacheRatio = cacheTotal ? Math.round(100*t.cache_read_tokens/cacheTotal) : 0;
  const measured = t.measured_requests > 0;
  const saved = measured ? t.real_tokens_saved : t.est_tokens_saved;
  const savedLabel = measured ? "Tokens saved (measured)" : "Est. tokens saved";
  const savedSub = measured
    ? (t.real_savings_pct!=null ? t.real_savings_pct+"% smaller (measured)" : "measured")
    : "estimate — run --measure for exact";
  const judging = (d.mode && d.mode.judge);
  const judged = t.judged_requests || 0;
  const tol = Math.round(100*(t.judge_tolerance||0.99));
  let qVal = "—", qSub = "run with --judge to grade quality", qCls = "idle";
  if (judging && !judged) {
    qSub = "judging " + Math.round(100*d.mode.judge_sample) + "% of compressed requests…";
  } else if (judged) {
    const q = t.quality_retained_pct;
    qVal = q.toFixed(1) + "%";
    qCls = q >= tol ? "good" : "bad";
    qSub = `${n(judged)} judged · ${t.judge_degraded} below ${tol}%`;
  }

  $("cards").innerHTML =
    card("Requests", n(t.requests), "since start") +
    card("Tokens in", n(t.input_tokens), "uncached") +
    card("Tokens out", n(t.output_tokens), "") +
    card(savedLabel, n(saved), savedSub, "save") +
    card("Quality retained", qVal, qSub, qCls) +
    card("Cost", "$"+(t.cost_usd||0).toFixed(4), "est. USD", "cost") +
    card("Cache reads", n(t.cache_read_tokens), "writes "+n(t.cache_write_tokens), "");

  renderJudgements(d);
  renderEval(d.eval);

  if (t.requests > 0 && d.series && d.series.length > 1) {
    $("sparkwrap").style.display = "block";
    $("sparklbl").textContent = "cumulative tokens saved" + (measured ? " (measured)" : " (est)");
    drawSpark(d.series);
  } else {
    $("sparkwrap").style.display = "none";
  }

  const c = $("cache");
  if (t.requests === 0) {
    c.className = "banner"; c.innerHTML = "No requests yet.";
  } else if (t.cache_read_tokens > 0) {
    c.className = "banner good";
    c.innerHTML = `✓ Cache healthy — ${cacheRatio}% of cached tokens are reads. `+
                  `Compression is not busting the prompt cache.`;
  } else {
    c.className = "banner warn";
    c.innerHTML = `⚠ No cache reads yet. If this stays zero once traffic warms up, `+
                  `compression may be busting the prompt cache — check the tail boundary.`;
  }

  const rows = d.recent;
  const tb = $("rows");
  if (!rows.length){ tb.innerHTML = `<tr><td class="empty" colspan="10">waiting for requests…</td></tr>`; }
  else {
    tb.innerHTML = rows.map(e => {
      const sc = "status-" + String(e.status)[0];
      const saved = e.saved ? `<span class="saved">-${n(e.saved)}</span>` : `<span class="zero">0</span>`;
      const cost = e.cost==null ? "—" : e.cost.toFixed(5);
      return `<tr><td class="l">${e.time}</td><td class="l">${e.model}</td>`+
             `<td class="${sc}">${e.status}</td>`+
             `<td>${n(e.input)}</td><td>${n(e.cache_read)}</td><td>${n(e.cache_write)}</td>`+
             `<td>${n(e.output)}</td><td>${saved}</td><td>${n(e.ms)}</td><td>${cost}</td></tr>`;
    }).join("");
  }
  if (d.mode) {
    const bits = ["compress=" + d.mode.compress];
    if (d.mode.compress === "llmlingua2") bits.push("rate=" + d.mode.rate);
    if (d.mode.judge) bits.push("judge=" + d.mode.judge_model);
    $("mode").textContent = bits.join(" · ");
  }
  $("meta").textContent = `uptime ${d.uptime_s}s · updated ${new Date().toLocaleTimeString()}`;
}

setInterval(tick, 1000);
tick();
</script>
</body>
</html>
"""
