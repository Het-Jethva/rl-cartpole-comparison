"use strict";

const $ = id => document.getElementById(id);
const names = {reinforce: "REINFORCE + baseline", dqn: "DQN", tabular: "Tabular Q-learning"};
const notes = {reinforce: "Learns a policy directly. A value baseline reduces variance.", dqn: "Learns action values with a replay buffer and target network.", tabular: "Learns a Q table over a discretized state space."};
let session = null;
let phase = "connecting";
let busy = false;
let timer = null;
let evaluationFrames = [];
let evaluationIndex = 0;

async function request(path, data) {
  const response = await fetch(path, data === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "The demo could not complete this request.");
  return body;
}

function controls() {
  const active = phase === "running" || phase === "evaluating";
  const settingsChanged = session && (session.method !== $("method").value || session.seed !== Number($("seed").value) || session.budget !== Number($("budget").value));
  $("start").disabled = busy || active || !session || session.complete || settingsChanged;
  $("start").textContent = session && session.steps ? "Resume training" : "Start training";
  $("pause").disabled = !active;
  $("reset").disabled = busy || active;
  $("evaluate").disabled = busy || active || !session || settingsChanged;
  for (const id of ["method", "seed", "budget"]) $(id).disabled = busy || active;
  $("status").textContent = ({connecting:"Connecting", ready:"Ready", running:"Training live", paused:"Paused", evaluating:"Evaluating policy", complete:"Budget complete", error:"Needs attention"})[phase];
  if (settingsChanged && !active && !busy && phase !== "error") $("status").textContent = "Settings changed · reset to apply";
}

function fail(error) {
  clearTimeout(timer);
  phase = "error";
  $("error").hidden = false;
  $("error").textContent = `${error.message} Reset the session to retry. If the server stopped, restart it and reload this page.`;
  controls();
}

function drawCart(frame) {
  const c = $("cartpole"), ctx = c.getContext("2d");
  const [x, , theta] = frame.observation;
  const scale = c.width / 6;
  const cartX = c.width / 2 + x * scale;
  const ground = c.height * .76;
  const pivotY = ground - 45;
  const length = scale;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, c.width, c.height);
  ctx.strokeStyle = "#d3ddd8"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(35, ground + 2); ctx.lineTo(c.width - 35, ground + 2); ctx.stroke();
  ctx.font = `${Math.max(15, 12 * c.width / c.clientWidth)}px Segoe UI`; ctx.fillStyle = "#536b70"; ctx.textAlign = "center";
  for (const position of [-2.4, 0, 2.4]) {
    const px = c.width / 2 + position * scale;
    ctx.beginPath(); ctx.moveTo(px, ground + 12); ctx.lineTo(px, ground + 23); ctx.stroke();
    ctx.fillText(`${position} m`, px, ground + 47);
  }
  ctx.setLineDash([5, 7]); ctx.strokeStyle = "#b7c7c3";
  ctx.beginPath(); ctx.moveTo(cartX, pivotY); ctx.lineTo(cartX, pivotY - length - 15); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#173d48"; ctx.fillRect(cartX - 39, ground - 39, 78, 30);
  for (const offset of [-24, 24]) {ctx.beginPath();ctx.arc(cartX + offset, ground - 7, 9, 0, Math.PI * 2);ctx.fill();}
  ctx.strokeStyle = frame.terminated ? "#ae4e1d" : "#006d6f";
  ctx.lineWidth = 12; ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(cartX, pivotY); ctx.lineTo(cartX + Math.sin(theta) * length, pivotY - Math.cos(theta) * length);ctx.stroke();
  ctx.fillStyle = "#173d48";ctx.beginPath();ctx.arc(cartX, pivotY, 7, 0, Math.PI * 2);ctx.fill();
  $("angle").textContent = `Pole angle ${(theta * 180 / Math.PI).toFixed(1)}°`;
  $("score").textContent = frame.score.toLocaleString();
  const action = frame.action === null ? "Ready for the first action" : `Action ${frame.action} · push ${frame.action ? "right" : "left"}`;
  $("frame-label").textContent = frame.terminated ? `${action} · episode ended` : frame.truncated ? `${action} · 500-step time limit` : action;
}

function chart(id, points, color, empty) {
  const c = $(id), ctx = c.getContext("2d"), left = 47, top = 15, width = c.width - 65, height = c.height - 58;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.font = `${Math.max(20, 12 * c.width / c.clientWidth)}px Segoe UI`;ctx.fillStyle = "#4c646b";ctx.lineWidth = 1;
  for (const y of [0, 250, 500]) {
    const py = top + height * (1 - y / 500);
    ctx.strokeStyle = "#d1dbd7";ctx.beginPath();ctx.moveTo(left, py);ctx.lineTo(left + width, py);ctx.stroke();
    ctx.textAlign = "right";ctx.fillText(y, left - 9, py + 5);
  }
  if (!points.length) {ctx.textAlign="center";ctx.fillText(empty,left+width/2,top+height/2);return;}
  const first = Math.min(0, points[0].x), max = Math.max(points.at(-1).x, 1);
  ctx.strokeStyle = color;ctx.lineWidth = 2.5;ctx.beginPath();
  points.forEach((p, i) => {const x = left + (p.x - first) / (max - first) * width;const y = top + height * (1 - p.y / 500);i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);});ctx.stroke();
  if (points.length === 1) {ctx.fillStyle=color;ctx.beginPath();ctx.arc(left+(points[0].x-first)/(max-first)*width,top+height*(1-points[0].y/500),4,0,Math.PI*2);ctx.fill();}
  ctx.fillStyle="#4c646b";ctx.textAlign="left";ctx.fillText("0",left,c.height-18);ctx.textAlign="right";ctx.fillText(`${max.toLocaleString()} steps`,left+width,c.height-18);
}

function render(state) {
  session = state;
  $("steps").textContent = state.steps.toLocaleString();
  $("budget-label").textContent = `of ${state.budget.toLocaleString()}`;
  $("episode").textContent = state.episode.toLocaleString();
  $("updates").textContent = state.updates.toLocaleString();
  const last = state.history.at(-1);
  $("last-score").textContent = last ? `Last return ${last.score}` : "No completed episodes";
  $("progress-fill").style.width = `${100 * state.steps / state.budget}%`;
  document.querySelector(".progress").setAttribute("aria-valuenow", Math.round(100 * state.steps / state.budget));
  drawCart(state.frame);
  chart("training-chart", state.history.map(p => ({x:p.step,y:p.score})), "#006d6f", "No completed episodes yet");
  chart("evaluation-chart", state.evaluations.map(p => ({x:p.step,y:p.mean})), "#ae4e1d", "No evaluations yet");
  if (last) $("training-summary").textContent = `Episode ${last.episode} returned ${last.score} at step ${last.step.toLocaleString()}.`;
  else $("training-summary").textContent = "Start training to see episode returns.";
  const evaluation = state.evaluations.at(-1);
  $("evaluation-summary").textContent = evaluation ? `Mean ${evaluation.mean.toFixed(1)} / 500 at step ${evaluation.step.toLocaleString()}. Range ${Math.min(...evaluation.scores)} to ${Math.max(...evaluation.scores)}.` : "Choose Evaluate policy to measure the agent.";
}

function trainingView() {
  $("view-title").textContent = `${names[session.method]} · training`;
  $("view-note").textContent = "Exploration enabled";
  $("score-label").textContent = "Current return";
}

async function train() {
  if (phase !== "running" || busy) return;
  busy = true; controls();
  const started = performance.now();
  try {
    const state = await request("/api/advance", {generation:session.generation, count:Number($("speed").value)});
    render(state);
    if (state.complete) phase = "complete";
  } catch (error) {fail(error);}
  finally {busy = false;controls();}
  if (phase === "running") timer = setTimeout(train, Math.max(0, ($("speed").value === "1" ? 20 : 30) - (performance.now() - started)));
}

$("start").addEventListener("click", () => {
  trainingView();drawCart(session.frame);phase="running";controls();train();
});
$("pause").addEventListener("click", () => {
  clearTimeout(timer);phase=session && session.complete ? "complete" : "paused";controls();
});
$("method").addEventListener("change", () => {$("method-note").textContent = notes[$("method").value];controls();});
for (const id of ["seed", "budget"]) $(id).addEventListener("input", controls);
$("settings").addEventListener("submit", async event => {
  event.preventDefault();
  if (busy || phase === "running" || phase === "evaluating") return;
  clearTimeout(timer);busy=true;controls();$("error").hidden=true;
  try {
    render(await request("/api/reset", {method:$("method").value,seed:Number($("seed").value),budget:Number($("budget").value)}));
    phase="ready";trainingView();
  } catch (error) {fail(error);}
  finally {busy=false;controls();}
});

function playEvaluation() {
  if (phase !== "evaluating") return;
  const frame = evaluationFrames[evaluationIndex++];
  if (!frame) {
    phase=session.complete ? "complete" : "paused";
    $("view-note").textContent="Evaluation finished · training paused";
    controls();return;
  }
  drawCart(frame);
  timer=setTimeout(playEvaluation,20);
}

$("evaluate").addEventListener("click", async () => {
  if (busy) return;
  clearTimeout(timer);phase="evaluating";busy=true;controls();$("error").hidden=true;
  $("view-title").textContent="Policy evaluation";
  $("view-note").textContent="Measuring ten episodes · no learning";
  $("score-label").textContent="Evaluation return";
  try {
    const result=await request("/api/evaluate",{generation:session.generation});
    render(result.state);evaluationFrames=result.frames;evaluationIndex=0;
    $("view-note").textContent=`${names[session.method]} · first evaluation seed · exploration off`;
    if (phase === "evaluating") playEvaluation();
    else {trainingView(); drawCart(session.frame);}
  } catch(error) {fail(error);}
  finally {busy=false;controls();}
});

async function initialize() {
  busy=true;controls();
  try {
    let state=await request("/api/state");
    if(!state) state=await request("/api/reset",{method:"reinforce",seed:0,budget:100000});
    $("method").value=state.method;$("seed").value=state.seed;$("budget").value=state.budget;
    $("method-note").textContent=notes[state.method];
    render(state);trainingView();phase=state.complete?"complete":state.steps?"paused":"ready";
  } catch(error) {fail(error);}
  finally {busy=false;controls();}
}
document.addEventListener("visibilitychange", () => {
  if(document.hidden && (phase === "running" || phase === "evaluating")) {clearTimeout(timer);phase="paused";controls();}
});
initialize();
