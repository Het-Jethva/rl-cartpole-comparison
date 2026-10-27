const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../rlcompare/demo_assets/demo.js"), "utf8");

async function checkEvaluation(cancellation) {
  const elements = new Map();
  const events = new Map();
  const context = new Proxy({}, {
    get: (object, key) => object[key] || (() => {}),
    set: (object, key, value) => (object[key] = value, true),
  });
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      value: "", textContent: "", width: 1000, height: 400, style: {}, handlers: {},
      setAttribute() {},
      getContext() { return context; },
      addEventListener(event, handler) { this.handlers[event] = handler; },
    });
    return elements.get(id);
  }
  const state = {
    generation: "session", method: "reinforce", seed: 0, budget: 100000,
    steps: 42, episode: 1, updates: 0, complete: false, history: [], evaluations: [],
    frame: { observation: [0, 0, 0.15, 0], action: 0, score: 42, terminated: false, truncated: false },
  };
  let finishEvaluation;
  let playbackTimers = 0;
  const document = {
    hidden: false,
    getElementById: element,
    querySelector: () => element("progress"),
    addEventListener(event, handler) { events.set(event, handler); },
  };
  const sandbox = {
    document, performance: { now: () => 0 }, clearTimeout() {},
    setTimeout() { playbackTimers += 1; return playbackTimers; },
    fetch(url) {
      if (url === "/api/state") return Promise.resolve({ ok: true, json: async () => state });
      assert.equal(url, "/api/evaluate");
      return new Promise(resolve => { finishEvaluation = resolve; });
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);
  await new Promise(setImmediate);
  const pending = element("evaluate").handlers.click();
  if (cancellation === "pause") element("pause").handlers.click();
  if (cancellation === "hidden") {
    document.hidden = true;
    events.get("visibilitychange")();
  }
  finishEvaluation({
    ok: true,
    json: async () => ({
      state,
      frames: [{ observation: [0, 0, 0, 0], action: null, score: 0, terminated: false, truncated: false }],
    }),
  });
  await pending;
  if (cancellation) {
    assert.equal(element("status").textContent, "Paused");
    assert.equal(element("view-title").textContent, "REINFORCE + baseline · training");
    assert.equal(element("view-note").textContent, "Exploration enabled");
    assert.equal(element("score-label").textContent, "Current return");
    assert.equal(element("score").textContent, "42");
    assert.equal(playbackTimers, 0);
  } else {
    assert.equal(element("status").textContent, "Evaluating policy");
    assert.equal(element("view-title").textContent, "Policy evaluation");
    assert.equal(element("score-label").textContent, "Evaluation return");
    assert.equal(element("score").textContent, "0");
    assert.equal(playbackTimers, 1);
  }
}

(async () => {
  for (const cancellation of ["pause", "hidden", null]) {
    await checkEvaluation(cancellation);
    console.log(`Passed evaluation response with ${cancellation || "normal playback"}.`);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
