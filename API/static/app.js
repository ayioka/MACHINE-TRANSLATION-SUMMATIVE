// Frontend for the Kidaw'ida <-> Kiswahili translator. Talks to the FastAPI backend under /api.
(() => {
  const NAMES = { swa: "Kiswahili", dav: "Kidaw'ida" };
  const PLACEHOLDER = {
    swa: "Andika sentensi kwa Kiswahili…",
    dav: "Andika sentensi kwa Kidaw'ida…",
  };
  const $ = (id) => document.getElementById(id);
  const el = {
    status: $("status"), statusText: $("status-text"),
    srcName: $("src-name"), tgtName: $("tgt-name"), swap: $("swap"),
    input: $("source-text"), count: $("count"), clear: $("clear"), translate: $("translate"),
    output: $("output"), meta: $("meta"), copy: $("copy"),
    examples: $("examples"), results: document.querySelector("#results tbody"),
    limits: $("limits"), metricNote: $("metric-note"),
  };

  const state = { source: "swa", target: "dav", ready: false, busy: false, examples: null, last: "" };

  // Remember the last direction (a per-browser convenience; safe to fail).
  try {
    const saved = localStorage.getItem("direction");
    if (saved === "dav-swa") { state.source = "dav"; state.target = "swa"; }
  } catch (_) { /* storage unavailable */ }

  // ------------------------------------------------------------ helpers
  const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  // Kidaw'ida's apostrophe is part of the spelling; it gets the accent colour in the output.
  const markApostrophes = (s) => escapeHtml(s).replace(/'/g, '<span class="apos">\'</span>');

  async function api(path, options = {}) {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON error page */ }
    if (!res.ok) {
      let msg = `The server answered ${res.status}.`;
      if (body && typeof body.detail === "string") msg = body.detail;
      else if (body && Array.isArray(body.detail)) msg = body.detail.map((d) => d.msg.replace(/^Value error, /, "")).join(" ");
      throw new Error(msg);
    }
    return body;
  }

  function setStatus(kind, text) {
    el.status.dataset.state = kind;
    el.statusText.textContent = text;
  }

  function showOutput(html, cls = "") {
    el.output.className = `output ${cls}`.trim();
    el.output.innerHTML = html;
  }

  // ---------------------------------------------------------- direction
  function renderDirection() {
    el.srcName.textContent = NAMES[state.source];
    el.tgtName.textContent = NAMES[state.target];
    el.input.placeholder = PLACEHOLDER[state.source];
    el.input.lang = state.source === "swa" ? "sw" : "dav";
    el.output.lang = state.target === "swa" ? "sw" : "dav";
    renderExamples();
  }

  el.swap.addEventListener("click", () => {
    [state.source, state.target] = [state.target, state.source];
    el.swap.classList.toggle("turned");
    // Carry the current translation over so it can be translated back.
    if (state.last) {
      el.input.value = state.last;
      updateCount();
      state.last = "";
      showOutput('<p class="placeholder">The translation appears here.</p>');
      el.copy.hidden = true; el.meta.textContent = "";
    }
    try { localStorage.setItem("direction", `${state.source}-${state.target}`); } catch (_) {}
    renderDirection();
    el.input.focus();
  });

  // -------------------------------------------------------------- input
  function updateCount() {
    const n = el.input.value.length;
    el.count.textContent = `${n} / 1000`;
    el.clear.hidden = n === 0;
  }
  el.input.addEventListener("input", updateCount);
  el.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); translate(); }
  });
  el.clear.addEventListener("click", () => {
    el.input.value = ""; updateCount(); el.input.focus();
  });

  // ---------------------------------------------------------- translate
  async function translate() {
    const text = el.input.value.trim();
    if (!text) { el.input.focus(); return; }
    if (state.busy) return;
    if (!state.ready) {
      showOutput('<p class="error">The model is still loading. It usually takes a minute after the server starts.</p>');
      return;
    }
    state.busy = true;
    el.translate.disabled = true;
    el.translate.textContent = "Translating…";
    showOutput(el.output.querySelector(".placeholder") ? "" : el.output.innerHTML, "is-busy");
    try {
      const r = await api("/api/translate", {
        method: "POST",
        body: JSON.stringify({ text, source: state.source, target: state.target }),
      });
      state.last = r.translation;
      showOutput(markApostrophes(r.translation));
      const secs = (r.time_ms / 1000).toFixed(1);
      el.meta.textContent = `${r.sentences} ${r.sentences === 1 ? "sentence" : "sentences"} in ${secs} s`;
      el.copy.hidden = false;
    } catch (err) {
      state.last = "";
      showOutput(`<p class="error">${escapeHtml(err.message)}</p>`);
      el.meta.textContent = ""; el.copy.hidden = true;
    } finally {
      state.busy = false;
      el.translate.disabled = false;
      el.translate.textContent = "Translate";
    }
  }
  el.translate.addEventListener("click", translate);

  el.copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(state.last);
      el.copy.textContent = "Copied";
    } catch (_) {
      el.copy.textContent = "Copy failed";
    }
    setTimeout(() => { el.copy.textContent = "Copy"; }, 1600);
  });

  // ----------------------------------------------------------- examples
  function renderExamples() {
    if (!state.examples) return;
    el.examples.innerHTML = "";
    for (const ex of state.examples[state.source] || []) {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.innerHTML = `${escapeHtml(ex.text)}<span class="note">${escapeHtml(ex.note || "")}</span>`;
      b.addEventListener("click", () => {
        el.input.value = ex.text; updateCount(); translate();
      });
      li.appendChild(b);
      el.examples.appendChild(li);
    }
  }

  // --------------------------------------------------------- model info
  function scoreCell(v) {
    if (v === null || v === undefined) return '<td><span class="na">not run</span></td>';
    return `<td><div class="score"><span>${v.toFixed(1)}</span><div class="bar"><span style="width:${v}%"></span></div></div></td>`;
  }
  function renderModelInfo(info) {
    el.results.innerHTML = info.results.map((r) =>
      `<tr class="${r.deployed ? "deployed" : ""}"><td>${escapeHtml(r.system)}</td>${scoreCell(r.swa_dav)}${scoreCell(r.dav_swa)}</tr>`
    ).join("");
    el.metricNote.textContent = `Scores are ${info.metric}. ${info.data.note}`;
    el.limits.innerHTML = info.limitations.map((l) => `<li>${escapeHtml(l)}</li>`).join("");
  }

  // -------------------------------------------------------------- health
  async function pollHealth() {
    try {
      const h = await api("/api/health");
      if (h.status === "ready") {
        state.ready = true;
        setStatus("ready", "Model ready");
        return;
      }
      if (h.status === "error") {
        setStatus("error", "Model failed to load");
        showOutput(`<p class="error">The model failed to load on the server: ${escapeHtml(h.error || "unknown error")}</p>`);
        return;
      }
      setStatus("loading", "Loading the model…");
    } catch (_) {
      setStatus("error", "Server unreachable");
    }
    setTimeout(pollHealth, 3000);
  }

  // ---------------------------------------------------------------- init
  renderDirection();
  updateCount();
  pollHealth();
  api("/api/examples").then((ex) => { state.examples = ex; renderExamples(); }).catch(() => {});
  api("/api/model-info").then(renderModelInfo).catch(() => {});
})();
