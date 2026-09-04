import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOAD = "MiniMaxH3MotionContextLoadLatent";
const SAVE = "MiniMaxH3MotionContextSaveLatent";
const CHAIN = "MiniMaxH3MotionContextChain";
const MAX = 9999;

const CSS = `
.h3mc-chain{display:flex;flex-direction:column;gap:6px;color:#ddd;font:12px/1.3 sans-serif;width:100%;box-sizing:border-box;padding:2px 0 4px;}
.h3mc-chain-row{display:flex;gap:6px;}
.h3mc-chain-row button{flex:1;cursor:pointer;border:1px solid #555;background:#2a2a2a;color:#ddd;border-radius:4px;padding:6px 4px;font:12px sans-serif;}
.h3mc-chain-row button:hover{background:#3a3a3a;}
.h3mc-chain-row button:disabled{opacity:.45;cursor:default;}
.h3mc-chain-row button.on{border-color:#6f8bbd;background:#1f2a3a;color:#c5d4ee;}
.h3mc-chain-meta{opacity:.75;font-size:11px;min-height:14px;}
`;

let cssOnce = false;
function injectCss() {
  if (cssOnce) return;
  cssOnce = true;
  const s = document.createElement("style");
  s.textContent = CSS;
  document.head.appendChild(s);
}

function swallow(el) {
  for (const type of ["pointerdown", "mousedown", "pointerup", "mouseup",
                      "click", "dblclick", "contextmenu"]) {
    el.addEventListener(type, (e) => e.stopPropagation());
  }
}

function graphNodes(graph) {
  return graph?._nodes || graph?.nodes || [];
}

function groupList(graph) {
  return graph?._groups || graph?.groups || [];
}

function groupMembers(group) {
  if (typeof group.recomputeInsideNodes === "function") {
    try { group.recomputeInsideNodes(); } catch (e) { /* older litegraph */ }
  }
  if (Array.isArray(group._nodes) && group._nodes.length) return group._nodes;
  if (Array.isArray(group.nodes) && group.nodes.length) return group.nodes;
  const kids = group._children || group.children;
  if (kids) return Array.from(kids);
  return [];
}

function inGroup(group, node) {
  const members = groupMembers(group);
  if (members.includes(node) || members.some((n) => n.id === node.id)) return true;
  const b = group._bounding || group.bounding;
  if (!b || b.length < 4 || !node.pos) return false;
  const x = node.pos[0] + (node.size?.[0] || 0) / 2;
  const y = node.pos[1] + (node.size?.[1] || 0) / 2;
  return x >= b[0] && x <= b[0] + b[2] && y >= b[1] && y <= b[1] + b[3];
}

function findPair(ctrl) {
  const graph = ctrl.graph || app.graph;
  const nodes = graphNodes(graph);
  const loads = nodes.filter((n) => n.comfyClass === LOAD);
  const saves = nodes.filter((n) => n.comfyClass === SAVE);
  for (const g of groupList(graph)) {
    if (!inGroup(g, ctrl)) continue;
    const members = groupMembers(g);
    const pool = members.length ? members : nodes.filter((n) => inGroup(g, n));
    const load = pool.find((n) => n.comfyClass === LOAD);
    const save = pool.find((n) => n.comfyClass === SAVE);
    if (load && save) return { load, save };
  }
  if (loads.length === 1 && saves.length === 1) return { load: loads[0], save: saves[0] };
  return null;
}

function clipWidget(node) {
  return node?.widgets?.find((w) => w.name === "clip_index");
}

function readClip(node) {
  return (clipWidget(node)?.value | 0) || 0;
}

function widgetValue(node, name) {
  return node?.widgets?.find((w) => w.name === name)?.value;
}

function loadPath(pair) {
  const p = widgetValue(pair.load, "latent_path");
  return (p == null || p === "") ? "h3_context" : p;
}

async function firstClipExists(pair) {
  try {
    const r = await api.fetchApi("/h3_motion_context/slot_exists", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ latent_path: loadPath(pair), clip_index: 1 }),
    });
    if (!r.ok) return false;
    const j = await r.json();
    return !!j.exists;
  } catch (e) {
    return false;
  }
}

async function clearLatents(pair) {
  try {
    const r = await api.fetchApi("/h3_motion_context/clear_latents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ latent_path: loadPath(pair) }),
    });
    if (!r.ok) return 0;
    const j = await r.json();
    return j.removed | 0;
  } catch (e) {
    return 0;
  }
}

function writeClip(node, value) {
  const w = clipWidget(node);
  if (!w) return false;
  const next = Math.max(0, Math.min(MAX, value | 0));
  w.value = next;
  return true;
}

function readSegments(ctrl) {
  return Math.max(0, widgetValue(ctrl, "segments") | 0);
}

function paint(ctrl) {
  const meta = ctrl._h3mc?.meta;
  if (!meta) return;
  const pair = findPair(ctrl);
  if (!pair) {
    meta.textContent = "Load, Save, and Chain must share one canvas group.";
    return;
  }
  const a = readClip(pair.load);
  const b = readClip(pair.save);
  const left = ctrl._h3mc.remaining | 0;
  if (ctrl._h3mc.chaining) {
    meta.textContent = left
      ? `Chaining  Load ${a} / Save ${b}  ·  ${left} left`
      : `Chaining  Load ${a} / Save ${b}`;
  } else {
    meta.textContent = `Load ${a} / Save ${b}`;
  }
  const chainBtn = ctrl._h3mc.chainBtn;
  if (chainBtn) {
    chainBtn.textContent = ctrl._h3mc.chaining ? "Stop" : "Chain";
    chainBtn.classList.toggle("on", !!ctrl._h3mc.chaining);
  }
}

let live = null;

function stopChain(ctrl) {
  if (ctrl) {
    ctrl._h3mc.chaining = false;
    ctrl._h3mc.awaiting = false;
    ctrl._h3mc.remaining = 0;
    paint(ctrl);
  }
  if (live === ctrl) live = null;
}

function advance(ctrl) {
  const pair = findPair(ctrl);
  if (!pair) return false;
  const load = readClip(pair.load) + 1;
  let save = readClip(pair.save) + 1;
  if (save < 1) save = 1;
  writeClip(pair.load, Math.min(load, MAX));
  writeClip(pair.save, Math.min(save, MAX));
  app.graph?.setDirtyCanvas?.(true, true);
  paint(ctrl);
  return true;
}

function resetFirst(ctrl) {
  const pair = findPair(ctrl);
  if (!pair) return false;
  writeClip(pair.load, 0);
  writeClip(pair.save, 1);
  app.graph?.setDirtyCanvas?.(true, true);
  paint(ctrl);
  return true;
}

async function startChain(ctrl) {
  const pair = findPair(ctrl);
  if (!pair) return false;
  const load = readClip(pair.load);
  const save = readClip(pair.save);
  const atFirst = load === 0 && save <= 1;
  if (atFirst && !(await firstClipExists(pair))) {
    await queueOnce(ctrl);
    return true;
  }
  if (!advance(ctrl)) return false;
  await queueOnce(ctrl);
  return true;
}

async function queueOnce(ctrl) {
  const pair = findPair(ctrl);
  if (!pair) return;
  ctrl._h3mc.awaiting = true;
  live = ctrl;
  await app.queuePrompt(0, 1);
}

function onPromptDone(ok) {
  const ctrl = live;
  if (!ctrl?._h3mc?.awaiting) return;
  ctrl._h3mc.awaiting = false;
  if (!ok) {
    stopChain(ctrl);
    return;
  }
  if (!ctrl._h3mc.chaining) {
    live = null;
    paint(ctrl);
    return;
  }
  const left = ctrl._h3mc.remaining | 0;
  if (left > 0) {
    ctrl._h3mc.remaining = left - 1;
    if (ctrl._h3mc.remaining === 0) {
      ctrl._h3mc.chaining = false;
      live = null;
      paint(ctrl);
      return;
    }
  }
  if (!advance(ctrl)) {
    stopChain(ctrl);
    return;
  }
  queueOnce(ctrl);
}

api.addEventListener("execution_success", () => onPromptDone(true));
api.addEventListener("execution_error", () => onPromptDone(false));
api.addEventListener("execution_interrupted", () => onPromptDone(false));

app.registerExtension({
  name: "h3_motion_context.chain",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== CHAIN) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      injectCss();
      const root = document.createElement("div");
      root.className = "h3mc-chain";
      const row = document.createElement("div");
      row.className = "h3mc-chain-row";
      const row2 = document.createElement("div");
      row2.className = "h3mc-chain-row";
      const row3 = document.createElement("div");
      row3.className = "h3mc-chain-row";
      const approve = document.createElement("button");
      approve.textContent = "Approve";
      approve.title = "Advance Load/Save, then run the next clip.";
      const reroll = document.createElement("button");
      reroll.textContent = "Run/Re-roll";
      reroll.title = "Queue at the current Load/Save indices. Use this instead of ComfyUI's Run button.";
      const chainBtn = document.createElement("button");
      chainBtn.textContent = "Chain";
      chainBtn.title = "Approve on a loop. segments > 0 stops after that many clips; 0 runs until Stop.";
      const resetBtn = document.createElement("button");
      resetBtn.textContent = "Reset";
      resetBtn.title = "Set Load 0 / Save 1. Does not queue or delete files.";
      const clearBtn = document.createElement("button");
      clearBtn.textContent = "Clear latents";
      clearBtn.title = "Delete numbered chain slots (clip_00001.safetensors and so on). Custom filenames are left alone. Does not change indices or queue.";
      row.append(approve, reroll);
      row2.append(chainBtn, resetBtn);
      row3.append(clearBtn);
      const meta = document.createElement("div");
      meta.className = "h3mc-chain-meta";
      root.append(row, row2, row3, meta);
      swallow(root);
      this.addDOMWidget("h3mc_chain", "CHAIN", root, { serialize: false });
      this._h3mc = { chaining: false, awaiting: false, remaining: 0, meta, chainBtn };
      approve.onclick = async (e) => {
        e.stopPropagation();
        if (this._h3mc.awaiting) return;
        stopChain(this);
        if (!advance(this)) return;
        await queueOnce(this);
      };
      reroll.onclick = async (e) => {
        e.stopPropagation();
        if (this._h3mc.awaiting) return;
        stopChain(this);
        await queueOnce(this);
      };
      chainBtn.onclick = async (e) => {
        e.stopPropagation();
        if (this._h3mc.chaining) {
          stopChain(this);
          return;
        }
        if (this._h3mc.awaiting) return;
        if (!findPair(this)) {
          paint(this);
          return;
        }
        this._h3mc.chaining = true;
        this._h3mc.remaining = readSegments(this);
        paint(this);
        if (!await startChain(this)) stopChain(this);
      };
      resetBtn.onclick = (e) => {
        e.stopPropagation();
        if (this._h3mc.awaiting) return;
        stopChain(this);
        resetFirst(this);
      };
      clearBtn.onclick = async (e) => {
        e.stopPropagation();
        if (this._h3mc.awaiting || this._h3mc.chaining) return;
        const pair = findPair(this);
        if (!pair) {
          paint(this);
          return;
        }
        const n = await clearLatents(pair);
        paint(this);
        if (this._h3mc?.meta) {
          this._h3mc.meta.textContent = n
            ? `Removed ${n} numbered slot${n === 1 ? "" : "s"}. Custom names kept.`
            : "No numbered chain slots to remove.";
        }
      };
      paint(this);
      this.setSize?.([270, 168]);
      return r;
    };
  },
});
