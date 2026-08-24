"use strict";

const state = { profiles: [], selected: null, result: null, charts: [] };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const tooltip = $("#chart-tooltip");
const COLORS = { missile: "#f16a45", target: "#5ed2d6", actual: "#eac469", grid: "rgba(126,161,170,.14)", text: "#80979d" };
const COLORS_3D = { ground: "rgba(95,132,141,.16)", axisX: "#f16a45", axisY: "#61c995", axisZ: "#5ed2d6", stage: "#eac469", burnout: "#d58cff", termination: "#ff8576", connector: "rgba(255,133,118,.45)" };

const PRESETS = {
  head_on: { loft_enabled: false, observation_mode: "ideal_truth", target_course_reference: "statshark_relative_to_los", launch_speed_kmh: 1200, launch_altitude_m: 6500, launch_pitch_deg: 0, launch_heading_deg: 0, target_speed_kmh: 1200, target_altitude_m: 6500, initial_distance_m: 12000, target_azimuth_deg: 0, target_heading_deg: 0, target_vertical_heading_deg: 0, target_constant_turn_g: 0, max_simulation_time_s: null },
  off_axis_10: { loft_enabled: false, observation_mode: "ideal_truth", target_course_reference: "statshark_relative_to_los", launch_speed_kmh: 1200, launch_altitude_m: 6500, launch_pitch_deg: 0, launch_heading_deg: 0, target_speed_kmh: 1200, target_altitude_m: 6500, initial_distance_m: 12000, target_azimuth_deg: 10, target_heading_deg: 0, target_vertical_heading_deg: 0, target_constant_turn_g: 0, max_simulation_time_s: null },
  off_axis_38: { loft_enabled: false, observation_mode: "ideal_truth", target_course_reference: "statshark_relative_to_los", launch_speed_kmh: 1200, launch_altitude_m: 6500, launch_pitch_deg: 0, launch_heading_deg: 0, target_speed_kmh: 1200, target_altitude_m: 6500, initial_distance_m: 15000, target_azimuth_deg: 38, target_heading_deg: 0, target_vertical_heading_deg: 0, target_constant_turn_g: 0, max_simulation_time_s: null },
  off_axis_70: { loft_enabled: false, observation_mode: "ideal_truth", target_course_reference: "statshark_relative_to_los", launch_speed_kmh: 1200, launch_altitude_m: 6500, launch_pitch_deg: 0, launch_heading_deg: 0, target_speed_kmh: 1200, target_altitude_m: 6500, initial_distance_m: 8000, target_azimuth_deg: 70, target_heading_deg: 0, target_vertical_heading_deg: 0, target_constant_turn_g: 0, max_simulation_time_s: null },
  sensorwhale_pl12_off_axis_38: { loft_enabled: false, observation_mode: "ideal_truth", target_course_reference: "sensorwhale_launch_axis", launch_speed_kmh: 1200, launch_altitude_m: 6500, launch_pitch_deg: 0, launch_heading_deg: 0, target_speed_kmh: 1200, target_altitude_m: 6500, initial_distance_m: 15000, target_azimuth_deg: 38, target_heading_deg: 0, target_vertical_heading_deg: 0, target_constant_turn_g: 0, max_simulation_time_s: null },
  tail_chase: { loft_enabled: false, observation_mode: "ideal_truth", target_course_reference: "statshark_relative_to_los", launch_speed_kmh: 1200, launch_altitude_m: 6500, launch_pitch_deg: 0, launch_heading_deg: 0, target_speed_kmh: 900, target_altitude_m: 6500, initial_distance_m: 8000, target_azimuth_deg: 0, target_heading_deg: 180, target_vertical_heading_deg: 0, target_constant_turn_g: 0, max_simulation_time_s: null }
};

function showAlert(element, message) { element.textContent = message; element.classList.remove("hidden"); }
function hideAlert(element) { element.classList.add("hidden"); element.textContent = ""; }
function number(value, digits = 1) { return Number(value).toLocaleString("zh-CN", { maximumFractionDigits: digits, minimumFractionDigits: digits }); }
function magnitude(vector) { return Math.hypot(...vector.map(Number)); }

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload;
  try { payload = await response.json(); } catch (_) { throw new Error("本地服务返回了无法解析的响应。请查看终端日志。"); }
  if (!response.ok || payload.ok === false) throw new Error(payload.error?.message || `请求失败（HTTP ${response.status}）。`);
  return payload;
}

async function loadMissiles() {
  try {
    const payload = await api("/api/missiles");
    state.profiles = payload.missiles;
    const select = $("#missile-select");
    select.innerHTML = "";
    for (const profile of state.profiles) {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent = `${profile.name} · ${profile.country} / ${profile.series}`;
      select.appendChild(option);
    }
    if (payload.library_errors.length) showAlert($("#library-alert"), `部分导弹配置未加载：\n${payload.library_errors.join("\n")}`);
    if (!state.profiles.length) {
      showAlert($("#error-alert"), "导弹库为空或所有 JSON 均无效。请查看上方导弹库错误和终端日志。");
      $("#run-button").disabled = true;
      return;
    }
    const firstRunnable = state.profiles.find((item) => item.runnable) || state.profiles[0];
    select.value = firstRunnable.id;
    selectMissile(firstRunnable.id);
  } catch (error) {
    showAlert($("#error-alert"), `无法读取导弹库：${error.message}`);
    $("#run-button").disabled = true;
  }
}

function selectMissile(id) {
  state.selected = state.profiles.find((profile) => profile.id === id) || null;
  if (!state.selected) return;
  const profile = state.selected;
  $("#missile-name").textContent = `${profile.name} · ${profile.country} / ${profile.series}`;
  $("#missile-detail").textContent = profile.status_reason || profile.description;
  const badge = $("#status-badge");
  badge.textContent = profile.status;
  badge.className = "status-badge " + ({ "Validated": "validated", "Experimental": "experimental", "Unsupported physics": "unsupported" }[profile.status] || "");
  const facts = [];
  const p = profile.parameters || {};
  if (p.lifetime_s != null) facts.push(["寿命", `${p.lifetime_s} s`]);
  if (p.maximum_range_m != null) facts.push(["射程门限", `${number(p.maximum_range_m / 1000, 0)} km`]);
  if (p.engine_stages != null) facts.push(["发动机", `${p.engine_stages} 段`]);
  if (p.loft_enabled != null) facts.push(["Profile Loft 能力", p.loft_enabled ? "支持" : "不支持"]);
  facts.push(["模型", profile.runnable ? "可运行" : "不可运行"]);
  $("#model-facts").innerHTML = facts.map(([key, value]) => `<div><dt>${key}</dt><dd>${value}</dd></div>`).join("");
  $("#run-button").disabled = !profile.runnable;
  $("#run-caption").textContent = profile.runnable ? "公共 Python H2 模型层 + 所选导弹 JSON；本次运行不会写入实验目录。" : "此条目的物理类型尚未被公共 H2 计算核心支持，不能运行。";
  hideAlert($("#error-alert"));
}

function applyScenario(values) {
  for (const [key, value] of Object.entries(values)) {
    const input = $(`[name="${key}"]`);
    if (!input) continue;
    if (key === "loft_enabled" && input.type === "checkbox") input.checked = Boolean(value);
    else input.value = value == null ? "" : value;
  }
  if (!("loft_enabled" in values)) $("#loft-enabled-toggle").checked = false;
  if (values.target_course_reference) $("#target-course-reference-select").value = values.target_course_reference;
}

function collectScenario() {
  const form = $("#scenario-form");
  if (!form.checkValidity()) {
    const invalid = form.querySelector(":invalid");
    invalid?.focus();
    throw new Error("场景中存在缺失或超范围的输入，请检查标红字段。");
  }
  const data = new FormData(form);
  const scenario = {};
  for (const [key, value] of data.entries()) {
    if (key === "loft_enabled") continue;
    scenario[key] = ["observation_mode", "target_course_reference"].includes(key) ? value : (value === "" ? null : Number(value));
  }
  scenario.observation_mode = $("#observation-mode-select").value;
  scenario.target_course_reference = $("#target-course-reference-select").value;
  scenario.loft_enabled = $("#loft-enabled-toggle").checked;
  return scenario;
}

function setBusy(busy) {
  $("#progress").classList.toggle("hidden", !busy);
  $("#run-button").disabled = busy || !state.selected?.runnable;
  $("#run-button .run-label").textContent = busy ? "正在计算" : "Calculate";
}

async function runSimulation() {
  hideAlert($("#error-alert"));
  let scenario;
  try { scenario = collectScenario(); } catch (error) { showAlert($("#error-alert"), error.message); return; }
  setBusy(true);
  try {
    const payload = await api("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ missile_id: state.selected.id, scenario })
    });
    state.result = payload.result;
    renderResult(payload.result);
  } catch (error) {
    showAlert($("#error-alert"), error.message);
  } finally { setBusy(false); }
}

const EVENT_NAMES = { hit: "命中", proximity_fuse: "近炸", ground: "撞地", lifetime: "寿命", max_range: "射程", numerical_failure: "数值失败" };
function renderResult(result) {
  $("#empty-state").classList.add("hidden");
  $("#result-content").classList.remove("hidden");
  const s = result.summary;
  const scenarioLoftEnabled = s.scenario_loft_enabled ?? s.loft_enabled;
  const items = [
    ["终止事件", EVENT_NAMES[s.termination_event] || s.termination_event, "", true],
    ["飞行时间", number(s.flight_time_s, 2), "s"],
    ["末端距离", number(s.terminal_distance_m, 1), "m"],
    ["末端速度", number(s.terminal_speed_kmh, 0), "km/h"],
    ["末端高度", number(s.terminal_altitude_m, 0), "m"],
    ["最小距离", number(s.minimum_distance_m, 1), "m"],
    ["最大速度", number(s.maximum_speed_kmh, 0), "km/h"],
    ["最大高度", number(s.maximum_altitude_m, 0), "m"],
    ["最大指令 G", number(s.maximum_commanded_g, 2), "G"],
    ["最大实际 G", number(s.maximum_actual_g, 2), "G"],
    ["场景 Loft 开关", scenarioLoftEnabled ? "开启" : "关闭", ""],
    ["Loft 实际状态", s.loft_enabled ? "按现有门槛计算" : "未运行", ""],
    ["发动机燃尽", number(s.burnout_time_s, 2), "s"],
    ["观测模式", s.observation_mode || result.model?.observation_mode || "ideal_truth", ""],
    ["观测 Provider", s.observation_provider || result.model?.observation_provider || "—", ""],
    ["Track", s.track_mode || "—", ""],
    ["Seeker", s.seeker_display_state || s.seeker_state || "—", ""],
    ["最大航迹误差", s.maximum_track_error_m == null ? "—" : number(s.maximum_track_error_m, 1), "m"],
    ["首次雷达 Track", s.first_radar_track_time_s == null ? "—" : number(s.first_radar_track_time_s, 3), "s"],
    ["首次脱锁", s.first_lock_loss_time_s == null ? "—" : number(s.first_lock_loss_time_s, 3), "s"],
    ["首次复锁", s.first_reacquire_time_s == null ? "—" : number(s.first_reacquire_time_s, 3), "s"],
    ["最后拒绝原因", s.last_observation_reject_reason || s.last_radar_reject_reason || "—", ""]
  ];
  $("#summary-grid").innerHTML = items.map(([label, value, unit, primary]) => `<div class="summary-item${primary ? " primary" : ""}"><span>${label}</span><strong>${value}</strong>${unit ? `<small>${unit}</small>` : ""}</div>`).join("");
  buildCharts(result);
}

class CanvasChart {
  constructor(canvas, options) {
    this.canvas = canvas; this.options = options; this.ctx = canvas.getContext("2d"); this.drag = null; this.hover = null;
    this.padding = { left: 56, right: 18, top: 20, bottom: 42 };
    this.reset(); this.bind(); this.resize();
    this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(canvas.parentElement);
  }
  bounds() {
    const points = this.options.datasets.flatMap((dataset) => dataset.points);
    let xs = points.map((point) => point.x).filter(Number.isFinite), ys = points.map((point) => point.y).filter(Number.isFinite);
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const xp = Math.max((x1 - x0) * .05, 1e-6), yp = Math.max((y1 - y0) * .10, 1e-6);
    if (this.options.zeroY) y0 = Math.min(0, y0);
    return { x0: x0 - xp, x1: x1 + xp, y0: y0 - yp, y1: y1 + yp };
  }
  reset() { this.view = this.bounds(); this.draw?.(); }
  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr)); this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
    this.canvas.style.width = `${rect.width}px`; this.canvas.style.height = `${rect.height}px`; this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); this.width = rect.width; this.height = rect.height; this.draw();
  }
  area() { return { left: this.padding.left, right: this.width - this.padding.right, top: this.padding.top, bottom: this.height - this.padding.bottom }; }
  sx(x) { const a = this.area(); return a.left + (x - this.view.x0) / (this.view.x1 - this.view.x0) * (a.right - a.left); }
  sy(y) { const a = this.area(); return a.bottom - (y - this.view.y0) / (this.view.y1 - this.view.y0) * (a.bottom - a.top); }
  ux(px) { const a = this.area(); return this.view.x0 + (px - a.left) / (a.right - a.left) * (this.view.x1 - this.view.x0); }
  uy(py) { const a = this.area(); return this.view.y1 - (py - a.top) / (a.bottom - a.top) * (this.view.y1 - this.view.y0); }
  formatTick(value) { const abs = Math.abs(value); if (abs >= 10000) return `${(value / 1000).toFixed(0)}k`; if (abs >= 1000) return `${(value / 1000).toFixed(1)}k`; if (abs < .1 && abs !== 0) return value.toExponential(1); return value.toFixed(abs < 10 ? 1 : 0); }
  draw() {
    if (!this.width || !this.options.datasets.length) return;
    const ctx = this.ctx, a = this.area(); ctx.clearRect(0, 0, this.width, this.height); ctx.font = "10px ui-monospace, monospace"; ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const x = a.left + (a.right - a.left) * i / 5, y = a.top + (a.bottom - a.top) * i / 5;
      ctx.strokeStyle = COLORS.grid; ctx.beginPath(); ctx.moveTo(x, a.top); ctx.lineTo(x, a.bottom); ctx.stroke(); ctx.beginPath(); ctx.moveTo(a.left, y); ctx.lineTo(a.right, y); ctx.stroke();
      ctx.fillStyle = COLORS.text; ctx.textAlign = "center"; ctx.fillText(this.formatTick(this.view.x0 + (this.view.x1 - this.view.x0) * i / 5), x, a.bottom + 18);
      ctx.textAlign = "right"; ctx.fillText(this.formatTick(this.view.y1 - (this.view.y1 - this.view.y0) * i / 5), a.left - 8, y + 3);
    }
    ctx.fillStyle = COLORS.text; ctx.textAlign = "center"; ctx.fillText(this.options.xLabel, (a.left + a.right) / 2, this.height - 8);
    ctx.save(); ctx.translate(12, (a.top + a.bottom) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(this.options.yLabel, 0, 0); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(a.left, a.top, a.right - a.left, a.bottom - a.top); ctx.clip();
    if (this.options.markers) for (const marker of this.options.markers) {
      const x = this.sx(marker.time_s); ctx.strokeStyle = marker.kind === "termination" ? "rgba(241,106,69,.8)" : "rgba(234,196,105,.55)"; ctx.setLineDash(marker.kind === "termination" ? [] : [4, 4]); ctx.beginPath(); ctx.moveTo(x, a.top); ctx.lineTo(x, a.bottom); ctx.stroke(); ctx.setLineDash([]);
    }
    for (const dataset of this.options.datasets) {
      ctx.strokeStyle = dataset.color; ctx.lineWidth = dataset.width || 1.7; ctx.beginPath(); let started = false;
      for (const point of dataset.points) { const x = this.sx(point.x), y = this.sy(point.y); if (!Number.isFinite(x + y)) continue; if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y); }
      ctx.stroke();
    }
    ctx.restore();
    let legendX = a.left + 8;
    for (const dataset of this.options.datasets) { ctx.fillStyle = dataset.color; ctx.fillRect(legendX, a.top + 5, 13, 2); ctx.fillStyle = "#aebfc2"; ctx.textAlign = "left"; ctx.fillText(dataset.label, legendX + 18, a.top + 9); legendX += ctx.measureText(dataset.label).width + 42; }
    if (this.hover) { ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(this.hover.px, this.hover.py, 3, 0, Math.PI * 2); ctx.fill(); }
  }
  bind() {
    this._listeners = [];
    const on = (type, handler, options) => { this.canvas.addEventListener(type, handler, options); this._listeners.push([type, handler]); };
    on("wheel", (event) => { event.preventDefault(); const rect = this.canvas.getBoundingClientRect(), px = event.clientX - rect.left, py = event.clientY - rect.top, cx = this.ux(px), cy = this.uy(py), factor = event.deltaY > 0 ? 1.16 : .86; this.view = { x0: cx + (this.view.x0 - cx) * factor, x1: cx + (this.view.x1 - cx) * factor, y0: cy + (this.view.y0 - cy) * factor, y1: cy + (this.view.y1 - cy) * factor }; this.draw(); }, { passive: false });
    on("pointerdown", (event) => { this.canvas.setPointerCapture(event.pointerId); this.drag = { x: event.clientX, y: event.clientY, view: { ...this.view } }; this.canvas.classList.add("dragging"); });
    on("pointermove", (event) => { if (this.drag) { const a = this.area(), dx = (event.clientX - this.drag.x) / (a.right - a.left) * (this.drag.view.x1 - this.drag.view.x0), dy = (event.clientY - this.drag.y) / (a.bottom - a.top) * (this.drag.view.y1 - this.drag.view.y0); this.view = { x0: this.drag.view.x0 - dx, x1: this.drag.view.x1 - dx, y0: this.drag.view.y0 + dy, y1: this.drag.view.y1 + dy }; tooltip.classList.add("hidden"); this.draw(); } else this.showHover(event); });
    on("pointerup", () => { this.drag = null; this.canvas.classList.remove("dragging"); });
    on("pointerleave", () => { this.hover = null; tooltip.classList.add("hidden"); this.draw(); });
    on("dblclick", () => this.reset());
  }
  showHover(event) {
    const rect = this.canvas.getBoundingClientRect(), mx = event.clientX - rect.left, my = event.clientY - rect.top; let best = null;
    for (const dataset of this.options.datasets) for (const point of dataset.points) { const px = this.sx(point.x), py = this.sy(point.y), distance = Math.hypot(px - mx, py - my); if (!best || distance < best.distance) best = { dataset, point, px, py, distance }; }
    if (!best || best.distance > 32) { this.hover = null; tooltip.classList.add("hidden"); this.draw(); return; }
    this.hover = best; tooltip.textContent = `${best.dataset.label}\n${this.options.xLabel}: ${number(best.point.x, 2)}\n${this.options.yLabel}: ${number(best.point.y, 2)}${best.point.time != null ? `\n时间: ${number(best.point.time, 2)} s` : ""}`; tooltip.style.left = `${Math.min(window.innerWidth - 270, event.clientX + 14)}px`; tooltip.style.top = `${Math.min(window.innerHeight - 100, event.clientY + 14)}px`; tooltip.classList.remove("hidden"); this.draw();
  }
  destroy() { this.observer.disconnect(); for (const [type, handler] of this._listeners || []) this.canvas.removeEventListener(type, handler); this._listeners = []; }
}

class Trajectory3D {
  constructor(canvas, result) {
    this.canvas = canvas; this.ctx = canvas.getContext("2d"); this.result = result; this.drag = null; this.projectedHits = [];
    this.camera = { yaw: -0.72, pitch: 0.38, distance: 3.25, panX: 0, panY: -.04 };
    this.defaultCamera = { ...this.camera };
    this.samples = result.samples;
    this.paths = [
      { label: "导弹", color: COLORS.missile, points: this.samples.map((s) => this.makePoint(s.position_m, s, "missile")) },
      { label: "目标", color: COLORS.target, points: this.samples.map((s) => this.makePoint(s.target_position_m, s, "target")) }
    ];
    this.computeWorld(); this.markers = this.makeMarkers(); this.bind(); this.resize();
    this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(canvas.parentElement);
  }
  makePoint(position, sample, kind) { return { x: Number(position[0]), y: Number(position[1]), z: Number(position[2]), sample, kind }; }
  computeWorld() {
    const all = this.paths.flatMap((path) => path.points), xs = all.map((p) => p.x), ys = all.map((p) => p.y).concat([0]), zs = all.map((p) => p.z);
    this.world = { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys), minZ: Math.min(...zs), maxZ: Math.max(...zs) };
    this.world.centerX = (this.world.minX + this.world.maxX) / 2; this.world.centerY = (this.world.minY + this.world.maxY) / 2; this.world.centerZ = (this.world.minZ + this.world.maxZ) / 2;
    this.world.span = Math.max(this.world.maxX - this.world.minX, this.world.maxY - this.world.minY, this.world.maxZ - this.world.minZ, 1);
  }
  nearestSample(time) { return this.samples.reduce((best, sample) => Math.abs(sample.time_s - time) < Math.abs(best.time_s - time) ? sample : best, this.samples[0]); }
  makeMarkers() {
    return this.result.markers.map((marker) => {
      const sample = this.nearestSample(marker.time_s), position = sample.position_m;
      return { ...this.makePoint(position, sample, marker.kind), label: marker.label, markerKind: marker.kind, time_s: marker.time_s };
    });
  }
  reset() { this.camera = { ...this.defaultCamera }; this.draw(); }
  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr)); this.canvas.height = Math.max(1, Math.round(rect.height * dpr)); this.canvas.style.width = `${rect.width}px`; this.canvas.style.height = `${rect.height}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); this.width = rect.width; this.height = rect.height; this.draw();
  }
  normalize(point) { return { x: (point.x - this.world.centerX) / this.world.span, y: (point.y - this.world.centerY) / this.world.span, z: (point.z - this.world.centerZ) / this.world.span }; }
  project(point) {
    const p = this.normalize(point), cy = Math.cos(this.camera.yaw), sy = Math.sin(this.camera.yaw), cp = Math.cos(this.camera.pitch), sp = Math.sin(this.camera.pitch);
    const x1 = cy * p.x - sy * p.z, z1 = sy * p.x + cy * p.z, y2 = cp * p.y - sp * z1, z2 = sp * p.y + cp * z1;
    const depth = Math.max(.18, this.camera.distance - z2), perspective = 2.2 / depth, scale = Math.min(this.width, this.height) * .82;
    return { x: this.width / 2 + (x1 + this.camera.panX) * scale * perspective, y: this.height / 2 - (y2 + this.camera.panY) * scale * perspective, depth, visible: depth > .18 };
  }
  niceStep(raw) { const power = 10 ** Math.floor(Math.log10(Math.max(raw, 1))), scaled = raw / power; return (scaled < 2 ? 2 : scaled < 5 ? 5 : 10) * power; }
  gridRange(minimum, maximum, step) { const values = []; for (let value = Math.floor(minimum / step) * step; value <= Math.ceil(maximum / step) * step + step * .1; value += step) values.push(value); return values; }
  drawLine3D(a, b, color, width = 1, dash = []) {
    const p1 = this.project(a), p2 = this.project(b); if (!p1.visible || !p2.visible) return; const ctx = this.ctx; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke(); ctx.setLineDash([]);
  }
  drawGround() {
    const step = this.niceStep(this.world.span / 8), xs = this.gridRange(this.world.minX, this.world.maxX, step), zs = this.gridRange(this.world.minZ, this.world.maxZ, step);
    for (const x of xs) this.drawLine3D({x,y:0,z:zs[0]}, {x,y:0,z:zs[zs.length-1]}, COLORS_3D.ground);
    for (const z of zs) this.drawLine3D({x:xs[0],y:0,z}, {x:xs[xs.length-1],y:0,z}, COLORS_3D.ground);
    const origin = {x:0,y:0,z:0}, axisLength = step * 1.35;
    this.drawLine3D(origin, {x:axisLength,y:0,z:0}, COLORS_3D.axisX, 2); this.drawLine3D(origin, {x:0,y:axisLength,z:0}, COLORS_3D.axisY, 2); this.drawLine3D(origin, {x:0,y:0,z:axisLength}, COLORS_3D.axisZ, 2);
    const ctx = this.ctx;
    for (const [end, label, color] of [[{x:axisLength,y:0,z:0},"X",COLORS_3D.axisX],[{x:0,y:axisLength,z:0},"高度",COLORS_3D.axisY],[{x:0,y:0,z:axisLength},"Z",COLORS_3D.axisZ]]) { const p = this.project(end); ctx.fillStyle = color; ctx.font = "600 10px ui-monospace, monospace"; ctx.fillText(label, p.x + 5, p.y - 5); }
  }
  drawPath(path) {
    const ctx = this.ctx; ctx.strokeStyle = path.color; ctx.lineWidth = 2.2; ctx.beginPath(); let started = false;
    for (let index = 0; index < path.points.length; index++) { const point = path.points[index], p = this.project(point); if (!p.visible) continue; if (!started) { ctx.moveTo(p.x,p.y); started=true; } else ctx.lineTo(p.x,p.y); if (index % 2 === 0) this.projectedHits.push({x:p.x,y:p.y,point,label:path.label,priority:1}); }
    ctx.stroke();
  }
  drawMarker(marker) {
    const p = this.project(marker); if (!p.visible) return; const ctx = this.ctx, kind = marker.markerKind, color = kind === "stage" ? COLORS_3D.stage : kind === "burnout" ? COLORS_3D.burnout : COLORS_3D.termination;
    ctx.save(); ctx.translate(p.x,p.y); ctx.fillStyle = color; ctx.strokeStyle = "rgba(7,16,20,.95)"; ctx.lineWidth = 2; ctx.beginPath();
    if (kind === "stage") { ctx.rotate(Math.PI/4); ctx.rect(-5,-5,10,10); }
    else if (kind === "burnout") ctx.rect(-5,-5,10,10);
    else ctx.arc(0,0,6,0,Math.PI*2);
    ctx.fill(); ctx.stroke(); ctx.restore();
    ctx.fillStyle = color; ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "left"; ctx.fillText(marker.label, p.x + 9, p.y - 8);
    this.projectedHits.push({x:p.x,y:p.y,point:marker,label:marker.label,priority:10});
  }
  drawConnector() {
    const terminal = this.samples[this.samples.length-1]; this.drawLine3D({x:terminal.position_m[0],y:terminal.position_m[1],z:terminal.position_m[2]}, {x:terminal.target_position_m[0],y:terminal.target_position_m[1],z:terminal.target_position_m[2]}, COLORS_3D.connector, 1.3, [4,4]);
  }
  drawOrientation() {
    const ctx = this.ctx; ctx.fillStyle = "rgba(7,16,20,.72)"; ctx.fillRect(12,12,122,25); ctx.fillStyle = COLORS.text; ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "left"; ctx.fillText(`方位 ${Math.round(this.camera.yaw*180/Math.PI)}°  仰角 ${Math.round(this.camera.pitch*180/Math.PI)}°`,20,28);
  }
  draw() {
    if (!this.width) return; const ctx = this.ctx; ctx.clearRect(0,0,this.width,this.height); this.projectedHits = []; this.drawGround();
    const ordered = [...this.paths].sort((a,b) => this.project(a.points[Math.floor(a.points.length/2)]).depth - this.project(b.points[Math.floor(b.points.length/2)]).depth).reverse();
    for (const path of ordered) this.drawPath(path); this.drawConnector(); for (const marker of this.markers) this.drawMarker(marker); this.drawOrientation();
  }
  showHover(event) {
    const rect = this.canvas.getBoundingClientRect(), x = event.clientX-rect.left, y=event.clientY-rect.top;
    const candidates = this.projectedHits.map((hit) => ({...hit,distance:Math.hypot(hit.x-x,hit.y-y)})).filter((hit) => hit.distance <= 22).sort((a,b) => b.priority-a.priority || a.distance-b.distance);
    if (!candidates.length) { tooltip.classList.add("hidden"); return; } const hit = candidates[0], sample = hit.point.sample;
    const position = hit.point.kind === "target" ? sample.target_position_m : sample.position_m, speed = magnitude(hit.point.kind === "target" ? sample.target_velocity_mps : sample.velocity_mps)*3.6;
    tooltip.textContent = `${hit.label}\n时间: ${number(sample.time_s,2)} s\nX / 高度 / Z: ${number(position[0],0)} / ${number(position[1],0)} / ${number(position[2],0)} m\n速度: ${number(speed,0)} km/h\n目标距离: ${number(sample.distance_to_target_m,1)} m`;
    tooltip.style.left = `${Math.min(window.innerWidth-270,event.clientX+14)}px`; tooltip.style.top = `${Math.min(window.innerHeight-120,event.clientY+14)}px`; tooltip.classList.remove("hidden");
  }
  bind() {
    this._listeners = [];
    const on = (type, handler, options) => { this.canvas.addEventListener(type, handler, options); this._listeners.push([type, handler]); };
    on("contextmenu", (event) => event.preventDefault());
    on("wheel", (event) => { event.preventDefault(); this.camera.distance = Math.max(1.35, Math.min(8, this.camera.distance * (event.deltaY > 0 ? 1.1 : .9))); tooltip.classList.add("hidden"); this.draw(); }, {passive:false});
    on("pointerdown", (event) => { this.canvas.setPointerCapture(event.pointerId); this.drag={x:event.clientX,y:event.clientY,camera:{...this.camera},mode:event.shiftKey||event.button===2?"pan":"rotate"}; this.canvas.classList.add("dragging"); tooltip.classList.add("hidden"); });
    on("pointermove", (event) => { if (!this.drag) { this.showHover(event); return; } const dx=event.clientX-this.drag.x,dy=event.clientY-this.drag.y; if(this.drag.mode==="rotate"){this.camera.yaw=this.drag.camera.yaw+dx*.007;this.camera.pitch=Math.max(-1.35,Math.min(1.35,this.drag.camera.pitch+dy*.007));}else{this.camera.panX=this.drag.camera.panX+dx/Math.min(this.width,this.height)*1.2;this.camera.panY=this.drag.camera.panY-dy/Math.min(this.width,this.height)*1.2;} this.draw(); });
    const end=()=>{this.drag=null;this.canvas.classList.remove("dragging");};
    on("pointerup", end); on("pointercancel", end);
    on("pointerleave", ()=>{if(!this.drag)tooltip.classList.add("hidden");});
    on("dblclick", ()=>this.reset());
    on("keydown", (event)=>{let used=true;if(event.key==="ArrowLeft")this.camera.yaw-=.08;else if(event.key==="ArrowRight")this.camera.yaw+=.08;else if(event.key==="ArrowUp")this.camera.pitch-=.08;else if(event.key==="ArrowDown")this.camera.pitch+=.08;else if(event.key==="+"||event.key==="=")this.camera.distance=Math.max(1.35,this.camera.distance*.9);else if(event.key==="-")this.camera.distance=Math.min(8,this.camera.distance*1.1);else if(event.key==="Home")this.reset();else used=false;if(used){event.preventDefault();this.draw();}});
  }
  destroy(){ this.observer.disconnect(); for (const [type, handler] of this._listeners || []) this.canvas.removeEventListener(type, handler); this._listeners = []; }
}

function point(x, y, time = null) { return { x: Number(x), y: Number(y), time }; }
function buildCharts(result) {
  for (const chart of state.charts) chart.destroy(); state.charts = [];
  const samples = result.samples;
  const missileHorizontal = samples.map((s) => point(s.position_m[0], s.position_m[2], s.time_s));
  const targetHorizontal = samples.map((s) => point(s.target_position_m[0], s.target_position_m[2], s.time_s));
  const missileProfile = samples.map((s) => point(Math.hypot(s.position_m[0], s.position_m[2]), s.position_m[1], s.time_s));
  const targetProfile = samples.map((s) => point(Math.hypot(s.target_position_m[0], s.target_position_m[2]), s.target_position_m[1], s.time_s));
  const timeSeries = (getter) => samples.map((s) => point(s.time_s, getter(s)));
  const configs = [
    ["#chart-horizontal", { xLabel: "X 距离 (m)", yLabel: "Z 距离 (m)", datasets: [{ label: "导弹", color: COLORS.missile, points: missileHorizontal }, { label: "目标", color: COLORS.target, points: targetHorizontal }] }],
    ["#chart-profile", { xLabel: "水平距离 (m)", yLabel: "高度 (m)", datasets: [{ label: "导弹", color: COLORS.missile, points: missileProfile }, { label: "目标", color: COLORS.target, points: targetProfile }] }],
    ["#chart-speed", { xLabel: "时间 (s)", yLabel: "速度 (km/h)", zeroY: true, markers: result.markers, datasets: [{ label: "导弹", color: COLORS.missile, points: timeSeries((s) => magnitude(s.velocity_mps) * 3.6) }, { label: "目标", color: COLORS.target, points: timeSeries((s) => magnitude(s.target_velocity_mps) * 3.6) }] }],
    ["#chart-altitude", { xLabel: "时间 (s)", yLabel: "高度 (m)", zeroY: true, markers: result.markers, datasets: [{ label: "导弹", color: COLORS.missile, points: timeSeries((s) => s.position_m[1]) }, { label: "目标", color: COLORS.target, points: timeSeries((s) => s.target_position_m[1]) }] }],
    ["#chart-distance", { xLabel: "时间 (s)", yLabel: "距离目标 (m)", zeroY: true, markers: result.markers, datasets: [{ label: "相对距离", color: COLORS.cyan || COLORS.target, points: timeSeries((s) => s.distance_to_target_m) }] }],
    ["#chart-g", { xLabel: "时间 (s)", yLabel: "过载 (G)", zeroY: true, markers: result.markers, datasets: [{ label: "指令 G", color: COLORS.missile, points: timeSeries((s) => Math.hypot(...s.commanded_acceleration_g)) }, { label: "实际 G", color: COLORS.actual, points: timeSeries((s) => s.actual_overload_g ?? s.trajectory_lateral_load_g) }] }]
  ];
  state.charts.push(new Trajectory3D($("#chart-3d"), result));
  for (const [selector, options] of configs) state.charts.push(new CanvasChart($(selector), options));
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type })), link = document.createElement("a"); link.href = url; link.download = name; document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function stamp() { return new Date().toISOString().replace(/[:.]/g, "-"); }
function exportScenario() {
  try { download(`scenario-${stamp()}.json`, JSON.stringify({ schema_version: 1, missile_id: state.selected?.id, preset: $("#preset-select").value, scenario: collectScenario() }, null, 2), "application/json"); }
  catch (error) { showAlert($("#error-alert"), error.message); }
}
async function importScenario(file) {
  try {
    const parsed = JSON.parse(await file.text()), scenario = parsed.scenario || parsed;
    if (!scenario || typeof scenario !== "object" || Array.isArray(scenario)) throw new Error("场景 JSON 顶层必须是对象。 ");
    applyScenario(scenario); $("#preset-select").value = "custom";
    if (parsed.missile_id && state.profiles.some((p) => p.id === parsed.missile_id)) { $("#missile-select").value = parsed.missile_id; selectMissile(parsed.missile_id); }
    hideAlert($("#error-alert"));
  } catch (error) { showAlert($("#error-alert"), `无法导入场景：${error.message}`); }
}
function exportCsv() {
  if (!state.result) return; const headers = ["time_s","missile_x_m","missile_altitude_m","missile_z_m","target_x_m","target_altitude_m","target_z_m","missile_speed_kmh","target_speed_kmh","distance_to_target_m","commanded_g","actual_g","mach","thrust_n","drag_n","mass_kg","loft_active"];
  const rows = state.result.samples.map((s) => [s.time_s,...s.position_m,...s.target_position_m,magnitude(s.velocity_mps)*3.6,magnitude(s.target_velocity_mps)*3.6,s.distance_to_target_m,Math.hypot(...s.commanded_acceleration_g),s.actual_overload_g,s.mach,s.thrust_n,s.drag_n,s.mass_kg,s.loft_active]);
  download(`trajectory-${state.result.missile.id}-${stamp()}.csv`, [headers, ...rows].map((row) => row.join(",")).join("\n") + "\n", "text/csv;charset=utf-8");
}

$("#missile-select").addEventListener("change", (event) => selectMissile(event.target.value));
$("#preset-select").addEventListener("change", (event) => { if (PRESETS[event.target.value]) applyScenario(PRESETS[event.target.value]); });
$("#scenario-form").addEventListener("input", () => { if ($("#preset-select").value !== "custom") $("#preset-select").value = "custom"; });
$("#run-button").addEventListener("click", runSimulation);
$("#reset-view").addEventListener("click", () => state.charts.forEach((chart) => chart.reset()));
$("#reset-3d").addEventListener("click", () => state.charts.find((chart) => chart instanceof Trajectory3D)?.reset());
$("#export-scenario").addEventListener("click", exportScenario);
$("#import-scenario").addEventListener("click", () => $("#scenario-file").click());
$("#scenario-file").addEventListener("change", (event) => { const file = event.target.files[0]; if (file) importScenario(file); event.target.value = ""; });
$("#export-result").addEventListener("click", () => state.result && download(`result-${state.result.missile.id}-${stamp()}.json`, JSON.stringify(state.result, null, 2), "application/json"));
$("#export-csv").addEventListener("click", exportCsv);

applyScenario(PRESETS.head_on);
loadMissiles();
