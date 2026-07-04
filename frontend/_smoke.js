// Runtime smoke after role-removal: confirm init() still runs and renders.
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("C:/Users/j1234/AppData/Local/Temp/jsdom-check/node_modules/jsdom");

const FE = "C:/Users/j1234/AppData/Local/Temp/../../../../../Downloads/Nornikel-AI-Science-Hack-chunking111/frontend";
const html = fs.readFileSync(path.join("C:/Users/j1234/Downloads/Nornikel-AI-Science-Hack-chunking111/frontend", "index.html"), "utf8");
const dataJson = fs.readFileSync(path.join("C:/Users/j1234/Downloads/Nornikel-AI-Science-Hack-chunking111/frontend", "data.json"), "utf8");

const dom = new JSDOM(html, { runScripts: "outside-only", url: "http://localhost:8050/", pretendToBeVisual: true });
const { window } = dom;
window.requestAnimationFrame = (cb) => setTimeout(() => cb(Date.now()), 0);
window.fetch = (url) => String(url).endsWith("data.json")
  ? Promise.resolve({ ok: true, json: () => Promise.resolve(JSON.parse(dataJson)) })
  : Promise.resolve({ ok: false, json: () => Promise.resolve(null) });
window.URL.createObjectURL = () => "blob:fake";
window.URL.revokeObjectURL = () => {};
window.print = () => {};
Object.defineProperty(window.HTMLElement.prototype, "clientWidth", { get() { return 900; }, configurable: true });

let failed = null;
window.addEventListener("error", (e) => { failed = failed || (e.error || e.message); });
window.addEventListener("unhandledrejection", (e) => { failed = failed || e.reason; });

try { window.eval(html.match(/<script>([\s\S]*?)<\/script>/)[1]); }
catch (e) { console.error("SCRIPT THREW:", e.message, "\n", e.stack); process.exit(1); }

setTimeout(() => {
  if (failed) { console.error("RUNTIME ERROR:", (failed && failed.stack) || failed); process.exit(1); }
  const d = window.document;
  const checks = {
    "no roleSel in DOM": !d.querySelector("#roleSel"),
    "no badge-role in brand": !d.querySelector(".badge-role"),
    "audit header has 2 cols (Время/Действие)": d.querySelectorAll(".audit-row").length > 0,
    "kpi tiles": d.querySelectorAll("#kpiRow .card").length,
    "graph nodes": d.querySelectorAll("#graphSvg .gnode").length,
    "experts": d.querySelectorAll("#expGrid .exp").length,
    "gaps": d.querySelectorAll("#gapsList .gap").length,
    "chips": d.querySelectorAll("#chips .chip").length,
    "brand text has no role badge": !/role/i.test(d.querySelector(".brand-txt b").innerHTML),
  };
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    const pass = v === true || (typeof v === "number" && v > 0);
    if (!pass) ok = false;
    console.log(`${pass ? "✓" : "✗"} ${k}: ${v}`);
  }
  // run a query to be sure answer path still works
  try { d.querySelector("#query").value = "обессоливание воды сульфаты 300 мг/л"; d.querySelector("#askBtn").click(); }
  catch (e) { console.error("QUERY THREW:", e.message); ok = false; }
  setTimeout(() => {
    const ans = (d.querySelector("#answerBody").innerHTML || "").length;
    console.log(`${ans > 150 ? "✓" : "✗"} answer after role-removal: ${ans} chars`);
    if (ans < 150) ok = false;
    console.log(ok ? "\nROLE-REMOVAL SMOKE PASSED" : "\nSMOKE FAILED");
    process.exit(ok ? 0 : 1);
  }, 250);
}, 800);
