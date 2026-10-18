// Dashboard page test (jsdom): loads the real page against a running detector and checks what it renders
// in three states: live, finished, server gone. Needs node and `npm i jsdom` (installed anywhere node can resolve it).
//
//   python3 src/live_detector.py detect --pcap pcaps/test/mixed.pcap --model models/live_model.npz \
//           --dashboard 8099 --realtime --speed 4 &
//   DETECTOR_PID=$! node tests/dashboard_page_test.js
//
const { JSDOM, VirtualConsole } = require("jsdom");
const BASE = "http://localhost:8099/";
const sleep = ms => new Promise(r => setTimeout(r, ms));
const state = async () => (await fetch(BASE + "api/state")).json();
const errors = [];
let checks = 0, failed = 0;
function check(name, cond, extra = "") { checks++; if (!cond) failed++; console.log((cond ? "  PASS " : "  FAIL ") + name + (extra ? "  -> " + extra : "")); }

(async () => {
  // wait for the replay to be under way
  while ((await state()).windows < 10) await sleep(300);
  const vc = new VirtualConsole();
  vc.on("jsdomError", e => errors.push("jsdomError: " + e.message));
  vc.on("error", (...a) => errors.push("console.error: " + a.join(" ")));
  const dom = await JSDOM.fromURL(BASE, {
    runScripts: "dangerously", resources: "usable", pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(w) { w.fetch = (u, o) => fetch(new URL(u, BASE), o); }
  });
  const d = dom.window.document, $ = id => d.getElementById(id);
  await sleep(1800);

  console.log("--- phase 1: live (mid-replay) ---");
  console.log("  pill:", $("pill").textContent, "| status:", $("slabel").textContent, "|", $("sdetail").textContent);
  check("pill shows LIVE", $("pill").textContent === "LIVE");
  check("status card has a class state", /status (ok|warn|alert)/.test($("status").className), $("status").className);
  check("packets KPI > 0", parseInt($("k-packets").textContent.replace(/,/g, "")) > 0, $("k-packets").textContent);
  check("timeline drew bars", d.querySelectorAll("#chart svg rect").length > 5, d.querySelectorAll("#chart svg rect").length + " rects");
  check("legend lists classes", d.querySelectorAll("#legend span").length >= 2, $("legend").textContent);
  check("class bars rendered", d.querySelectorAll("#bars .barrow").length >= 1);

  // tooltip interaction
  const svg = $("svg");
  svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 1000, height: 170 });
  svg.onmousemove({ clientX: 990, clientY: 50 });
  check("tooltip shows window details", !$("tip").hidden && /t\+/.test($("tip").textContent), $("tip").textContent.slice(0, 50));

  console.log("--- phase 2: replay finished ---");
  while (!(await state()).finished) await sleep(400);
  await sleep(1600);
  console.log("  pill:", $("pill").textContent, "| alerts KPI:", $("k-alerts").textContent, "| alert rows:", d.querySelectorAll("#alerts tr").length - 1);
  check("pill shows FINISHED", $("pill").textContent === "FINISHED");
  check("alert log has the 3 episodes", d.querySelectorAll("#alerts tr").length - 1 === 3);
  const labels = [...d.querySelectorAll("#alerts tr")].slice(1).map(r => r.children[1].textContent).reverse();
  check("alert order = syn flood, udp flood, port scan", labels.join("|") === "syn flood|udp flood|port scan", labels.join(" | "));
  check("alert episodes KPI = 3", $("k-alerts").textContent === "3");

  console.log("--- phase 3: server goes away ---");
  process.kill(parseInt(process.env.DETECTOR_PID));
  await sleep(2500);
  console.log("  pill:", $("pill").textContent);
  check("pill shows OFFLINE", $("pill").textContent === "OFFLINE");

  check("no JavaScript errors during the whole run", errors.length === 0, errors.join("; "));
  console.log(`\n${checks - failed}/${checks} checks passed`);
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error("test crashed:", e); process.exit(2); });
