import { fetchText, DataLoadError, resourceExists } from "./data.js";

/**
 * @param {string} id
 * @param {boolean} ok
 * @param {string} text
 */
function line(id, ok, text) {
  const el = document.getElementById(id);
  if (!el) return;
  const msg = el.querySelector(".status-value");
  if (msg) {
    msg.className = ok ? "status-value status-ok" : "status-value status-bad";
    msg.textContent = text;
    return;
  }
  el.className = ok ? "status-ok" : "status-bad";
  el.textContent = text;
}

async function probe(path, label) {
  try {
    await fetchText(path);
    return { ok: true, msg: `${label}: OK` };
  } catch (e) {
    if (e instanceof DataLoadError && e.status === 404) {
      return { ok: false, msg: `${label}: no encontrado` };
    }
    return { ok: false, msg: `${label}: error (${e instanceof Error ? e.message : String(e)})` };
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  const r1 = await probe("processed/summary_by_config.csv", "summary_by_config.csv");
  line("st-summary", r1.ok, r1.msg);
  const r2 = await probe("processed/paired_metrics.csv", "paired_metrics.csv");
  line("st-paired", r2.ok, r2.msg);
  const r3 = await probe("processed/energy_curves_agg.csv", "energy_curves_agg.csv");
  line("st-curves", r3.ok, r3.msg);
  const imgOk = await resourceExists("images/energy_history_mean_cudaq_qubo_vs_tqudo_virtual_n5.png");
  line("st-images", imgOk, imgOk ? "figuras PNG (ej.): OK" : "figuras PNG (ej.): no encontrado");
});
