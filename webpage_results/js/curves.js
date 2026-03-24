import { fetchText, parseCsv, DataLoadError } from "./data.js";

function seriesKey(row) {
  const d = row.qaoa_depth;
  const depth =
    d === null || d === undefined || d === "" || Number.isNaN(d) ? "" : ` / p=${d}`;
  return `${row.solver} / ${row.formulation} / n=${row.n_cities}${depth}`;
}

const palette = [
  "rgba(94, 184, 255, 1)",
  "rgba(105, 219, 124, 1)",
  "rgba(255, 180, 100, 1)",
  "rgba(255, 107, 107, 1)",
  "rgba(200, 150, 255, 1)",
  "rgba(255, 230, 120, 1)",
];

document.addEventListener("DOMContentLoaded", async () => {
  const Chart = globalThis.Chart;
  const errEl = document.getElementById("curves-error");
  const chartEl = document.getElementById("chart-curves");
  errEl.textContent = "";

  let rows;
  try {
    const text = await fetchText("processed/energy_curves_agg.csv");
    const parsed = parseCsv(text);
    rows = parsed.data.filter((r) => Object.keys(r).length > 0);
  } catch (e) {
    errEl.textContent =
      e instanceof DataLoadError
        ? `No hay curvas: ${e.message}. Necesitas historiales de energía en los JSON y metrics.`
        : String(e);
    return;
  }

  if (!rows.length) {
    errEl.textContent = "energy_curves_agg.csv está vacío.";
    return;
  }

  if (typeof Chart === "undefined" || !chartEl) {
    errEl.textContent = "Chart.js no cargado.";
    return;
  }

  Chart.defaults.color = "#e7ecf3";
  Chart.defaults.borderColor = "#2a3a4f";

  const byKey = new Map();
  for (const row of rows) {
    const k = seriesKey(row);
    if (!byKey.has(k)) byKey.set(k, []);
    byKey.get(k).push(row);
  }

  const datasets = [...byKey.entries()].map(([label, pts], i) => {
    const sorted = [...pts].sort((a, b) => Number(a.step) - Number(b.step));
    return {
      label,
      data: sorted.map((r) => ({ x: Number(r.step), y: Number(r.p50) })),
      borderColor: palette[i % palette.length],
      backgroundColor: "transparent",
      tension: 0.1,
      pointRadius: 0,
      borderWidth: 2,
    };
  });

  new Chart(chartEl, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        title: {
          display: true,
          text: "Energía mediana (p50) por paso del optimizador",
        },
        legend: {
          position: "bottom",
          labels: { boxWidth: 12, font: { size: 10 } },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "step" },
        },
        y: {
          title: { display: true, text: "p50 (unidades escaladas)" },
        },
      },
    },
  });
});
