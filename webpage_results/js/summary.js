import { fetchText, parseCsv, DataLoadError } from "./data.js";

function labelRow(row) {
  const parts = [
    String(row.solver ?? ""),
    String(row.formulation ?? ""),
    `n=${row.n_cities ?? ""}`,
  ];
  if (row.qaoa_depth !== null && row.qaoa_depth !== undefined && row.qaoa_depth !== "") {
    parts.push(`p=${row.qaoa_depth}`);
  }
  return parts.join(" / ");
}

document.addEventListener("DOMContentLoaded", async () => {
  const Tabulator = globalThis.Tabulator;
  const Chart = globalThis.Chart;
  const errEl = document.getElementById("summary-error");
  const chartEl = document.getElementById("chart-feas");
  const tableHost = document.getElementById("table-summary");
  errEl.textContent = "";

  let rows;
  try {
    const text = await fetchText("processed/summary_by_config.csv");
    const parsed = parseCsv(text);
    rows = parsed.data.filter((r) => Object.keys(r).length > 0);
  } catch (e) {
    errEl.textContent =
      e instanceof DataLoadError
        ? `No hay datos: ${e.message}. Ejecuta data_analysis.metrics primero.`
        : String(e);
    return;
  }

  if (!rows.length) {
    errEl.textContent = "summary_by_config.csv está vacío.";
    return;
  }

  if (typeof Tabulator === "undefined") {
    errEl.textContent = "Tabulator no cargado.";
    return;
  }

  const cols = Object.keys(rows[0]).map((field) => ({
    title: field,
    field,
    headerFilter: "input",
  }));

  new Tabulator(tableHost, {
    data: rows,
    columns: cols,
    layout: "fitDataStretch",
    height: "420px",
    pagination: "local",
    paginationSize: 25,
  });

  if (typeof Chart === "undefined" || !chartEl) {
    return;
  }

  Chart.defaults.color = "#e8edf4";
  Chart.defaults.borderColor = "#2a3441";

  const withFeas = rows.filter((r) => r.feas_rate !== null && r.feas_rate !== undefined);
  if (!withFeas.length) {
    return;
  }

  const labels = withFeas.map(labelRow);
  const values = withFeas.map((r) => Number(r.feas_rate));

  new Chart(chartEl, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "feas_rate",
          data: values,
          backgroundColor: "rgba(78, 205, 196, 0.5)",
          borderColor: "rgba(78, 205, 196, 1)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: "Factibilidad de rutas (precedencias) por configuración experimental",
        },
      },
      scales: {
        x: {
          min: 0,
          max: 1.05,
          title: { display: true, text: "feas_rate (fracción de ejecuciones factibles)" },
        },
      },
    },
  });
});
