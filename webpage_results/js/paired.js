import { fetchText, parseCsv, DataLoadError } from "./data.js";

document.addEventListener("DOMContentLoaded", async () => {
  const Tabulator = globalThis.Tabulator;
  const errEl = document.getElementById("paired-error");
  const tableHost = document.getElementById("table-paired");
  errEl.textContent = "";

  let rows;
  try {
    const text = await fetchText("processed/paired_metrics.csv");
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
    errEl.textContent = "paired_metrics.csv está vacío.";
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
    width: field === "path" ? 320 : undefined,
  }));

  const sortField = Object.prototype.hasOwnProperty.call(rows[0], "path")
    ? "path"
    : Object.keys(rows[0])[0];

  new Tabulator(tableHost, {
    data: rows,
    columns: cols,
    layout: "fitDataStretch",
    height: "70vh",
    pagination: "local",
    paginationSize: 50,
    initialSort: [{ column: sortField, dir: "asc" }],
  });
});
