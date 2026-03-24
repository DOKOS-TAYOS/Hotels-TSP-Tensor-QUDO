import { outputUrl } from "./config.js";

const knownFigures = [
  [
    "img-feas",
    "images/feasibility_by_config.png",
    "Factibilidad de rutas vs configuración (solver × formulación × n)",
  ],
  [
    "img-violin",
    "images/approx_ratio_real_violin.png",
    "Distribución del ratio de aproximación sobre el coste real",
  ],
  [
    "img-energy",
    "images/energy_history_median.png",
    "Historial de energía (mediana) durante la optimización QAOA",
  ],
];

document.addEventListener("DOMContentLoaded", () => {
  for (const [id, rel, caption] of knownFigures) {
    const img = document.getElementById(id);
    const cap = document.querySelector(`figcaption[data-for="${id}"]`);
    if (!img) continue;
    const url = outputUrl(rel);
    img.src = `${url}?t=${Date.now()}`;
    img.alt = caption;
    img.onerror = () => {
      img.style.display = "none";
      if (cap) cap.textContent = `${caption} — no generada (ejecuta data_analysis.plot).`;
    };
    img.onload = () => {
      img.style.display = "";
      if (cap) cap.textContent = caption;
    };
  }
});
