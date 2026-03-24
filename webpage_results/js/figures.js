import { outputUrl } from "./config.js";

const knownFigures = [
  ["img-feas", "images/feasibility_by_config.png", "Feasibilidad por configuración (matplotlib)"],
  ["img-violin", "images/approx_ratio_real_violin.png", "approx_ratio_real (violin)"],
  ["img-energy", "images/energy_history_median.png", "Historial de energía (mediana)"],
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
