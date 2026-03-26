import { outputUrl } from "./config.js";

const knownFigures = [
  [
    "img-energy",
    "images/energy_history_mean_cudaq_qubo_vs_tqudo_virtual_n5.png",
    "Historial de energía (media): CUDA-Q QUBO vs TQUDO virtual, n=5",
  ],
  [
    "img-energy-cirq-cuda",
    "images/energy_history_mean_cirq_tqudo_vs_cudaq_tvirt_n5.png",
    "Historial de energía (media): Cirq TQUDO vs CUDA-Q virtual, n=5",
  ],
  [
    "img-energy-cirq-n",
    "images/energy_history_mean_cirq_tqudo_by_ncities.png",
    "Historial de energía (media): Cirq TQUDO por n_cities",
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
