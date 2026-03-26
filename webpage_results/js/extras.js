import {
  bustCache,
  fetchJson,
  fetchText,
  parseDirectoryJsonLinks,
  resourceExists,
} from "./data.js";
import { getOutputBase, outputUrl } from "./config.js";

async function fetchDirListing(relDir) {
  const url = bustCache(outputUrl(relDir));
  const r = await fetch(url);
  if (!r.ok) return { ok: false, links: [], status: r.status };
  const html = await r.text();
  return { ok: true, links: parseDirectoryJsonLinks(html), status: r.status };
}

document.addEventListener("DOMContentLoaded", async () => {
  const t0 = await fetchDirListing("T0sampling/");
  const hostT0 = document.getElementById("t0-list");
  if (hostT0) {
    if (!t0.ok) {
      hostT0.innerHTML = `<p class="muted">T0sampling/ no listado (${t0.status}). ¿Existe la carpeta?</p>`;
    } else if (!t0.links.length) {
      hostT0.innerHTML = "<p class=\"muted\">No hay .json en T0sampling/</p>";
    } else {
      hostT0.innerHTML = `<ul class="json-file-list">${t0.links
        .map(
          (name) =>
            `<li><button type="button" class="btn load-json" data-path="T0sampling/${name}">${name}</button></li>`,
        )
        .join("")}</ul>`;
    }
  }

  const lam = await fetchDirListing("lambdasSampling/");
  const hostL = document.getElementById("lambda-list");
  if (hostL) {
    if (!lam.ok) {
      hostL.innerHTML = `<p class="muted">lambdasSampling/ no listado (${lam.status}). ¿Existe la carpeta?</p>`;
    } else if (!lam.links.length) {
      hostL.innerHTML = "<p class=\"muted\">No hay .json en lambdasSampling/</p>";
    } else {
      hostL.innerHTML = `<ul class="json-file-list">${lam.links
        .map(
          (name) =>
            `<li><button type="button" class="btn load-json" data-path="lambdasSampling/${name}">${name}</button></li>`,
        )
        .join("")}</ul>`;
    }
  }

  const viewer = document.getElementById("json-picked");
  document.querySelectorAll("button.load-json").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const path = btn.getAttribute("data-path");
      if (!path || !viewer) return;
      viewer.textContent = "Cargando…";
      try {
        const data = await fetchJson(path);
        viewer.textContent = JSON.stringify(data, null, 2);
      } catch (e) {
        viewer.textContent = String(e);
      }
    });
  });

  const manifestNote = document.getElementById("manifest-note");
  if (manifestNote) {
    try {
      await fetchText("processed/manifest.csv");
      manifestNote.textContent = "manifest.csv disponible (puedes añadir una vista en el futuro).";
      manifestNote.className = "status-ok";
    } catch {
      if (await resourceExists("processed/manifest.parquet")) {
        manifestNote.textContent =
          "Solo manifest.parquet: el navegador no lo lee directamente; usa ingest --format csv para CSV.";
        manifestNote.className = "muted";
      } else {
        manifestNote.textContent = "No hay manifest en processed/.";
        manifestNote.className = "muted";
      }
    }
  }

  const baseEl = document.getElementById("extras-base");
  if (baseEl) {
    baseEl.textContent = `Raíz de datos: ${getOutputBase()}`;
  }
});
