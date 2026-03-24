document.addEventListener("DOMContentLoaded", () => {
  const el = document.getElementById("site-nav");
  if (!el) return;
  const pages = [
    ["index.html", "Inicio"],
    ["summary.html", "Resumen por config"],
    ["paired.html", "Paired metrics"],
    ["curves.html", "Curvas de energía"],
    ["figures.html", "Figuras PNG"],
    ["extras.html", "Extras / calibración"],
  ];
  const parts = window.location.pathname.split("/").filter(Boolean);
  let current = parts[parts.length - 1] ?? "index.html";
  if (!current.includes(".html")) {
    current = "index.html";
  }
  const lis = pages
    .map(([href, label]) => {
      const active = href === current ? ' class="active"' : "";
      return `<li${active}><a href="${href}">${label}</a></li>`;
    })
    .join("");
  el.innerHTML = `
    <ul class="nav-list">
      ${lis}
    </ul>
    <p class="nav-meta">
      <button type="button" id="btn-refresh" class="btn">Refrescar datos</button>
      <span id="nav-output-base" class="muted"></span>
    </p>
  `;
  const baseEl = el.querySelector("#nav-output-base");
  if (baseEl) {
    import("./config.js").then(({ getOutputBase }) => {
      baseEl.textContent = `output: ${getOutputBase()}`;
    });
  }
  const btn = el.querySelector("#btn-refresh");
  if (btn) {
    btn.addEventListener("click", () => {
      window.location.reload();
    });
  }
});
