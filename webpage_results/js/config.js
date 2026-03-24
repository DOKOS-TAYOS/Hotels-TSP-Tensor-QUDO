/**
 * Base URL for output artifacts (processed/, images/, T0sampling/, etc.).
 * Override with ?output=/path/to/output/ (must end with slash or it will be added).
 */
export function getOutputBase() {
  const params = new URLSearchParams(window.location.search);
  const q = params.get("output");
  if (q) {
    const trimmed = q.trim();
    return trimmed.endsWith("/") ? trimmed : `${trimmed}/`;
  }
  return "../output/";
}

/** Resolve a path under the output root (e.g. "processed/summary_by_config.csv"). */
export function outputUrl(path) {
  const base = getOutputBase();
  const p = path.replace(/^\/+/, "");
  return new URL(p, new URL(base, window.location.href)).href;
}
