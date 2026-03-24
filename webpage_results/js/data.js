/**
 * Fetch helpers for CSV/JSON under the output directory (requires HTTP server; not file://).
 */

import { outputUrl } from "./config.js";

export class DataLoadError extends Error {
  /** @param {string} message */
  /** @param {number} [status] */
  constructor(message, status) {
    super(message);
    this.name = "DataLoadError";
    this.status = status;
  }
}

/** @param {string} path */
export function bustCache(path) {
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}t=${Date.now()}`;
}

/**
 * @param {string} relativePath
 * @returns {Promise<string>}
 */
export async function fetchText(relativePath) {
  const url = bustCache(outputUrl(relativePath));
  const r = await fetch(url);
  if (!r.ok) {
    throw new DataLoadError(`No se pudo cargar ${relativePath} (${r.status})`, r.status);
  }
  return r.text();
}

/**
 * @param {string} relativePath
 * @returns {Promise<object>}
 */
export async function fetchJson(relativePath) {
  const text = await fetchText(relativePath);
  return JSON.parse(text);
}

/** True if the resource exists (uses HEAD to avoid downloading large files). */
export async function resourceExists(relativePath) {
  const url = bustCache(outputUrl(relativePath));
  let r = await fetch(url, { method: "HEAD" });
  if (r.status === 405 || r.status === 501) {
    r = await fetch(url, { method: "GET" });
  }
  return r.ok;
}

/**
 * Parse CSV text with Papa Parse (must be loaded globally as Papa).
 * @param {string} text
 * @returns {import('papaparse').ParseResult<Record<string, unknown>>}
 */
export function parseCsv(text) {
  if (typeof Papa === "undefined") {
    throw new Error("Papa Parse no está cargado");
  }
  return Papa.parse(text, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true,
  });
}

/**
 * @param {string} html
 * @returns {string[]}
 */
export function parseDirectoryJsonLinks(html) {
  const out = new Set();
  const re = /href="([^"]+\.json)"/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    out.add(m[1]);
  }
  return [...out];
}
