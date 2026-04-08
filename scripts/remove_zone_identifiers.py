#!/usr/bin/env python3
"""Elimina archivos *Zone.Identifier* (marcas de zona de Internet de Windows)
recorriendo un directorio.

Soporta dos formas habituales:

- Fichero suelto llamado ``Zone.Identifier`` (copias desde algunos entornos).
- En **WSL sobre NTFS**, el ADS a veces aparece como nombre compuesto
  ``archivo.json:Zone.Identifier`` (un solo nombre de fichero con dos puntos).

Para quitar marcas solo como flujos en Windows nativo (sin nombre de fichero con
``:``), en PowerShell::

  Get-ChildItem -Recurse -File | Unblock-File

Uso:
  python scripts/remove_zone_identifiers.py [ directorio_raíz ]
  python scripts/remove_zone_identifiers.py --dry-run .
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _is_zone_identifier(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name.lower()
    return name == "zone.identifier" or name.endswith(":zone.identifier")


def remove_zone_identifiers(root: Path, *, dry_run: bool) -> tuple[int, list[str]]:
    removed: list[str] = []
    for p in root.rglob("*"):
        try:
            if not _is_zone_identifier(p):
                continue
        except OSError:
            continue
        rel = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
        if dry_run:
            removed.append(rel)
            continue
        try:
            p.unlink()
            removed.append(rel)
        except OSError as e:
            print(f"omitido (error): {p} — {e}", file=sys.stderr)
    return len(removed), removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Borra todos los archivos llamados Zone.Identifier bajo un directorio.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        type=Path,
        help="Directorio raíz (por defecto: .)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo lista rutas; no borra.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"No es un directorio: {root}", file=sys.stderr)
        return 2
    n, paths = remove_zone_identifiers(root, dry_run=args.dry_run)
    label = "encontrados" if args.dry_run else "eliminados"
    for rel in paths:
        print(rel)
    print(f"Total {label}: {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
