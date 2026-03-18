#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[install.sh] $1"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

python_ok() {
  if ! command_exists python3; then
    return 1
  fi
  python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'
}

detect_package_manager() {
  if command_exists apt-get; then
    echo "apt-get"
    return 0
  fi
  if command_exists dnf; then
    echo "dnf"
    return 0
  fi
  if command_exists yum; then
    echo "yum"
    return 0
  fi
  if command_exists pacman; then
    echo "pacman"
    return 0
  fi
  if command_exists zypper; then
    echo "zypper"
    return 0
  fi
  if command_exists brew; then
    echo "brew"
    return 0
  fi
  return 1
}

install_with_manager() {
  local manager="$1"
  local requirement="$2"
  local package_name="$requirement"

  if [[ "$requirement" == "python3" && "$manager" == "brew" ]]; then
    package_name="python@3.12"
  fi

  log "Installing ${package_name} using ${manager}..."
  if [[ "$manager" == "apt-get" ]]; then
    sudo apt-get update
    sudo apt-get install -y "${package_name}"
  elif [[ "$manager" == "dnf" ]]; then
    sudo dnf install -y "${package_name}"
  elif [[ "$manager" == "yum" ]]; then
    sudo yum install -y "${package_name}"
  elif [[ "$manager" == "pacman" ]]; then
    sudo pacman -Sy --noconfirm "${package_name}"
  elif [[ "$manager" == "zypper" ]]; then
    sudo zypper --non-interactive install "${package_name}"
  elif [[ "$manager" == "brew" ]]; then
    brew install "${package_name}"
  else
    log "Unsupported package manager: ${manager}"
    return 1
  fi
}

prompt_install_missing() {
  local manager="$1"
  shift
  local missing=("$@")

  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi

  log "Missing prerequisites: ${missing[*]}"
  if [[ -z "${manager}" ]]; then
    log "No supported package manager detected. Install requirements manually and rerun."
    return 1
  fi

  read -r -p "Install missing prerequisites with ${manager}? [y/N]: " reply
  case "${reply}" in
    y|Y|yes|YES)
      for req in "${missing[@]}"; do
        install_with_manager "${manager}" "${req}" || return 1
      done
      ;;
    *)
      log "Installation cancelled by user."
      return 1
      ;;
  esac
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXTRAS="${1:-dev,ui,cudaq}"
PACKAGE_MANAGER="$(detect_package_manager || true)"

log "Step 1/3: Checking prerequisites (Python 3.12+ and Git)"

missing=()
if ! command_exists git; then
  missing+=("git")
fi
if ! python_ok; then
  missing+=("python3")
fi

if [[ ${#missing[@]} -gt 0 ]]; then
  prompt_install_missing "${PACKAGE_MANAGER}" "${missing[@]}"
fi

if ! command_exists git; then
  log "ERROR: Git is still missing."
  exit 1
fi

if ! python_ok; then
  log "ERROR: Python 3.12+ is still missing."
  exit 1
fi

log "Step 2/3: Prerequisites OK"
log "Step 3/3: Running setup script"
"${SCRIPT_DIR}/bin/setup.sh" "${EXTRAS}"
log "Installer completed successfully."
