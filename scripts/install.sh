#!/usr/bin/env bash
#
# OffScan — dependency installation script
# Installs all system and Python dependencies for a fully offline development setup.
#
# Usage:
#   ./scripts/install.sh          # install everything
#   ./scripts/install.sh --system # system deps only
#   ./scripts/install.sh --python  # Python deps only
#
set -euo pipefail

# ── Colors ────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}ℹ  $1${NC}"; }
success() { echo -e "${GREEN}✓  $1${NC}"; }
warn()    { echo -e "${YELLOW}⚠  $1${NC}"; }
error()   { echo -e "${RED}✗  $1${NC}" >&2; exit 1; }

# ── Detect OS ─────────────────────────────────────────────
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &>/dev/null; then
            echo "debian"
        elif command -v dnf &>/dev/null; then
            echo "fedora"
        elif command -v pacman &>/dev/null; then
            echo "arch"
        elif command -v zypper &>/dev/null; then
            echo "opensuse"
        else
            echo "unknown-linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
info "Detected OS: $OS"

# ── System dependencies ──────────────────────────────────
install_system() {
    case "$OS" in
        debian)
            info "Installing system dependencies (Debian/Ubuntu)..."
            sudo apt-get update
            sudo apt-get install -y \
                tesseract-ocr \
                tesseract-ocr-eng \
                libopencv-dev \
                python3-dev \
                python3-venv \
                pkg-config \
                espeak-ng \
                libsndfile1
            ;;
        fedora)
            info "Installing system dependencies (Fedora/RHEL)..."
            sudo dnf install -y \
                tesseract \
                tesseract-langpack-eng \
                opencv-devel \
                python3-devel \
                espeak-ng \
                libsndfile
            ;;
        arch)
            info "Installing system dependencies (Arch)..."
            sudo pacman -S --noconfirm \
                tesseract \
                tesseract-data-eng \
                opencv \
                espeak-ng \
                libsndfile
            ;;
        opensuse)
            info "Installing system dependencies (openSUSE)..."
            sudo zypper install -y \
                tesseract-ocr \
                tesseract-ocr-data-eng \
                opencv-devel \
                python3-devel \
                espeak \
                libsndfile1
            ;;
        macos)
            info "Installing system dependencies (macOS)..."
            if ! command -v brew &>/dev/null; then
                warn "Homebrew not found. Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install tesseract opencv espeak
            ;;
        windows)
            warn "Windows detected. Please install manually:"
            warn "  1. Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
            warn "  2. Add Tesseract to PATH"
            warn "  3. Set TESSDATA_PREFIX environment variable"
            ;;
        *)
            error "Unsupported OS: $OS. Please install Tesseract and OpenCV manually."
            ;;
    esac
    success "System dependencies installed."
}

# ── Python dependencies ──────────────────────────────────
install_python() {
    info "Setting up Python virtual environment..."

    if [[ ! -d ".venv" ]]; then
        python3 -m venv .venv
        success "Created .venv"
    fi

    # shellcheck disable=SC1091
    source .venv/bin/activate

    info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel

    info "Installing OffScan (editable, with dev dependencies)..."
    pip install -e ".[dev]"

    success "Python dependencies installed."
}

# ── Verify ───────────────────────────────────────────────
verify() {
    info "Verifying installation..."

    if command -v tesseract &>/dev/null; then
        success "Tesseract: $(tesseract --version 2>&1 | head -1)"
    else
        warn "Tesseract not found on PATH."
    fi

    if [[ -d ".venv" ]]; then
        source .venv/bin/activate
    fi

    if python3 -c "import cv2; import pytesseract; import pyttsx3; import onnxruntime; import sklearn" 2>/dev/null; then
        success "All Python packages importable."
    else
        warn "Some Python packages failed to import. Check the output above."
    fi

    success "OffScan is ready! Run 'make test' to verify."
}

# ── Main ─────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    --system)
        install_system
        ;;
    --python)
        install_python
        ;;
    all|"")
        install_system
        install_python
        verify
        ;;
    *)
        error "Usage: $0 [--system|--python|all]"
        ;;
esac
