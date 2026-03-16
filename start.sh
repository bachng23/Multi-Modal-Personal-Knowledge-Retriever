#!/usr/bin/env bash
set -e

ORANGE='\033[38;5;208m'
GREEN='\033[92m'
YELLOW='\033[93m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

clear

echo -e "${ORANGE}${BOLD}"
echo "     ██╗ █████╗ ███╗   ██╗"
echo "     ██║██╔══██╗████╗  ██║"
echo "     ██║███████║██╔██╗ ██║"
echo "██   ██║██╔══██║██║╚██╗██║"
echo "╚█████╔╝██║  ██║██║ ╚████║"
echo " ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝"
echo -e "${RESET}"
echo -e "  ${DIM}Personal Knowledge Assistant${RESET}"
echo ""

cd "$(dirname "$0")"

# ── Check Python ──────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo -e "  ${YELLOW}✗ Python 3 not found. Please install Python 3.12+${RESET}"
    exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "  ${GREEN}✓${RESET} Python ${PY_VER}"

# ── Virtual environment ──────────────────────────────────
if [ ! -d ".venv" ]; then
    echo -e "  ${DIM}Creating virtual environment...${RESET}"
    python3 -m venv .venv
fi
source .venv/bin/activate
echo -e "  ${GREEN}✓${RESET} Virtual environment"

# ── Install dependencies ─────────────────────────────────
echo -e "  ${DIM}Installing dependencies...${RESET}"
if command -v uv &>/dev/null; then
    uv pip install -e . --quiet 2>/dev/null
else
    pip install -e . --quiet 2>/dev/null
fi
echo -e "  ${GREEN}✓${RESET} Dependencies installed"

# ── Check Qdrant ─────────────────────────────────────────
if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${RESET} Qdrant is running"
else
    echo -e "  ${YELLOW}⚠ Qdrant is not running at localhost:6333${RESET}"
    if command -v docker &>/dev/null; then
        # Check if container exists but is stopped
        if docker ps -a --format '{{.Names}}' | grep -q '^qdrant$'; then
            echo -e "  ${DIM}Starting existing Qdrant container...${RESET}"
            docker start qdrant > /dev/null
        else
            read -p "  Start Qdrant with Docker? (y/n): " start_qdrant
            if [[ "$start_qdrant" == "y" || "$start_qdrant" == "Y" ]]; then
                docker run -d --name qdrant \
                    -p 6333:6333 -p 6334:6334 \
                    -v "$(pwd)/data/qdrant_storage:/qdrant/storage" \
                    qdrant/qdrant > /dev/null
            fi
        fi
        # Wait a moment for Qdrant to start
        sleep 2
        if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓${RESET} Qdrant started"
        else
            echo -e "  ${YELLOW}⚠ Qdrant failed to start. Check Docker.${RESET}"
        fi
    else
        echo -e "  ${DIM}Install Docker, then run:${RESET}"
        echo -e "  ${DIM}docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant${RESET}"
    fi
fi

# ── Setup .env if needed ─────────────────────────────────
if [ ! -f ".env" ]; then
    echo ""
    echo -e "  ${ORANGE}${BOLD}First-time setup${RESET}"
    echo -e "  ${DIM}Let's configure your environment${RESET}"
    echo ""
    python3 scripts/setup.py
fi

echo -e "  ${GREEN}✓${RESET} Configuration loaded"
echo ""

# ── Launch ────────────────────────────────────────────────
python3 -m src.main "$@"
