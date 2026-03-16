#!/usr/bin/env bash
set -e

ORANGE='\033[38;5;208m'
GREEN='\033[92m'
YELLOW='\033[93m'
RED='\033[91m'
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
    echo -e "  ${RED}✗ Python 3 not found. Please install Python 3.12+${RESET}"
    exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "  ${GREEN}✓${RESET} Python ${PY_VER}"

# ── Check Docker ──────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo -e "  ${RED}✗ Docker not found.${RESET}"
    echo -e "  ${DIM}Qdrant requires Docker. Install it from https://docs.docker.com/get-docker/${RESET}"
    exit 1
fi

if ! docker info &>/dev/null; then
    echo -e "  ${RED}✗ Docker is not running. Please start Docker Desktop first.${RESET}"
    exit 1
fi
echo -e "  ${GREEN}✓${RESET} Docker"

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

# ── Ensure Qdrant is running ─────────────────────────────
ensure_qdrant() {
    if curl -sf http://localhost:6333/healthz &>/dev/null; then
        echo -e "  ${GREEN}✓${RESET} Qdrant is running"
        return 0
    fi

    # Container exists but stopped → start it
    if docker ps -a --format '{{.Names}}' | grep -q '^qdrant$'; then
        echo -e "  ${DIM}Starting Qdrant container...${RESET}"
        docker start qdrant &>/dev/null
    else
        # No container → pull image and create one
        echo -e "  ${DIM}Pulling Qdrant image...${RESET}"
        docker pull qdrant/qdrant:latest --quiet 2>/dev/null || docker pull qdrant/qdrant:latest

        echo -e "  ${DIM}Creating Qdrant container...${RESET}"
        mkdir -p data/qdrant_storage
        docker run -d --name qdrant \
            -p 6333:6333 -p 6334:6334 \
            -v "$(pwd)/data/qdrant_storage:/qdrant/storage:z" \
            -e QDRANT__SERVICE__GRPC_PORT=6334 \
            --restart unless-stopped \
            qdrant/qdrant:latest &>/dev/null
    fi

    # Wait for Qdrant to become healthy (up to 15s)
    echo -ne "  ${DIM}Waiting for Qdrant"
    for i in $(seq 1 15); do
        if curl -sf http://localhost:6333/healthz &>/dev/null; then
            echo -e "${RESET}"
            echo -e "  ${GREEN}✓${RESET} Qdrant is running"
            return 0
        fi
        echo -n "."
        sleep 1
    done

    echo -e "${RESET}"
    echo -e "  ${RED}✗ Qdrant failed to start after 15s${RESET}"
    echo -e "  ${DIM}Check: docker logs qdrant${RESET}"
    exit 1
}

ensure_qdrant

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
