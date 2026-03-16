"""Interactive setup wizard for JAN - Personal Knowledge Assistant.

Creates the .env configuration file by prompting the user for required values.
"""

import os
import sys
from pathlib import Path

ORANGE = "\033[38;5;208m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def ask(prompt: str, default: str = "", required: bool = False, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"  {GREEN}▸{RESET} {prompt}{DIM}{suffix}{RESET}: ").strip()
        if not value and default:
            return default
        if not value and required:
            print(f"    {YELLOW}This field is required.{RESET}")
            continue
        return value


def validate_path(path_str: str) -> str:
    expanded = os.path.expanduser(path_str)
    p = Path(expanded)
    if not p.exists():
        print(f"    {YELLOW}Warning: path '{expanded}' does not exist yet.{RESET}")
    return expanded


def main():
    if ENV_PATH.exists():
        print(f"\n  {YELLOW}A .env file already exists.{RESET}")
        overwrite = input(f"  {GREEN}▸{RESET} Overwrite? (y/n) [{DIM}n{RESET}]: ").strip().lower()
        if overwrite != "y":
            print(f"  {DIM}Setup cancelled. Keeping existing .env{RESET}\n")
            return
        print()

    print(f"\n  {ORANGE}{BOLD}JAN Setup Wizard{RESET}")
    print(f"  {'─' * 40}\n")

    # --- Required ---
    print(f"  {BOLD}1. Obsidian Vault{RESET}")
    print(f"  {DIM}Path to your Obsidian vault folder{RESET}")
    vault_path = validate_path(ask("Vault path", required=True))
    print()

    print(f"  {BOLD}2. OpenRouter API Key{RESET}")
    print(f"  {DIM}Get one at https://openrouter.ai/keys{RESET}")
    openrouter_key = ask("API key", required=True)
    print()

    print(f"  {BOLD}3. LLM Model{RESET}")
    print(f"  {DIM}OpenRouter model ID for chat{RESET}")
    llm_model = ask("Model", default="deepseek/deepseek-v3.2")
    print()

    print(f"  {BOLD}4. Embedding Model{RESET}")
    print(f"  {DIM}OpenRouter model ID for embeddings{RESET}")
    embed_model = ask("Model", default="openai/text-embedding-3-small")
    embed_dims = ask("Dimensions", default="1536")
    print()

    # --- Qdrant ---
    print(f"  {BOLD}5. Qdrant Vector Database{RESET}")
    print(f"  {DIM}URL of your Qdrant instance{RESET}")
    qdrant_url = ask("Qdrant URL", default="http://localhost:6333")
    qdrant_grpc = ask("Qdrant gRPC port", default="6334")
    qdrant_api_key = ask("Qdrant API key (Cloud only, leave empty for local)", default="")
    print()

    # --- Optional ---
    print(f"  {BOLD}6. Optional Services{RESET}")
    print(f"  {DIM}Press Enter to skip{RESET}")
    tavily_key = ask("Tavily API key (web search)", default="")
    langsmith_key = ask("LangSmith API key (tracing)", default="")
    print()

    # --- Build .env ---
    lines = [
        "# --- Obsidian Vault ---",
        f"OBSIDIAN_VAULT_PATH={vault_path}",
        "",
        "# --- OpenRouter ---",
        f"OPENROUTER_API_KEY={openrouter_key}",
        "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1",
        "",
        "# --- LLM ---",
        f"LLM_MODEL={llm_model}",
        "",
        "# --- Embedding ---",
        f"EMBED_MODEL={embed_model}",
        f"EMBED_DIMENSIONS={embed_dims}",
        "",
        "# --- Database ---",
        "DATA_DIR=./data",
        "",
        "# --- Qdrant ---",
        f"QDRANT_URL={qdrant_url}",
        f"QDRANT_GRPC_PORT={qdrant_grpc}",
    ]

    if qdrant_api_key:
        lines.append(f"QDRANT_API_KEY={qdrant_api_key}")

    if tavily_key:
        lines += ["", "# --- Web Search ---", f"TAVILY_API_KEY={tavily_key}"]

    if langsmith_key:
        lines += [
            "",
            "# --- LangSmith ---",
            "LANGSMITH_TRACING=true",
            "LANGSMITH_ENDPOINT=https://api.smith.langchain.com",
            f"LANGSMITH_API_KEY={langsmith_key}",
            'LANGSMITH_PROJECT=Multi-Modal-Personal-Knowledge-Retriever',
        ]

    lines.append("")
    ENV_PATH.write_text("\n".join(lines))

    print(f"  {GREEN}{BOLD}✓ Configuration saved to .env{RESET}")
    print(f"  {DIM}You can edit it later at: {ENV_PATH}{RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print(f"\n\n  {DIM}Setup cancelled.{RESET}\n")
        sys.exit(1)
