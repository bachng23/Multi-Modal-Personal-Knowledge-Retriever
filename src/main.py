"""Standalone CLI for the Knowledge Assistant with SQLite persistent memory.

Usage:
    python -m src.main                     # start the assistant
    python -m src.main --thread mythread   # resume a conversation
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger

from src.agents.graph import builder
from src.agents.memory import create_memory_checkpointer
from src.core.config import config as app_config

ORANGE = "\033[38;5;208m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

JAN_ART = [
    "     ██╗ █████╗ ███╗   ██╗",
    "     ██║██╔══██╗████╗  ██║",
    "     ██║███████║██╔██╗ ██║",
    "██   ██║██╔══██║██║╚██╗██║",
    "╚█████╔╝██║  ██║██║ ╚████║",
    " ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝",
]


def _suppress_logs():
    logger.remove()
    logger.add(os.devnull, level="CRITICAL")
    logging.basicConfig(level=logging.CRITICAL, force=True)
    for name in ("httpx", "httpcore", "openai", "langchain", "qdrant_client", "grpc"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def print_banner(thread_id: str):
    print()
    for line in JAN_ART:
        print(f"  {ORANGE}{line}{RESET}")
    print()
    print(f"  {DIM}Personal Knowledge Assistant{RESET}")
    print(f"  {DIM}LangGraph + Qdrant + OpenRouter{RESET}")
    print()
    print(f"  {DIM}Thread:{RESET} {BOLD}{thread_id}{RESET}")
    print(f"  {DIM}Commands:{RESET} {GREEN}/new{RESET} new thread  "
          f"{GREEN}/status{RESET} vault info  "
          f"{GREEN}/quit{RESET} exit")
    print(f"  {'─' * 50}")
    print()


_checkpointer = None


async def build_agent():
    global _checkpointer
    _checkpointer = await create_memory_checkpointer()
    return builder.compile(
        checkpointer=_checkpointer,
        interrupt_before=["sensitive_tools"],
    )


def _clear_status():
    sys.stdout.write("\r" + " " * 50 + "\r")
    sys.stdout.flush()


def _show_status(text: str):
    sys.stdout.write(f"\r{DIM}{text}{RESET}")
    sys.stdout.flush()


_TOOL_LABELS = {
    "search_knowledge": "Searching your vault...",
    "web_search": "Searching the web...",
    "reindex_vault": "Reindexing vault...",
    "get_vault_status": "Checking vault status...",
    "get_document_info": "Looking up document info...",
}


def _tool_label(tool_name: str) -> str:
    return _TOOL_LABELS.get(tool_name, f"Using {tool_name}...")


async def stream_turn(agent, input_data, config):
    """Stream one agent turn. Handles tool execution and interrupts."""
    printed_header = False
    started_text = False
    summarizing = False
    needs_separator = False
    active_tools: set[str] = set()

    if input_data is not None:
        _show_status("Thinking...")

    try:
        async for event in agent.astream_events(
            input_data, config=config, version="v2"
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                node = event.get("metadata", {}).get("langgraph_node")
                if node == "summarize":
                    if not summarizing:
                        summarizing = True
                        if printed_header:
                            print("\n")
                        _show_status("Saving memory...")
                    continue
                chunk = event["data"]["chunk"]
                if chunk.content and not getattr(chunk, "tool_call_chunks", None):
                    text = chunk.content
                    if not started_text:
                        text = text.lstrip("\n")
                        if not text:
                            continue
                        started_text = True
                    if not printed_header:
                        _clear_status()
                        sys.stdout.write(f"\n{ORANGE}{BOLD}JAN ▸{RESET} ")
                        sys.stdout.flush()
                        printed_header = True
                    elif needs_separator:
                        sys.stdout.write("\n\n")
                        sys.stdout.flush()
                        started_text = False
                        text = text.lstrip("\n")
                        if not text:
                            needs_separator = False
                            continue
                        started_text = True
                    needs_separator = False
                    sys.stdout.write(text)
                    sys.stdout.flush()

            elif kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                active_tools.add(tool_name)
                label = _tool_label(tool_name)
                if printed_header:
                    needs_separator = True
                    started_text = False
                _clear_status()
                print(f"  {DIM}🔍 {label}{RESET}")
                _show_status("Working...")

            elif kind == "on_tool_end":
                tool_name = event.get("name", "tool")
                active_tools.discard(tool_name)
                if not printed_header:
                    _show_status("Generating response...")

    except Exception as e:
        _clear_status()
        print(f"\n{YELLOW}Error: {e}{RESET}\n")
        return

    if summarizing:
        _clear_status()
        print()
    elif printed_header:
        print("\n")
    else:
        _clear_status()

    state = await agent.aget_state(config)

    if state.next:
        pending_tool_calls = []
        for msg in reversed(state.values["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                pending_tool_calls = msg.tool_calls
                break

        if pending_tool_calls:
            tool_names = [tc["name"] for tc in pending_tool_calls]
            print(f"{YELLOW}Agent wants to run: {BOLD}{', '.join(tool_names)}{RESET}")
            confirm = input(f"{YELLOW}Approve? (y/n): {RESET}").strip().lower()
            if confirm in ("y", "yes"):
                await stream_turn(agent, None, config)
            else:
                print(f"{DIM}Cancelled.{RESET}\n")
        return

    if not printed_header:
        for msg in reversed(state.values.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\n{ORANGE}{BOLD}JAN ▸{RESET} {msg.content}\n")
                break


async def chat_loop(agent, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    print_banner(thread_id)

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}You ▸ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print(f"{DIM}Goodbye!{RESET}")
            break

        if user_input.lower() in ("/new", "new"):
            thread_id = str(uuid.uuid4())[:8]
            config = {"configurable": {"thread_id": thread_id}}
            os.system("cls" if os.name == "nt" else "clear")
            print_banner(thread_id)
            continue

        if user_input.lower() in ("/status", "status"):
            user_input = "What is the current vault status?"

        await stream_turn(
            agent,
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )


async def _auto_index_if_needed():
    """Run indexing automatically on first launch when the vault is not yet indexed."""
    from src.infrastructure.database.parent_store_manager import ParentStoreManager

    db_path = app_config.SQLITE_DB_PATH
    if not db_path.exists():
        needs_index = True
    else:
        store = ParentStoreManager(db_path=db_path)
        all_states = store.get_all_index_states()
        needs_index = len(all_states) == 0

    if not needs_index:
        return

    print(f"  {ORANGE}{BOLD}First run detected — indexing your vault{RESET}")
    print(f"  {DIM}This only happens once and may take a few minutes...{RESET}\n")

    from src.services.indexing import IndexingService

    indexing_service = IndexingService()
    stats = await indexing_service.run()

    print(f"  {GREEN}✓{RESET} Indexing complete: "
          f"{stats.indexed} indexed, {stats.skipped} skipped, "
          f"{stats.failed} failed ({stats.total} total)\n")


def main():
    os.system("cls" if os.name == "nt" else "clear")

    parser = argparse.ArgumentParser(description="JAN - Knowledge Assistant CLI")
    parser.add_argument(
        "--thread", default=None, help="Thread ID to resume a conversation"
    )
    args = parser.parse_args()

    _suppress_logs()

    thread_id = args.thread or str(uuid.uuid4())[:8]

    async def _run():
        await _auto_index_if_needed()
        agent = await build_agent()
        try:
            await chat_loop(agent, thread_id)
        finally:
            if _checkpointer and hasattr(_checkpointer, "conn"):
                await _checkpointer.conn.close()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
