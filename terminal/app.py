"""
Opus Live Terminal — Interactive Research Session
==================================================
Entry point for live mode: `python -m terminal.app "research goal"`

Architecture:
- Main thread: user input (prompt_toolkit)
- Thread 1: supervisor._run_loop()
- Thread 2: display loop (drains DisplayEvents → Rich console)

The supervisor runs exactly as in batch mode. We capture its output
via print() interception and inject user messages at cycle boundaries.
"""

import os
import sys
import json
import threading
import time
import argparse

from rich.console import Console

from terminal.message_bus import MessageBus, UserMessage, DisplayEvent
from terminal.display import TerminalDisplay
from terminal.input_handler import parse_input, format_help
from terminal import print_interceptor


def _run_supervisor(goal: str, bus: MessageBus, max_cycles: int = 15,
                    llm_backend: str = "minimax"):
    """Run the supervisor in a background thread."""
    try:
        # Import here to avoid circular imports at module level
        from core.llm import MiniMaxClient
        from core.tool_registry import ToolRegistry
        from core.event_bus import EventBus
        from supervisor.supervisor import Supervisor
        from config import WORKSPACE_BASE

        # Build system
        llm = MiniMaxClient(backend=llm_backend)
        registry = ToolRegistry()
        event_bus = EventBus()

        # Create mission directory
        import re
        slug = re.sub(r'[^a-z0-9]+', '_', goal.lower())[:40].strip('_')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mission_id = f"mission_{timestamp}_{slug}"
        mission_dir = os.path.join(WORKSPACE_BASE, mission_id)

        bus.emit_status("system", f"Mission: {mission_id}")

        # Create supervisor with message_bus
        supervisor = Supervisor(
            llm=llm,
            registry=registry,
            event_bus=event_bus,
            goal=goal,
            mission_dir=mission_dir,
            max_cycles=max_cycles,
        )
        supervisor.message_bus = bus

        # Run
        bus.emit_status("system", "Starting research pipeline...")
        supervisor.run()

        # Completion
        score_path = os.path.join(supervisor.mission_ctx.workspace_dir, "mission_score.json")
        grade, total = "", 0
        if os.path.exists(score_path):
            score_data = json.load(open(score_path))
            grade = score_data.get("grade", "")
            total = score_data.get("overall", 0)

        report_path = ""
        reports_dir = os.path.join(mission_dir, "reports")
        if os.path.isdir(reports_dir):
            reports = [f for f in os.listdir(reports_dir) if f.startswith("research_report")]
            if reports:
                report_path = os.path.join(reports_dir, reports[0])

        bus.emit(DisplayEvent(
            source="system", event_type="complete",
            content="Mission complete",
            metadata={"grade": grade, "score": total, "report_path": report_path},
        ))

    except Exception as e:
        bus.emit(DisplayEvent(
            source="system", event_type="error",
            content=f"Mission failed: {e}",
        ))
    finally:
        bus.shutdown()


def _display_loop(bus: MessageBus, display: TerminalDisplay):
    """Drain display events and render them. Runs in a thread."""
    while bus.active:
        event = bus.next_display_event(timeout=0.2)
        if event is None:
            continue

        if event.event_type == "complete":
            meta = event.metadata
            display.render_completion(
                grade=meta.get("grade", ""),
                score=meta.get("score", 0),
                report_path=meta.get("report_path", ""),
            )
            break

        display.render_event(event)


def _quick_query(text: str, llm_backend: str = "minimax"):
    """Handle a simple query without full pipeline."""
    from core.llm import MiniMaxClient

    console = Console()
    console.print(f"\n [dim]Searching...[/dim]")

    llm = MiniMaxClient(backend=llm_backend)
    response = llm.chat([
        {"role": "system", "content": "You are a helpful research assistant. Be concise."},
        {"role": "user", "content": text},
    ])
    answer = response["choices"][0]["message"]["content"]
    console.print(f"\n{answer}\n")


def _is_research_goal(text: str) -> bool:
    """Classify if input is a research goal vs simple question."""
    text_lower = text.lower()
    # Research indicators
    research_kw = [
        "compare", "implement", "benchmark", "evaluate", "研究",
        "experiment", "train", "fine-tune", "ablation", "比較",
        "vs", "versus",
    ]
    if any(kw in text_lower for kw in research_kw):
        return True
    # Short questions are queries
    if len(text) < 80 and text.endswith("?"):
        return False
    # Default: research
    return True


def main():
    parser = argparse.ArgumentParser(description="Opus Live Terminal")
    parser.add_argument("goal", nargs="?", default=None,
                        help="Research goal or query")
    parser.add_argument("--max-cycles", type=int, default=15)
    parser.add_argument("--llm", default="minimax")
    parser.add_argument("--query", action="store_true",
                        help="Force simple query mode")
    args = parser.parse_args()

    console = Console()

    # Interactive mode: no goal provided, enter REPL
    if not args.goal:
        console.print("\n[bold bright_white]◆ Opus Research Agent[/bold bright_white]")
        console.print("[dim]Type a research goal to start, or ask a question.[/dim]\n")

        try:
            from prompt_toolkit import PromptSession
            session = PromptSession()
            text = session.prompt("opus> ")
        except (ImportError, EOFError, KeyboardInterrupt):
            text = input("opus> ")

        if not text.strip():
            return
        args.goal = text.strip()

    # Quick query mode
    if args.query or not _is_research_goal(args.goal):
        _quick_query(args.goal, args.llm)
        return

    # ── Full research mode ──────────────────────────────────
    bus = MessageBus()
    display = TerminalDisplay(console)

    # Install print interceptor
    print_interceptor.install(bus)

    # Show header
    display.render_header(args.goal, args.max_cycles)

    # Start supervisor thread
    sup_thread = threading.Thread(
        target=_run_supervisor,
        args=(args.goal, bus, args.max_cycles, args.llm),
        daemon=True,
    )
    sup_thread.start()

    # Start display thread
    disp_thread = threading.Thread(
        target=_display_loop,
        args=(bus, display),
        daemon=True,
    )
    disp_thread.start()

    # ── Input loop (main thread) ────────────────────────────
    try:
        # Try prompt_toolkit for better UX
        from prompt_toolkit import PromptSession
        from prompt_toolkit.patch_stdout import patch_stdout

        session = PromptSession()

        with patch_stdout(raw=True):
            while bus.active and sup_thread.is_alive():
                try:
                    text = session.prompt("opus> ")
                    if not text.strip():
                        continue

                    msg = parse_input(text)
                    if msg is None:
                        continue

                    # Handle local commands
                    if msg.msg_type == "command" and msg.text.startswith("/help"):
                        console.print(f"\n{format_help()}\n")
                        continue

                    # Show in display
                    display.render_user_message(text)

                    # Send to supervisor
                    bus.send_user_message(msg)

                    # For abort, wait briefly then exit
                    if msg.msg_type == "abort":
                        console.print("[dim]Aborting... generating report with existing results.[/dim]")
                        time.sleep(2)
                        break

                except (EOFError, KeyboardInterrupt):
                    bus.send_user_message(UserMessage(text="", msg_type="abort"))
                    console.print("\n[dim]Session ended.[/dim]")
                    break

    except ImportError:
        # Fallback: basic input() without prompt_toolkit
        while bus.active and sup_thread.is_alive():
            try:
                text = input("opus> ")
                if not text.strip():
                    continue
                msg = parse_input(text)
                if msg:
                    display.render_user_message(text)
                    bus.send_user_message(msg)
                    if msg.msg_type == "abort":
                        break
            except (EOFError, KeyboardInterrupt):
                break

    # Wait for threads to finish
    sup_thread.join(timeout=30)
    disp_thread.join(timeout=5)

    # Restore print
    print_interceptor.uninstall()


if __name__ == "__main__":
    main()
