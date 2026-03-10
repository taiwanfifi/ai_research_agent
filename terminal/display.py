"""
Display — Rich Terminal Formatter
===================================
Renders DisplayEvents as colored, formatted terminal output.
"""

import time
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from terminal.message_bus import DisplayEvent


# Source → color/icon mapping
_STYLES = {
    "supervisor": {"color": "bright_blue", "icon": "◆"},
    "explorer":   {"color": "bright_green", "icon": "🔍"},
    "coder":      {"color": "bright_yellow", "icon": "⚙"},
    "reviewer":   {"color": "bright_magenta", "icon": "📊"},
    "system":     {"color": "dim", "icon": "·"},
    "user":       {"color": "bright_cyan", "icon": "▶"},
    "opus":       {"color": "bright_white", "icon": "◈"},
}

# Event type → prefix style
_EVENT_STYLES = {
    "error": "bold red",
    "decision": "italic",
    "tool_call": "dim",
    "tool_result": "bold",
    "progress": "bold bright_blue",
    "complete": "bold green",
}


class TerminalDisplay:
    """Renders DisplayEvents to the terminal using Rich."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._mission_start = time.time()
        self._last_source = None

    def render_event(self, event: DisplayEvent):
        """Render a single DisplayEvent to the terminal."""
        style_info = _STYLES.get(event.source, _STYLES["system"])
        icon = style_info["icon"]
        color = style_info["color"]
        event_style = _EVENT_STYLES.get(event.event_type, "")

        # Elapsed time
        elapsed = time.time() - self._mission_start
        time_str = f"{elapsed:6.0f}s"

        # Format content
        content = event.content
        if len(content) > 300:
            content = content[:297] + "..."

        # Build the line
        source_label = event.source.upper()
        text = Text()
        text.append(f" {time_str} ", style="dim")
        text.append(f"{icon} ", style=color)
        text.append(f"[{source_label}] ", style=f"bold {color}")

        if event_style:
            text.append(content, style=event_style)
        else:
            text.append(content)

        # Metadata (cycle, worker, etc.)
        meta = event.metadata
        if meta.get("cycle"):
            text.append(f"  (cycle {meta['cycle']})", style="dim")

        self.console.print(text)
        self._last_source = event.source

    def render_user_message(self, text: str):
        """Render a user message in the chat flow."""
        t = Text()
        t.append(f"\n {'▶':>7} ", style="bright_cyan")
        t.append("[YOU] ", style="bold bright_cyan")
        t.append(text, style="bright_cyan")
        t.append("")
        self.console.print(t)

    def render_opus_reply(self, text: str):
        """Render Opus's reply to a user question."""
        t = Text()
        t.append(f" {'◈':>7} ", style="bright_white")
        t.append("[OPUS] ", style="bold bright_white")
        t.append(text)
        self.console.print(t)

    def render_header(self, goal: str, max_cycles: int = 15):
        """Render the session header."""
        self.console.print()
        self.console.print(Panel(
            f"[bold]Goal:[/bold] {goal}\n"
            f"[dim]Max cycles: {max_cycles} | Type anytime to interact[/dim]",
            title="[bold bright_white]◆ Opus Research Agent[/bold bright_white]",
            border_style="bright_blue",
            padding=(0, 1),
        ))
        self.console.print()

    def render_completion(self, grade: str = "", score: float = 0,
                          report_path: str = ""):
        """Render mission completion summary."""
        elapsed = time.time() - self._mission_start
        minutes = elapsed / 60

        parts = [f"[bold]Mission complete[/bold] ({minutes:.1f} min)"]
        if grade:
            parts.append(f"Grade: [bold]{grade}[/bold] ({score:.1f}/10)")
        if report_path:
            parts.append(f"Report: {report_path}")

        self.console.print()
        self.console.print(Panel(
            "\n".join(parts),
            title="[bold green]✓ Done[/bold green]",
            border_style="green",
            padding=(0, 1),
        ))
