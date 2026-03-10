"""
Print Interceptor — Capture Status from Existing print() Calls
===============================================================
The codebase has 200+ print() calls with patterns like:
    [Supervisor] Planning...
    [explorer] Found 12 papers
    [LLM] Request took 2.3s

This interceptor captures them as DisplayEvents without
modifying any existing code.
"""

import builtins
import re
from terminal.message_bus import MessageBus, DisplayEvent

# Pattern: [source] content  or  (source) content
_SOURCE_PATTERN = re.compile(
    r'^\s*[\[\(]'           # opening [ or (
    r'([\w\-\.]+)'          # source name
    r'[\]\)]\s*'            # closing ] or )
    r'(.*)$',               # content
    re.DOTALL,
)

# Known source → display source mapping
_SOURCE_MAP = {
    "supervisor": "supervisor",
    "planner": "supervisor",
    "goaltracker": "supervisor",
    "flowmonitor": "supervisor",
    "failureanalyzer": "supervisor",
    "policyselector": "supervisor",
    "explorer": "explorer",
    "coder": "coder",
    "reviewer": "reviewer",
    "llm": "system",
    "minimax": "system",
    "tool_guard": "system",
    "sanity": "system",
    "evolution": "system",
}

_original_print = builtins.print
_active_bus: MessageBus | None = None


def _parse_source(text: str) -> tuple[str, str]:
    """Parse [source] prefix from print output."""
    match = _SOURCE_PATTERN.match(text)
    if match:
        raw_source = match.group(1).lower().strip()
        content = match.group(2).strip()
        source = _SOURCE_MAP.get(raw_source, raw_source)
        return source, content
    return "system", text.strip()


def _classify_event(source: str, content: str) -> str:
    """Classify the event type from content."""
    content_lower = content.lower()
    if any(kw in content_lower for kw in ["error", "failed", "fail", "crash"]):
        return "error"
    if any(kw in content_lower for kw in ["→", "queued", "dispatching"]):
        return "decision"
    if any(kw in content_lower for kw in ["tool:", "run_python", "write_file"]):
        return "tool_call"
    if any(kw in content_lower for kw in ["accuracy", "loss", "score", "metric"]):
        return "tool_result"
    if any(kw in content_lower for kw in ["cycle", "phase", "planning"]):
        return "progress"
    return "status"


def _intercepted_print(*args, **kwargs):
    """Drop-in replacement for print() that also emits DisplayEvents."""
    # Always call original print (for logging / batch mode)
    _original_print(*args, **kwargs)

    if not _active_bus or not _active_bus.active:
        return

    # Build text from args
    sep = kwargs.get("sep", " ")
    text = sep.join(str(a) for a in args)
    if not text.strip():
        return

    source, content = _parse_source(text)
    event_type = _classify_event(source, content)

    _active_bus.emit(DisplayEvent(
        source=source,
        event_type=event_type,
        content=content,
    ))


def install(bus: MessageBus):
    """Install the print interceptor. Call once at startup."""
    global _active_bus
    _active_bus = bus
    builtins.print = _intercepted_print


def uninstall():
    """Restore original print."""
    global _active_bus
    _active_bus = None
    builtins.print = _original_print
