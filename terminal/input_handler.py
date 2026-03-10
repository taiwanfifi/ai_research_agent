"""
Input Handler — Parse User Input During Live Sessions
======================================================
Handles commands (/status, /pause, /abort) and free-text messages.
"""

from terminal.message_bus import UserMessage


# Built-in commands
COMMANDS = {
    "/status": "Show current mission progress",
    "/abort": "Stop the mission (generates report with what exists)",
    "/help": "Show available commands",
    "/direction": "Change research direction (e.g. /direction focus on accuracy not speed)",
}


def parse_input(text: str) -> UserMessage:
    """Parse user input into a UserMessage."""
    text = text.strip()
    if not text:
        return None

    # Commands
    if text.startswith("/"):
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/abort":
            return UserMessage(text=text, msg_type="abort")
        elif cmd == "/status":
            return UserMessage(text=text, msg_type="command")
        elif cmd == "/direction":
            if arg:
                return UserMessage(text=arg, msg_type="direction")
            return UserMessage(text="What direction?", msg_type="command")
        elif cmd == "/help":
            return UserMessage(text=text, msg_type="command")
        else:
            return UserMessage(text=text, msg_type="chat")

    # Free text — classify intent
    text_lower = text.lower()

    # Direction changes
    direction_keywords = [
        "focus on", "instead", "change to", "switch to",
        "try", "don't", "stop", "skip", "重點", "改成", "不要",
    ]
    if any(kw in text_lower for kw in direction_keywords):
        return UserMessage(text=text, msg_type="direction")

    # Default: chat (question or comment)
    return UserMessage(text=text, msg_type="chat")


def format_help() -> str:
    """Format help text for available commands."""
    lines = ["Available commands:"]
    for cmd, desc in COMMANDS.items():
        lines.append(f"  {cmd:15s} {desc}")
    lines.append("")
    lines.append("Or type anything — questions, feedback, direction changes.")
    lines.append("Opus reads your messages at each cycle boundary.")
    return "\n".join(lines)
