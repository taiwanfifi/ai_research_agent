"""
MessageBus — Thread-Safe Bidirectional Channel
================================================
Connects user input ↔ supervisor loop ↔ terminal display.
Two queues: user→supervisor, supervisor→display.
"""

import queue
import time
from dataclasses import dataclass, field


@dataclass
class UserMessage:
    """A message from the user during a live session."""
    text: str
    msg_type: str = "chat"  # chat, command, direction, abort
    timestamp: float = field(default_factory=time.time)


@dataclass
class DisplayEvent:
    """A status update for the terminal display."""
    source: str         # supervisor, explorer, coder, reviewer, system, user
    event_type: str     # status, thinking, tool_call, tool_result,
                        # decision, progress, error, complete, user_ack
    content: str
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    """Thread-safe bidirectional message channel.

    Usage:
        bus = MessageBus()
        # User thread:
        bus.send_user_message(UserMessage("focus on accuracy"))
        # Supervisor thread:
        msgs = bus.check_user_messages()  # non-blocking
        bus.emit(DisplayEvent(source="coder", event_type="status", content="Training..."))
        # Display thread:
        event = bus.next_display_event(timeout=0.1)  # blocking with timeout
    """

    def __init__(self):
        self._user_queue: queue.Queue[UserMessage] = queue.Queue()
        self._display_queue: queue.Queue[DisplayEvent] = queue.Queue()
        self._active = True

    # ── User → Supervisor ──────────────────────────────────

    def send_user_message(self, msg: UserMessage):
        """Send a message from user to supervisor (non-blocking)."""
        self._user_queue.put(msg)

    def check_user_messages(self) -> list[UserMessage]:
        """Non-blocking: get all pending user messages."""
        messages = []
        while True:
            try:
                messages.append(self._user_queue.get_nowait())
            except queue.Empty:
                break
        return messages

    # ── Supervisor/Workers → Display ───────────────────────

    def emit(self, event: DisplayEvent):
        """Emit a display event (non-blocking)."""
        self._display_queue.put(event)

    def emit_status(self, source: str, content: str, **metadata):
        """Convenience: emit a status event."""
        self.emit(DisplayEvent(
            source=source, event_type="status",
            content=content, metadata=metadata,
        ))

    def next_display_event(self, timeout: float = 0.1) -> DisplayEvent | None:
        """Blocking: get next display event, or None on timeout."""
        try:
            return self._display_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_display_events(self, max_events: int = 50) -> list[DisplayEvent]:
        """Non-blocking: get all pending display events (up to max)."""
        events = []
        for _ in range(max_events):
            try:
                events.append(self._display_queue.get_nowait())
            except queue.Empty:
                break
        return events

    # ── Lifecycle ──────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self._active

    def shutdown(self):
        """Signal that the session is ending."""
        self._active = False
        self.emit(DisplayEvent(
            source="system", event_type="complete",
            content="Session ended.",
        ))
