"""
Event Bus (Synchronous Pub/Sub)
================================
Borrowed from OpenHands event stream pattern.
Enables decoupled communication between supervisor, workers, and knowledge tree.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventType(Enum):
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKER_STARTED = "worker_started"
    WORKER_FINISHED = "worker_finished"
    KNOWLEDGE_ADDED = "knowledge_added"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    REORG_NEEDED = "reorg_needed"
    REORG_COMPLETED = "reorg_completed"
    SKILL_EVOLVED = "skill_evolved"
    MCP_SERVER_GENERATED = "mcp_server_generated"
    REPORT_GENERATED = "report_generated"
    STATE_CHANGED = "state_changed"
    ERROR = "error"
    # Round 9: Quality Guardrails & Self-Evolution
    GOAL_SUBGOAL_COMPLETED = "goal_subgoal_completed"
    GOAL_ALL_COMPLETE = "goal_all_complete"
    FLOW_ADVISORY = "flow_advisory"
    EVOLUTION_LEARNING_ADDED = "evolution_learning_added"


@dataclass
class Event:
    type: EventType
    data: dict = field(default_factory=dict)
    source: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp,
        }


class EventBus:
    """Synchronous event bus with typed subscribers."""

    def __init__(self):
        self._subscribers: dict[EventType, list[tuple[str, Callable]]] = {}
        self._history: list[Event] = []
        self._max_history = 500

    def subscribe(self, event_type: EventType, callback: Callable, subscriber_id: str = ""):
        """Register a callback for an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append((subscriber_id, callback))

    def unsubscribe(self, event_type: EventType, subscriber_id: str):
        """Remove a subscriber by ID."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                (sid, cb) for sid, cb in self._subscribers[event_type]
                if sid != subscriber_id
            ]

    def publish(self, event: Event):
        """Publish an event to all subscribers of its type."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        for subscriber_id, callback in self._subscribers.get(event.type, []):
            try:
                callback(event)
            except Exception as e:
                print(f"  [EventBus] Error in {subscriber_id}: {e}")

    def emit(self, event_type: EventType, data: dict = None, source: str = ""):
        """Convenience: create and publish an event."""
        self.publish(Event(type=event_type, data=data or {}, source=source))

    def get_history(self, event_type: EventType = None, limit: int = 50) -> list[Event]:
        """Get recent events, optionally filtered by type."""
        events = self._history
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def clear_history(self):
        """Clear event history."""
        self._history.clear()
