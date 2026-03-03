"""
Base Skill Definition
======================
Skill dataclass representing a reusable research workflow.
"""

from dataclasses import dataclass, field
import json
import os
import time


@dataclass
class Skill:
    """A versioned, evolvable research skill."""
    name: str
    version: int = 1
    description: str = ""
    prompt: str = ""              # System prompt for the skill
    tools: list[str] = field(default_factory=list)    # Required tool names
    workflow_steps: list[str] = field(default_factory=list)  # Step descriptions
    success_criteria: str = ""    # How to judge if skill succeeded
    worker_type: str = "explorer"  # Which worker runs this skill

    # Performance tracking
    runs: int = 0
    successes: int = 0
    failures: int = 0
    avg_elapsed_s: float = 0.0
    last_run: str = ""
    failure_log: list[str] = field(default_factory=list)

    def success_rate(self) -> float:
        return self.successes / self.runs if self.runs > 0 else 0.0

    def record_run(self, success: bool, elapsed_s: float, error: str = ""):
        """Record a skill execution result."""
        self.runs += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
            if error:
                self.failure_log.append(f"{time.strftime('%Y-%m-%d %H:%M')}: {error[:200]}")
                self.failure_log = self.failure_log[-10:]  # Keep last 10
        # Running average
        self.avg_elapsed_s = (self.avg_elapsed_s * (self.runs - 1) + elapsed_s) / self.runs
        self.last_run = time.strftime("%Y-%m-%dT%H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "name": self.name, "version": self.version,
            "description": self.description, "prompt": self.prompt,
            "tools": self.tools, "workflow_steps": self.workflow_steps,
            "success_criteria": self.success_criteria,
            "worker_type": self.worker_type,
            "runs": self.runs, "successes": self.successes,
            "failures": self.failures, "avg_elapsed_s": self.avg_elapsed_s,
            "last_run": self.last_run, "failure_log": self.failure_log,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, directory: str):
        """Save skill to JSON file."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.name}_v{self.version}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Skill":
        """Load skill from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
