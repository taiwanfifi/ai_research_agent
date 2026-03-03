"""
Skill Registry
================
Manages skill lifecycle: load, evolve, version.
"""

import json
import os
from skills.base_skill import Skill


class SkillRegistry:
    """Manages the lifecycle of skills: loading, versioning, evolution."""

    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir
        self._skills: dict[str, Skill] = {}
        os.makedirs(skills_dir, exist_ok=True)

    def register(self, skill: Skill):
        """Register a skill (or update if newer version)."""
        existing = self._skills.get(skill.name)
        if not existing or skill.version > existing.version:
            self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """List all registered skills."""
        return list(self._skills.values())

    def load_from_directory(self, directory: str = None):
        """Load all skill JSON files from a directory."""
        directory = directory or self.skills_dir
        if not os.path.isdir(directory):
            return
        for fname in os.listdir(directory):
            if fname.endswith(".json"):
                try:
                    skill = Skill.load(os.path.join(directory, fname))
                    self.register(skill)
                except Exception as e:
                    print(f"  [SkillRegistry] Failed to load {fname}: {e}")

    def load_builtin(self):
        """Load built-in skills from the builtin/ subdirectory."""
        builtin_dir = os.path.join(os.path.dirname(__file__), "builtin")
        if os.path.isdir(builtin_dir):
            for fname in os.listdir(builtin_dir):
                if fname.endswith(".py") and not fname.startswith("_"):
                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(
                            fname[:-3], os.path.join(builtin_dir, fname)
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        if hasattr(module, "SKILL"):
                            self.register(module.SKILL)
                    except Exception as e:
                        print(f"  [SkillRegistry] Failed to load builtin {fname}: {e}")

    def save_all(self, directory: str = None):
        """Save all skills to disk."""
        directory = directory or self.skills_dir
        for skill in self._skills.values():
            skill.save(directory)

    def record_run(self, skill_name: str, success: bool, elapsed_s: float, error: str = ""):
        """Record a skill execution and check if evolution is needed."""
        skill = self._skills.get(skill_name)
        if skill:
            skill.record_run(success, elapsed_s, error)
            skill.save(self.skills_dir)

    def needs_evolution(self, skill_name: str) -> bool:
        """Check if a skill should be evolved based on failure patterns."""
        skill = self._skills.get(skill_name)
        if not skill or skill.runs < 3:
            return False
        # Evolve if success rate below 50% after at least 3 runs
        return skill.success_rate() < 0.5

    def get_performance_summary(self) -> list[dict]:
        """Get performance summary for all skills."""
        return [
            {
                "name": s.name,
                "version": s.version,
                "runs": s.runs,
                "success_rate": f"{s.success_rate():.0%}",
                "avg_time": f"{s.avg_elapsed_s:.1f}s",
                "needs_evolution": self.needs_evolution(s.name),
            }
            for s in self._skills.values()
        ]
