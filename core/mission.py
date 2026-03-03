"""
Mission Manager
================
Manages isolated mission directories — each mission gets its own
knowledge, state, workspace, and reports folder.

Supports fuzzy resume (by timestamp or slug) and cross-mission knowledge search.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict


@dataclass
class MissionContext:
    """Everything needed to identify and run a single mission."""
    mission_id: str           # "mission_20260303_185200_flash_attention_search"
    goal: str
    direction: str            # can differ from goal on resume
    slug: str                 # "flash_attention_search"
    created_at: str           # ISO timestamp
    language: str = "en"      # "en" or "zh"
    cross_knowledge: bool = False
    status: str = "running"

    # Derived paths (set by _set_paths)
    root_dir: str = ""
    state_dir: str = ""
    knowledge_dir: str = ""
    workspace_dir: str = ""
    reports_dir: str = ""

    def _set_paths(self, missions_dir: str):
        self.root_dir = os.path.join(missions_dir, self.mission_id)
        self.state_dir = os.path.join(self.root_dir, "state")
        self.knowledge_dir = os.path.join(self.root_dir, "knowledge")
        self.workspace_dir = os.path.join(self.root_dir, "workspace")
        self.reports_dir = os.path.join(self.root_dir, "reports")

    def ensure_dirs(self):
        """Create all mission subdirectories."""
        for d in [self.root_dir, self.state_dir, self.knowledge_dir,
                  self.workspace_dir, self.reports_dir]:
            os.makedirs(d, exist_ok=True)

    def to_manifest(self) -> dict:
        """Serialise to mission.json fields."""
        return {
            "mission_id": self.mission_id,
            "goal": self.goal,
            "direction": self.direction,
            "slug": self.slug,
            "created_at": self.created_at,
            "language": self.language,
            "cross_knowledge": self.cross_knowledge,
            "status": self.status,
        }


def _slugify_fallback(text: str) -> str:
    """Best-effort slug from raw text (no LLM)."""
    ascii_text = text.encode("ascii", errors="ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
    parts = slug.split("_")[:4]
    return "_".join(p for p in parts if p) or "mission"


class MissionManager:
    """Create / list / find / load / save missions."""

    def __init__(self, missions_dir: str, llm=None):
        self.missions_dir = missions_dir
        self.llm = llm
        os.makedirs(missions_dir, exist_ok=True)

    # ── Create ────────────────────────────────────────────────────────

    def create_mission(self, goal: str, language: str = "en",
                       cross_knowledge: bool = False) -> MissionContext:
        slug = self._generate_slug(goal)
        ts = time.strftime("%Y%m%d_%H%M%S")
        mission_id = f"mission_{ts}_{slug}"
        created_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        ctx = MissionContext(
            mission_id=mission_id,
            goal=goal,
            direction=goal,
            slug=slug,
            created_at=created_at,
            language=language,
            cross_knowledge=cross_knowledge,
            status="running",
        )
        ctx._set_paths(self.missions_dir)
        ctx.ensure_dirs()
        self.save_mission(ctx)
        return ctx

    # ── List ──────────────────────────────────────────────────────────

    def list_missions(self) -> list[dict]:
        """Return manifests for all missions, newest first."""
        results = []
        if not os.path.isdir(self.missions_dir):
            return results
        for name in sorted(os.listdir(self.missions_dir), reverse=True):
            manifest = os.path.join(self.missions_dir, name, "mission.json")
            if os.path.isfile(manifest):
                with open(manifest) as f:
                    results.append(json.load(f))
        return results

    # ── Find (fuzzy) ──────────────────────────────────────────────────

    def find_mission(self, partial: str = None) -> list[MissionContext]:
        """
        Fuzzy-match missions.

        - None / empty → return the most recent mission
        - Otherwise search mission_id, timestamp part, and slug
        """
        missions = self.list_missions()
        if not missions:
            return []

        if not partial:
            return [self._manifest_to_ctx(missions[0])]

        partial_lower = partial.lower()
        matched = []
        for m in missions:
            mid = m["mission_id"].lower()
            slug = m.get("slug", "").lower()
            # timestamp portion: everything between first and second '_'
            parts = mid.split("_", 2)
            ts_part = parts[1] if len(parts) > 1 else ""
            if (partial_lower in mid
                    or partial_lower in slug
                    or partial_lower in ts_part
                    or partial_lower in m.get("goal", "").lower()):
                matched.append(self._manifest_to_ctx(m))

        return matched

    # ── Load / Save ───────────────────────────────────────────────────

    def load_mission(self, mission_id: str) -> MissionContext:
        manifest = os.path.join(self.missions_dir, mission_id, "mission.json")
        with open(manifest) as f:
            data = json.load(f)
        return self._manifest_to_ctx(data)

    def save_mission(self, ctx: MissionContext):
        manifest = os.path.join(ctx.root_dir, "mission.json")
        with open(manifest, "w") as f:
            json.dump(ctx.to_manifest(), f, ensure_ascii=False, indent=2)

    # ── Cross-mission knowledge ───────────────────────────────────────

    def get_all_knowledge_dirs(self, exclude_mission_id: str = None) -> list[dict]:
        """Return [{mission_id, knowledge_dir, goal}] for all other missions."""
        result = []
        for m in self.list_missions():
            mid = m["mission_id"]
            if mid == exclude_mission_id:
                continue
            kdir = os.path.join(self.missions_dir, mid, "knowledge")
            if os.path.isdir(kdir):
                result.append({
                    "mission_id": mid,
                    "knowledge_dir": kdir,
                    "goal": m.get("goal", ""),
                })
        return result

    # ── Internal helpers ──────────────────────────────────────────────

    def _generate_slug(self, goal: str) -> str:
        """Use LLM to compress goal into 2-4 English words; fallback to regex."""
        if self.llm:
            try:
                prompt = (
                    "Compress the following research goal into 2-4 lowercase English words "
                    "joined by underscores. Reply with ONLY the slug, nothing else.\n\n"
                    f"Goal: {goal}"
                )
                response = self.llm.chat([
                    {"role": "system", "content": "Reply with a short slug only."},
                    {"role": "user", "content": prompt},
                ])
                raw = response["choices"][0]["message"]["content"].strip()
                slug = re.sub(r"[^a-z0-9_]", "", raw.lower()).strip("_")
                if slug and 2 <= len(slug) <= 80:
                    return slug
            except Exception:
                pass
        return _slugify_fallback(goal)

    def _manifest_to_ctx(self, data: dict) -> MissionContext:
        ctx = MissionContext(
            mission_id=data["mission_id"],
            goal=data["goal"],
            direction=data.get("direction", data["goal"]),
            slug=data.get("slug", ""),
            created_at=data.get("created_at", ""),
            language=data.get("language", "en"),
            cross_knowledge=data.get("cross_knowledge", False),
            status=data.get("status", "unknown"),
        )
        ctx._set_paths(self.missions_dir)
        return ctx
