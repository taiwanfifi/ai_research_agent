"""
Domain Brain — Structured Research Knowledge
=============================================
Replaces flat evolution_store learnings with organized domain knowledge.

Inspired by Kael's brain regions: sparse activation, self-organizing,
accumulative understanding rather than tactical tips.

Structure:
  brain/
    regularization/
      principles.json   - Mechanistic understanding
      experiments.json   - What we've tested and found
      open_questions.json - What we don't know yet
    optimization/
      ...
    _meta.json          - Region metadata, access counts

Each mission reads relevant regions and writes back what it learned.
Knowledge grows across missions, not just within them.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from core.llm import MiniMaxClient, strip_think


BRAIN_DIR_NAME = "_domain_brain"


class DomainBrain:
    """Persistent, topic-organized research knowledge."""

    def __init__(self, missions_dir: str, llm: MiniMaxClient = None):
        self.brain_dir = Path(missions_dir) / BRAIN_DIR_NAME
        self.brain_dir.mkdir(exist_ok=True)
        self.llm = llm
        self._meta = self._load_meta()

    # ── Reading ─────────────────────────────────────────────────

    def get_relevant_context(self, goal: str, max_regions: int = 3) -> str:
        """Load brain regions relevant to a research goal.

        Returns formatted context string for injection into planner/critic.
        Like Kael's router: sparse activation, not full load.
        """
        regions = self._route(goal, max_regions)
        if not regions:
            return ""

        parts = []
        for region_name in regions:
            region_data = self._load_region(region_name)
            if not region_data:
                continue

            # Track access
            self._record_access(region_name)

            parts.append(f"### Domain: {region_name}")

            # Principles (most valuable)
            principles = region_data.get("principles", [])
            if principles:
                parts.append("**Established Principles:**")
                for p in principles[-5:]:  # Last 5 (most recent)
                    parts.append(f"- {p['principle']}")
                    if p.get("evidence"):
                        parts.append(f"  Evidence: {p['evidence'][:100]}")

            # Past experiments
            experiments = region_data.get("experiments", [])
            if experiments:
                parts.append("**Past Experiments:**")
                for e in experiments[-3:]:
                    result = e.get("result", "unknown")
                    parts.append(f"- {e.get('description', '?')} → {result}")

            # Open questions
            questions = region_data.get("open_questions", [])
            if questions:
                parts.append("**Open Questions:**")
                for q in questions[-3:]:
                    parts.append(f"- {q}")

            parts.append("")

        return "\n".join(parts)

    def get_all_principles(self) -> list[dict]:
        """Get all principles across all regions (for cross-domain reasoning)."""
        all_principles = []
        for region_name in self._list_regions():
            data = self._load_region(region_name)
            if data:
                for p in data.get("principles", []):
                    p["domain"] = region_name
                    all_principles.append(p)
        return all_principles

    # ── Writing (after mission) ─────────────────────────────────

    def learn_from_mission(self, goal: str, analysis_summary: dict,
                           hypothesis_chain: list = None):
        """Extract and store knowledge from a completed mission.

        This is the key difference from evolution_store: we extract
        PRINCIPLES (why things work) not just TACTICS (what to do).

        Args:
            goal: The mission goal
            analysis_summary: The analysis_summary.json contents
            hypothesis_chain: Optional hypothesis history
        """
        if not self.llm:
            return

        # Determine which region(s) this belongs to
        regions = self._route(goal, max_regions=2)
        if not regions:
            # Create a new region based on the goal
            region_name = self._suggest_region_name(goal)
            regions = [region_name]

        # Extract knowledge using LLM
        knowledge = self._extract_knowledge(goal, analysis_summary,
                                             hypothesis_chain)
        if not knowledge:
            return

        # Write to each relevant region
        for region_name in regions:
            self._ensure_region(region_name)
            self._append_knowledge(region_name, knowledge)

        print(f"  [Brain] Learned {len(knowledge.get('principles', []))} principles, "
              f"{len(knowledge.get('experiments', []))} experiments → "
              f"{', '.join(regions)}")

    def _extract_knowledge(self, goal: str, analysis: dict,
                           hypotheses: list = None) -> dict | None:
        """Use LLM to extract principles from mission results."""
        analysis_str = json.dumps(analysis, indent=2, default=str)[:3000]
        hyp_str = ""
        if hypotheses:
            hyp_str = "\n".join(
                f"- [{h.get('status', '?')}] {h.get('claim', '')}"
                for h in hypotheses[-5:]
            )

        prompt = f"""Extract research knowledge from this completed experiment.

## Experiment
{goal}

## Results
{analysis_str}

{f"## Hypothesis Chain{chr(10)}{hyp_str}" if hyp_str else ""}

Extract three types of knowledge:

1. **Principles** — Mechanistic understanding (WHY things work, not just WHAT happened)
   Example: "L1 regularization promotes weight sparsity, but requires sufficient training
   epochs and model capacity to show measurable effects vs L2"

2. **Experiment Record** — What we tested and found (for future reference)
   Example: "L1 vs L2 on SimpleCNN/CIFAR-10, 2000 samples, 5 epochs → no significant
   difference (p=0.74, d=0.16). Likely insufficient training for regularization effects."

3. **Open Questions** — What this experiment makes us want to know
   Example: "At what training duration does L1 vs L2 difference become significant?"

Respond with JSON:
{{
  "principles": [
    {{"principle": "...", "evidence": "...", "confidence": 0.8}}
  ],
  "experiments": [
    {{"description": "...", "result": "...", "p_value": null, "effect_size": null}}
  ],
  "open_questions": ["..."],
  "cross_domain_connections": ["connects to X because Y"]
}}

Rules:
- Extract UNDERSTANDING, not just data points
- If the result was null, explain WHY (insufficient power? genuine no-effect?)
- Principles should be generalizable beyond this specific experiment
- Be concise but precise"""

        try:
            response = self.llm.chat([
                {"role": "system", "content": (
                    "You are a research knowledge curator. Extract deep understanding "
                    "from experimental results. Focus on principles and mechanisms, "
                    "not surface-level observations."
                )},
                {"role": "user", "content": prompt},
            ])

            raw = strip_think(response["choices"][0]["message"]["content"])
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                return None
            return json.loads(json_match.group())

        except Exception as e:
            print(f"  [Brain] Knowledge extraction failed: {e}")
            return None

    # ── Routing (sparse activation) ─────────────────────────────

    def _route(self, goal: str, max_regions: int = 3) -> list[str]:
        """Select relevant brain regions for a goal.
        Simple keyword matching (like Kael's router).
        """
        goal_lower = goal.lower()
        scores = {}

        for region_name in self._list_regions():
            data = self._load_region(region_name)
            if not data:
                continue

            # Score by keyword overlap
            keywords = data.get("keywords", [region_name])
            score = sum(1 for kw in keywords if kw.lower() in goal_lower)

            # Boost regions with more content (they're more useful)
            content_size = (len(data.get("principles", []))
                           + len(data.get("experiments", [])))
            if content_size > 0:
                score += 0.5

            if score > 0:
                scores[region_name] = score

        # Sort by score, take top N
        sorted_regions = sorted(scores, key=scores.get, reverse=True)
        return sorted_regions[:max_regions]

    # ── Region Management ───────────────────────────────────────

    def _ensure_region(self, name: str):
        """Create a region if it doesn't exist."""
        region_dir = self.brain_dir / name
        region_dir.mkdir(exist_ok=True)
        data_file = region_dir / "knowledge.json"
        if not data_file.exists():
            default = {
                "keywords": [name],
                "principles": [],
                "experiments": [],
                "open_questions": [],
                "created": datetime.now().isoformat(),
            }
            data_file.write_text(json.dumps(default, indent=2))
            # Update meta
            self._meta.setdefault("regions", {})[name] = {
                "created": datetime.now().isoformat(),
                "access_count": 0,
            }
            self._save_meta()

    def _append_knowledge(self, region_name: str, knowledge: dict):
        """Append extracted knowledge to a region."""
        data = self._load_region(region_name) or {
            "keywords": [region_name],
            "principles": [], "experiments": [], "open_questions": [],
        }

        timestamp = datetime.now().isoformat()

        for p in knowledge.get("principles", []):
            p["added"] = timestamp
            data["principles"].append(p)

        for e in knowledge.get("experiments", []):
            e["added"] = timestamp
            data["experiments"].append(e)

        for q in knowledge.get("open_questions", []):
            if q not in data["open_questions"]:
                data["open_questions"].append(q)

        # Cap sizes to prevent bloat
        data["principles"] = data["principles"][-20:]
        data["experiments"] = data["experiments"][-15:]
        data["open_questions"] = data["open_questions"][-10:]

        self._save_region(region_name, data)

    def _list_regions(self) -> list[str]:
        """List all brain regions."""
        if not self.brain_dir.exists():
            return []
        return [
            d.name for d in self.brain_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ]

    def _load_region(self, name: str) -> dict | None:
        """Load a brain region's knowledge."""
        data_file = self.brain_dir / name / "knowledge.json"
        if not data_file.exists():
            return None
        try:
            return json.loads(data_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_region(self, name: str, data: dict):
        """Save a brain region."""
        region_dir = self.brain_dir / name
        region_dir.mkdir(exist_ok=True)
        data_file = region_dir / "knowledge.json"
        data_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _record_access(self, region_name: str):
        """Track access for routing optimization."""
        regions = self._meta.setdefault("regions", {})
        region_meta = regions.setdefault(region_name, {"access_count": 0})
        region_meta["access_count"] = region_meta.get("access_count", 0) + 1
        region_meta["last_accessed"] = datetime.now().isoformat()
        self._save_meta()

    def _suggest_region_name(self, goal: str) -> str:
        """Suggest a region name from a goal string."""
        # Simple heuristic: extract key ML concepts
        concepts = {
            "regularization": ["l1", "l2", "dropout", "weight decay", "regulariz"],
            "optimization": ["adam", "sgd", "optimizer", "learning rate", "lr", "momentum"],
            "architecture": ["cnn", "mlp", "transformer", "resnet", "attention", "layer"],
            "normalization": ["batch norm", "layer norm", "rmsnorm", "normaliz"],
            "training": ["epoch", "training", "convergence", "loss", "overfit"],
            "data": ["dataset", "augment", "sample", "cifar", "mnist", "imagenet"],
            "evaluation": ["benchmark", "metric", "accuracy", "f1", "compare"],
        }
        goal_lower = goal.lower()
        for name, keywords in concepts.items():
            if any(kw in goal_lower for kw in keywords):
                return name
        return "general"

    # ── Meta persistence ────────────────────────────────────────

    def _load_meta(self) -> dict:
        meta_file = self.brain_dir / "_meta.json"
        if meta_file.exists():
            try:
                return json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"regions": {}, "created": datetime.now().isoformat()}

    def _save_meta(self):
        meta_file = self.brain_dir / "_meta.json"
        meta_file.write_text(json.dumps(self._meta, indent=2))

    # ── Concept Mining (Opus v3) ────────────────────────────────

    def consolidate(self, force: bool = False) -> int:
        """Knowledge Pressure: compress findings into concepts.

        Runs after each mission. When a region has >5 principles,
        uses LLM to extract higher-level concepts.

        Returns number of new concepts created.
        """
        if not self.llm:
            return 0

        total_concepts = 0
        for region_name in self._list_regions():
            data = self._load_region(region_name)
            if not data:
                continue

            principles = data.get("principles", [])
            concepts = data.get("concepts", [])
            experiments = data.get("experiments", [])

            # Knowledge Pressure trigger: >5 principles without recent consolidation
            if not force and len(principles) < 5:
                continue

            # Check if we already consolidated recently
            last_consolidation = data.get("last_consolidation", "")
            if not force and last_consolidation:
                # Don't consolidate if we did it in the last mission
                added_since = [p for p in principles
                               if p.get("added", "") > last_consolidation]
                if len(added_since) < 3:
                    continue

            # Mine concepts from principles + experiments
            new_concepts = self._mine_concepts(
                region_name, principles, experiments, concepts
            )

            if new_concepts:
                # Append concepts with anchors (anti-drift)
                for c in new_concepts:
                    c["created"] = datetime.now().isoformat()
                    c["version"] = 1
                    # Evidence tracking: link to supporting principles
                    c.setdefault("supporting_evidence", [])

                data.setdefault("concepts", []).extend(new_concepts)
                data["concepts"] = data["concepts"][-15:]  # Cap
                data["last_consolidation"] = datetime.now().isoformat()
                self._save_region(region_name, data)

                total_concepts += len(new_concepts)
                print(f"  [Brain] {region_name}: mined {len(new_concepts)} concepts "
                      f"from {len(principles)} principles")

        return total_concepts

    def _mine_concepts(self, region_name: str, principles: list,
                       experiments: list, existing_concepts: list) -> list:
        """Use LLM to extract higher-level concepts from principles.

        This is the core of Knowledge Pressure: force the LLM to identify
        patterns, not just summarize.
        """
        principles_str = "\n".join(
            f"- {p.get('principle', '')} (confidence={p.get('confidence', '?')})"
            for p in principles[-15:]
        )
        experiments_str = "\n".join(
            f"- {e.get('description', '')} → {e.get('result', '')}"
            for e in experiments[-10:]
        )
        existing_str = "\n".join(
            f"- {c.get('concept', '')} (anchored: {c.get('anchor_definition', '')})"
            for c in (existing_concepts or [])
        )

        prompt = f"""You are a research knowledge curator for the domain "{region_name}".

## Current Principles (individual findings)
{principles_str}

## Past Experiments
{experiments_str}

{f"## Existing Concepts (do not duplicate){chr(10)}{existing_str}" if existing_str else ""}

## Your Task
Identify HIGHER-LEVEL CONCEPTS that explain multiple principles.

A concept is NOT a summary — it is a **general principle** that:
- Explains WHY multiple findings are true
- Can predict outcomes of untested experiments
- Has clear scope and limitations

Example:
Principles: "dropout 0.3 good", "L2 helps small data", "early stopping helps"
→ Concept: "Regularization-data tradeoff: model capacity must match dataset size.
   More regularization helps when data is limited, hurts when data is abundant."

Respond with JSON:
{{
  "concepts": [
    {{
      "concept": "short name",
      "anchor_definition": "precise definition that should NOT drift over time",
      "explanation": "why this concept explains the underlying principles",
      "supporting_evidence": ["principle 1 text", "principle 2 text"],
      "predictions": ["what this concept predicts for untested scenarios"],
      "scope": "when does this concept apply vs not apply",
      "novelty": "empirical|textbook"
    }}
  ]
}}

Rules:
- Only create concepts with 2+ supporting principles
- Anchor definition must be specific and falsifiable
- If no patterns emerge, return {{"concepts": []}}
- Do NOT duplicate existing concepts
- NOVELTY FILTER: Prefer concepts that contain SPECIFIC quantitative findings
  from our experiments (e.g. "Adam outperforms SGD by 7-9% on n=2000")
  over generic ML textbook knowledge (e.g. "Adam converges faster than SGD").
  Mark each concept's "novelty" field as "empirical" (from our data) or
  "textbook" (common knowledge). Only "empirical" concepts are valuable.
- Each prediction must be SPECIFIC and TESTABLE with a concrete experiment"""

        try:
            response = self.llm.chat([
                {"role": "system", "content": (
                    "Extract higher-level concepts from research findings. "
                    "Focus on explanatory power, not summarization. JSON only."
                )},
                {"role": "user", "content": prompt},
            ])
            raw = strip_think(response["choices"][0]["message"]["content"])
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                return []
            data = json.loads(json_match.group())
            return data.get("concepts", [])

        except Exception as e:
            print(f"  [Brain] Concept mining failed: {e}")
            return []

    def get_concepts_for_prompt(self, goal: str, max_concepts: int = 5) -> str:
        """Get relevant concepts for injection into planner/supervisor.

        Retrieves concepts (not raw findings) for efficient reasoning.
        This implements Hierarchical Retrieval: concept first, then findings.
        """
        regions = self._route(goal, max_regions=3)
        if not regions:
            return ""

        all_concepts = []
        for region_name in regions:
            data = self._load_region(region_name)
            if not data:
                continue
            for c in data.get("concepts", []):
                c["_region"] = region_name
                all_concepts.append(c)

        if not all_concepts:
            return ""

        parts = ["## Established Concepts (higher-level understanding)"]
        for c in all_concepts[-max_concepts:]:
            parts.append(f"**{c.get('concept', '?')}** [{c.get('_region', '')}]")
            parts.append(f"  Definition: {c.get('anchor_definition', c.get('explanation', ''))}")
            if c.get("predictions"):
                parts.append(f"  Predicts: {c['predictions'][0] if isinstance(c['predictions'], list) else c['predictions']}")
            if c.get("scope"):
                parts.append(f"  Scope: {c['scope']}")
            parts.append("")

        return "\n".join(parts)

    def get_knowledge_health(self) -> dict:
        """Knowledge health metrics (anti-entropy monitoring)."""
        total_findings = 0
        total_concepts = 0
        total_experiments = 0
        regions_without_concepts = []

        for region_name in self._list_regions():
            data = self._load_region(region_name)
            if not data:
                continue
            n_principles = len(data.get("principles", []))
            n_concepts = len(data.get("concepts", []))
            n_experiments = len(data.get("experiments", []))
            total_findings += n_principles
            total_concepts += n_concepts
            total_experiments += n_experiments
            if n_principles >= 5 and n_concepts == 0:
                regions_without_concepts.append(region_name)

        return {
            "total_findings": total_findings,
            "total_concepts": total_concepts,
            "total_experiments": total_experiments,
            "abstraction_ratio": total_concepts / max(total_findings, 1),
            "regions_needing_consolidation": regions_without_concepts,
        }

    # ── Concept Prediction Verification ─────────────────────────

    def verify_concepts(self, goal: str, analysis_summary: dict) -> list[dict]:
        """Verify concept predictions against new mission results.

        After each mission, check if any concept's predictions match or
        conflict with the new data. Update concept credibility accordingly.

        Returns list of verification results for logging.
        """
        if not self.llm or not analysis_summary:
            return []

        regions = self._route(goal, max_regions=3)
        if not regions:
            return []

        verifications = []

        for region_name in regions:
            data = self._load_region(region_name)
            if not data:
                continue

            concepts = data.get("concepts", [])
            if not concepts:
                continue

            # Find concepts with predictions
            concepts_with_predictions = [
                (i, c) for i, c in enumerate(concepts)
                if c.get("predictions") and isinstance(c["predictions"], list)
                and len(c["predictions"]) > 0
            ]
            if not concepts_with_predictions:
                continue

            # Ask LLM to check predictions against new results
            results = self._check_predictions(
                concepts_with_predictions, analysis_summary, goal
            )

            # Update concept credibility
            for result in results:
                idx = result.get("concept_index")
                if idx is None or idx >= len(concepts):
                    continue

                concept = concepts[idx]
                old_cred = concept.get("credibility", 0.5)

                if result["verdict"] == "confirmed":
                    concept["credibility"] = min(1.0, old_cred + 0.15)
                    concept.setdefault("confirmed_by", []).append(goal[:80])
                elif result["verdict"] == "refuted":
                    concept["credibility"] = max(0.0, old_cred - 0.2)
                    concept.setdefault("refuted_by", []).append(goal[:80])
                    # If credibility drops below 0.2, mark for revision
                    if concept["credibility"] < 0.2:
                        concept["needs_revision"] = True
                # "inconclusive" → no change

                concept["last_verified"] = datetime.now().isoformat()
                result["region"] = region_name
                result["concept_name"] = concept.get("concept", "?")
                result["new_credibility"] = concept.get("credibility", 0.5)
                verifications.append(result)

            # Save updated concepts
            if results:
                self._save_region(region_name, data)

        if verifications:
            confirmed = sum(1 for v in verifications if v["verdict"] == "confirmed")
            refuted = sum(1 for v in verifications if v["verdict"] == "refuted")
            print(f"  [Brain] Concept verification: {confirmed} confirmed, "
                  f"{refuted} refuted, "
                  f"{len(verifications) - confirmed - refuted} inconclusive")

        return verifications

    def _check_predictions(self, concepts_with_predictions: list,
                           analysis: dict, goal: str) -> list[dict]:
        """Use LLM to check concept predictions against new results."""
        analysis_str = json.dumps(analysis, indent=2, default=str)[:3000]

        concepts_str = ""
        for idx, c in concepts_with_predictions:
            preds = c.get("predictions", [])
            preds_str = "; ".join(preds[:3]) if isinstance(preds, list) else str(preds)
            concepts_str += (
                f"\n[{idx}] {c.get('concept', '?')}: "
                f"predictions=[{preds_str}] "
                f"(anchor: {c.get('anchor_definition', '')[:100]})"
            )

        prompt = f"""Check if any concept predictions are confirmed or refuted by this new experiment.

## New Experiment
Goal: {goal}
Results:
{analysis_str}

## Concepts with predictions
{concepts_str}

For each concept, determine if the new data CONFIRMS, REFUTES, or is INCONCLUSIVE for its predictions.
Only mark "confirmed" if the data directly supports a prediction.
Only mark "refuted" if the data directly contradicts a prediction.

Respond with JSON:
{{
  "results": [
    {{
      "concept_index": 0,
      "verdict": "confirmed|refuted|inconclusive",
      "evidence": "brief explanation of why"
    }}
  ]
}}

Rules:
- Only include concepts where the new data is RELEVANT to their predictions
- Skip concepts where the experiment doesn't test anything related
- Be conservative: require clear evidence, not vague overlap"""

        try:
            response = self.llm.chat([
                {"role": "system", "content": (
                    "You verify research concept predictions against new data. "
                    "Be strict: only confirm/refute with clear evidence. JSON only."
                )},
                {"role": "user", "content": prompt},
            ])
            raw = strip_think(response["choices"][0]["message"]["content"])
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                return []
            data = json.loads(json_match.group())
            return data.get("results", [])
        except Exception as e:
            print(f"  [Brain] Prediction check failed: {e}")
            return []

    def get_concept_predictions_for_goal(self, goal: str) -> str:
        """Get testable predictions from concepts relevant to a new mission.

        Injected into planner so it can design experiments that also test
        existing concept predictions (closing the loop).
        """
        regions = self._route(goal, max_regions=3)
        if not regions:
            return ""

        predictions = []
        for region_name in regions:
            data = self._load_region(region_name)
            if not data:
                continue
            for c in data.get("concepts", []):
                preds = c.get("predictions", [])
                if not preds:
                    continue
                cred = c.get("credibility", 0.5)
                name = c.get("concept", "?")
                for p in (preds[:2] if isinstance(preds, list) else [str(preds)]):
                    predictions.append(
                        f"- [{name}] (credibility={cred:.1f}): {p}"
                    )

        if not predictions:
            return ""

        return (
            "## Concept Predictions to Verify\n"
            "These predictions come from established concepts. "
            "If your experiment can test any of them, note whether results "
            "confirm or refute the prediction.\n"
            + "\n".join(predictions[:8])
        )

    # ── Bootstrap: seed from existing evolution store ───────────

    def bootstrap_from_evolution(self, evolution_store):
        """One-time migration: convert flat learnings to brain regions."""
        try:
            learnings_file = Path(evolution_store.store_dir) / "learnings.json"
            if not learnings_file.exists():
                return

            with open(learnings_file) as f:
                data = json.load(f)

            learnings = data.get("learnings", [])
            findings = [l for l in learnings if l.get("type") == "research_finding"]

            count = 0
            for finding in findings:
                content = finding.get("content", "")
                context = finding.get("context", "")

                # Determine region
                region = self._suggest_region_name(content + " " + context)
                self._ensure_region(region)

                # Convert to principle
                self._append_knowledge(region, {
                    "principles": [{
                        "principle": content,
                        "evidence": context,
                        "confidence": 0.6,
                        "source": "evolution_store_migration",
                    }],
                })
                count += 1

            if count:
                print(f"  [Brain] Bootstrapped {count} findings from evolution store")

        except Exception as e:
            print(f"  [Brain] Bootstrap failed: {e}")
