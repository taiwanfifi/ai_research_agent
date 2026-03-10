"""
Progress Reporter
==================
Generates markdown progress reports from mission state.
Two modes:
  1. Progress report (for system/LLM) — concise task log
  2. Research report (for human PM) — structured like a paper

Supports English (default) and Traditional Chinese (繁體中文).
"""

import os
import time


class Reporter:
    """Generates structured progress reports in English or Chinese."""

    def __init__(self, reports_dir: str, language: str = "en",
                 workspace_dir: str = None):
        self.reports_dir = reports_dir
        self.workspace_dir = workspace_dir
        self.language = language
        os.makedirs(reports_dir, exist_ok=True)

    def _list_workspace_files(self) -> list[dict]:
        """List files in workspace directory for report inclusion."""
        if not self.workspace_dir or not os.path.isdir(self.workspace_dir):
            return []
        files = []
        for root, dirs, fnames in os.walk(self.workspace_dir):
            for fn in fnames:
                if '__pycache__' in root or fn.startswith('.') or '.code_store' in root:
                    continue
                filepath = os.path.join(root, fn)
                rel_path = os.path.relpath(filepath, self.workspace_dir)
                ext = os.path.splitext(fn)[1].lower()
                file_type = ("code" if ext == ".py" else
                             "figure" if ext in (".png", ".jpg", ".svg") else
                             "data" if ext in (".csv", ".json", ".txt") else
                             "report" if ext == ".md" else "other")
                files.append({
                    "path": rel_path,
                    "name": fn,
                    "size": os.path.getsize(filepath),
                    "type": file_type,
                    "ext": ext,
                })
        return files

    def _load_workspace_metrics(self) -> dict:
        """Load all JSON result files from workspace and extract metrics.
        Returns {"files": {filename: data}, "all_metrics": {name: value}}."""
        import json
        result = {"files": {}, "all_metrics": {}}
        if not self.workspace_dir or not os.path.isdir(self.workspace_dir):
            return result
        for root, _, fnames in os.walk(self.workspace_dir):
            for fn in fnames:
                if not fn.endswith('.json') or fn.startswith('.'):
                    continue
                filepath = os.path.join(root, fn)
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    rel = os.path.relpath(filepath, self.workspace_dir)
                    result["files"][rel] = data
                    # Recursively extract numeric metrics
                    self._extract_metrics(data, result["all_metrics"], prefix=fn.replace('.json', ''))
                except Exception:
                    continue
        return result

    @staticmethod
    def _extract_metrics(obj, out: dict, prefix: str = ""):
        """Recursively extract numeric values from nested dict/list."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[key] = v
                elif isinstance(v, (dict, list)):
                    Reporter._extract_metrics(v, out, key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                Reporter._extract_metrics(item, out, f"{prefix}[{i}]")

    def generate(self, goal: str, completed_tasks: list[dict],
                 pending_tasks: list[dict], knowledge_stats: dict,
                 errors: list[str] = None,
                 working_memory: str = "") -> str:
        """Generate a markdown progress report in the configured language."""
        if self.language == "zh":
            report = self._build_zh_report(goal, completed_tasks, pending_tasks,
                                           knowledge_stats, errors,
                                           working_memory)
        else:
            report = self._build_en_report(goal, completed_tasks, pending_tasks,
                                           knowledge_stats, errors,
                                           working_memory)

        # Save progress report
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"progress_{self.language}_{ts}.md"
        filepath = os.path.join(self.reports_dir, filename)
        with open(filepath, "w") as f:
            f.write(report)

        # Also generate the research report (for human reading)
        research_report = self._build_research_report(
            goal, completed_tasks, pending_tasks,
            knowledge_stats, errors, working_memory
        )
        research_filename = f"research_report_{self.language}_{ts}.md"
        research_filepath = os.path.join(self.reports_dir, research_filename)
        with open(research_filepath, "w") as f:
            f.write(research_report)

        return report

    # ── Research Report (paper-quality) ─────────────────────────────

    @staticmethod
    def _clean_output(output: str) -> str:
        """Strip LLM narration and tool call artifacts from output for clean report."""
        import re
        # Remove tool call XML blocks (MiniMax format)
        cleaned = re.sub(r'<minimax:tool_call>.*?</invoke>\s*', '', output, flags=re.DOTALL)
        cleaned = re.sub(r'<invoke\s+name=.*?</invoke>\s*', '', cleaned, flags=re.DOTALL)
        # Remove code blocks that are tool call arguments (not results)
        cleaned = re.sub(r'<parameter\s+name="code">.*?</parameter>', '', cleaned, flags=re.DOTALL)
        # Remove decision envelope JSON
        cleaned = re.sub(r'```json\s*\{["\']decision["\'].*?\}\s*```', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\{"decision":\s*\{.*?\}\}', '', cleaned)
        # Remove "Let me...", "I'll...", "Now let me..." narration lines
        lines = cleaned.split('\n')
        filtered = []
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(p) for p in [
                "Let me ", "I'll ", "Now let me", "Now I'll", "Looking at",
                "The output was truncated", "There's an error",
                "Good - I found", "Excellent!", "I found", "Good progress!",
                "That wasn't the correct", "Now I have",
            ]):
                continue
            filtered.append(line)
        return '\n'.join(filtered).strip()

    @staticmethod
    def _extract_tables_and_findings(output: str) -> str:
        """Extract results tables, key findings, and analysis from output."""
        import re
        sections = []

        # Extract markdown tables
        table_pattern = re.compile(r'(\|.+\|(?:\n\|[-:|\s]+\|)?(?:\n\|.+\|)+)', re.MULTILINE)
        for match in table_pattern.finditer(output):
            sections.append(match.group())

        # Extract key findings / analysis sections
        for marker in ['### Key Findings', '### Analysis', '### Results', '## Results',
                       '### Summary', '## Summary', '### Experimental Setup',
                       '### Effect Sizes', '### Statistical']:
            idx = output.find(marker)
            if idx >= 0:
                end = len(output)
                for next_marker in ['## ', '### ']:
                    next_idx = output.find(next_marker, idx + len(marker) + 1)
                    if next_idx > 0:
                        end = min(end, next_idx)
                section = output[idx:end].strip()
                if section not in '\n'.join(sections):
                    sections.append(section)

        return '\n\n'.join(sections) if sections else output[:2000]

    def _build_research_report(self, goal, completed_tasks, pending_tasks,
                               knowledge_stats, errors, working_memory):
        """Build a paper-quality structured research report.

        Key design: grounded in workspace JSON files (ground truth),
        not LLM narration. Structured like an academic paper.
        """
        import json as _json
        import re
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        total = len(completed_tasks) + len(pending_tasks)

        explorer_tasks = [t for t in completed_tasks if t.get("worker") == "explorer" and t.get("success")]
        coder_tasks = [t for t in completed_tasks if t.get("worker") == "coder" and t.get("success")]
        reviewer_tasks = [t for t in completed_tasks if t.get("worker") == "reviewer" and t.get("success")]

        total_time = sum(t.get("elapsed_s", 0) or 0 for t in completed_tasks)
        failed_count = sum(1 for t in completed_tasks if not t.get("success"))
        success_count = len(completed_tasks) - failed_count

        # ── Load ground-truth metrics from workspace JSON files ──
        ws_metrics = self._load_workspace_metrics()
        ws_files = self._list_workspace_files()
        figure_files = [f for f in ws_files if f["type"] == "figure"]
        code_files = [f for f in ws_files if f["type"] == "code"]
        data_files = [f for f in ws_files if f["type"] == "data"]

        # ── Extract key result numbers from JSON files ──
        key_metrics = {}
        for fname, data in ws_metrics["files"].items():
            if isinstance(data, dict):
                key_metrics[fname] = data

        sections = []

        # ── Title & Metadata ──────────────────────────────────────
        sections.append(f"# {goal}")
        sections.append(f"\n*Automated Research Report — Generated {timestamp}*")
        sections.append(f"*{success_count}/{total} tasks completed | Total compute: {total_time:.0f}s*\n")

        # ── Abstract (synthesized, not grep'd) ────────────────────
        sections.append("## Abstract\n")
        abstract = self._synthesize_abstract(goal, key_metrics, working_memory,
                                              len(explorer_tasks), len(coder_tasks))
        sections.append(abstract)
        sections.append("")

        # ── 1. Introduction ──────────────────────────────────────
        sections.append("## 1. Introduction\n")
        sections.append(f"**Research Question:** {goal}\n")
        # Extract motivation from working memory
        if working_memory:
            for line in working_memory.split('\n'):
                stripped = line.strip()
                if any(kw in stripped.lower() for kw in ['key paper', 'citations', 'state-of-the-art', 'gap']):
                    sections.append(f"- {stripped.lstrip('- ')}")
        sections.append("")

        # ── 2. Related Work (deduplicated across explorer runs) ──
        if explorer_tasks:
            sections.append("## 2. Related Work\n")
            seen_titles = set()  # Deduplicate papers by title
            all_paper_entries = []
            all_meta_sections = []  # Key Methods, Research Gaps, etc.

            for t in explorer_tasks:
                output = self._clean_output(t.get("output", "") or "")
                if not output:
                    continue

                # Extract structured paper listings
                paper_sections = re.findall(
                    r'(###\s*\d+\..*?)(?=###\s*\d+\.|### Key|### Open|### Research|### Critical|$)',
                    output, re.DOTALL
                )
                for ps in paper_sections:
                    cleaned = ps.strip()
                    if len(cleaned) < 20:
                        continue
                    # Extract title for dedup (first bold text or first line)
                    title_match = re.search(r'\*\*(.+?)\*\*', cleaned)
                    title = title_match.group(1)[:60] if title_match else cleaned[:60]
                    if title.lower() not in seen_titles:
                        seen_titles.add(title.lower())
                        all_paper_entries.append(cleaned)

                # Extract taxonomy/methods/gaps (take the LAST/most complete version)
                for marker in ['### Key Methods', '### Open-Source', '### Research Gaps', '### Critical']:
                    idx = output.find(marker)
                    if idx >= 0:
                        end = len(output)
                        for nm in ['### ', '## ']:
                            ni = output.find(nm, idx + len(marker) + 1)
                            if ni > 0:
                                end = min(end, ni)
                        section = output[idx:end].strip()
                        # Replace older version of same section type
                        replaced = False
                        for i, existing in enumerate(all_meta_sections):
                            if existing.startswith(marker):
                                all_meta_sections[i] = section
                                replaced = True
                                break
                        if not replaced:
                            all_meta_sections.append(section)

                if not paper_sections:
                    # Fallback: use truncated output but only if no papers found yet
                    if not all_paper_entries:
                        all_paper_entries.append(output[:2000])

            # Write deduplicated papers
            for entry in all_paper_entries:
                sections.append(entry)
                sections.append("")

            # Write meta sections (methods, gaps, etc.)
            for ms in all_meta_sections:
                sections.append(ms)
                sections.append("")

        # ── 3. Methodology ───────────────────────────────────────
        if coder_tasks:
            sections.append("## 3. Methodology\n")

            # Extract experimental setup from JSON config files
            for fname, data in key_metrics.items():
                if any(kw in fname.lower() for kw in ['config', 'setup', 'param']):
                    sections.append(f"**Configuration** (`{fname}`):")
                    for k, v in (data.items() if isinstance(data, dict) else []):
                        if isinstance(v, (str, int, float, bool)):
                            sections.append(f"- {k}: {v}")
                    sections.append("")

            for t in coder_tasks:
                output = self._clean_output(t.get("output", "") or "")
                if not output:
                    continue

                for marker in ['### Architecture', '## Architecture', '### Experimental',
                               '### Files Created', '## Summary']:
                    idx = output.find(marker)
                    if idx >= 0:
                        end = len(output)
                        for nm in ['### ', '## ']:
                            ni = output.find(nm, idx + len(marker) + 1)
                            if ni > 0:
                                end = min(end, ni)
                        sections.append(output[idx:end].strip())
                        sections.append("")
                        break

        # ── 4. Results (grounded in workspace JSON) ──────────────
        sections.append("## 4. Results\n")

        # First: tables from result JSON files (ground truth)
        if key_metrics:
            sections.append("### Quantitative Results (from result files)\n")
            for fname, data in key_metrics.items():
                if any(kw in fname.lower() for kw in ['result', 'analysis', 'summary', 'metric']):
                    sections.append(f"**`{fname}`**")
                    # Format as readable key-value or table
                    self._format_json_as_markdown(data, sections)
                    sections.append("")

        # Second: tables from reviewer output
        if reviewer_tasks:
            for t in reviewer_tasks:
                output = self._clean_output(t.get("output", "") or "")
                if output:
                    findings = self._extract_tables_and_findings(output)
                    sections.append(findings)
                    sections.append("")
        elif coder_tasks:
            for t in coder_tasks:
                output = t.get("output", "") or ""
                findings = self._extract_tables_and_findings(output)
                if '|' in findings:
                    sections.append(findings)
                    sections.append("")

        # ── 5. Discussion ────────────────────────────────────────
        sections.append("## 5. Discussion\n")

        # Extract interpretation from reviewer output (try multiple section names)
        found_discussion = False
        for t in reviewer_tasks:
            output = t.get("output", "") or ""
            for marker in ['### Discussion & Interpretation', '### Discussion', '## Discussion',
                           '### Analysis', '### Why', '## Analysis']:
                idx = output.find(marker)
                if idx >= 0:
                    end = len(output)
                    for nm in ['### ', '## ']:
                        ni = output.find(nm, idx + len(marker) + 1)
                        if ni > 0:
                            end = min(end, ni)
                    interp = output[idx:end].strip()
                    if len(interp) > 50:
                        sections.append(interp)
                        sections.append("")
                        found_discussion = True
                    break
            if found_discussion:
                break

        # Extract analytical observations from working memory
        if working_memory:
            discussion_lines = []
            for line in working_memory.split('\n'):
                stripped = line.strip()
                if any(kw in stripped.lower() for kw in [
                    'root cause', 'finding', 'over-regulariz', 'underperform',
                    'outperform', 'suggest', 'implicat', 'because', 'hypothesis',
                    'surprising', 'unexpected', 'practical',
                ]):
                    discussion_lines.append(stripped.lstrip('- '))
            if discussion_lines:
                for dl in discussion_lines[:8]:
                    sections.append(f"- {dl}")
                sections.append("")

        # ── 5.1 Limitations ──────────────────────────────────────
        sections.append("### Limitations\n")
        # Generate specific limitations from actual mission data
        limitations = []
        if total_time < 600:
            limitations.append(f"Total compute time was {total_time:.0f}s, limiting model convergence and hyperparameter exploration.")
        if len(coder_tasks) > 0:
            seed_count = sum(1 for m in ws_metrics["all_metrics"] if 'seed' in m.lower())
            if seed_count < 5:
                limitations.append(f"Only {seed_count} seed(s) detected; ≥5 seeds recommended for reliable statistical conclusions.")
        if len(explorer_tasks) < 2:
            limitations.append("Literature review was limited to a single search iteration; a comprehensive review would require broader coverage.")
        if failed_count > 0:
            limitations.append(f"{failed_count} task(s) failed during execution, potentially leaving gaps in the analysis.")
        limitations.append("For publication-quality results, cross-validation and ablation studies would strengthen the conclusions.")
        for lim in limitations:
            sections.append(f"- {lim}")
        sections.append("")

        if errors:
            sections.append("### Challenges Encountered\n")
            for e in errors[:5]:
                sections.append(f"- {e}")
            sections.append("")

        # ── 6. Conclusion ────────────────────────────────────────
        sections.append("## 6. Conclusion\n")
        conclusion = self._synthesize_conclusion(goal, key_metrics, working_memory)
        sections.append(conclusion)
        sections.append("")

        # ── 7. Artifacts ─────────────────────────────────────────
        if ws_files:
            sections.append("## 7. Artifacts\n")

            if figure_files:
                sections.append("### Figures")
                for f in figure_files:
                    # Add caption hint based on filename
                    caption = self._figure_caption(f["name"], key_metrics)
                    sections.append(f"- **`{f['path']}`**: {caption}")
                sections.append("")

            if code_files:
                sections.append("### Code")
                for f in code_files:
                    sections.append(f"- `{f['path']}` ({f['size']:,} bytes)")
                sections.append("")

            if data_files:
                sections.append("### Data")
                for f in data_files:
                    sections.append(f"- `{f['path']}`")
                sections.append("")

        # ── Appendix ─────────────────────────────────────────────
        sections.append("\n---\n")
        sections.append("## Appendix: Execution Log\n")
        sections.append("| # | Worker | Task | Status | Time |")
        sections.append("|---|--------|------|--------|------|")
        for i, t in enumerate(completed_tasks):
            worker = t.get("worker", "?")
            task_text = (t.get("task", "") or "")[:60].replace("|", "/")
            status = "OK" if t.get("success") else "FAIL"
            elapsed = t.get("elapsed_s", 0) or 0
            sections.append(f"| {i+1} | {worker} | {task_text} | {status} | {elapsed:.0f}s |")

        return "\n".join(sections)

    # ── Report synthesis helpers ─────────────────────────────────

    @staticmethod
    def _synthesize_abstract(goal, key_metrics, working_memory, n_explorer, n_coder):
        """Write a concise academic abstract from ground-truth metrics.
        Structure: background → method → key result → significance."""
        import re as _re

        # 1. Background sentence (strip implementation details from goal)
        goal_short = goal.split('.')[0].strip()
        parts = [f"We investigate {goal_short.lower()}."]

        # 2. Method sentence
        method_parts = []
        if n_explorer > 0:
            method_parts.append("a literature review")
        if n_coder > 0:
            method_parts.append(f"{n_coder} experimental configuration(s)")
        if method_parts:
            parts.append(f"Our approach combines {' and '.join(method_parts)}.")

        # 3. Key result — find the best method and its accuracy from analysis_summary
        best_method = None
        conclusion_text = None
        for fname, data in key_metrics.items():
            if not isinstance(data, dict):
                continue
            if 'conclusion' in data:
                conclusion_text = data['conclusion']
            if 'best_method' in data:
                best_method = data['best_method']

        if conclusion_text:
            parts.append(conclusion_text)
        else:
            # Fallback: extract headline metrics
            headline = []
            for fname, data in key_metrics.items():
                if not isinstance(data, dict):
                    continue
                # Look for methods with mean_accuracy
                methods = data.get('methods', data.get('results', {}))
                if isinstance(methods, dict):
                    for method_name, method_data in methods.items():
                        if isinstance(method_data, dict):
                            acc = method_data.get('mean_accuracy', method_data.get('accuracy_mean'))
                            if acc is not None:
                                std = method_data.get('std_accuracy', method_data.get('accuracy_std', 0))
                                headline.append(f"{method_name}: {acc:.1f}%±{std:.1f}%")
                # Look for statistical significance
                stats = data.get('statistics', data.get('statistical_tests', {}))
                if isinstance(stats, dict):
                    for test_name, test_data in stats.items():
                        if isinstance(test_data, dict):
                            p = test_data.get('p_value')
                            if p is not None and p < 0.05:
                                headline.append(f"p={p:.4f}")
            if headline:
                parts.append(f"Results: {'; '.join(headline[:4])}.")

        # 4. Core finding from working memory
        if working_memory and not conclusion_text:
            for line in working_memory.split('\n'):
                stripped = line.strip()
                if stripped.startswith('**Core Finding') or stripped.startswith('- **Core'):
                    clean = _re.sub(r'\*\*', '', stripped).lstrip('- ').strip()
                    parts.append(clean)
                    break

        return ' '.join(parts)

    @staticmethod
    def _synthesize_conclusion(goal, key_metrics, working_memory):
        """Write conclusion grounded in actual results.
        Structure: restate finding → statistical evidence → practical implication → future work."""
        import re as _re
        parts = []

        # 1. Core finding from analysis_summary.json conclusion field
        conclusion_from_json = None
        best_method = None
        for fname, data in key_metrics.items():
            if not isinstance(data, dict):
                continue
            if 'conclusion' in data and isinstance(data['conclusion'], str):
                conclusion_from_json = data['conclusion']
            if 'best_method' in data:
                best_method = data['best_method']

        if conclusion_from_json:
            parts.append(conclusion_from_json)

        # 2. Core finding from working memory (if no JSON conclusion)
        if not conclusion_from_json and working_memory:
            for line in working_memory.split('\n'):
                stripped = line.strip()
                if stripped.startswith('**Core Finding') or stripped.startswith('- **Core'):
                    clean = _re.sub(r'\*\*', '', stripped).lstrip('- ').strip()
                    parts.append(clean)
                elif stripped.startswith('**Statistical') or stripped.startswith('- **Statistical'):
                    clean = _re.sub(r'\*\*', '', stripped).lstrip('- ').strip()
                    parts.append(clean)

        # 3. Statistical evidence from JSON
        if key_metrics:
            stat_evidence = []
            for fname, data in key_metrics.items():
                if not isinstance(data, dict):
                    continue
                stats = data.get('statistics', data.get('statistical_tests', {}))
                if isinstance(stats, dict):
                    for test_name, test_data in stats.items():
                        if isinstance(test_data, dict):
                            p = test_data.get('p_value')
                            d = test_data.get('cohens_d')
                            if p is not None:
                                sig = "statistically significant" if p < 0.05 else "not statistically significant"
                                stat_evidence.append(f"{test_name}: p={p:.4f} ({sig})")
                            if d is not None and isinstance(d, (int, float)):
                                magnitude = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
                                stat_evidence.append(f"Cohen's d={d:.2f} ({magnitude} effect)")
            if stat_evidence and not conclusion_from_json:
                parts.append("Statistical analysis: " + "; ".join(stat_evidence[:3]) + ".")

        # 4. Future work from working memory
        if working_memory:
            for line in working_memory.split('\n'):
                stripped = line.strip()
                if any(kw in stripped.lower() for kw in ['next priority', 'future', 'next step']):
                    clean = _re.sub(r'\*\*', '', stripped).lstrip('- ').strip()
                    parts.append(f"Future work: {clean}")
                    break

        if not parts:
            goal_short = goal.split('.')[0].strip()
            parts.append(f"This study addressed {goal_short.lower()}. See Results section for quantitative findings and statistical analysis.")

        return '\n\n'.join(parts)

    @staticmethod
    def _figure_caption(filename, key_metrics):
        """Generate a descriptive caption for a figure based on its filename."""
        name = filename.lower().replace('.png', '').replace('.jpg', '').replace('_', ' ')
        if 'comparison' in name or 'bar' in name:
            return "Comparison of methods with error bars (see Results for statistical significance)."
        elif 'loss' in name or 'training' in name or 'curve' in name:
            return "Training loss curves across experimental conditions."
        elif 'confusion' in name:
            return "Confusion matrix showing classification performance."
        elif 'ablation' in name:
            return "Ablation study results showing component contributions."
        elif 'chart' in name:
            return "Performance comparison chart."
        return f"Visualization: {name}."

    @staticmethod
    def _format_json_as_markdown(data, sections, indent=0):
        """Format a JSON dict as readable markdown."""
        if not isinstance(data, dict):
            return
        prefix = "  " * indent
        for k, v in data.items():
            if isinstance(v, dict):
                sections.append(f"{prefix}- **{k}**:")
                Reporter._format_json_as_markdown(v, sections, indent + 1)
            elif isinstance(v, list):
                if v and isinstance(v[0], (int, float)):
                    sections.append(f"{prefix}- **{k}**: {v}")
                else:
                    sections.append(f"{prefix}- **{k}**: {len(v)} items")
            elif isinstance(v, (int, float)):
                sections.append(f"{prefix}- **{k}**: {v}")
            elif isinstance(v, str) and len(v) < 200:
                sections.append(f"{prefix}- **{k}**: {v}")

    # ── Progress Report (for system) ────────────────────────────────

    def _build_en_report(self, goal: str, completed_tasks: list[dict],
                         pending_tasks: list[dict], knowledge_stats: dict,
                         errors: list[str] = None,
                         working_memory: str = "") -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        total = len(completed_tasks) + len(pending_tasks)

        sections = [
            f"# Research Mission Report",
            f"\nGenerated: {timestamp}",
            f"\n## Mission Goal\n\n{goal}",
        ]

        if working_memory:
            sections.append(f"\n## Current Understanding\n\n{working_memory}")

        sections.append(f"\n## Progress ({len(completed_tasks)}/{total} tasks)")
        sections.append("")

        if completed_tasks:
            sections.append("### Completed")
            for t in completed_tasks:
                worker = t.get("worker", "unknown")
                task = t.get("task", "")
                elapsed = t.get("elapsed_s", "?")
                sections.append(f"- [{worker}] {task} ({elapsed}s)")

        if pending_tasks:
            sections.append("\n### Pending")
            for t in pending_tasks:
                worker = t.get("worker", "unknown")
                task = t.get("task", "")
                sections.append(f"- [{worker}] {task}")

        sections.append(f"\n## Knowledge Acquired")
        sections.append(f"\n- Total items: {knowledge_stats.get('total_items', 0)}")
        for cat, count in knowledge_stats.get("by_category", {}).items():
            if count > 0:
                sections.append(f"- {cat}: {count} items")

        if errors:
            sections.append(f"\n## Errors")
            for e in errors:
                sections.append(f"- {e}")

        if pending_tasks:
            sections.append(f"\n## Next Steps")
            next_task = pending_tasks[0]
            sections.append(f"\nNext: [{next_task.get('worker', '?')}] {next_task.get('task', '?')}")

        return "\n".join(sections)

    def _build_zh_report(self, goal: str, completed_tasks: list[dict],
                         pending_tasks: list[dict], knowledge_stats: dict,
                         errors: list[str] = None,
                         working_memory: str = "") -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        total = len(completed_tasks) + len(pending_tasks)

        sections = [
            f"# 研究任務報告",
            f"\n產生時間：{timestamp}",
            f"\n## 任務目標\n\n{goal}",
        ]

        if working_memory:
            sections.append(f"\n## 目前研究理解\n\n{working_memory}")

        sections.extend([
            f"\n## 進度（{len(completed_tasks)}/{total} 項任務）",
            "",
        ])

        if completed_tasks:
            sections.append("### 已完成")
            for t in completed_tasks:
                worker = t.get("worker", "unknown")
                task = t.get("task", "")
                elapsed = t.get("elapsed_s", "?")
                sections.append(f"- [{worker}] {task}（{elapsed}秒）")

        if pending_tasks:
            sections.append("\n### 待執行")
            for t in pending_tasks:
                worker = t.get("worker", "unknown")
                task = t.get("task", "")
                sections.append(f"- [{worker}] {task}")

        sections.append(f"\n## 知識庫統計")
        sections.append(f"\n- 總計：{knowledge_stats.get('total_items', 0)} 項")
        for cat, count in knowledge_stats.get("by_category", {}).items():
            if count > 0:
                sections.append(f"- {cat}：{count} 項")

        if errors:
            sections.append(f"\n## 錯誤記錄")
            for e in errors:
                sections.append(f"- {e}")

        if pending_tasks:
            sections.append(f"\n## 後續步驟")
            next_task = pending_tasks[0]
            sections.append(f"\n下一步：[{next_task.get('worker', '?')}] {next_task.get('task', '?')}")

        return "\n".join(sections)
