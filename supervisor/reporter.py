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

    # ── Research Report (for human PM) ──────────────────────────────

    def _build_research_report(self, goal, completed_tasks, pending_tasks,
                               knowledge_stats, errors, working_memory):
        """Build a structured research report readable by humans."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        total = len(completed_tasks) + len(pending_tasks)

        # Categorize tasks by worker
        explorer_tasks = [t for t in completed_tasks if t.get("worker") == "explorer"]
        coder_tasks = [t for t in completed_tasks if t.get("worker") == "coder"]
        reviewer_tasks = [t for t in completed_tasks if t.get("worker") == "reviewer"]

        # Calculate total time
        total_time = sum(t.get("elapsed_s", 0) or 0 for t in completed_tasks)
        failed_count = sum(1 for t in completed_tasks if not t.get("success"))

        sections = []

        # Header
        sections.append(f"# Research Report: {goal}")
        sections.append(f"\nGenerated: {timestamp}")
        sections.append(f"Status: {len(completed_tasks)}/{total} tasks completed | "
                        f"{failed_count} failures | Total time: {total_time:.0f}s")
        sections.append(f"\n---\n")

        # 1. Background & Objective
        sections.append("## 1. Background & Objective\n")
        sections.append(f"**Research Goal:** {goal}\n")
        if working_memory:
            sections.append("**Current Understanding:**\n")
            sections.append(f"{working_memory}\n")

        # 2. Literature Review
        if explorer_tasks:
            sections.append("## 2. Literature Review\n")
            for t in explorer_tasks:
                status = "Completed" if t.get("success") else "Failed"
                elapsed = t.get("elapsed_s", 0) or 0
                sections.append(f"### Search: {(t.get('task', '') or '')[:100]}")
                sections.append(f"*Status: {status} | Time: {elapsed:.0f}s*\n")
                output = (t.get("output", "") or "")
                if output:
                    # Show up to 1500 chars of the explorer's findings
                    sections.append(output[:1500])
                    if len(output) > 1500:
                        sections.append(f"\n*... ({len(output)} chars total, truncated)*")
                sections.append("")

        # 3. Implementation
        if coder_tasks:
            sections.append("## 3. Implementation\n")
            for t in coder_tasks:
                status = "Completed" if t.get("success") else "Failed"
                elapsed = t.get("elapsed_s", 0) or 0
                sections.append(f"### Task: {(t.get('task', '') or '')[:100]}")
                sections.append(f"*Status: {status} | Time: {elapsed:.0f}s*\n")
                output = (t.get("output", "") or "")
                if output:
                    sections.append(output[:1500])
                    if len(output) > 1500:
                        sections.append(f"\n*... ({len(output)} chars total, truncated)*")
                sections.append("")

        # 4. Experimental Results
        if reviewer_tasks:
            sections.append("## 4. Experimental Results\n")
            for t in reviewer_tasks:
                status = "Completed" if t.get("success") else "Failed"
                elapsed = t.get("elapsed_s", 0) or 0
                sections.append(f"### Evaluation: {(t.get('task', '') or '')[:100]}")
                sections.append(f"*Status: {status} | Time: {elapsed:.0f}s*\n")
                output = (t.get("output", "") or "")
                if output:
                    sections.append(output[:1500])
                    if len(output) > 1500:
                        sections.append(f"\n*... ({len(output)} chars total, truncated)*")
                sections.append("")

        # 5. Discussion
        sections.append("## 5. Discussion\n")
        if errors:
            sections.append("### Issues Encountered\n")
            for e in errors:
                sections.append(f"- {e}")
            sections.append("")

        sections.append("### Knowledge Acquired\n")
        sections.append(f"Total knowledge items: {knowledge_stats.get('total_items', 0)}\n")
        for cat, count in knowledge_stats.get("by_category", {}).items():
            if count > 0:
                sections.append(f"- **{cat}**: {count} items")
        sections.append("")

        # 6. Conclusion & Next Steps
        sections.append("## 6. Conclusion & Next Steps\n")
        if pending_tasks:
            sections.append("### Remaining Tasks\n")
            for t in pending_tasks[:5]:
                worker = t.get("worker", "unknown")
                task = t.get("task", "")
                sections.append(f"- [{worker}] {task}")
        else:
            sections.append("All planned tasks have been completed.\n")

        # 7. Artifacts
        ws_files = self._list_workspace_files()
        if ws_files:
            sections.append("## 7. Research Artifacts\n")
            code_files = [f for f in ws_files if f["type"] == "code"]
            figure_files = [f for f in ws_files if f["type"] == "figure"]
            data_files = [f for f in ws_files if f["type"] == "data"]
            report_files = [f for f in ws_files if f["type"] == "report"]

            if code_files:
                sections.append("### Code Files")
                for f in code_files:
                    sections.append(f"- `{f['path']}` ({f['size']:,} bytes)")
                sections.append("")

            if figure_files:
                sections.append("### Generated Figures")
                for f in figure_files:
                    sections.append(f"- `{f['path']}` ({f['size']:,} bytes)")
                sections.append("")

            if data_files:
                sections.append("### Data Files")
                for f in data_files:
                    sections.append(f"- `{f['path']}` ({f['size']:,} bytes)")
                sections.append("")

            if report_files:
                sections.append("### Reports")
                for f in report_files:
                    sections.append(f"- `{f['path']}` ({f['size']:,} bytes)")
                sections.append("")

        # Appendix: Full task log
        sections.append("\n---\n")
        sections.append("## Appendix: Task Execution Log\n")
        sections.append("| # | Worker | Task | Status | Time |")
        sections.append("|---|--------|------|--------|------|")
        for i, t in enumerate(completed_tasks):
            worker = t.get("worker", "?")
            task_text = (t.get("task", "") or "")[:60].replace("|", "/")
            status = "OK" if t.get("success") else "FAIL"
            elapsed = t.get("elapsed_s", 0) or 0
            sections.append(f"| {i+1} | {worker} | {task_text} | {status} | {elapsed:.0f}s |")

        return "\n".join(sections)

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
