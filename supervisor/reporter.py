"""
Progress Reporter
==================
Generates markdown progress reports from mission state.
Supports English (default) and Traditional Chinese (繁體中文).
"""

import os
import time


class Reporter:
    """Generates structured progress reports in English or Chinese."""

    def __init__(self, reports_dir: str, language: str = "en"):
        self.reports_dir = reports_dir
        self.language = language
        os.makedirs(reports_dir, exist_ok=True)

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

        # Save to file
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"progress_{self.language}_{ts}.md"
        filepath = os.path.join(self.reports_dir, filename)
        with open(filepath, "w") as f:
            f.write(report)

        return report

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
