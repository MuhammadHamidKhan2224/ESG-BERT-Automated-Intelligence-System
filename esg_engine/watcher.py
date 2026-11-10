"""
ESGWatcher: Automated Batch Processor & Directory Watcher.
Monitors an incoming directory for corporate disclosures or reports,
audits them through the ESGBERT pipeline, and writes output reports.
"""

import os
import time
import json
from pathlib import Path
from typing import List, Optional

from .report_analyzer import ESGReportAnalyzer


class ESGWatcher:
    """
    Automated batch watcher for corporate ESG document pipelines.
    """

    def __init__(
        self,
        incoming_dir: str = "data/incoming",
        output_dir: str = "data/output_reports",
        analyzer: Optional[ESGReportAnalyzer] = None,
    ):
        self.incoming_dir = Path(incoming_dir)
        self.output_dir = Path(output_dir)
        self.analyzer = analyzer or ESGReportAnalyzer()
        self.processed_files = set()

        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_file(self, file_path: Path) -> Optional[Path]:
        """
        Audits a single file and saves Markdown and JSON reports.
        """
        try:
            print(f"[ESGWatcher] Processing: {file_path.name}")
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            title = file_path.stem.replace("_", " ").title()
            summary = self.analyzer.audit_document(content, document_title=title)

            # Export Markdown report
            md_path = self.output_dir / f"{file_path.stem}_esg_audit.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(summary.generate_markdown_report())

            # Export JSON scorecard
            json_path = self.output_dir / f"{file_path.stem}_esg_audit.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary.to_dict(), f, indent=2)

            print(f"[ESGWatcher] Successfully audited {file_path.name} -> {md_path.name}")
            self.processed_files.add(str(file_path.resolve()))
            return md_path

        except Exception as e:
            print(f"[ESGWatcher] Error processing {file_path.name}: {e}")
            return None

    def run_batch(self) -> List[Path]:
        """
        Processes all unprocessed supported files in the incoming directory.
        """
        supported_exts = {".txt", ".md", ".csv"}
        generated_reports = []

        for p in self.incoming_dir.iterdir():
            if p.is_file() and p.suffix.lower() in supported_exts:
                if str(p.resolve()) not in self.processed_files:
                    out = self.process_file(p)
                    if out:
                        generated_reports.append(out)

        return generated_reports

    def watch(self, interval_seconds: int = 5, max_iterations: Optional[int] = None):
        """
        Runs a continuous watch loop on the incoming directory.
        """
        print(f"[ESGWatcher] Watching directory: {self.incoming_dir.resolve()}")
        print(f"[ESGWatcher] Saving audit reports to: {self.output_dir.resolve()}")
        iterations = 0

        try:
            while True:
                self.run_batch()
                iterations += 1
                if max_iterations and iterations >= max_iterations:
                    break
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("[ESGWatcher] Monitoring stopped by user.")
