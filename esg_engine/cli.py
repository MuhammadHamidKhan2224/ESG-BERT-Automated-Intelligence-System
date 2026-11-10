"""
Command Line Interface for the ESG BERT Automation Engine.
"""

import argparse
import json
import sys
from pathlib import Path

from .classifier import ESGBertClassifier
from .dataset_loader import ESGDatasetLoader, HF_ESG_DATASETS
from .report_analyzer import ESGReportAnalyzer
from .watcher import ESGWatcher


def cmd_analyze(args):
    print(f"\n🔍 Analyzing text with ESGBERT Suite...")
    classifier = ESGBertClassifier(threshold=args.threshold)
    res = classifier.analyze_text(args.text)

    print("\n" + "=" * 60)
    print(f"Text: \"{res.text}\"")
    print(f"Dominant Pillar : {res.dominant_pillar.upper()}")
    print(f"Active Pillars  : {', '.join(res.active_pillars) if res.active_pillars else 'None'}")
    print(f"Max ESG Score   : {res.max_esg_score:.4f}")
    print(f"Action Orientation: {res.action_orientation}")
    print(f"Sub-Topics      : {', '.join(res.sub_topics) if res.sub_topics else 'None'}")
    print("-" * 60)
    print(f"🌱 Environmental : {res.environmental.score:.4f} ({res.environmental.confidence_label}) [Active: {res.environmental.is_pillar}]")
    print(f"🤝 Social        : {res.social.score:.4f} ({res.social.confidence_label}) [Active: {res.social.is_pillar}]")
    print(f"🏛️ Governance    : {res.governance.score:.4f} ({res.governance.confidence_label}) [Active: {res.governance.is_pillar}]")
    print("=" * 60 + "\n")

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))


def cmd_audit(args):
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"\n📄 Auditing Corporate Report: {file_path.name}...")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    analyzer = ESGReportAnalyzer()
    summary = analyzer.audit_document(content, document_title=file_path.stem)

    md_report = summary.generate_markdown_report()
    print("\n" + md_report + "\n")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix == ".json":
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(summary.to_dict(), f, indent=2)
        else:
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(md_report)
        print(f"💾 Report saved to: {out_p.resolve()}")


def cmd_stream_hf(args):
    pillar = args.pillar.lower()
    print(f"\n🌐 Streaming {args.limit} records from Hugging Face dataset: '{HF_ESG_DATASETS.get(pillar, pillar)}'...")
    loader = ESGDatasetLoader()

    if args.search:
        print(f"Filtering records matching keyword: '{args.search}'")
        records = loader.search_stream(pillar, args.search, max_matches=args.limit)
    else:
        records = loader.stream_dataset(pillar, limit=args.limit, offset=args.offset)

    print(f"Retrieved {len(records)} sentences:\n")
    for idx, r in enumerate(records, 1):
        sentence = r.get("sentence", "")
        print(f"[{idx}] {sentence}\n")


def cmd_benchmark(args):
    print(f"\n📊 Creating balanced ESG benchmark from Hugging Face ({args.samples} samples per pillar)...")
    loader = ESGDatasetLoader()
    df = loader.create_balanced_benchmark(samples_per_pillar=args.samples, output_path=args.output)
    print(f"Benchmark created with {len(df)} total samples.")
    print("Class breakdown:\n", df["pillar"].value_counts())
    if args.output:
        print(f"Saved benchmark to {args.output}")


def cmd_watch(args):
    print(f"\n👀 Starting Automated ESG Watcher on {args.incoming}...")
    watcher = ESGWatcher(incoming_dir=args.incoming, output_dir=args.output)
    watcher.watch(interval_seconds=args.interval)


def main():
    parser = argparse.ArgumentParser(
        prog="esg_engine",
        description="Automated ESG BERT Intelligence & Corporate Auditing Suite",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: analyze
    p_analyze = subparsers.add_parser("analyze", help="Classify a single sentence or text")
    p_analyze.add_argument("text", type=str, help="Text to classify")
    p_analyze.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    p_analyze.add_argument("--json", action="store_true", help="Output raw JSON")

    # Command: audit
    p_audit = subparsers.add_parser("audit", help="Audit an entire corporate document file")
    p_audit.add_argument("file", type=str, help="Path to report file (.txt, .md)")
    p_audit.add_argument("-o", "--output", type=str, help="Path to save output report (.md or .json)")

    # Command: stream-hf
    p_stream = subparsers.add_parser("stream-hf", help="Stream live rows from Hugging Face ESG datasets")
    p_stream.add_argument("pillar", choices=["environmental", "social", "governance"], help="Pillar dataset")
    p_stream.add_argument("--limit", type=int, default=5, help="Number of records to stream")
    p_stream.add_argument("--offset", type=int, default=0, help="Initial record offset")
    p_stream.add_argument("--search", type=str, help="Filter stream by keyword")

    # Command: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Create a balanced benchmark dataset")
    p_bench.add_argument("--samples", type=int, default=10, help="Samples per pillar")
    p_bench.add_argument("-o", "--output", type=str, default="data/esg_benchmark.csv", help="Output path")

    # Command: watch
    p_watch = subparsers.add_parser("watch", help="Run automated directory watcher")
    p_watch.add_argument("--incoming", type=str, default="data/incoming", help="Incoming folder")
    p_watch.add_argument("--output", type=str, default="data/output_reports", help="Output folder")
    p_watch.add_argument("--interval", type=int, default=5, help="Poll interval in seconds")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "audit":
        cmd_audit(args)
    elif args.command == "stream-hf":
        cmd_stream_hf(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "watch":
        cmd_watch(args)


if __name__ == "__main__":
    main()
