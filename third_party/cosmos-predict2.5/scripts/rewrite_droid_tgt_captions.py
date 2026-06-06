#!/usr/bin/env python3
"""Insert a TAViD-style [TGT] marker before the target object phrase.

The DROID target-aware datasets already carry object masks, but the original
Cosmos captions are plain task commands. This script rewrites ``metas/*.txt`` so
the text side also contains an explicit target-object marker, e.g.

    put the blue object in the drawer
    put the [TGT] blue object in the drawer

It keeps the first direct object of common robot-manipulation verbs. The rewrite
is intentionally conservative and emits a report for auditing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


LEGACY_SUFFIX_RE = re.compile(
    r"\s*The robot interacts with the \[TGT\] target object\.?\s*$",
    flags=re.IGNORECASE,
)

VERB_PATTERN = re.compile(
    r"\b("
    r"pick up|switch off|switch on|straighten out|press on|spread out|pile up|scoop up|"
    r"stretch out|left click|right click|click on|correctly position|spill out|scrunch up|"
    r"group up|pack away|fix up|slightly untwist|cut out|heap up|mix in|pick|grab|grasp|take|get|"
    r"put|place|move|bring|shift|"
    r"lift|remove|pull|push|open|close|flip|turn|slide|pour|hang|wipe|fold|unfold|"
    r"insert|stack|cover|uncover|rotate|separate|use|make|rest|plug|throw|erase|scrub|"
    r"press|unwind|clean|stir|squeeze|empty|create|face|unhook|wrap|unhang|unstack|"
    r"straighten|transfer|twist|lay|spread|set|click|align|arrange|draw|position|pile|"
    r"center|release|drop|scoop|roll|switch|sweep|untangle|flick|swap|connect|"
    r"rearrange|tap|stretch|adjust|shove|mix|tip|wind|unlock|tear|stuff|unravel|unwrap|"
    r"attach|fix|touch|hook|point|spill|replace|scrunch|hold|spell|gather|rub|closed|"
    r"coil|lean|iron|group|pack|stick|detach|turnover|screw|spoon|shake|pu|lock|add|"
    r"spin|untwist|join|form|retrieve|fill|spray|unlatch|sprinkle|untie|heap|carry|"
    r"nudge|organize|hit|dial|divide|split|poke|hung|plcae|tilt|uncoil|loop|"
    r"disconnect|pur|pus|collect|unassemble|seal|bunch up|bunch|scribble on|write on|"
    r"interact with|peel|tuck|readjust|ford|mount|moe|pout|pace|cut|partially torn|"
    r"unroll|rip off|drag|rip up|build|flush|raise|correct|brush"
    r"|dip|crumple up|crumple|swipe|tighten|secure|pump|unscrew|knock down|knock|"
    r"extend|unpack|ppick|convert|clip|unplug from|unplug|restack|change|shut"
    r")\s+",
    flags=re.IGNORECASE,
)

STOP_PATTERN = re.compile(
    r"(?="
    r"\s+(?:in|into|inside|on|onto|to|from|out of|off|over|under|closer|away|"
    r"toward|towards|near|next to|with|using|through|back|around|behind|beside)\b"
    r"|\s*,"
    r"|\s+and\s+(?:then\s+)?(?:put|place|move|pick|pick up|take|open|close|flip|turn|slide|hang|pour|remove|push|pull)\b"
    r"|\s+then\s+"
    r"|[.]"
    r"|$"
    r")",
    flags=re.IGNORECASE,
)

ARTICLE_RE = re.compile(
    r"^(?P<article>(?:the|a|an|this|that|these|those|another|one)\s+)(?P<rest>.+)$",
    flags=re.IGNORECASE,
)

BAD_OBJECTS = {"it", "them", "this", "that", "there"}
PROMPT_TEMPLATE = "A Franka robotic arm with a parallel-jaw gripper {task}."


@dataclass
class RewriteResult:
    path: str
    changed: bool
    status: str
    verb: str = ""
    object_phrase: str = ""
    original: str = ""
    rewritten: str = ""


def strip_legacy_suffix(caption: str) -> str:
    return LEGACY_SUFFIX_RE.sub("", caption).strip()


def insert_marker_in_phrase(phrase: str) -> str:
    phrase = phrase.strip()
    match = ARTICLE_RE.match(phrase)
    if match:
        return f"{match.group('article')}[TGT] {match.group('rest')}"
    return f"[TGT] {phrase}"


def rewrite_caption(caption: str) -> tuple[str, str, str, str]:
    original = caption.strip()
    caption = strip_legacy_suffix(original)
    if "[TGT]" in caption:
        return caption, "already_marked", "", ""

    for verb_match in VERB_PATTERN.finditer(caption):
        obj_start = verb_match.end()
        stop_match = STOP_PATTERN.search(caption, obj_start)
        obj_end = stop_match.start() if stop_match else len(caption)
        phrase = caption[obj_start:obj_end].strip()
        phrase = phrase.strip(" ,.;")
        if not phrase:
            continue
        if phrase.lower() in BAD_OBJECTS:
            continue

        marked_phrase = insert_marker_in_phrase(phrase)
        rewritten = caption[:obj_start] + marked_phrase + caption[obj_end:]
        return rewritten, "rewritten", verb_match.group(1), phrase

    return caption, "no_match", "", ""


def normalize_task(task: str) -> str:
    task = task.strip().rstrip(".").strip()
    if not task:
        return "performs a manipulation task"
    return task[0].lower() + task[1:]


def load_metadata_captions(dataset_dir: Path) -> dict[Path, str]:
    metadata = dataset_dir / "metadata.csv"
    if not metadata.exists():
        return {}
    captions: dict[Path, str] = {}
    with metadata.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "name" not in reader.fieldnames or "task_orig" not in reader.fieldnames:
            return {}
        for row in reader:
            name = row["name"]
            task = normalize_task(row.get("task_orig", ""))
            captions[dataset_dir / "metas" / f"{name}.txt"] = PROMPT_TEMPLATE.format(task=task)
    return captions


def backup_metas(dataset_dir: Path, backup_name: str, overwrite_backup: bool) -> Path:
    metas = dataset_dir / "metas"
    backup = dataset_dir / backup_name
    if backup.exists():
        if overwrite_backup:
            shutil.rmtree(backup)
        else:
            return backup
    shutil.copytree(metas, backup, symlinks=True)
    return backup


def process_dataset(
    dataset_dir: Path,
    dry_run: bool,
    backup_name: str,
    overwrite_backup: bool,
    prefer_metadata_csv: bool,
) -> dict:
    metas = dataset_dir / "metas"
    if not metas.is_dir():
        raise FileNotFoundError(f"Missing metas directory: {metas}")

    backup = None
    if not dry_run:
        backup = backup_metas(dataset_dir, backup_name, overwrite_backup)

    csv_captions = load_metadata_captions(dataset_dir) if prefer_metadata_csv else {}
    paths = sorted(csv_captions) if csv_captions else sorted(metas.glob("*.txt"))

    rows: list[RewriteResult] = []
    counts = {"rewritten": 0, "already_marked": 0, "no_match": 0}
    for path in paths:
        original = csv_captions[path] if csv_captions else path.read_text().strip()
        rewritten, status, verb, phrase = rewrite_caption(original)
        counts[status] = counts.get(status, 0) + 1
        changed = rewritten != original
        rows.append(
            RewriteResult(
                path=str(path),
                changed=changed,
                status=status,
                verb=verb,
                object_phrase=phrase,
                original=original,
                rewritten=rewritten,
            )
        )
        if changed and not dry_run:
            path.write_text(rewritten.rstrip() + "\n")

    report = {
        "dataset_dir": str(dataset_dir),
        "dry_run": dry_run,
        "backup": str(backup) if backup else None,
        "caption_source": "metadata.csv" if csv_captions else "metas/*.txt",
        "num_files": len(rows),
        "counts": counts,
        "num_changed": sum(row.changed for row in rows),
        "examples": [asdict(row) for row in rows[:20]],
        "no_match_examples": [asdict(row) for row in rows if row.status == "no_match"][:50],
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json = dataset_dir / f"tgt_caption_rewrite_report_{stamp}.json"
    report_csv = dataset_dir / f"tgt_caption_rewrite_report_{stamp}.csv"
    if not dry_run:
        report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        with report_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else ["path"])
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        report["report_json"] = str(report_json)
        report["report_csv"] = str(report_csv)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup-name", default="metas_before_tgt_marker")
    parser.add_argument("--overwrite-backup", action="store_true")
    parser.add_argument("--no-metadata-csv", action="store_true", help="Read existing metas/*.txt instead of metadata.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        process_dataset(
            path.resolve(),
            args.dry_run,
            args.backup_name,
            args.overwrite_backup,
            prefer_metadata_csv=not args.no_metadata_csv,
        )
        for path in args.dataset_dirs
    ]
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
