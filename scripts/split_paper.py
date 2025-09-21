"""
Split docs/paper/draft_paper.md into chapter files under docs/paper/chapters/.

Modes:
- Default (granular): preserves preface items as individual files and splits on
  all-caps numeric headings and appendices.
- Consolidated (--consolidate):
  * Combines title block, contents, abstract, conventions, definitions, version
    and scope into a single 00-preface.md.
  * Merges the alternate introduction ("INTRODUCTION: A NEW APPROACH…") into
    the "1. INTRODUCTION" chapter so only one Introduction file exists.
  * Subsequent chapters are split on uppercase numeric headings 2..13 and
    appendices, into 01-introduction.md, 02-..., etc.

Each output file is named with a zero-padded numeric prefix and a slug.
Writes an index listing the mapping of headings to filenames.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC_DEFAULT = ROOT / "docs" / "paper" / "draft_paper.md"
OUT_DIR_DEFAULT = ROOT / "docs" / "paper" / "chapters"


@dataclass
class Anchor:
    line: int  # 1-based line number where chapter starts
    title: str
    slug: str


def slugify(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "section"


def is_numeric_all_caps_heading(text: str) -> bool:
    m = re.match(r"^(\d+)\.\s+(.+?)\s*$", text)
    if not m:
        return False
    rest = m.group(2)
    has_lower = re.search(r"[a-z]", rest) is not None
    return not has_lower


def is_appendix_heading(text: str) -> bool:
    t = text.strip()
    return t.startswith("APPENDIX ") and t == t.upper()


def find_preface_anchors(lines: List[str]) -> List[Anchor]:
    preface_titles = [
        "ABSTRACT – OUTDATED.",
        "WRITING CONVENTIONS",
        "FREQUENTLY USED DEFINITIONS",
        "DOCUMENT VERSION AND METADATA- OUTDATED.",
        "DOCUMENT SCOPE - OUTDATED.",
        "INTRODUCTION: A NEW APPROACH FOR ADAPTIVE SYSTEMS - UPDATED.",
    ]
    anchors: List[Anchor] = []
    for i, text in enumerate(lines, start=1):
        if text.strip() in preface_titles:
            anchors.append(Anchor(i, text.strip(), slugify(text)))
    return anchors


def find_main_anchors(lines: List[str]) -> List[Anchor]:
    anchors: List[Anchor] = []
    for i, text in enumerate(lines, start=1):
        t = text.rstrip("\n")
        if is_numeric_all_caps_heading(t) or is_appendix_heading(t):
            if i < 140:  # skip TOC area
                continue
            anchors.append(Anchor(i, t.strip(), slugify(t)))
    return anchors


def write_section(out_path: Path, section_lines: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(section_lines), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split draft paper into chapters")
    parser.add_argument("--src", type=str, default=str(SRC_DEFAULT), help="Source markdown file")
    parser.add_argument("--outdir", type=str, default=str(OUT_DIR_DEFAULT), help="Output directory for chapters")
    parser.add_argument("--consolidate", action="store_true", help="Create consolidated chapter set")
    parser.add_argument("--yaml", action="store_true", help="Add YAML front matter to each output file")
    parser.add_argument("--coauthor", type=str, default="Codex CLI (OpenAI)", help="Name to attribute as co-author in Preface and YAML")
    args = parser.parse_args(argv)

    src = Path(args.src)
    out_dir = Path(args.outdir)

    if not src.exists():
        print(f"Source not found: {src}", file=sys.stderr)
        return 1

    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    n = len(lines)

    if args.consolidate:
        # Locate alternate intro and main intro
        alt_intro_title = "INTRODUCTION: A NEW APPROACH FOR ADAPTIVE SYSTEMS - UPDATED."
        alt_intro_line = next((i for i, t in enumerate(lines, start=1) if t.strip() == alt_intro_title), None)
        intro_line = next((i for i, t in enumerate(lines, start=1) if is_numeric_all_caps_heading(t.rstrip("\n")) and t.strip().startswith("1.")), None)
        if intro_line is None:
            print("Could not locate '1. INTRODUCTION' heading", file=sys.stderr)
            return 2
        # Preface is from beginning to just before alt intro or intro
        preface_end = (alt_intro_line or intro_line) - 1
        if preface_end < 1:
            preface_end = 1

        ranges: List[Tuple[str, str, int, int]] = []  # (title, slug, start, end)
        ranges.append(("Preface", "preface", 1, preface_end))

        # Introduction from earliest of alt intro / main intro to before chapter 2
        intro_start = min([x for x in [alt_intro_line, intro_line] if x is not None])
        chap2_line = next((i for i, t in enumerate(lines, start=1) if is_numeric_all_caps_heading(t.rstrip("\n")) and t.strip().startswith("2.")), None)
        intro_end = (chap2_line - 1) if chap2_line else n
        ranges.append(("1. INTRODUCTION (with prelude)", "01-introduction", intro_start, intro_end))

        # Remaining numeric chapters (2..) and appendices
        main_anchors: List[Anchor] = []
        for i, text in enumerate(lines, start=1):
            t = text.rstrip("\n")
            if is_numeric_all_caps_heading(t):
                if t.strip().startswith("1."):
                    continue
                if chap2_line and i < chap2_line:
                    continue
                main_anchors.append(Anchor(i, t.strip(), slugify(t)))
            elif is_appendix_heading(t):
                main_anchors.append(Anchor(i, t.strip(), slugify(t)))
        main_anchors.sort(key=lambda a: a.line)
        for idx, a in enumerate(main_anchors):
            start = a.line
            end = (main_anchors[idx + 1].line - 1) if idx + 1 < len(main_anchors) else n
            ranges.append((a.title, a.slug, start, end))

        # Helpers
        def build_yaml_block(display_title: str, chapter: int | None, appendix: str | None, start: int, end: int) -> str:
            if not args.yaml:
                return ""
            lines_yaml = [
                "---\n",
                f"title: {display_title}\n",
                f"source: {src}\n",
                f"source_lines: {start}-{end}\n",
                "split_mode: consolidated\n",
            ]
            if chapter is not None:
                lines_yaml.append(f"chapter: {chapter}\n")
            if appendix is not None:
                lines_yaml.append(f"appendix: \"{appendix}\"\n")
            lines_yaml.append("coauthors:\n")
            lines_yaml.append("  - John Morrissey\n")
            lines_yaml.append(f"  - {args.coauthor}\n")
            lines_yaml.append("generated_by: scripts/split_paper.py\n")
            lines_yaml.append("---\n\n")
            return "".join(lines_yaml)

        def to_title_case(s: str) -> str:
            words = s.split()
            out = []
            for w in words:
                if w.isupper():
                    out.append(w)
                else:
                    out.append(w.capitalize())
            return " ".join(out)

        index_lines = ["# Chapters Index (Consolidated)\n\n", f"Generated from {src}\n\n"]
        seq = 0
        for title, slug, start, end in ranges:
            seq += 1
            if title.startswith("1. INTRODUCTION"):
                fname = "01-introduction.md"
                display_title = "Introduction"
                chapter_num = 1
                appendix_letter = None
            elif (m := re.match(r"^(\d+)\.\s+(.*)$", title)):
                num = int(m.group(1))
                name = slugify(m.group(2))
                fname = f"{num:02d}-{name}.md"
                display_title = to_title_case(m.group(2).strip())
                chapter_num = num
                appendix_letter = None
            elif title.upper().startswith("APPENDIX "):
                fname = f"{slug}.md"
                m_app = re.match(r"^APPENDIX\s+([A-Z])[:\s]+(.*)$", title)
                appendix_letter = m_app.group(1) if m_app else None
                display_title = to_title_case(m_app.group(2)) if m_app else title.title()
                chapter_num = None
            elif title == "Preface":
                fname = "00-preface.md"
                display_title = "Preface"
                chapter_num = None
                appendix_letter = None
            else:
                fname = f"{seq:02d}-{slug}.md"
                display_title = to_title_case(title)
                chapter_num = None
                appendix_letter = None

            out_file = out_dir / fname
            section = lines[start - 1 : end]
            yaml_block = build_yaml_block(display_title, chapter_num, appendix_letter, start, end)
            body = section
            if title == "Preface":
                new_body: List[str] = []
                inserted = False
                for ln in body:
                    new_body.append(ln)
                    if not inserted and ln.strip().startswith("Author:"):
                        new_body.append(f"Co-author: {args.coauthor}\n")
                        inserted = True
                body = new_body
            content = [yaml_block] if yaml_block else []
            content.extend(body)
            write_section(out_file, content)
            index_lines.append(f"{fname}  <-  {title}  (lines {start}-{end})\n")

        write_section(out_dir / "README.md", index_lines)
        print(f"Wrote {len(ranges)} consolidated chapter files to {out_dir}")
        return 0

    # Granular mode
    anchors: List[Anchor] = []
    try:
        contents_line = next(i for i, t in enumerate(lines, start=1) if t.strip() == "CONTENTS")
    except StopIteration:
        contents_line = 0
    if contents_line > 1:
        anchors.append(Anchor(1, "Front Matter", "front-matter"))
        anchors.append(Anchor(contents_line, "Contents", "contents"))
    anchors.extend(find_preface_anchors(lines))
    anchors.extend(find_main_anchors(lines))
    anchors.sort(key=lambda a: a.line)
    dedup: List[Anchor] = []
    seen_lines = set()
    for a in anchors:
        if a.line in seen_lines:
            continue
        seen_lines.add(a.line)
        dedup.append(a)
    anchors = dedup
    if not anchors:
        print("No anchors found; aborting.", file=sys.stderr)
        return 2
    ranges2: List[Tuple[Anchor, int, int]] = []
    for idx, a in enumerate(anchors):
        start = a.line
        end = (anchors[idx + 1].line - 1) if idx + 1 < len(anchors) else n
        ranges2.append((a, start, end))
    index_lines = ["# Chapters Index\n\n", f"Generated from {src}\n\n"]
    for seq, (a, start, end) in enumerate(ranges2, start=1):
        fname = f"{seq:02d}-{a.slug}.md"
        out_file = out_dir / fname
        section = lines[start - 1 : end]
        write_section(out_file, section)
        index_lines.append(f"{seq:02d}. {a.title} -> {fname}\n")
    write_section(out_dir / "README.md", index_lines)
    print(f"Wrote {len(ranges2)} chapter files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

