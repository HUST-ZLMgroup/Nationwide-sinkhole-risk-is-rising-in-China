from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import tempfile
from pathlib import Path

import mistune

from lingam_pipeline_v1.pre_attribution.attribution_config import FONT_FAMILY


DEFAULT_INPUT = Path("code/4_attribution/lingam_pipeline_v1/outputs/figures/final/figure_text.md")
DEFAULT_OUTPUT = DEFAULT_INPUT.with_suffix(".pdf")

HEADING_SIZES = {
    1: "xx-large",
    2: "x-large",
    3: "large",
    4: "medium",
    5: "medium",
    6: "medium",
}

BLOCK_QUOTE_COLOR = "#5F5A55"
CODE_BACKGROUND = "#F4F1EC"
RULE_COLOR = "#A39C95"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Markdown file to PDF via mistune + pango-view.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input Markdown path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PDF path.")
    parser.add_argument("--font-size", type=float, default=11.0, help="Base font size passed to pango-view.")
    parser.add_argument("--width", type=int, default=540, help="Wrapped text width in points.")
    parser.add_argument("--margin", type=int, default=42, help="Margin in pixels for pango-view.")
    parser.add_argument("--dpi", type=int, default=144, help="Rendering DPI for pango-view.")
    parser.add_argument(
        "--keep-markup",
        action="store_true",
        help="Keep the intermediate Pango markup file next to the output PDF for debugging.",
    )
    return parser.parse_args()


def _escape(text: str) -> str:
    return html.escape(text, quote=False)


def _render_inlines(tokens: list[dict]) -> str:
    parts: list[str] = []
    for token in tokens:
        token_type = token["type"]
        if token_type == "text":
            parts.append(_escape(token.get("raw", "")))
        elif token_type == "strong":
            parts.append(f"<b>{_render_inlines(token.get('children', []))}</b>")
        elif token_type == "emphasis":
            parts.append(f"<i>{_render_inlines(token.get('children', []))}</i>")
        elif token_type == "codespan":
            code = _escape(token.get("raw", ""))
            parts.append(
                f"<span font_family='DejaVu Sans Mono' background='{CODE_BACKGROUND}'>{code}</span>"
            )
        elif token_type == "link":
            label = _render_inlines(token.get("children", [])) or _escape(token.get("attrs", {}).get("url", ""))
            url = _escape(token.get("attrs", {}).get("url", ""))
            parts.append(f"<u>{label}</u>" + (f" ({url})" if url else ""))
        elif token_type in {"softbreak", "linebreak"}:
            parts.append("\n")
        elif "children" in token:
            parts.append(_render_inlines(token["children"]))
        else:
            parts.append(_escape(token.get("raw", "")))
    return "".join(parts)


def _indent_lines(text: str, prefix: str) -> str:
    return "\n".join((prefix + line) if line.strip() else line for line in text.splitlines())


def _render_list_item(token: dict, depth: int, index: int, ordered: bool) -> str:
    bullet = f"{index}. " if ordered else "• "
    indent = "    " * depth
    nested_blocks: list[str] = []
    text_parts: list[str] = []

    for child in token.get("children", []):
        child_type = child["type"]
        if child_type == "block_text":
            text_parts.append(_render_inlines(child.get("children", [])))
        elif child_type == "paragraph":
            text_parts.append(_render_inlines(child.get("children", [])))
        elif child_type == "list":
            nested_blocks.append(_render_blocks([child], depth=depth + 1).strip())
        else:
            nested_blocks.append(_render_blocks([child], depth=depth + 1).strip())

    first_line = indent + bullet + " ".join(part for part in text_parts if part).strip()
    parts = [first_line.rstrip()]
    for nested in nested_blocks:
        if nested:
            parts.append(nested)
    return "\n".join(parts).rstrip()


def _render_block_quote(token: dict, depth: int) -> str:
    body = _render_blocks(token.get("children", []), depth=depth).strip()
    if not body:
        return ""
    body = _indent_lines(body, "    ")
    return f"<span foreground='{BLOCK_QUOTE_COLOR}' style='italic'>{body}</span>"


def _render_blocks(tokens: list[dict], depth: int = 0) -> str:
    parts: list[str] = []
    for token in tokens:
        token_type = token["type"]
        if token_type == "blank_line":
            continue
        if token_type == "heading":
            level = int(token.get("attrs", {}).get("level", 2))
            size = HEADING_SIZES.get(level, "medium")
            text = _render_inlines(token.get("children", []))
            parts.append(f"<span size='{size}' weight='bold'>{text}</span>")
            parts.append("")
            continue
        if token_type == "paragraph":
            parts.append(_render_inlines(token.get("children", [])))
            parts.append("")
            continue
        if token_type == "list":
            ordered = bool(token.get("attrs", {}).get("ordered", False))
            start = int(token.get("attrs", {}).get("start", 1))
            for offset, item in enumerate(token.get("children", [])):
                parts.append(_render_list_item(item, depth=depth, index=start + offset, ordered=ordered))
            parts.append("")
            continue
        if token_type == "block_quote":
            quote = _render_block_quote(token, depth=depth)
            if quote:
                parts.append(quote)
                parts.append("")
            continue
        if token_type == "block_code":
            raw = _escape(token.get("raw", ""))
            code = _indent_lines(raw, "    ")
            parts.append(
                f"<span font_family='DejaVu Sans Mono' background='{CODE_BACKGROUND}'>{code}</span>"
            )
            parts.append("")
            continue
        if token_type == "thematic_break":
            parts.append(f"<span foreground='{RULE_COLOR}'>────────────────────────────────────────</span>")
            parts.append("")
            continue
        if "children" in token:
            fallback = _render_blocks(token["children"], depth=depth)
            if fallback.strip():
                parts.append(fallback.strip())
                parts.append("")

    while parts and parts[-1] == "":
        parts.pop()
    return "\n".join(parts)


def markdown_to_pango_markup(markdown_text: str) -> str:
    parser = mistune.create_markdown(renderer="ast")
    ast = parser(markdown_text)
    body = _render_blocks(ast).strip()
    if not body:
        body = _escape(markdown_text)
    return f"<span>{body}</span>\n"


def render_pdf(markup_path: Path, output_pdf: Path, font_size: float, width: int, margin: int, dpi: int) -> None:
    pango_view = shutil.which("pango-view")
    if pango_view is None:
        raise FileNotFoundError("pango-view was not found in PATH.")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    font_desc = f"{FONT_FAMILY} {font_size:g}"
    cmd = [
        pango_view,
        "--no-display",
        "--markup",
        f"--font={font_desc}",
        f"--width={width}",
        f"--margin={margin}",
        f"--dpi={dpi}",
        "--align=left",
        "--wrap=word-char",
        "--line-spacing=1.18",
        "--language=zh_CN",
        f"--output={output_pdf}",
        str(markup_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    markdown_text = input_path.read_text(encoding="utf-8")
    markup_text = markdown_to_pango_markup(markdown_text)

    if args.keep_markup:
        markup_path = output_path.with_suffix(".pango")
        markup_path.write_text(markup_text, encoding="utf-8")
        render_pdf(markup_path, output_path, args.font_size, args.width, args.margin, args.dpi)
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".pango", encoding="utf-8", delete=False) as tmp:
            tmp.write(markup_text)
            markup_path = Path(tmp.name)
        try:
            render_pdf(markup_path, output_path, args.font_size, args.width, args.margin, args.dpi)
        finally:
            markup_path.unlink(missing_ok=True)

    print(f"[OK] Exported PDF -> {output_path}")


if __name__ == "__main__":
    main()
