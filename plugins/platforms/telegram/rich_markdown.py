"""Render Telegram Bot API Rich Messages (rich_message) into Markdown.

Incoming rich messages from the Telegram Rich Text Editor arrive with empty
``message.text`` — their content lives in ``message.api_kwargs["rich_message"]``
as a recursive structure of blocks and inline text nodes:

    RichMessage   = { blocks: [RichBlock], is_rtl? }
    RichBlock     = paragraph | heading | pre | footer | divider | ...
                    | list | blockquote | pullquote | table | details
                    | collage | slideshow | map
                    | photo | video | audio | voice_note | animation
    RichText      = str | [RichText] | one of {bold, italic, underline,
                    strikethrough, spoiler, code, custom_emoji, url,
                    email_address, phone_number, bank_card_number, mention,
                    hashtag, cashtag, bot_command, subscript, superscript,
                    marked, date_time, mathematical_expression, text_mention,
                    anchor, anchor_link, reference, reference_link}

This module converts that structure into faithful Markdown so headings, code
blocks, blockquotes, lists, tables, emphasis, links, and media placement
survive intact. Media blocks (photo/video/audio/animation/voice_note) are both
emitted as Markdown placeholders and collected into ``media`` descriptors so a
caller can download/cache them for vision/transcription analysis.

Pure functions; no I/O. Safe against missing/unknown node types (degrades to
plain text rather than raising).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# A media descriptor: (kind, node_dict, caption_plaintext)
MediaDesc = Tuple[str, Dict[str, Any], str]

# Inline text node type -> Markdown wrapper.
# Types we deliberately do NOT wrap (no Markdown equivalent / keep literal):
#   date_time, text_mention, custom_emoji, mathematical_expression, anchor,
#   reference — their *text* is emitted as-is (still recursed).
_STRONG_WRAP = {"bold": "**", "marked": "=="}
_EMPH_WRAP = {"italic": "*", "underline": "__", "strikethrough": "~~"}
_CODE_WRAP = {"code": "`"}
_SPOILER = "||"


def _rich_text_str(value: Any) -> str:
    """Best-effort recursive flattener to PLAIN text (no markdown)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_rich_text_str(item) for item in value)
    if isinstance(value, dict):
        ntype = value.get("type")
        # custom emoji has no nested text; fall back to alternative text
        if ntype == "custom_emoji":
            return value.get("alternative_text") or ""
        if ntype == "mathematical_expression":
            return value.get("expression") or ""
        # url/mention/etc carry semantic value in their own field too
        for key in ("url", "username", "email_address", "phone_number",
                    "bank_card_number", "hashtag", "cashtag", "bot_command"):
            if key in value and not value.get("text"):
                return str(value[key])
        text = value.get("text")
        if text is not None:
            return _rich_text_str(text)
        children = value.get("children")
        if children is not None:
            return _rich_text_str(children)
    return ""


def _render_inline(value: Any) -> str:
    """Recursively render a RichText node (or list/string) as Markdown."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_render_inline(item) for item in value)
    if isinstance(value, dict):
        ntype = value.get("type", "")

        if ntype == "custom_emoji":
            return value.get("alternative_text") or ""
        if ntype == "mathematical_expression":
            expr = (value.get("expression") or "").strip()
            return f"${expr}$" if expr else ""
        if ntype == "url":
            url = value.get("url") or ""
            label = _render_inline(value.get("text")) or url
            return f"[{label}]({url})" if url else label
        if ntype == "button":
            # Telegram Rich Text can wrap a clickable link as {type: button,
            # button: {text, url}} (pasted links, buttons, tg:// deep links).
            # url is arbitrary — scheme, host and path are preserved verbatim.
            raw_btn = value.get("button")
            btn = raw_btn if isinstance(raw_btn, dict) else {}
            btn_url = str(btn.get("url") or "")
            label = _render_inline(btn.get("text")) or _render_inline(value.get("text"))
            label = str(label or btn_url)
            return f"[{label}]({btn_url})" if btn_url else label
        if ntype == "email_address":
            addr = value.get("email_address") or ""
            label = _render_inline(value.get("text")) or addr
            return f"[{label}](mailto:{addr})" if addr else label
        if ntype in ("mention", "hashtag", "cashtag", "bot_command"):
            # entity already carries the leading @/#/$/; just recurse text
            return _render_inline(value.get("text"))
        if ntype == "text_mention":
            # mention by id; plain text is the visible name
            return _render_inline(value.get("text"))
        if ntype == "anchor_link":
            name = value.get("anchor_name") or ""
            label = _render_inline(value.get("text"))
            return f"[{label}](#{name})" if name else label
        if ntype == "reference_link":
            name = value.get("reference_name") or ""
            label = _render_inline(value.get("text"))
            return f"[{label}](#{name})" if name else label

        inner = _render_inline(value.get("text")) if "text" in value else ""
        if ntype in _STRONG_WRAP:
            w = _STRONG_WRAP[ntype]
            return f"{w}{inner}{w}" if inner else ""
        if ntype in _EMPH_WRAP:
            w = _EMPH_WRAP[ntype]
            return f"{w}{inner}{w}" if inner else ""
        if ntype == "code":
            return f"`{inner}`" if inner else ""
        if ntype == "spoiler":
            return f"{_SPOILER}{inner}{_SPOILER}" if inner else ""
        if ntype == "superscript" and inner:
            return f"<sup>{inner}</sup>"
        if ntype == "subscript" and inner:
            return f"<sub>{inner}</sub>"
        # anchor, reference, date_time -> plain
        if inner:
            return inner
        # anything else with nested children
        children = value.get("children")
        if children is not None:
            return _render_inline(children)
        return _rich_text_str(value)
    return ""


def _render_caption(cap: Any) -> str:
    """Render a RichBlockCaption / RichText caption to plain text."""
    if cap is None:
        return ""
    if isinstance(cap, dict):
        return _rich_text_str(cap.get("text"))
    return _rich_text_str(cap)


def _collect_media_from_node(block: Dict[str, Any], media: List[MediaDesc]) -> None:
    """Pull a media descriptor out of a media block, if present."""
    kind = block.get("type")
    caption = _render_caption(block.get("caption")) or ""
    if kind == "photo":
        media.append(("photo", block, caption))
    elif kind == "video":
        media.append(("video", block, caption))
    elif kind == "audio":
        media.append(("audio", block, caption))
    elif kind == "voice_note":
        media.append(("voice_note", block, caption))
    elif kind == "animation":
        media.append(("animation", block, caption))


def _render_blocks(blocks: Any, media: List[MediaDesc]) -> str:
    """Render a list of RichBlocks to Markdown, collecting media along the way."""
    if not isinstance(blocks, list):
        return ""
    out: List[str] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")

        # ── media blocks ───────────────────────────────────────────────
        if btype in ("photo", "video", "audio", "voice_note", "animation"):
            _collect_media_from_node(block, media)
            cap = _render_caption(block.get("caption"))
            label = {"photo": "Изображение", "video": "Видео",
                     "audio": "Аудио", "voice_note": "Голосовое",
                     "animation": "Анимация/GIF"}.get(btype, btype)
            marker = f"[{label}"
            if cap:
                marker += f": {cap}"
            out.append(marker + "]")
            continue

        if btype == "collage" or btype == "slideshow":
            inner = _render_blocks(block.get("blocks"), media)
            if inner:
                out.append(inner)
            cap = _render_caption((block.get("caption") or {}).get("text")
                                  if isinstance(block.get("caption"), dict)
                                  else block.get("caption"))
            # caption comes from nested _render_blocks captions already
            continue

        if btype == "divider":
            out.append("---")
            continue

        if btype == "anchor":
            name = block.get("name")
            if name:
                out.append(f'<a id="{name}"></a>')
            continue

        if btype == "heading":
            size = int(block.get("size") or 1)
            hashes = "#" * min(max(size, 1), 6)
            text = _render_inline(block.get("text")).strip()
            if text:
                out.append(f"{hashes} {text}")
            continue

        if btype == "pre":
            lang = block.get("language") or ""
            text = _rich_text_str(block.get("text"))
            fence = "```"
            # escape a longer fence if content contains triple backticks
            if "```" in text:
                fence = "````"
            out.append(f"{fence}{lang}\n{text.strip()}\n{fence}")
            continue

        if btype == "footer":
            text = _render_inline(block.get("text")).strip()
            if text:
                out.append(f"_{text}_")
            continue

        if btype == "mathematical_expression":
            expr = (block.get("expression") or "").strip()
            if expr:
                out.append(f"$$\n{expr}\n$$")
            continue

        if btype == "list":
            rendered_items = _render_list(block.get("items"), media)
            if rendered_items:
                out.extend(rendered_items)
            continue

        if btype == "blockquote":
            inner = _render_blocks(block.get("blocks"), media)
            if inner:
                quoted = "\n".join(f"> {ln}" if ln else ">" for ln in inner.splitlines())
                out.append(quoted)
            credit = _render_caption(block.get("credit"))
            if credit:
                out.append(f"_{credit}_")
            continue

        if btype == "pullquote":
            text = _render_inline(block.get("text")).strip()
            if text:
                quoted = "\n".join(f"> {ln}" if ln else ">" for ln in text.splitlines())
                out.append(quoted)
            credit = _render_caption(block.get("credit"))
            if credit:
                out.append(f"_{credit}_")
            continue

        if btype == "table":
            rendered = _render_table(block)
            if rendered:
                out.append(rendered)
            cap = block.get("caption")
            if cap:
                out.append(_render_caption(cap))
            continue

        if btype == "details":
            summary = _render_inline(block.get("summary")).strip() if block.get("summary") else ""
            inner = _render_blocks(block.get("blocks"), media)
            if summary:
                out.append(f"**{summary}**")
            if inner:
                out.append(inner)
            continue

        if btype == "map":
            loc = block.get("location") or {}
            lat = loc.get("latitude") if isinstance(loc, dict) else None
            lon = loc.get("longitude") if isinstance(loc, dict) else None
            if lat is not None and lon is not None:
                out.append(f"[Карта: {lat}, {lon}]")
            cap = _render_caption((block.get("caption") or {}).get("text")
                                  if isinstance(block.get("caption"), dict)
                                  else block.get("caption"))
            continue

        # ── default: any paragraph-like block with a "text" ────────────
        text = _render_inline(block.get("text")).strip()
        if text:
            out.append(text)

    # join with blank line between block-level chunks
    joined = "\n\n".join(chunk.strip("\n") for chunk in out if chunk and chunk.strip())
    return joined.strip()


def _render_list(items: Any, media: List[MediaDesc]) -> List[str]:
    """Render list items. Bulleted by default; checkbox markers when present."""
    if not isinstance(items, list):
        return []
    lines: List[str] = []
    ordered = False
    # detect ordering from first item that declares a numeric value/type
    for it in items:
        if isinstance(it, dict) and (it.get("value") is not None or it.get("type") in ("a", "A", "i", "I", "1")):
            ordered = True
            break
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        inner = _render_blocks(item.get("blocks"), media)
        # label is separate from checkbox/value (list-item label)
        label = item.get("label")
        has_cb = bool(item.get("has_checkbox"))
        checked = bool(item.get("is_checked"))
        marker = ""
        if has_cb:
            marker = "[x] " if checked else "[ ] "
        elif ordered:
            marker = f"{idx}. "
        else:
            marker = "- "
        # label often repeats the numeric/letter marker; keep only if non-numeric
        if label and not _label_is_marker(label, item):
            marker = f"{marker}{label} ".strip() + " "
        if inner:
            inner_lines = inner.splitlines()
            first = marker + inner_lines[0].lstrip()
            lines.append(first)
            for extra in inner_lines[1:]:
                lines.append("  " + extra)
        elif marker.strip():
            lines.append(marker.rstrip())
    return lines


def _label_is_marker(label: Any, item: Dict[str, Any]) -> bool:
    """True when the list-item label duplicates the auto marker (1., a., i., ...)."""
    if label is None:
        return False
    lab = str(label).strip()
    itype = item.get("type")
    if itype == "1" or (not itype and item.get("value") is not None):
        return lab.lstrip(".)").isdigit()
    if itype in ("a", "A", "i", "I"):
        return bool(re.fullmatch(r"[a-zA-ZivxIVX. )]+", lab))
    return False


def _render_table(block: Dict[str, Any]) -> str:
    """Render RichBlockTable cells to a GFM table."""
    cells = block.get("cells")
    if not isinstance(cells, list) or not cells:
        return ""
    # normalize: pad ragged rows to the widest row
    widths = []
    rows_norm = []
    for row in cells:
        if not isinstance(row, list):
            rows_norm.append([])
            continue
        rows_norm.append([_cell_text(c) for c in row])
    if not rows_norm:
        return ""
    max_cols = max(len(r) for r in rows_norm)
    for r in rows_norm:
        r.extend([""] * (max_cols - len(r)))
    widths = [max(len(r[c]) for r in rows_norm) for c in range(max_cols)]

    # header row = first row if any cell is_header, else first row as header anyway
    def fmt_row(row):
        return "| " + " | ".join(
            cell.ljust(widths[i]) if i < len(widths) else cell
            for i, cell in enumerate(row)
        ) + " |"

    lines = [fmt_row(rows_norm[0])]
    lines.append("| " + " | ".join("-" * max(w, 3) for w in widths) + " |")
    for row in rows_norm[1:]:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def _cell_text(cell: Any) -> str:
    if not isinstance(cell, dict):
        return _rich_text_str(cell)
    # RichBlockTableCell nests its content under either "text" (RichText) or
    # nested "blocks"
    if "blocks" in cell:
        return _rich_text_str(cell.get("blocks"))
    return _render_inline(cell.get("text")).strip()


def rich_message_to_markdown(rich_message: Any) -> Tuple[str, List[MediaDesc]]:
    """Convert a parsed ``rich_message`` object/dict into (markdown, media)."""
    if isinstance(rich_message, dict):
        blocks = rich_message.get("blocks")
    else:
        getter = getattr(rich_message, "get", None)
        blocks = getter("blocks") if callable(getter) else None
    media: List[MediaDesc] = []
    text = _render_blocks(blocks, media)
    return text, media
