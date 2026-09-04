"""Tests for rich_markdown: Bot API 10.1/10.2 rich_message -> Markdown + media.

Rich messages from the Telegram Rich Text Editor carry structure only in
``rich_message.blocks`` (recursive blocks + inline text nodes). These tests
verify the converter preserves headings, code, blockquotes, lists, tables,
emphasis, links, spoilers, sup/sub, and collects embedded media.
"""
import sys

sys.path.insert(0, "/home/magnum/.hermes/hermes-agent")
from plugins.platforms.telegram import rich_markdown as R  # noqa: E402


def _render(blocks):
    return R.rich_message_to_markdown({"blocks": blocks})


def test_paragraph_inline_styles():
    md, media = _render([{
        "type": "paragraph",
        "text": [
            {"type": "bold", "text": "b"},
            " ",
            {"type": "italic", "text": "i"},
            " ",
            {"type": "underline", "text": "u"},
            " ",
            {"type": "strikethrough", "text": "s"},
            " ",
            {"type": "spoiler", "text": "hidden"},
            " ",
            {"type": "code", "text": "x=1"},
        ],
    }])
    assert md == "**b** *i* __u__ ~~s~~ ||hidden|| `x=1`"
    assert media == []


def test_heading_sizes():
    md, _ = _render([
        {"type": "heading", "size": 1, "text": "One"},
        {"type": "heading", "size": 3, "text": "Three"},
        {"type": "heading", "size": 6, "text": "Six"},
    ])
    assert md == "# One\n\n### Three\n\n###### Six"


def test_preformatted_code_block_with_language():
    md, _ = _render([{
        "type": "pre", "language": "python",
        "text": "def f():\n    return 1",
    }])
    assert "```python" in md
    assert "def f():" in md
    assert "return 1" in md


def test_url_email_mention():
    md, _ = _render([{
        "type": "paragraph",
        "text": [
            {"type": "url", "text": "site", "url": "https://x.com"},
            " ",
            {"type": "email_address", "text": "me@x.io",
             "email_address": "me@x.io"},
            " ",
            {"type": "mention", "text": "@user"},
        ],
    }])
    assert "[site](https://x.com)" in md
    assert "[me@x.io](mailto:me@x.io)" in md
    assert "@user" in md


def test_button_link_renders_as_markdown_url():
    # Real structure seen in live dumps: a pasted app-store / web link arrives
    # as {type: button, button: {text, url}}, NOT {type: url}.
    md, _ = _render([{
        "type": "paragraph",
        "text": [
            {"type": "button", "button": {
                "text": "Ссылка",
                "url": "https://apps.apple.com/app/id6761788641",
            }},
            " на игру.",
        ],
    }])
    assert md == "[Ссылка](https://apps.apple.com/app/id6761788641) на игру."
    assert "https://apps.apple.com/app/id6761788641" in md


def test_divider_and_footer():
    md, _ = _render([
        {"type": "divider"},
        {"type": "footer", "text": "small"},
    ])
    assert "---" in md
    assert "_small_" in md


def test_list_bulleted_with_nested_and_checkbox():
    md, _ = _render([{
        "type": "list",
        "items": [
            {"blocks": [{"type": "paragraph", "text": "top"}]},
            {"has_checkbox": True, "is_checked": True,
             "blocks": [{"type": "paragraph", "text": "done"}]},
        ],
    }])
    assert "- top" in md
    assert "[x] done" in md


def test_list_ordered_numeric():
    md, _ = _render([{
        "type": "list",
        "items": [
            {"value": 1, "type": "1",
             "blocks": [{"type": "paragraph", "text": "first"}]},
            {"value": 2, "type": "1",
             "blocks": [{"type": "paragraph", "text": "second"}]},
        ],
    }])
    assert "1. first" in md
    assert "2. second" in md


def test_blockquote():
    md, _ = _render([{
        "type": "blockquote",
        "blocks": [{"type": "paragraph", "text": ["quote ", {"type": "bold", "text": "bold"}]}],
    }])
    assert "> quote **bold**" in md


def test_table_with_header_row():
    md, _ = _render([{
        "type": "table",
        "cells": [
            [{"text": "A", "is_header": True}, {"text": "B", "is_header": True}],
            [{"text": "1"}, {"text": "2"}],
        ],
    }])
    lines = md.splitlines()
    assert "| A" in lines[0]
    assert "| ---" in lines[1] or "| --- " in lines[1]
    assert "| 1" in lines[2]


def test_details_and_math_block():
    md, _ = _render([
        {"type": "details", "summary": "More", "blocks": [
            {"type": "paragraph", "text": "hidden content"},
        ]},
        {"type": "mathematical_expression", "expression": "\\sum x"},
    ])
    assert "**More**" in md
    assert "hidden content" in md
    assert "\\sum x" in md


def test_superscript_subscript():
    md, _ = _render([{
        "type": "paragraph",
        "text": ["E=mc", {"type": "superscript", "text": "2"}],
    }])
    assert "E=mc<sup>2</sup>" in md


def test_photo_media_collected():
    md, media = _render([{
        "type": "photo",
        "photo": [{"file_id": "F1", "width": 800, "height": 400}],
        "caption": {"text": "скриншот"},
    }])
    assert "Изображение" in md and "скриншот" in md
    assert media == [("photo", {"type": "photo",
                                "photo": [{"file_id": "F1", "width": 800, "height": 400}],
                                "caption": {"text": "скриншот"}},
                      "скриншот")]


def test_video_audio_voice_animation_media_collected():
    md, media = _render([
        {"type": "video", "video": {"file_id": "V1"}},
        {"type": "audio", "audio": {"file_id": "A1"}},
        {"type": "voice_note", "voice_note": {"file_id": "VN1"}},
        {"type": "animation", "animation": {"file_id": "G1"}},
    ])
    kinds = [m[0] for m in media]
    assert kinds == ["video", "audio", "voice_note", "animation"]


def test_collage_collects_nested_photos():
    md, media = _render([{
        "type": "collage",
        "blocks": [
            {"type": "photo", "photo": [{"file_id": "C1"}]},
            {"type": "photo", "photo": [{"file_id": "C2"}]},
        ],
    }])
    assert len(media) == 2
    assert media[0][1]["photo"][0]["file_id"] == "C1"
    assert media[1][1]["photo"][0]["file_id"] == "C2"


def test_unknown_block_degrades_gracefully():
    # unknown block types / malformed input must not raise
    md, media = _render([
        {"type": "totally_unknown_type", "text": "fallback"},
        None,
        "not a dict",
        {"type": "paragraph"},
    ])
    assert "fallback" in md  # default text branch still runs
    assert media == []


def test_pullquote_and_map():
    md, _ = _render([
        {"type": "pullquote", "text": "центрированная цитата",
         "credit": "автор"},
        {"type": "map", "location": {"latitude": 55.75, "longitude": 37.61}},
    ])
    assert "> центрированная цитата" in md
    assert "_автор_" in md
    assert "55.75" in md and "37.61" in md


def test_slideshow_collects_nested_photos():
    md, media = _render([{
        "type": "slideshow",
        "blocks": [{"type": "photo", "photo": [{"file_id": "S1"}]}],
    }])
    assert len(media) == 1
    assert media[0][1]["photo"][0]["file_id"] == "S1"


def test_anchor_block_and_reference_links():
    md, _ = _render([
        {"type": "anchor", "name": "top"},
        {"type": "paragraph", "text": [
            {"type": "reference_link", "text": "см. выше",
             "reference_name": "top"}]},
    ])
    assert "top" in md
    assert "см. выше" in md


def test_full_editor_styles_integration():
    """A single message exercising every Rich Editor block type at once
    (as in the dark-UI style menu: Heading 1-6, Quote, Pullquote, Code Block,
    Footer, Divider) plus emphasis and links."""
    md, _ = _render([
        {"type": "heading", "size": 1, "text": "Заголовок 1"},
        {"type": "heading", "size": 4, "text": "Подраздел"},
        {"type": "paragraph", "text": [
            {"type": "bold", "text": "bold"},
            " ",
            {"type": "italic", "text": "italic"},
            " ",
            {"type": "code", "text": "inline"},
            " ",
            {"type": "url", "text": "сайт", "url": "https://x.com"},
        ]},
        {"type": "blockquote", "blocks": [
            {"type": "paragraph", "text": "цитата"}]},
        {"type": "pullquote", "text": "pull-цитата", "credit": "кто-то"},
        {"type": "pre", "language": "python", "text": "print('hi')"},
        {"type": "footer", "text": "мелкий"},
        {"type": "divider"},
        {"type": "list", "items": [
            {"blocks": [{"type": "paragraph", "text": "пункт"}]},
            {"has_checkbox": True, "is_checked": False,
             "blocks": [{"type": "paragraph", "text": "todo"}]},
        ]},
        {"type": "table", "cells": [
            [{"text": "A", "is_header": True}, {"text": "B", "is_header": True}],
            [{"text": "1"}, {"text": "2"}],
        ]},
        {"type": "mathematical_expression", "expression": "E = mc^2"},
        {"type": "paragraph", "text": [
            {"type": "button", "button": {"text": "App",
             "url": "https://apps.apple.com/app/id6761788641"}}]},
    ])
    # structural assertions for every style option
    assert "# Заголовок 1" in md
    assert "#### Подраздел" in md
    assert "**bold**" in md and "*italic*" in md and "`inline`" in md
    assert "[сайт](https://x.com)" in md
    assert "> цитата" in md
    assert "> pull-цитата" in md and "_кто-то_" in md
    assert "```python" in md and "print('hi')" in md
    assert "_мелкий_" in md
    assert "---" in md
    assert "- пункт" in md and "[ ] todo" in md
    assert "| A |" in md and "| 1 |" in md
    assert "E = mc^2" in md
    assert "apps.apple.com/app/id6761788641" in md
