"""Inbound Rich Message handling for the Telegram platform.

Telegram Bot API 10.1+ Rich Messages (sent from the Rich Text Editor) arrive
with an empty ``message.text`` — their content lives only in
``message.api_kwargs[\"rich_message\"][\"blocks\"]`` (PTB <22.6 has no
``message.rich_message`` attribute). Because ``message.text`` is empty these
messages match none of the normal TEXT/COMMAND/media filters and were silently
dropped before this handler existed.

This module owns the inbound rich surface as a mixin so ``TelegramAdapter``
does not grow (god-file decomposition plan #78792: rich/inbound clusters live
in their own module, mirroring the ``TelegramAuthorizationMixin`` pattern from
PR #75742). It converts the structured rich content to Markdown via
``rich_markdown``, downloads any embedded media for vision/STT, and forwards
the result through the normal text pipeline.
"""

import logging
import os
from typing import Any, List, Optional

try:
    from telegram import Update, Message
    from telegram.ext import ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:  # pragma: no cover - mirrors the adapter's import guard
    TELEGRAM_AVAILABLE = False
    Update = Message = Any
    # Keep ContextTypes.DEFAULT_TYPE annotations from crashing without the lib.
    class _MockContextTypes:
        DEFAULT_TYPE = Any
    ContextTypes = _MockContextTypes

from plugins.platforms.telegram import rich_markdown as rich_md

from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SUPPORTED_VIDEO_TYPES,
    cache_audio_from_bytes_async,
    cache_image_from_bytes_async,
    cache_video_from_bytes_async,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _rich_media_mime(kind: str, ext: str) -> str:
    """Map a rich-media kind + file extension to a MIME type for event.media_types."""
    if kind == "photo":
        return f"image/{ext.lstrip('.')}" if ext else "image/jpeg"
    if kind in ("video", "animation"):
        if ext == ".gif":
            return "image/gif"
        if ext == ".webp":
            return "image/webp"
        return "video/mp4" if ext == ".mp4" else SUPPORTED_VIDEO_TYPES.get(ext, "video/mp4")
    if kind == "audio":
        return "audio/mpeg" if ext in (".mp3",) else "audio/ogg"
    if kind == "voice_note":
        return "audio/ogg"
    return "application/octet-stream"


class TelegramRichInboundMixin:
    """Inbound Rich Message handlers for ``TelegramAdapter``.

    Pure mixin: methods call ``self.*`` that live on the final
    ``TelegramAdapter`` (auth/gating/topics/event-build/enqueue) plus the
    helpers defined here. Registered as the first base so its methods resolve
    ahead of ``BasePlatformAdapter``.
    """

    async def _cache_rich_media(self, rich_message: Any, event: MessageEvent) -> List[str]:
        """Download and cache media blocks embedded in a rich message.

        Rich-message media blocks (photo/video/audio/voice_note/animation)
        carry raw dicts (not PTB objects), so we resolve each ``file_id`` via
        the bot's ``get_file`` and cache bytes locally — the same cache the
        ordinary media path uses, so downstream vision/STT can read the file
        even after Telegram's ephemeral file URL expires.

        Returns the list of newly cached local paths (also appended to
        ``event.media_urls`` / ``event.media_types``).
        """
        if not rich_message:
            return []
        added: List[str] = []
        try:
            _markdown, media = rich_md.rich_message_to_markdown(rich_message)
        except Exception:
            logger.debug("[Telegram] rich_md parse error", exc_info=True)
            return added
        for kind, node, _caption in media:
            file_id = None
            # pull the largest photo size, else the single media object's file_id
            for key in ("photo", "video", "audio", "voice_note", "animation"):
                val = node.get(key)
                if isinstance(val, list) and val:
                    last = val[-1]
                    if isinstance(last, dict):
                        file_id = last.get("file_id")
                    else:
                        file_id = getattr(last, "file_id", None)
                elif isinstance(val, dict):
                    file_id = val.get("file_id")
                if file_id:
                    break
            if not file_id or not self._bot:
                continue
            try:
                file_obj = await self._bot.get_file(file_id)
                data = bytes(await file_obj.download_as_bytearray())
                ext = os.path.splitext(getattr(file_obj, "file_path", "") or "")[1]
                if kind == "photo":
                    if ext not in _IMAGE_EXTENSIONS:
                        ext = ".jpg"
                    cached_path = await cache_image_from_bytes_async(data, ext=ext)
                elif kind == "voice_note":
                    cached_path = await cache_audio_from_bytes_async(data, ext=".ogg")
                elif kind == "audio":
                    mime_ext = ext if ext in (".mp3", ".ogg", ".m4a", ".wav", ".opus") else ".mp3"
                    cached_path = await cache_audio_from_bytes_async(data, ext=mime_ext)
                elif kind == "video":
                    mime_ext = ext if ext in (".mp4", ".mov", ".mkv", ".webm") else ".mp4"
                    cached_path = await cache_video_from_bytes_async(data, ext=mime_ext)
                elif kind == "animation":
                    mime_ext = ext if ext in (".gif", ".webp", ".mp4") else ".mp4"
                    cached_path = await cache_video_from_bytes_async(data, ext=mime_ext)
                else:
                    continue
                event.media_urls.append(cached_path)
                event.media_types.append(_rich_media_mime(kind, ext))
                added.append(cached_path)
                logger.info("[Telegram] Cached rich %s at %s (file_id=%s...)",
                            kind, cached_path, str(file_id)[:16])
            except Exception as exc:
                logger.warning(
                    "[Telegram] Failed to cache rich %s (file_id=%s...): %s",
                    kind, str(file_id)[:16], exc, exc_info=True,
                )
        return added

    async def _handle_rich_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming Rich Message (Bot API 10.1+).

        Rich messages from the Telegram Rich Text Editor arrive with empty
        ``message.text`` — their content lives only in
        ``message.api_kwargs[\"rich_message\"][\"blocks\"]`` (PTB <22.6 has no
        ``message.rich_message`` attribute). Because ``message.text`` is empty
        these messages match none of the normal handlers (TEXT/COMMAND/media)
        and were silently dropped. This handler extracts the structured rich
        content, converts it to Markdown (headings, code, blockquotes, lists,
        tables, emphasis, links/buttons, spoilers preserved), downloads any
        embedded media for vision/STT, and forwards the result through the
        normal text pipeline.
        """
        msg = self._effective_update_message(update)
        if not msg:
            return

        # rich_message may be a first-class attr (PTB 22.6+) or live in
        # api_kwargs (older). api_kwargs is duck-typed (mapping-like), NOT
        # necessarily a dict — never isinstance-check it.
        rich_message = getattr(msg, "rich_message", None)
        if not rich_message:
            api_kwargs = getattr(msg, "api_kwargs", None)
            getter = getattr(api_kwargs, "get", None)
            if callable(getter):
                rich_message = getter("rich_message")
        if not rich_message:
            # Not a rich message after all — this handler only fires for
            # messages no other handler claimed.
            return

        rich_getter = getattr(rich_message, "get", None)
        blocks = rich_getter("blocks") if callable(rich_getter) else None
        try:
            text, _media = rich_md.rich_message_to_markdown(rich_message)
        except Exception:
            logger.debug("[Telegram] rich_md render error", exc_info=True)
            text = self._flatten_rich_blocks(blocks).strip()
        text = (text or "").strip()
        if not text:
            logger.debug("[Telegram] Rich Message with no extractable text, ignoring")
            return

        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg, what="rich message")
            return
        if not self._gate_or_observe(msg, update, MessageType.TEXT):
            return
        await self._ensure_forum_commands(msg)

        try:
            event = self._build_message_event(msg, MessageType.TEXT, update_id=update.update_id)
            event.text = self._clean_bot_trigger_text(text)
            try:
                await self._cache_rich_media(rich_message, event)
            except Exception as _mce:
                logger.debug("[Telegram] rich media cache error: %s", _mce)
            await self._cache_replied_media(msg, event)
            event = self._apply_telegram_group_observe_attribution(event)
            self._enqueue_text_event(event)
        except Exception:
            logger.exception(
                "[Telegram] Failed to enqueue rich message %s",
                getattr(msg, "message_id", None),
            )
