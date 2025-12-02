"""Socket.IO server setup for student portal realtime features."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional
from urllib.parse import parse_qs

import socketio

from ..config import settings
from ..services import student_portal_service
from ..utils.student_token import decode_student_token
from fastapi import HTTPException

logger = logging.getLogger(__name__)

if settings.cors_origins == ["*"]:
    _CORS_ORIGINS: object = "*"
else:
    _CORS_ORIGINS = settings.cors_origins

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=_CORS_ORIGINS,
    engineio_options={
        "cors_allowed_origins": _CORS_ORIGINS,
        "transports": ["websocket"],
        "allow_upgrades": False,
    },
)

# sid -> session data
_ACTIVE_CLIENTS: Dict[str, Dict[str, object]] = {}
# enrollment_number -> set of active sids
_STUDENT_CONNECTIONS: Dict[str, set[str]] = {}


def _personal_room(admin_id: int, enrollment_number: str) -> str:
    return f"student:{admin_id}:{enrollment_number}".lower()


def _class_room(context: Dict[str, Optional[str]]) -> str:
    division = (context.get("division") or "").strip() or "all"
    return f"class:{context['admin_id']}:{context['std']}:{division}".lower()


async def _broadcast_presence(context: Dict[str, Optional[str]], enrollment: str, status: str) -> None:
    payload = {
        "enrollment": enrollment,
        "status": status,
    }
    await sio.emit("presence:update", payload, room=_class_room(context))

async def broadcast_chat_message(*, admin_id: int, enrollments: Iterable[str], payload: dict, skip_sid: str | None = None) -> None:
    unique_enrollments = {enrollment.lower() for enrollment in enrollments if enrollment}
    for enrollment in unique_enrollments:
        await sio.emit(
            "message:new",
            payload,
            room=_personal_room(admin_id, enrollment),
            skip_sid=skip_sid,
        )


@sio.event
async def connect(sid: str, environ: dict) -> None:
    query_string: bytes = environ.get("asgi.scope", {}).get("query_string", b"")
    params = parse_qs(query_string.decode())
    token = params.get("token", [""])[0]

    try:
        enrollment = decode_student_token(token)
        context = student_portal_service.get_roster_context(enrollment)
    except Exception as exc:  # pragma: no cover - handshake failure
        logger.warning("Socket handshake failed: %s", exc)
        raise ConnectionRefusedError("unauthorized") from exc

    session_payload = {
        "enrollment": enrollment,
        "context": context,
    }
    await sio.save_session(sid, session_payload)
    _ACTIVE_CLIENTS[sid] = session_payload

    _STUDENT_CONNECTIONS.setdefault(enrollment, set()).add(sid)

    await sio.enter_room(sid, _personal_room(context["admin_id"], enrollment))
    await sio.enter_room(sid, _class_room(context))

    await _broadcast_presence(context, enrollment, "online")


@sio.event
async def disconnect(sid: str) -> None:
    session = _ACTIVE_CLIENTS.pop(sid, None)
    if not session:
        return

    enrollment = session["enrollment"]
    context = session["context"]

    sid_set = _STUDENT_CONNECTIONS.get(enrollment)
    if sid_set is not None:
        sid_set.discard(sid)
        if not sid_set:
            _STUDENT_CONNECTIONS.pop(enrollment, None)
            await _broadcast_presence(context, enrollment, "offline")


@sio.on("signal")
async def handle_signal(sid: str, data: dict) -> None:
    session = _ACTIVE_CLIENTS.get(sid)
    if not session:
        return

    peer_enrollment = (data or {}).get("peer_enrollment")
    signal_type = (data or {}).get("signal_type")
    payload = data.get("payload")

    if not peer_enrollment or not signal_type:
        return

    current_context = session["context"]
    try:
        peer_context = student_portal_service.ensure_same_classmate(current_context, peer_enrollment)
    except HTTPException:  # pragma: no cover - invalid peer
        return

    await sio.emit(
        "signal",
        {
            "sender_enrollment": session["enrollment"],
            "signal_type": signal_type,
            "payload": payload,
        },
        room=_personal_room(peer_context["admin_id"], peer_enrollment),
        skip_sid=sid,
    )


@sio.on("typing")
async def handle_typing(sid: str, data: dict) -> None:
    session = _ACTIVE_CLIENTS.get(sid)
    if not session:
        return

    peer_enrollment = (data or {}).get("peer_enrollment")
    is_typing = bool((data or {}).get("typing", False))
    if not peer_enrollment:
        return

    context = session["context"]
    await sio.emit(
        "typing",
        {
            "sender_enrollment": session["enrollment"],
            "typing": is_typing,
        },
        room=_personal_room(context["admin_id"], peer_enrollment),
        skip_sid=sid,
    )


@sio.on("presence:request")
async def handle_presence_request(sid: str) -> None:
    session = _ACTIVE_CLIENTS.get(sid)
    if not session:
        return

    context = session["context"]
    enrollment = session["enrollment"]
    room = _class_room(context)

    # compile list of currently online peers in class
    online = [other for other, sids in _STUDENT_CONNECTIONS.items() if sids]
    await sio.emit("presence:snapshot", {"online": online}, room=sid)
    await _broadcast_presence(context, enrollment, "online")


__all__ = ["sio", "broadcast_chat_message"]
