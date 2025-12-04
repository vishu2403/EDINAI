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
    logger.info(f"Broadcasting chat message to enrollments: {unique_enrollments}")
    logger.info(f"Payload: {payload}")
    
    for enrollment in unique_enrollments:
        room = _personal_room(admin_id, enrollment)
        logger.info(f"Emitting message:new to room: {room}")
        await sio.emit(
            "message:new",
            payload,
            room=room,
            skip_sid=skip_sid,
        )
    logger.info(f"Broadcast complete to {len(unique_enrollments)} enrollments")


@sio.event
async def connect(sid: str, environ: dict, auth: dict | None = None) -> None:
    # Try to get token from query string first
    query_string: bytes = environ.get("asgi.scope", {}).get("query_string", b"")
    params = parse_qs(query_string.decode())
    token = params.get("token", [""])[0]
    
    # If not in query string, try auth dict
    if not token and auth:
        token = auth.get("token", "")

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

    personal_room = _personal_room(context["admin_id"], enrollment)
    class_room = _class_room(context)
    
    logger.info(f"Student {enrollment} joining rooms:")
    logger.info(f"  - Personal room: {personal_room}")
    logger.info(f"  - Class room: {class_room}")
    
    await sio.enter_room(sid, personal_room)
    await sio.enter_room(sid, class_room)

    await _broadcast_presence(context, enrollment, "online")
    
    logger.info(f"Student {enrollment} connected via Socket.IO (sid: {sid})")


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


@sio.on("send_message")
async def handle_send_message(sid: str, data: dict) -> None:
    """Handle incoming chat messages via Socket.IO (pure websocket, no REST API)"""
    session = _ACTIVE_CLIENTS.get(sid)
    if not session:
        logger.warning(f"send_message from unknown sid: {sid}")
        return

    current_enrollment = session["enrollment"]
    current_context = session["context"]
    
    peer_enrollment = (data or {}).get("peer_enrollment")
    message_text = (data or {}).get("message", "").strip()
    share_metadata = (data or {}).get("share_metadata")

    logger.info(f"[Socket.IO] send_message from {current_enrollment} to {peer_enrollment}: {message_text[:50]}...")

    if not peer_enrollment or not message_text:
        logger.warning("send_message missing peer_enrollment or message")
        return

    # Verify peer is a classmate
    try:
        peer_context = student_portal_service.ensure_same_classmate(current_context, peer_enrollment)
    except HTTPException:
        logger.warning(f"send_message: {peer_enrollment} is not a classmate of {current_enrollment}")
        return

    # Create message payload for sending to service
    from ..schemas.student_portal_schema import SendChatMessageRequest
    
    payload = SendChatMessageRequest(
        peer_enrollment=peer_enrollment,
        message=message_text,
        share_metadata=share_metadata
    )
    
    # Save message to database
    record = student_portal_service.send_chat_message(
        payload=payload,
        current_context=current_context,
        peer_context=peer_context,
    )
    
    logger.info(f"[Socket.IO] Message saved to DB with ID: {record.get('id')}")

    # Broadcast to both participants in real-time
    broadcast_payload = {
        "message": record,
        "participants": [current_enrollment, peer_enrollment],
    }

    sender_room = _personal_room(current_context["admin_id"], current_enrollment)
    receiver_room = _personal_room(current_context["admin_id"], peer_enrollment)

    logger.info(f"[Socket.IO] Broadcasting message to sender room: {sender_room}")
    logger.info(f"[Socket.IO] Broadcasting message to receiver room: {receiver_room}")

    # Emit to sender
    await sio.emit("message:new", broadcast_payload, room=sender_room)
    
    # Emit to receiver  
    await sio.emit("message:new", broadcast_payload, room=receiver_room)
    
    logger.info(f"[Socket.IO] Message broadcast complete!")


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
