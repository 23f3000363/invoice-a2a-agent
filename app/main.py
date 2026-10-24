"""A2A 1.0 HTTP+JSON Invoice Action Agent."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

load_dotenv()

from . import storage
from .ai_decision import decide_batch
from .auth import require_principal
from .protocol import *


PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    os.getenv("A2A_BASE_URL", "http://127.0.0.1:8000/a2a"),
).rstrip("/")

app = FastAPI(
    title="Invoice Action Agent",
    version="1.0.0",
    default_response_class=JSONResponse,
)

# Per-principal message locks so concurrent equivalent sends share one Task.
_message_locks: Dict[str, asyncio.Lock] = {}
_message_locks_guard = asyncio.Lock()


async def _lock_for(principal: str, message_id: str) -> asyncio.Lock:
    key = f"{principal}:{message_id}"
    async with _message_locks_guard:
        lock = _message_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _message_locks[key] = lock
        return lock



@app.on_event("startup")
async def startup() -> None:
    await storage.init_db()


@app.exception_handler(HTTPException)
async def a2a_http_error(request: Request, exc: HTTPException):
    detail = (
        exc.detail
        if isinstance(exc.detail, dict)
        else {"code": "BAD_REQUEST", "message": str(exc.detail)}
    )
    return JSONResponse(
        {"error": detail},
        status_code=exc.status_code,
        media_type=A2A_JSON,
    )


def _version(value: Optional[str]) -> None:
    if (value or "").strip() != "1.0":
        raise HTTPException(400, GENERIC_VERSION["error"])


def _media(content_type: Optional[str]) -> None:
    actual = (content_type or "").split(";", 1)[0].strip().lower()
    if actual != A2A_JSON:
        raise HTTPException(
            400,
            {
                "code": "INVALID_MEDIA_TYPE",
                "message": "Content-Type must be application/a2a+json",
            },
        )


def _json(data: Any, status: int = 200):
    return JSONResponse(data, status_code=status, media_type=A2A_JSON)


def _public(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: task[key]
        for key in (
            "id",
            "contextId",
            "status",
            "history",
            "artifacts",
            "createdAt",
            "updatedAt",
        )
        if key in task
    }


def _part(message: Dict[str, Any], media_type: str) -> Dict[str, Any]:
    parts = message.get("parts")
    if (
        not isinstance(parts, list)
        or len(parts) != 1
        or not isinstance(parts[0], dict)
        or parts[0].get("mediaType") != media_type
        or not isinstance(parts[0].get("data"), dict)
    ):
        raise HTTPException(400, GENERIC_BAD_REQUEST["error"])
    return parts[0]


@app.get("/.well-known/agent-card.json")
async def agent_card():
    path = Path(__file__).resolve().parent.parent / "agent_card.json"
    card = json.loads(path.read_text(encoding="utf-8"))
    card["supportedInterfaces"] = [
        {
            "url": PUBLIC_BASE_URL,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }
    ]
    return JSONResponse(card, media_type="application/json")


@app.post("/a2a/message:send")
async def message_send(
    request: Request,
    principal: str = Depends(require_principal),
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
    content_type: Optional[str] = Header(None, alias="Content-Type"),
):
    _version(a2a_version)
    _media(content_type)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, GENERIC_BAD_REQUEST["error"])

    if not isinstance(body, dict) or not isinstance(body.get("message"), dict):
        raise HTTPException(400, GENERIC_BAD_REQUEST["error"])

    message = body["message"]
    message_id = message.get("messageId")

    if (
        message.get("role") != ROLE_USER
        or not isinstance(message_id, str)
        or not message_id.strip()
    ):
        raise HTTPException(400, GENERIC_BAD_REQUEST["error"])

    digest = message_hash(message)

    lock = await _lock_for(principal, message_id)
    async with lock:
        return await _message_send_locked(
            principal, message, message_id, digest, body
        )


async def _message_send_locked(
    principal: str,
    message: Dict[str, Any],
    message_id: str,
    digest: str,
    body: Dict[str, Any],
):
    # Check idempotency before dispatching either message type.
    existing_id, existing_hash = await storage.check_idempotency(
        principal, message_id, digest
    )

    if existing_id:
        if existing_hash != digest:
            return _json(GENERIC_CONFLICT, 409)

        existing = await storage.get_task(principal, existing_id)

        if existing and existing["status"]["state"] == TASK_STATE_WORKING:
            # Concurrent equivalent request: wait for creator to finish.
            for _ in range(600):
                await asyncio.sleep(0.1)
                existing = await storage.get_task(principal, existing_id)
                if (
                    not existing
                    or existing["status"]["state"] != TASK_STATE_WORKING
                ):
                    break

        if existing:
            return _json({"task": _public(existing)})

    parts = message.get("parts")
    if not isinstance(parts, list) or len(parts) != 1:
        raise HTTPException(400, GENERIC_BAD_REQUEST["error"])

    part = parts[0]
    media_type = part.get("mediaType") if isinstance(part, dict) else None

    if media_type == INVOICE_BATCH:
        _part(message, INVOICE_BATCH)
        return await _initial(principal, message, digest)

    if media_type == INVOICE_RESULTS:
        _part(message, INVOICE_RESULTS)
        return await _continue(principal, message, digest)

    raise HTTPException(400, GENERIC_BAD_REQUEST["error"])


async def _initial(
    principal: str,
    message: Dict[str, Any],
    digest: str,
):
    data = message["parts"][0]["data"]
    batch_id = data.get("batchId")
    packages = data.get("packages")

    if (
        not isinstance(batch_id, str)
        or not batch_id
        or not isinstance(packages, list)
        or not packages
        or len(packages) > 20
    ):
        raise HTTPException(400, GENERIC_BAD_REQUEST["error"])

    package_ids = [
        package.get("packageId") or package.get("id")
        for package in packages
        if isinstance(package, dict)
    ]

    if (
        len(package_ids) != len(packages)
        or any(not isinstance(pid, str) or not pid for pid in package_ids)
        or len(set(package_ids)) != len(package_ids)
    ):
        raise HTTPException(400, GENERIC_BAD_REQUEST["error"])

    task_id = storage.new_task_id()
    context_id = message.get("contextId") or storage.new_context_id()
    now = time.time()

    shell = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": TASK_STATE_WORKING},
        "history": [message],
        "artifacts": [],
        "batchId": batch_id,
        "createdAt": now,
        "updatedAt": now,
    }

    created, existing_hash, existing_id = await storage.reserve_initial(
        shell, principal, message["messageId"], digest
    )

    if not created:
        if existing_hash != digest:
            return _json(GENERIC_CONFLICT, 409)

        existing = await storage.get_task(principal, existing_id)

        for _ in range(600):
            if not existing or existing["status"]["state"] != TASK_STATE_WORKING:
                break
            await asyncio.sleep(0.1)
            existing = await storage.get_task(principal, existing_id)

        if existing:
            return _json({"task": _public(existing)})

        raise HTTPException(409, GENERIC_CONFLICT["error"])

    try:
        decisions = await decide_batch(packages)

        if len(decisions) != len(packages):
            raise ValueError("Decision count mismatch")

        proposals = []

        for index, decision in enumerate(decisions):
            package_id = package_ids[index]
            action = decision.get("action")
            facts = decision.get("facts")
            refs = decision.get("evidenceRefs")
            rationale = decision.get("rationale")

            valid = (
                action in VALID_ACTIONS
                and isinstance(facts, dict)
                and isinstance(facts.get("vendorName"), str)
                and bool(facts["vendorName"].strip())
                and isinstance(facts.get("invoiceNumber"), str)
                and bool(facts["invoiceNumber"].strip())
                and isinstance(facts.get("amountMinor"), int)
                and not isinstance(facts.get("amountMinor"), bool)
                and facts["amountMinor"] >= 0
                and isinstance(facts.get("currency"), str)
                and bool(facts["currency"].strip())
                and isinstance(refs, list)
                and len(refs) == 3
                and len(set(refs)) == 3
                and all(
                    isinstance(ref, str)
                    and ref.startswith("[")
                    and ref.endswith("]")
                    for ref in refs
                )
                and isinstance(rationale, str)
                and 60 <= len(rationale.strip()) <= 1500
                and action in rationale
                and sum(ref in rationale for ref in refs) >= 2
            )

            # Do not manufacture evidence references or issue a proposal
            # when the decision fails schema validation.
            if not valid:
                raise ValueError(
                    f"Decision failed proposal validation for package {package_id}"
                )

            proposals.append(
                {
                    "packageId": package_id,
                    "actionId": storage.new_action_id(),
                    "action": action,
                    "facts": facts,
                    "evidenceRefs": refs,
                    "rationale": rationale.strip(),
                }
            )

        task = {
            **shell,
            "status": {"state": TASK_STATE_INPUT_REQUIRED},
            "artifacts": [
                {
                    "artifactId": storage.new_action_id(),
                    "parts": [
                        {
                            "mediaType": INVOICE_PROPOSALS,
                            "data": {
                                "batchId": batch_id,
                                "proposals": proposals,
                            },
                        }
                    ],
                }
            ],
            "_proposals": {
                proposal["actionId"]: proposal for proposal in proposals
            },
            "updatedAt": time.time(),
        }

        await storage.save_task(task, principal)
        return _json({"task": _public(task)})

    except Exception as exc:
        # Log the real failure so Render logs show the root cause.
        import logging
        logging.getLogger("invoice_a2a").exception(
            "Decision failed for batch: %s", exc
        )
        # Keep the reservation terminal and replayable; do not trigger a
        # second model call by retrying the same message ID.
        failed = {
            **shell,
            "status": {"state": TASK_STATE_FAILED},
            "history": [message],
            "artifacts": [],
            "updatedAt": time.time(),
        }
        await storage.transition_task(failed, principal, TASK_STATE_WORKING)
        raise HTTPException(
            503,
            {
                "code": "DECISION_SERVICE_UNAVAILABLE",
                "message": f"Could not evaluate this batch: {type(exc).__name__}: {exc}",
            },
        )


async def _continue(
    principal: str,
    message: Dict[str, Any],
    digest: str,
):
    task_id = message.get("taskId")
    context_id = message.get("contextId")
    message_id = message.get("messageId")

    if (
        not isinstance(task_id, str)
        or not task_id
        or not isinstance(context_id, str)
        or not context_id
        or not isinstance(message_id, str)
        or not message_id.strip()
    ):
        raise HTTPException(400, GENERIC_BAD_REQUEST["error"])

    task = await storage.get_task(principal, task_id)
    if not task:
        # Do not reveal whether the task exists for another principal.
        raise HTTPException(404, GENERIC_NOT_FOUND["error"])

    # Defense-in-depth: refuse if stored owner does not match (should be impossible).
    if task.get("_principal") and task.get("_principal") != principal:
        raise HTTPException(404, GENERIC_NOT_FOUND["error"])

    if task.get("contextId") != context_id:
        raise HTTPException(403, GENERIC_FORBIDDEN["error"])

    if task["status"]["state"] != TASK_STATE_INPUT_REQUIRED:
        raise HTTPException(409, GENERIC_CONFLICT["error"])

    data = message["parts"][0]["data"]
    if data.get("batchId") != task.get("batchId"):
        raise HTTPException(400, GENERIC_BAD_REQUEST["error"])

    results = data.get("results")
    proposals = task.get("_proposals") or {}

    if not isinstance(results, list):
        raise HTTPException(400, GENERIC_BAD_REQUEST["error"])

    seen = set()
    executions = []
    by_package = {
        proposal["packageId"]: proposal
        for proposal in proposals.values()
    }

    for result in results:
        if not isinstance(result, dict):
            raise HTTPException(400, GENERIC_BAD_REQUEST["error"])

        action_id = result.get("actionId")
        package_id = result.get("packageId")
        action = result.get("action")
        outcome = result.get("outcome")
        nonce = result.get("receiptNonce")

        if not isinstance(action_id, str) or action_id in seen:
            raise HTTPException(400, GENERIC_BAD_REQUEST["error"])
        seen.add(action_id)

        proposal = proposals.get(action_id)

        if (
            not proposal
            or proposal["packageId"] != package_id
            or proposal["action"] != action
            or package_id not in by_package
            or outcome not in ("ACCEPTED", "REJECTED")
        ):
            raise HTTPException(400, GENERIC_BAD_REQUEST["error"])

        if outcome == "ACCEPTED":
            if not isinstance(nonce, str) or not nonce:
                raise HTTPException(400, GENERIC_BAD_REQUEST["error"])
            executions.append(
                {
                    "packageId": package_id,
                    "actionId": action_id,
                    "action": action,
                    "receiptNonce": nonce,
                    "facts": proposal["facts"],
                    "evidenceRefs": proposal["evidenceRefs"],
                }
            )

    updated_task = dict(task)
    updated_task["status"] = {"state": TASK_STATE_COMPLETED}
    updated_task["history"] = list(task.get("history") or []) + [message]
    updated_task["artifacts"] = list(task.get("artifacts") or []) + [
        {
            "artifactId": storage.new_action_id(),
            "parts": [
                {
                    "mediaType": INVOICE_RECEIPTS,
                    "data": {
                        "batchId": task["batchId"],
                        "executions": executions,
                    },
                }
            ],
        }
    ]
    updated_task["updatedAt"] = time.time()

    # The state transition and continuation idempotency key must be committed
    # together by the storage transaction.
    committed = await storage.commit_continuation(
        updated_task,
        principal,
        message_id,
        digest,
        TASK_STATE_INPUT_REQUIRED,
    )

    if not committed:
        # A concurrent continuation or cancellation may have won.
        existing_id, existing_hash = await storage.check_idempotency(
            principal, message_id, digest
        )

        if existing_id:
            if existing_hash != digest:
                return _json(GENERIC_CONFLICT, 409)

            existing_task = await storage.get_task(principal, existing_id)
            if existing_task:
                return _json({"task": _public(existing_task)})

        # Cancel likely won — return current task or 409
        current = await storage.get_task(principal, task_id)
        if current:
            return _json({"task": _public(current)}) if current["status"]["state"] in (
                TASK_STATE_COMPLETED, TASK_STATE_CANCELED
            ) else (_json(GENERIC_CONFLICT, 409))
        raise HTTPException(409, GENERIC_CONFLICT["error"])

    final = await storage.get_task(principal, task_id)
    return _json({"task": _public(final or updated_task)})


@app.get("/a2a/tasks/{task_id}")
async def get_task(
    task_id: str,
    principal: str = Depends(require_principal),
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
):
    _version(a2a_version)
    task = await storage.get_task(principal, task_id)

    if not task:
        raise HTTPException(404, GENERIC_NOT_FOUND["error"])

    return _json(_public(task))


@app.get("/a2a/tasks")
async def list_tasks(
    principal: str = Depends(require_principal),
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
):
    _version(a2a_version)
    tasks = await storage.list_tasks(principal)
    return _json({"tasks": [_public(task) for task in tasks]})


@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(
    task_id: str,
    principal: str = Depends(require_principal),
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
):
    _version(a2a_version)
    # Cancel has no body — do not require Content-Type.

    task = await storage.get_task(principal, task_id)
    if not task:
        raise HTTPException(404, GENERIC_NOT_FOUND["error"])

    state = task["status"]["state"]
    if state == TASK_STATE_CANCELED:
        return _json(_public(task))
    if state != TASK_STATE_INPUT_REQUIRED:
        raise HTTPException(409, GENERIC_CONFLICT["error"])

    canceled = dict(task)
    canceled["status"] = {"state": TASK_STATE_CANCELED}
    canceled["updatedAt"] = time.time()

    ok = await storage.transition_task(
        canceled,
        principal,
        TASK_STATE_INPUT_REQUIRED,
    )
    if not ok:
        raise HTTPException(409, GENERIC_CONFLICT["error"])

    return _json(_public(canceled))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"service": "Invoice Action Agent", "a2a": PUBLIC_BASE_URL}
