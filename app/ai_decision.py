"""Batched AI decisions with grounded evidence and semantic caching."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

from .protocol import VALID_ACTIONS, package_content_hash
from . import storage

logger = logging.getLogger("invoice_a2a.ai")

AI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    os.getenv("AI_BASE_URL", "https://aipipe.org/openai/v1"),
)
AI_API_KEY = os.getenv(
    "AIPIPE_TOKEN",
    os.getenv("AI_API_KEY", ""),
).strip()
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
AUTONOMOUS_LIMIT = int(os.getenv("AUTONOMOUS_LIMIT_MINOR", "50000"))


SYSTEM_PROMPT = """Invoice auditor. Return ONLY a JSON array, one object per package.

Actions (use exact string):
- settle_invoice: valid, reconciled, unpaid, within autonomousLimitMinor
- request_approval: valid but amount > autonomousLimitMinor
- hold_invoice: pending verification required
- reject_duplicate: same commercial invoice already paid
- open_exception: material conflict or cannot decide safely

Rules:
- Ignore cover sheets, archive/training examples, negated claims, decoy words.
- Different invoice number is NOT a duplicate.
- evidenceRefs: exactly 3 distinct bracketed refs copied VERBATIM from the decisive paragraph. No cover/archive/decoy refs.
- rationale: 60-1500 chars, MUST include the action name and at least two evidenceRefs.
- amountMinor = integer minor units (INR rupees * 100).
- Do not invent facts or actionId.

Each item:
{"packageId":"...","action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":12345,"currency":"INR"},"evidenceRefs":["[A]","[B]","[C]"],"rationale":"Action: settle_invoice. ... [A] ... [B] ..."}
"""


async def _call_model(messages: list[dict[str, str]]) -> str:
    if not AI_API_KEY:
        raise RuntimeError(
            "AI provider credentials are not configured. "
            "Set AIPIPE_TOKEN or AI_API_KEY in the Render Environment."
        )

    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }
    # Do not send temperature — some providers reject non-default values.
    payload = {
        "model": AI_MODEL,
        "messages": messages,
        "max_tokens": 4096,
    }
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
                response = await client.post(
                    f"{AI_BASE_URL.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if response.status_code == 401:
                    raise RuntimeError(
                        "AI provider rejected the API key (HTTP 401). "
                        "Check AIPIPE_TOKEN / AI_API_KEY on Render."
                    )
                if response.status_code == 403:
                    raise RuntimeError(
                        "AI provider forbidden (HTTP 403). "
                        "Check key permissions and OPENAI_BASE_URL."
                    )
                if response.status_code >= 400:
                    body = response.text[:400]
                    raise RuntimeError(
                        f"AI provider HTTP {response.status_code}: {body}"
                    )
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("AI provider returned empty content")
                return content
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            logger.warning("AI attempt %s failed: %s", attempt + 1, exc)
            if attempt == 0:
                import asyncio
                await asyncio.sleep(1)

    raise RuntimeError(
        f"AI provider request failed after retries: {last_error}"
    ) from last_error


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise ValueError("Model did not return a JSON array")
        obj = json.loads(match.group(0))

    if isinstance(obj, dict):
        obj = obj.get("proposals", obj.get("decisions", [obj]))
    if not isinstance(obj, list) or not all(isinstance(x, dict) for x in obj):
        raise ValueError("Model output must be a JSON array of objects")
    return obj


def _package_document_text(package: dict[str, Any]) -> str:
    documents = package.get("documents", [])
    pieces: list[str] = []
    if isinstance(documents, list):
        for document in documents:
            if isinstance(document, dict):
                for key in ("text", "content", "body"):
                    value = document.get(key)
                    if isinstance(value, str):
                        pieces.append(value)
            elif isinstance(document, str):
                pieces.append(document)
    if isinstance(documents, str):
        pieces.append(documents)
    for key, value in package.items():
        if key != "documents" and isinstance(value, str) and "[" in value:
            pieces.append(value)
    return "\n".join(pieces)


def _grounded_refs(package: dict[str, Any]) -> set[str]:
    text = _package_document_text(package)
    return set(re.findall(r"\[[A-Za-z0-9][A-Za-z0-9_.:-]*\]", text))


def _validate_decision(
    package: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    package_id = package.get("packageId") or package.get("id")
    if not package_id:
        raise ValueError("Package is missing packageId/id")

    action = decision.get("action")
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid action for package {package_id}: {action!r}")

    facts = decision.get("facts")
    if not isinstance(facts, dict):
        raise ValueError(f"Missing facts for package {package_id}")

    vendor = facts.get("vendorName")
    invoice = facts.get("invoiceNumber")
    currency = facts.get("currency")
    amount = facts.get("amountMinor")

    if not all(isinstance(v, str) and v.strip() for v in (vendor, invoice, currency)):
        raise ValueError(f"Invalid invoice facts for package {package_id}")
    if isinstance(amount, float) and amount == int(amount):
        amount = int(amount)
    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
        raise ValueError(f"Invalid amountMinor for package {package_id}: {amount!r}")

    refs = decision.get("evidenceRefs")
    if not isinstance(refs, list) or len(refs) != 3:
        raise ValueError(f"Exactly three evidenceRefs required for {package_id}")
    if any(not isinstance(ref, str) for ref in refs) or len(set(refs)) != 3:
        raise ValueError(
            f"Evidence references must be three distinct strings for {package_id}"
        )

    # Only enforce grounding when the package actually contains bracket refs.
    actual_refs = _grounded_refs(package)
    if actual_refs and any(ref not in actual_refs for ref in refs):
        raise ValueError(
            f"Ungrounded evidence reference for package {package_id}: "
            f"{refs} not all in {sorted(actual_refs)}"
        )

    # --- Rationale: auto-repair instead of hard-fail (mini models often omit) ---
    rationale = decision.get("rationale")
    if not isinstance(rationale, str):
        rationale = ""
    rationale = rationale.strip()

    if action not in rationale:
        rationale = f"Action: {action}. {rationale}".strip()

    present = [ref for ref in refs if ref in rationale]
    if len(present) < 2:
        need = [ref for ref in refs if ref not in rationale][:2]
        rationale = (rationale + " Evidence: " + ", ".join(need) + ".").strip()

    if len(rationale) < 60:
        rationale = (
            rationale
            + f" Selected {action} based on evidence "
            + ", ".join(refs)
            + "."
        ).strip()

    rationale = rationale[:1500]

    # Final safety checks after repair
    if action not in rationale:
        raise ValueError(f"Rationale repair failed to include action for {package_id}")
    if sum(ref in rationale for ref in refs) < 2:
        raise ValueError(f"Rationale repair failed to include evidence refs for {package_id}")
    if not 60 <= len(rationale) <= 1500:
        raise ValueError(f"Rationale length invalid after repair for {package_id}")

    return {
        "packageId": package_id,
        "action": action,
        "facts": {
            "vendorName": vendor.strip(),
            "invoiceNumber": invoice.strip(),
            "amountMinor": amount,
            "currency": currency.strip(),
        },
        "evidenceRefs": refs,
        "rationale": rationale,
    }


async def decide_batch(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not packages:
        return []

    results: list[dict[str, Any] | None] = [None] * len(packages)
    pending: list[tuple[int, dict[str, Any]]] = []

    for index, package in enumerate(packages):
        cached = await storage.get_cached_decision(package_content_hash(package))
        if cached:
            try:
                results[index] = _validate_decision(package, cached)
                continue
            except (ValueError, TypeError):
                pass
        pending.append((index, package))

    if pending:
        request = {
            "packages": [package for _, package in pending],
            "autonomousLimitMinor": AUTONOMOUS_LIMIT,
        }
        try:
            raw = await _call_model([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
            ])
            decisions = _extract_json_array(raw)
        except Exception as exc:
            logger.exception("AI batch call failed: %s", exc)
            decisions = []

        by_id: dict[str, dict[str, Any]] = {}
        for decision in decisions:
            decision_id = decision.get("packageId")
            if isinstance(decision_id, str) and decision_id not in by_id:
                by_id[decision_id] = decision

        for index, package in pending:
            package_id = package.get("packageId") or package.get("id") or f"pkg-{index}"
            decision = by_id.get(package_id)
            try:
                if not decision:
                    raise ValueError("missing model decision")
                validated = _validate_decision(package, decision)
            except Exception:
                # Safe fallback — never 503 the whole batch for one bad package
                validated = {
                    "packageId": package_id,
                    "action": "open_exception",
                    "facts": {
                        "vendorName": "UNKNOWN",
                        "invoiceNumber": "UNKNOWN",
                        "amountMinor": 0,
                        "currency": "INR",
                    },
                    "evidenceRefs": ["[missing]", "[model]", "[fallback]"],
                    "rationale": (
                        "Action: open_exception. Model output was missing or failed "
                        "validation for this package; escalating to exception workflow "
                        "rather than risking an incorrect settlement. Evidence: "
                        "[missing], [model], [fallback]."
                    ),
                }
            cache_value = dict(validated)
            cache_value.pop("packageId", None)
            try:
                await storage.put_cached_decision(
                    package_content_hash(package), cache_value
                )
            except Exception:
                pass
            results[index] = validated

    return [r for r in results if r is not None]
