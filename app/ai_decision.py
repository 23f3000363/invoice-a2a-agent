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
AI_API_KEY = os.getenv("AIPIPE_TOKEN", os.getenv("AI_API_KEY", "")).strip()
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
AUTONOMOUS_LIMIT = int(os.getenv("AUTONOMOUS_LIMIT_MINOR", "50000"))


SYSTEM_PROMPT = """You are a careful invoice-claims auditor.

Return ONLY a JSON array with exactly one decision per supplied package.
Each decision must use that package's exact packageId.

DECISION POLICY
Choose exactly one action:
- settle_invoice: valid, reconciled, unpaid, and within autonomous authority.
- request_approval: commercially valid but above autonomous authority.
- hold_invoice: a specific required verification is still pending.
- reject_duplicate: the same commercial invoice has already been paid.
- open_exception: material evidence conflicts or the decision cannot be
  made safely from the supplied records.

EVIDENCE RULES
1. Read the entire package. Identify the decisive evidence paragraph before
   choosing an action.
2. Distinguish current transaction evidence from cover sheets, archive
   examples, training examples, irrelevant action words, and negated claims.
3. A different invoice number is not a duplicate. Confirm that the same
   commercial invoice was already paid before choosing reject_duplicate.
4. A stated pending verification requires hold_invoice unless stronger
   evidence resolves that requirement.
5. A commercially valid invoice above the autonomous limit requires
   request_approval. Never settle above that limit.
6. If records materially conflict or decisive facts are missing, use
   open_exception rather than inventing facts.
7. Use only facts and references present in the supplied package.
8. Return exactly three distinct, decisive bracketed evidence references
   copied verbatim from the paragraph that determines the action.
9. Never use cover-sheet, archive, training, or decoy references.
10. The rationale must be 60–1500 characters, name the selected action,
    explain the decision, and cite at least two of the three evidence refs.

FACTS
- vendorName and invoiceNumber must match the source records.
- amountMinor is an integer in minor currency units. For INR, convert
  rupees to paise (multiply by 100).
- currency must match the source records.
- Do not invent missing invoice details.
- Do not generate actionId; the application supplies it.

AUTONOMOUS AUTHORITY
Use the autonomousLimitMinor value supplied in the request. Do not invent
a different limit. An otherwise valid invoice above that limit requires
request_approval.

OUTPUT
Return only a JSON array. Include exactly one object per input package:
[
  {
    "packageId": "exact package ID",
    "action": "one allowed action",
    "facts": {
      "vendorName": "source vendor",
      "invoiceNumber": "source invoice number",
      "amountMinor": 12345,
      "currency": "INR"
    },
    "evidenceRefs": ["[ref1]", "[ref2]", "[ref3]"],
    "rationale": "60-1500 characters citing at least two references"
  }
]
"""


async def _call_model(messages: list[dict[str, str]]) -> str:
    if not AI_API_KEY:
        raise RuntimeError(
            "AI provider credentials are not configured. "
            "Set AIPIPE_TOKEN (or AI_API_KEY) in the Render Environment."
        )

    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": AI_MODEL,
        "messages": messages,
        "temperature": 0.1,
    }
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    f"{AI_BASE_URL.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if response.status_code == 401:
                    raise RuntimeError(
                        "AI provider rejected the API key (HTTP 401). "
                        "Check AIPIPE_TOKEN on Render."
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
            if attempt < 2:
                import asyncio
                await asyncio.sleep(1 + attempt * 2)

    raise RuntimeError(f"AI provider request failed after retries: {last_error}") from last_error


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    # Strip markdown fences if the model wraps JSON
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
    # Also scan other string fields that sometimes hold evidence
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
    # Accept int-like floats from some models
    if isinstance(amount, float) and amount == int(amount):
        amount = int(amount)
    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
        raise ValueError(f"Invalid amountMinor for package {package_id}: {amount!r}")

    refs = decision.get("evidenceRefs")
    if not isinstance(refs, list) or len(refs) != 3:
        raise ValueError(f"Exactly three evidenceRefs required for {package_id}")
    if any(not isinstance(ref, str) for ref in refs) or len(set(refs)) != 3:
        raise ValueError(f"Evidence references must be three distinct strings for {package_id}")

    actual_refs = _grounded_refs(package)
    if actual_refs and any(ref not in actual_refs for ref in refs):
        raise ValueError(
            f"Ungrounded evidence reference for package {package_id}: "
            f"{refs} not all in {sorted(actual_refs)}"
        )

    rationale = decision.get("rationale")
    if not isinstance(rationale, str) or not 60 <= len(rationale.strip()) <= 1500:
        raise ValueError(f"Rationale must be 60-1500 characters for {package_id}")
    if action not in rationale:
        raise ValueError(f"Rationale must name the selected action for {package_id}")
    if sum(ref in rationale for ref in refs) < 2:
        raise ValueError(f"Rationale must cite at least two evidenceRefs for {package_id}")

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
        "rationale": rationale.strip(),
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
                validated = _validate_decision(package, cached)
                results[index] = validated
                continue
            except (ValueError, TypeError):
                pass
        pending.append((index, package))

    if pending:
        request = {
            "packages": [package for _, package in pending],
            "autonomousLimitMinor": AUTONOMOUS_LIMIT,
        }
        raw = await _call_model([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ])
        decisions = _extract_json_array(raw)

        ids = [package.get("packageId") or package.get("id") for _, package in pending]
        if len(decisions) != len(pending):
            raise ValueError(
                f"Model must return exactly one decision per package "
                f"(got {len(decisions)}, expected {len(pending)})"
            )

        by_id: dict[str, dict[str, Any]] = {}
        for decision in decisions:
            decision_id = decision.get("packageId")
            if not isinstance(decision_id, str) or decision_id in by_id:
                raise ValueError("Model output contains a missing or duplicate packageId")
            by_id[decision_id] = decision
        if set(by_id) != set(ids):
            raise ValueError(
                f"Model output package IDs do not match the requested batch: "
                f"got {sorted(by_id)}, expected {sorted(ids)}"
            )

        for (index, package), package_id in zip(pending, ids):
            validated = _validate_decision(package, by_id[package_id])
            cache_value = dict(validated)
            cache_value.pop("packageId", None)
            await storage.put_cached_decision(package_content_hash(package), cache_value)
            results[index] = validated

    if any(result is None for result in results):
        raise RuntimeError("Internal error: missing package decision")

    return [result for result in results if result is not None]
