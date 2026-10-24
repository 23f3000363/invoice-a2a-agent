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

# NVIDIA is the primary provider. Existing Render variables override these defaults,
# so update OPENAI_BASE_URL and AI_MODEL in Render when switching providers.
AI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    os.getenv("AI_BASE_URL", "https://integrate.api.nvidia.com/v1"),
).strip().rstrip("/")
AI_API_KEY = (
    os.getenv("NVIDIA_API_KEY")
    or os.getenv("AIPIPE_TOKEN")
    or os.getenv("AI_API_KEY", "")
).strip()
AI_MODEL = os.getenv(
    "AI_MODEL",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
).strip()
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
