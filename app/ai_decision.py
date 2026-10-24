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
