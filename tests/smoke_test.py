"""Offline protocol smoke test. Does not call the AI provider or grader."""
import os
import tempfile
import uuid
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["DATABASE_PATH"] = str(
    Path(tempfile.gettempdir()) / f"invoice_a2a_smoke_test_{uuid.uuid4().hex}.db"
)
os.environ["PUBLIC_BASE_URL"] = "http://127.0.0.1:8000/a2a"

from fastapi.testclient import TestClient
from app import main


async def fake_decide(packages):
    return [
        {
            "packageId": p.get("packageId"),
            "action": "hold_invoice",
            "facts": {
                "vendorName": "Acme",
                "invoiceNumber": "INV-1",
                "amountMinor": 12345,
                "currency": "INR",
            },
            "evidenceRefs": ["[ref-1]", "[ref-2]", "[ref-3]"],
            "rationale": (
                "hold_invoice because verification is incomplete and payment "
                "must remain paused until invoice details are confirmed. "
                "[ref-1] [ref-2]"
            ),
        }
        for p in packages
    ]


main.decide_batch = fake_decide

with TestClient(main.app) as client:
    routes = {r.path for r in main.app.routes}
    required = {
        "/.well-known/agent-card.json",
        "/a2a/message:send",
        "/a2a/tasks",
        "/a2a/tasks/{task_id}",
        "/a2a/tasks/{task_id}:cancel",
    }

    assert required <= routes, f"Missing routes: {required - routes}"
    assert client.get("/.well-known/agent-card.json").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}

    assert client.get(
        "/a2a/tasks", headers={"A2A-Version": "1.0"}
    ).status_code == 401

    h = {
        "Authorization": "Bearer smoke-user",
        "A2A-Version": "1.0",
        "Content-Type": "application/a2a+json",
    }

    # Initial invoice batch.
    message = {
        "messageId": "smoke-msg-1",
        "role": "ROLE_USER",
        "parts": [
            {
                "mediaType": "application/vnd.ga5.invoice-claim-batch+json",
                "data": {
                    "batchId": "smoke-batch",
                    "policyRevision": "p1",
                    "packages": [
                        {"packageId": "pkg-1", "documents": []}
                    ],
                },
            }
        ],
    }

    response = client.post(
        "/a2a/message:send", headers=h, json={"message": message}
    )
    assert response.status_code == 200, response.text
    task = response.json()["task"]
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"

    # Initial-message replay.
    replay = client.post(
        "/a2a/message:send",
        headers=h,
        json={
            "message": message,
            "configuration": {"returnImmediately": True},
        },
    )
    assert replay.status_code == 200, replay.text
    assert replay.json()["task"]["id"] == task["id"]

    # Principal isolation.
    other_headers = {
        "Authorization": "Bearer other-user",
        "A2A-Version": "1.0",
    }
    assert client.get(
        "/a2a/tasks/" + task["id"], headers=other_headers
    ).status_code == 404

    proposal = task["artifacts"][0]["parts"][0]["data"]["proposals"][0]

    continuation = {
        "messageId": "smoke-msg-2",
        "taskId": task["id"],
        "contextId": task["contextId"],
        "role": "ROLE_USER",
        "parts": [
            {
                "mediaType": "application/vnd.ga5.invoice-action-results+json",
                "data": {
                    "batchId": "smoke-batch",
                    "results": [
                        {
                            "packageId": proposal["packageId"],
                            "actionId": proposal["actionId"],
                            "action": proposal["action"],
                            "outcome": "ACCEPTED",
                            "receiptNonce": "test-nonce",
                        }
                    ],
                },
            }
        ],
    }

    # Complete the task normally.
    result = client.post(
        "/a2a/message:send",
        headers=h,
        json={"message": continuation},
    )
    assert result.status_code == 200, result.text

    finished = result.json()["task"]
    assert finished["status"]["state"] == "TASK_STATE_COMPLETED"
    assert len(finished["artifacts"]) == 2

    executions = finished["artifacts"][-1]["parts"][0]["data"]["executions"]
    assert len(executions) == 1
    assert executions[0]["receiptNonce"] == "test-nonce"

    # Identical continuation retry returns the original task.
    retry = client.post(
        "/a2a/message:send",
        headers=h,
        json={"message": continuation},
    )
    assert retry.status_code == 200, retry.text
    assert retry.json()["task"]["id"] == finished["id"]
    assert len(retry.json()["task"]["artifacts"]) == 2

    # Changed content with the same message ID conflicts.
    original_result = continuation["parts"][0]["data"]["results"][0]
    conflicting = {
        **continuation,
        "parts": [
            {
                **continuation["parts"][0],
                "data": {
                    **continuation["parts"][0]["data"],
                    "results": [
                        {
                            **original_result,
                            "receiptNonce": "different-nonce",
                        }
                    ],
                },
            }
        ],
    }
    conflict = client.post(
        "/a2a/message:send", headers=h, json={"message": conflicting}
    )
    assert conflict.status_code == 409, conflict.text

    # Protocol version validation.
    assert client.get(
        "/a2a/tasks",
        headers={**h, "A2A-Version": "2.0"},
    ).status_code == 400

    # Create a second task specifically for the cancellation race.
    race_message = {
        **message,
        "messageId": "smoke-race-msg-1",
        "parts": [
            {
                **message["parts"][0],
                "data": {
                    **message["parts"][0]["data"],
                    "batchId": "smoke-race-batch",
                },
            }
        ],
    }
    race_response = client.post(
        "/a2a/message:send",
        headers=h,
        json={"message": race_message},
    )
    assert race_response.status_code == 200, race_response.text
    race_task = race_response.json()["task"]
    race_proposal = race_task["artifacts"][0]["parts"][0]["data"]["proposals"][0]

    race_continuation = {
        "messageId": "smoke-race-msg-2",
        "taskId": race_task["id"],
        "contextId": race_task["contextId"],
        "role": "ROLE_USER",
        "parts": [
            {
                "mediaType": "application/vnd.ga5.invoice-action-results+json",
                "data": {
                    "batchId": "smoke-race-batch",
                    "results": [
                        {
                            "packageId": race_proposal["packageId"],
                            "actionId": race_proposal["actionId"],
                            "action": race_proposal["action"],
                            "outcome": "ACCEPTED",
                            "receiptNonce": "race-nonce",
                        }
                    ],
                },
            }
        ],
    }

    # Release both requests together to exercise the competing state changes.
    barrier = threading.Barrier(2)

    def continue_request():
        with TestClient(main.app) as race_client:
            barrier.wait()
            return race_client.post(
                "/a2a/message:send",
                headers=h,
                json={"message": race_continuation},
            )

    def cancel_request():
        with TestClient(main.app) as race_client:
            barrier.wait()
            return race_client.post(
                f"/a2a/tasks/{race_task['id']}:cancel",
                headers=h,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        continuation_future = pool.submit(continue_request)
        cancellation_future = pool.submit(cancel_request)
        continuation_response = continuation_future.result()
        cancellation_response = cancellation_future.result()

    # Exactly one transition must win.
    assert (
        (continuation_response.status_code == 200
         and cancellation_response.status_code == 409)
        or
        (continuation_response.status_code == 409
         and cancellation_response.status_code == 200)
    ), (
        f"Continuation={continuation_response.status_code}, "
        f"Cancellation={cancellation_response.status_code}"
    )

    final_response = client.get(
        f"/a2a/tasks/{race_task['id']}", headers=h
    )
    assert final_response.status_code == 200
    final_task = final_response.json()

    if final_task["status"]["state"] == "TASK_STATE_COMPLETED":
        assert len(final_task["artifacts"]) == 2
        receipt = final_task["artifacts"][-1]["parts"][0]["data"]
        assert len(receipt["executions"]) == 1
    else:
        assert final_task["status"]["state"] == "TASK_STATE_CANCELED"
        assert len(final_task["artifacts"]) == 1

print(
    "PASS: routes, auth, version, idempotency, isolation, proposal/receipt, "
    "continuation conflict, cancellation/continuation race"
)
