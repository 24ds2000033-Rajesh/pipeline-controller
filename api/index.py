import copy
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

MAX_SAFE = 9007199254740991

NODES = [
    "verify_data",
    "prepare",
    "train",
    "evaluate",
    "register",
    "publish",
]

INPUTS = [
    "generation",
    "checksum",
    "canonicalData",
    "prepareCode",
    "prepareConfig",
    "trainCode",
    "trainConfig",
    "runtime",
    "evaluateCode",
    "evaluateConfig",
    "schemaDigest",
    "publishConfig",
]

EVENT_FIELDS = [
    "eventId",
    "revision",
    "node",
    "attempt",
    "status",
    "key",
    "artifactDigest",
    "receiptId",
]

STATUSES = {
    "started",
    "succeeded",
    "retryable_failed",
    "terminal_failed",
}

NODE_DEPS = {
    "verify_data": [
        "generation",
        "checksum",
    ],
    "prepare": [
        "canonicalData",
        "prepareCode",
        "prepareConfig",
        "prepareArtifact",
    ],
    "train": [
        "prepareArtifact",
        "trainCode",
        "trainConfig",
        "runtime",
    ],
    "evaluate": [
        "trainArtifact",
        "canonicalData",
        "evaluateCode",
        "evaluateConfig",
    ],
    "register": [
        "evaluateArtifact",
        "schemaDigest",
    ],
    "publish": [
        "registerArtifact",
        "publishConfig",
    ],
}

# ---------------------------------------------------------------------
# Persistent Redis configuration
# ---------------------------------------------------------------------

REDIS_URL = os.environ.get(
    "UPSTASH_REDIS_REST_URL",
    ""
).rstrip("/")

REDIS_TOKEN = os.environ.get(
    "UPSTASH_REDIS_REST_TOKEN",
    "")


# ---------------------------------------------------------------------
# JSON / hashing helpers
# ---------------------------------------------------------------------

def compact(obj: Any) -> str:
    """
    Canonical compact JSON.

    sort_keys=True makes equivalent JSON objects canonical while
    separators removes insignificant whitespace.
    """
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def digest_array(values: List[Any]) -> str:
    """
    Lowercase SHA-256 over UTF-8 compact JSON array.
    """
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------
# Redis persistence
# ---------------------------------------------------------------------

def redis_get(key: str) -> Optional[Dict[str, Any]]:
    if not REDIS_URL or not REDIS_TOKEN:
        raise RuntimeError("Redis is not configured")

    response = requests.get(
        f"{REDIS_URL}/get/{key}",
        headers={
            "Authorization": f"Bearer {REDIS_TOKEN}"
        },
        timeout=8,
    )

    response.raise_for_status()

    result = response.json().get("result")

    if result is None:
        return None

    return json.loads(result)


def redis_set(key: str, value: Dict[str, Any]) -> None:
    if not REDIS_URL or not REDIS_TOKEN:
        raise RuntimeError("Redis is not configured")

    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    response = requests.post(
        f"{REDIS_URL}/set/{key}",
        headers={
            "Authorization": f"Bearer {REDIS_TOKEN}",
            "Content-Type": "text/plain",
        },
        data=payload,
        timeout=8,
    )

    response.raise_for_status()


def state_key(session: str) -> str:
    """
    Session is hashed so arbitrary session strings cannot create
    problematic Redis keys.
    """
    return (
        "pipeline:v1:session:"
        + sha256_text(session)
    )


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

def new_state(session: str) -> Dict[str, Any]:
    return {
        "session": session,

        "revision": None,

        # Canonical complete inputs object.
        "inputCanonical": None,

        "inputs": None,

        # eventId -> canonical event JSON
        "events": {},

        # eventId -> original event object
        "eventObjects": {},

        # cacheKey -> immutable evidence
        #
        # {
        #   "artifactDigest": "...",
        #   "eventId": "...",
        #   "node": "..."
        # }
        #
        "cache": {},

        # Current attempt/terminal state.
        "nodes": {
            node: None
            for node in NODES
        },
    }


def load_state(session: str) -> Dict[str, Any]:
    state = redis_get(state_key(session))

    if state is None:
        return new_state(session)

    return state


def save_state(state: Dict[str, Any]) -> None:
    redis_set(
        state_key(state["session"]),
        state,
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def is_safe_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= MAX_SAFE
    )


def nonempty_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) > 0
    )


def validate_request(body: Any) -> bool:
    if not isinstance(body, dict):
        return False

    if not nonempty_string(body.get("session")):
        return False

    if not is_safe_int(body.get("revision")):
        return False

    if not isinstance(body.get("inputs"), dict):
        return False

    inputs = body["inputs"]

    for name in INPUTS:
        if not nonempty_string(inputs.get(name)):
            return False

    if not isinstance(body.get("events"), list):
        return False

    return True


def event_shape_valid(event: Any) -> bool:
    """
    Structural defects cause INVALID_EVENT.

    Semantic defects such as invalid attempt/status/artifact/receipt
    are ignored according to the specification.
    """
    if not isinstance(event, dict):
        return False

    if set(event.keys()) != set(EVENT_FIELDS):
        return False

    if not nonempty_string(event["eventId"]):
        return False

    if not is_safe_int(event["revision"]):
        return False

    if event["node"] not in NODES:
        return False

    if not nonempty_string(event["key"]):
        return False

    return True


def event_semantically_usable(event: Dict[str, Any]) -> bool:
    # Explicitly ignored by the specification.
    if not is_safe_int(event["attempt"]):
        return False

    if event["status"] not in STATUSES:
        return False

    if event["status"] == "succeeded":
        if not nonempty_string(
            event["artifactDigest"]
        ):
            return False

    else:
        if event["artifactDigest"] is not None:
            return False

    if (
        event["node"] in ("register", "publish")
        and event["status"] == "succeeded"
    ):
        expected = (
            f"receipt:{event['node']}:{event['key']}"
        )

        if event["receiptId"] != expected:
            return False

    elif event["receiptId"] is not None:
        return False

    return True


def canonical_event(event: Dict[str, Any]) -> str:
    return compact(event)


# ---------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------

def parent(node: str) -> Optional[str]:
    index = NODES.index(node)

    if index == 0:
        return None

    return NODES[index - 1]


def node_cache_key(
    node: str,
    inputs: Dict[str, str],
    parent_artifact: Optional[str],
) -> Optional[str]:

    if node == "verify_data":
        return digest_array([
            inputs["generation"],
            inputs["checksum"],
        ])

    if node == "prepare":
        if not parent_artifact:
            return None

        return digest_array([
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ])

    if node == "train":
        if not parent_artifact:
            return None

        return digest_array([
            parent_artifact,
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ])

    if node == "evaluate":
        if not parent_artifact:
            return None

        return digest_array([
            parent_artifact,
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ])

    if node == "register":
        if not parent_artifact:
            return None

        return digest_array([
            parent_artifact,
            inputs["schemaDigest"],
        ])

    if node == "publish":
        if not parent_artifact:
            return None

        return digest_array([
            parent_artifact,
            inputs["publishConfig"],
        ])

    return None


def current_artifact(
    state: Dict[str, Any],
    node: str,
    inputs: Dict[str, str],
) -> Optional[str]:

    p = parent(node)

    if p is None:
        return None

    parent_state = state["nodes"].get(p)

    if (
        parent_state
        and parent_state.get("status") == "succeeded"
    ):
        return parent_state.get(
            "artifactDigest"
        )

    # Parent may be reusable entirely through cache.
    parent_parent_artifact = current_artifact(
        state,
        p,
        inputs,
    )

    parent_key = node_cache_key(
        p,
        inputs,
        parent_parent_artifact,
    )

    if (
        parent_key
        and parent_key in state["cache"]
    ):
        return state["cache"][
            parent_key
        ]["artifactDigest"]

    return None


# ---------------------------------------------------------------------
# Dependency digest response
# ---------------------------------------------------------------------

def dependency_digests(
    node: str,
    inputs: Dict[str, str],
    cache_key: Optional[str],
    parent_artifact: Optional[str],
) -> Dict[str, str]:

    result = {}

    for dependency in NODE_DEPS[node]:

        if dependency.endswith("Artifact"):
            if parent_artifact is not None:
                result[dependency] = (
                    parent_artifact
                )

        else:
            result[dependency] = sha256_text(
                inputs[dependency]
            )

    if cache_key is not None:
        result["cacheKey"] = cache_key

    return result


# ---------------------------------------------------------------------
# Event transition engine
# ---------------------------------------------------------------------

def transition(
    state: Dict[str, Any],
    event: Dict[str, Any],
    accepted: List[str],
    ignored: List[str],
) -> None:

    event_id = event["eventId"]
    event_json = canonical_event(event)

    # ---------------------------------------------------------------
    # Global event-id idempotency
    # ---------------------------------------------------------------

    if event_id in state["events"]:

        if (
            state["events"][event_id]
            == event_json
        ):
            ignored.append(event_id)
            return

        raise ValueError(
            "EVENT_ID_CONFLICT"
        )

    # ---------------------------------------------------------------
    # Revision filtering
    # ---------------------------------------------------------------

    if event["revision"] != state["revision"]:
        ignored.append(event_id)
        return

    node = event["node"]
    inputs = state["inputs"]

    # ---------------------------------------------------------------
    # Parent availability
    # ---------------------------------------------------------------

    parent_artifact = current_artifact(
        state,
        node,
        inputs,
    )

    key = node_cache_key(
        node,
        inputs,
        parent_artifact,
    )

    # Wrong key or unavailable parent.
    if (
        key is None
        or event["key"] != key
    ):
        ignored.append(event_id)
        return

    # ---------------------------------------------------------------
    # Immutable cache evidence
    # ---------------------------------------------------------------

    cached = state["cache"].get(key)

    if cached:

        if event["status"] == "succeeded":

            if (
                event["artifactDigest"]
                != cached["artifactDigest"]
            ):
                raise ValueError(
                    "EVIDENCE_CONFLICT"
                )

        raise ValueError(
            "STATUS_CONFLICT"
        )

    node_state = state["nodes"].get(node)

    # Stale state for a different current key.
    if (
        node_state
        and node_state.get("key") != key
    ):
        ignored.append(event_id)
        return

    # ---------------------------------------------------------------
    # No current state
    # ---------------------------------------------------------------

    if node_state is None:

        # Only started(1) can initiate a node.
        if (
            event["status"] == "started"
            and event["attempt"] == 1
        ):
            state["nodes"][node] = {
                "key": key,
                "status": "started",
                "attempt": 1,
                "startEventId": event_id,
                "successEventId": None,
                "artifactDigest": None,
            }

            state["events"][event_id] = (
                event_json
            )

            state["eventObjects"][
                event_id
            ] = copy.deepcopy(event)

            accepted.append(event_id)
            return

        # Completion or attempt > 1 without start.
        ignored.append(event_id)
        return

    status = node_state["status"]
    attempt = node_state["attempt"]

    # ---------------------------------------------------------------
    # started(n)
    # ---------------------------------------------------------------

    if status == "started":

        if event["attempt"] < attempt:
            ignored.append(event_id)
            return

        if event["attempt"] != attempt:
            raise ValueError(
                "STATUS_CONFLICT"
            )

        if event["status"] not in {
            "succeeded",
            "retryable_failed",
            "terminal_failed",
        }:
            raise ValueError(
                "STATUS_CONFLICT"
            )

        node_state["status"] = (
            event["status"]
        )

        if event["status"] == "succeeded":

            node_state[
                "artifactDigest"
            ] = event["artifactDigest"]

            node_state[
                "successEventId"
            ] = event_id

            # Permanently bind key -> first evidence.
            state["cache"][key] = {
                "artifactDigest":
                    event["artifactDigest"],
                "eventId": event_id,
                "node": node,
            }

        state["events"][event_id] = (
            event_json
        )

        state["eventObjects"][
            event_id
        ] = copy.deepcopy(event)

        accepted.append(event_id)
        return

    # ---------------------------------------------------------------
    # retryable_failed(n)
    # ---------------------------------------------------------------

    if status == "retryable_failed":

        if (
            event["status"] == "started"
            and event["attempt"]
            == attempt + 1
        ):
            node_state["status"] = "started"
            node_state["attempt"] = (
                event["attempt"]
            )
            node_state["startEventId"] = (
                event_id
            )

            state["events"][event_id] = (
                event_json
            )

            state["eventObjects"][
                event_id
            ] = copy.deepcopy(event)

            accepted.append(event_id)
            return

        if event["attempt"] < attempt:
            ignored.append(event_id)
            return

        raise ValueError(
            "STATUS_CONFLICT"
        )

    # ---------------------------------------------------------------
    # terminal_failed
    # ---------------------------------------------------------------

    if status == "terminal_failed":
        raise ValueError(
            "STATUS_CONFLICT"
        )

    # ---------------------------------------------------------------
    # succeeded
    # ---------------------------------------------------------------

    if status == "succeeded":

        if event["status"] == "succeeded":

            if (
                event["artifactDigest"]
                != node_state[
                    "artifactDigest"
                ]
            ):
                raise ValueError(
                    "EVIDENCE_CONFLICT"
                )

        raise ValueError(
            "STATUS_CONFLICT"
        )

    raise ValueError(
        "STATUS_CONFLICT"
    )


# ---------------------------------------------------------------------
# Response construction
# ---------------------------------------------------------------------

def response_nodes(
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:

    inputs = state["inputs"]

    result = []

    upstream_terminal = False
    upstream_pending = False

    for node in NODES:

        parent_node = parent(node)

        parent_state = (
            state["nodes"].get(
                parent_node
            )
            if parent_node
            else None
        )

        parent_artifact = current_artifact(
            state,
            node,
            inputs,
        )

        key = node_cache_key(
            node,
            inputs,
            parent_artifact,
        )

        node_state = state["nodes"].get(
            node
        )

        dependencies = dependency_digests(
            node,
            inputs,
            key,
            parent_artifact,
        )

        triggering = []

        # -----------------------------------------------------------
        # Upstream terminal
        # -----------------------------------------------------------

        if upstream_terminal:

            action = "block"
            reason = "UPSTREAM_TERMINAL"

        # -----------------------------------------------------------
        # Upstream pending/running
        # -----------------------------------------------------------

        elif upstream_pending:

            action = "block"
            reason = "UPSTREAM_PENDING"

        else:

            cache = (
                state["cache"].get(key)
                if key
                else None
            )

            # -------------------------------------------------------
            # Cache hit
            # -------------------------------------------------------

            if cache:

                action = "reuse"
                reason = "CACHE_HIT"

                triggering = [
                    cache["eventId"]
                ]

            # -------------------------------------------------------
            # Terminal failure
            # -------------------------------------------------------

            elif (
                node_state
                and node_state.get(
                    "status"
                ) == "terminal_failed"
            ):

                action = "block"
                reason = (
                    "TERMINAL_FAILURE"
                )

            # -------------------------------------------------------
            # Currently running
            # -------------------------------------------------------

            elif (
                node_state
                and node_state.get(
                    "status"
                ) == "started"
            ):

                action = "block"
                reason = "RUNNING"

                triggering = [
                    node_state[
                        "startEventId"
                    ]
                ]

            # -------------------------------------------------------
            # Retryable failure
            # -------------------------------------------------------

            elif (
                node_state
                and node_state.get(
                    "status"
                ) == "retryable_failed"
            ):

                action = "rerun"
                reason = (
                    "RETRYABLE_FAILURE"
                )

            # -------------------------------------------------------
            # No cache
            # -------------------------------------------------------

            else:

                if (
                    parent_node
                    and parent_artifact is None
                ):

                    action = "block"
                    reason = (
                        "UPSTREAM_PENDING"
                    )

                else:

                    action = "rerun"
                    reason = "CACHE_MISS"

        result.append({
            "node": node,
            "action": action,
            "reasonCodes": [reason],
            "dependencyDigests":
                dependencies,
            "triggeringEventIds":
                triggering,
        })

        # Descendants of terminal failure.
        if (
            action == "block"
            and reason == "TERMINAL_FAILURE"
        ):
            upstream_terminal = True
            upstream_pending = False

        # Descendants of running/pending node.
        elif (
            action == "block"
            and reason in {
                "RUNNING",
                "UPSTREAM_PENDING",
            }
        ):
            upstream_pending = True

        # A reusable node makes descendants eligible.
        if action == "reuse":
            upstream_terminal = False
            upstream_pending = False

    return result


# ---------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------

def error(
    code: str,
    status: int = 409,
):
    return JSONResponse(
        status_code=status,
        content={
            "error": code
        },
    )


# ---------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------

@app.post("/pipeline")
async def pipeline(request: Request):

    # ---------------------------------------------------------------
    # Parse request
    # ---------------------------------------------------------------

    try:
        body = await request.json()

    except Exception:
        return error(
            "INVALID_REQUEST",
            400,
        )

    if not validate_request(body):
        return error(
            "INVALID_REQUEST",
            400,
        )

    session = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    # ---------------------------------------------------------------
    # Load persistent session state
    # ---------------------------------------------------------------

    try:
        state = load_state(session)

    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                "error":
                    "STATE_STORE_UNAVAILABLE"
            },
        )

    input_canonical = compact(inputs)

    # ---------------------------------------------------------------
    # First revision
    # ---------------------------------------------------------------

    if state["revision"] is None:

        state["revision"] = revision

        state[
            "inputCanonical"
        ] = input_canonical

        state["inputs"] = copy.deepcopy(
            inputs
        )

    # ---------------------------------------------------------------
    # Older revision
    # ---------------------------------------------------------------

    elif revision < state["revision"]:

        # Do not mutate state.
        pass

    # ---------------------------------------------------------------
    # Same revision
    # ---------------------------------------------------------------

    elif revision == state["revision"]:

        if (
            state["inputCanonical"]
            != input_canonical
        ):
            return error(
                "REVISION_CONFLICT"
            )

    # ---------------------------------------------------------------
    # New revision
    # ---------------------------------------------------------------

    else:

        state["revision"] = revision

        state[
            "inputCanonical"
        ] = input_canonical

        state["inputs"] = copy.deepcopy(
            inputs
        )

        # Attempt and terminal state are cleared.
        # Immutable successful cache remains.
        state["nodes"] = {
            node: None
            for node in NODES
        }

    # ---------------------------------------------------------------
    # Stale request: ignore its events.
    # ---------------------------------------------------------------

    if revision != state["revision"]:

        accepted = []

        ignored = []

        for event in events:

            if (
                isinstance(event, dict)
                and nonempty_string(
                    event.get("eventId")
                )
            ):
                ignored.append(
                    event["eventId"]
                )

    else:

        # -----------------------------------------------------------
        # Atomic copy.
        #
        # Nothing is committed until the complete event batch
        # succeeds.
        # -----------------------------------------------------------

        working = copy.deepcopy(
            state
        )

        accepted = []
        ignored = []

        try:

            for event in events:

                # Structural error => 409 INVALID_EVENT.
                if not event_shape_valid(
                    event
                ):
                    raise ValueError(
                        "INVALID_EVENT"
                    )

                # Invalid semantic event is ignored.
                if not event_semantically_usable(
                    event
                ):
                    ignored.append(
                        event["eventId"]
                    )
                    continue

                transition(
                    working,
                    event,
                    accepted,
                    ignored,
                )

        except ValueError as exc:

            # Nothing has been saved.
            # Therefore the entire batch rolls back.
            return error(
                str(exc)
            )

        state = working

        # -----------------------------------------------------------
        # Single state write after complete successful batch.
        # -----------------------------------------------------------

        try:
            save_state(state)

        except Exception:
            return JSONResponse(
                status_code=500,
                content={
                    "error":
                        "STATE_STORE_UNAVAILABLE"
                },
            )

    # ---------------------------------------------------------------
    # Response
    # ---------------------------------------------------------------

    return JSONResponse(
        content={
            "revision":
                state["revision"],

            "acceptedEventIds":
                accepted,

            "ignoredEventIds":
                ignored,

            "nodes":
                response_nodes(state),
        }
    )


# ---------------------------------------------------------------------
# Health/root endpoint
# ---------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "pipeline",
        "ok": True,
    }
