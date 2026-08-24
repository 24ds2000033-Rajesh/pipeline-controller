import copy
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()


# ============================================================
# CONSTANTS
# ============================================================

MAX_SAFE_INTEGER = 9007199254740991

NODES = [
    "verify_data",
    "prepare",
    "train",
    "evaluate",
    "register",
    "publish",
]

INPUT_NAMES = [
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


# ============================================================
# REDIS
# ============================================================

REDIS_URL = os.getenv("REDIS_URL", "").strip()

if not REDIS_URL:
    raise RuntimeError(
        "REDIS_URL environment variable is missing"
    )

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=10,
    socket_timeout=10,
    health_check_interval=30,
)


# ============================================================
# JSON / HASH HELPERS
# ============================================================

def compact_json(value: Any) -> str:
    """
    Compact canonical JSON.

    sort_keys=True ensures that object key ordering does not
    affect revision/input identity.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def compact_array(values: List[Any]) -> str:
    """
    Compact JSON array preserving the supplied array order.
    """
    return json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_utf8(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def sha256_array(values: List[Any]) -> str:
    """
    Lowercase SHA-256 over UTF-8 compact JSON array.
    """
    return sha256_utf8(
        compact_array(values)
    )


# ============================================================
# PERSISTENT SESSION STATE
# ============================================================

def session_storage_key(session: str) -> str:
    return (
        "pipeline:v1:session:"
        + sha256_utf8(session)
    )


def create_state(session: str) -> Dict[str, Any]:
    return {
        "session": session,

        "revision": None,

        # Canonical representation of ALL input metadata.
        "inputCanonical": None,

        # Complete input object.
        "inputs": None,

        # eventId -> canonical compact event JSON
        "events": {},

        # eventId -> event object
        "eventObjects": {},

        # cacheKey -> immutable successful evidence
        #
        # {
        #   "artifactDigest": "...",
        #   "eventId": "...",
        #   "node": "..."
        # }
        #
        "cache": {},

        # Current state of every node.
        "nodes": {
            node: None
            for node in NODES
        },
    }


def load_state(session: str) -> Dict[str, Any]:
    key = session_storage_key(session)

    value = redis_client.get(key)

    if value is None:
        return create_state(session)

    return json.loads(value)


def save_state(state: Dict[str, Any]) -> None:
    key = session_storage_key(
        state["session"]
    )

    redis_client.set(
        key,
        json.dumps(
            state,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


# ============================================================
# VALIDATION
# ============================================================

def safe_positive_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
        and value <= MAX_SAFE_INTEGER
    )


def nonempty_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) > 0
    )


def validate_request(body: Any) -> bool:

    if not isinstance(body, dict):
        return False

    if not nonempty_string(
        body.get("session")
    ):
        return False

    if not safe_positive_integer(
        body.get("revision")
    ):
        return False

    if not isinstance(
        body.get("inputs"),
        dict,
    ):
        return False

    inputs = body["inputs"]

    # All 12 required inputs must be
    # non-empty strings.
    for name in INPUT_NAMES:
        if not nonempty_string(
            inputs.get(name)
        ):
            return False

    if not isinstance(
        body.get("events"),
        list,
    ):
        return False

    return True


def validate_event_shape(
    event: Any,
) -> bool:
    """
    Structural validation.

    Structural failures are INVALID_EVENT.
    """

    if not isinstance(event, dict):
        return False

    # Exactly eight fields.
    if set(event.keys()) != set(
        EVENT_FIELDS
    ):
        return False

    if not nonempty_string(
        event.get("eventId")
    ):
        return False

    if not safe_positive_integer(
        event.get("revision")
    ):
        return False

    if event.get("node") not in NODES:
        return False

    if not safe_positive_integer(
        event.get("attempt")
    ):
        return False

    if not nonempty_string(
        event.get("key")
    ):
        return False

    return True


def event_semantically_valid(
    event: Dict[str, Any],
) -> bool:
    """
    Semantic-invalid events are ignored rather than
    causing a batch conflict.
    """

    status = event["status"]

    if status not in STATUSES:
        return False

    attempt = event["attempt"]

    if not safe_positive_integer(
        attempt
    ):
        return False

    artifact = event[
        "artifactDigest"
    ]

    receipt = event[
        "receiptId"
    ]

    # Success requires artifact digest.
    if status == "succeeded":

        if not nonempty_string(
            artifact
        ):
            return False

    # Every non-success status requires
    # artifactDigest == null.
    else:

        if artifact is not None:
            return False

    node = event["node"]

    # Register/publish successful events
    # require the exact receipt.
    if (
        status == "succeeded"
        and node in {
            "register",
            "publish",
        }
    ):

        expected = (
            f"receipt:{node}:{event['key']}"
        )

        if receipt != expected:
            return False

    # All other events require null receipt.
    else:

        if receipt is not None:
            return False

    return True


# ============================================================
# DAG HELPERS
# ============================================================

def parent_node(
    node: str,
) -> Optional[str]:

    index = NODES.index(node)

    if index == 0:
        return None

    return NODES[index - 1]


def node_has_current_success(
    state: Dict[str, Any],
    node: str,
    key: str,
) -> bool:

    cache = state["cache"].get(key)

    if cache is not None:
        return (
            cache["node"] == node
        )

    node_state = state[
        "nodes"
    ].get(node)

    if node_state is None:
        return False

    return (
        node_state.get("key") == key
        and node_state.get("status")
        == "succeeded"
    )


def reusable_artifact_for_node(
    state: Dict[str, Any],
    node: str,
    inputs: Dict[str, str],
) -> Optional[str]:
    """
    Return the parent's reusable artifact.

    A parent is reusable if:
      * its current state succeeded, or
      * its content-addressed cache contains
        immutable successful evidence.
    """

    p = parent_node(node)

    if p is None:
        return None

    # First resolve the parent's own parent.
    parent_parent_artifact = (
        reusable_artifact_for_node(
            state,
            p,
            inputs,
        )
    )

    parent_key = calculate_cache_key(
        p,
        inputs,
        parent_parent_artifact,
    )

    if parent_key is None:
        return None

    cache = state[
        "cache"
    ].get(parent_key)

    if cache is not None:
        return cache[
            "artifactDigest"
        ]

    current = state[
        "nodes"
    ].get(p)

    if (
        current is not None
        and current.get("key")
        == parent_key
        and current.get("status")
        == "succeeded"
    ):
        return current[
            "artifactDigest"
        ]

    return None


def calculate_cache_key(
    node: str,
    inputs: Dict[str, str],
    parent_artifact: Optional[str],
) -> Optional[str]:
    """
    IMPORTANT:
    The arrays exactly follow the specification.

    verify_data:
      [generation, checksum]

    prepare:
      [canonicalData, prepareCode, prepareConfig]

    train:
      [prepareArtifact, trainCode, trainConfig, runtime]

    evaluate:
      [trainArtifact, canonicalData, evaluateCode, evaluateConfig]

    register:
      [evaluateArtifact, schemaDigest]

    publish:
      [registerArtifact, publishConfig]
    """

    if node == "verify_data":

        return sha256_array([
            inputs["generation"],
            inputs["checksum"],
        ])

    if node == "prepare":

        # Key itself does not contain the parent artifact,
        # but it is unavailable until parent is reusable.
        if parent_artifact is None:
            return None

        return sha256_array([
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ])

    if node == "train":

        if parent_artifact is None:
            return None

        return sha256_array([
            parent_artifact,
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ])

    if node == "evaluate":

        if parent_artifact is None:
            return None

        return sha256_array([
            parent_artifact,
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ])

    if node == "register":

        if parent_artifact is None:
            return None

        return sha256_array([
            parent_artifact,
            inputs["schemaDigest"],
        ])

    if node == "publish":

        if parent_artifact is None:
            return None

        return sha256_array([
            parent_artifact,
            inputs["publishConfig"],
        ])

    return None


# ============================================================
# DEPENDENCY DIGESTS
# ============================================================

def build_dependency_digests(
    node: str,
    inputs: Dict[str, str],
    cache_key: Optional[str],
    parent_artifact: Optional[str],
) -> Dict[str, str]:

    result: Dict[str, str] = {}

    if node == "verify_data":

        result["generation"] = sha256_utf8(
            inputs["generation"]
        )

        result["checksum"] = sha256_utf8(
            inputs["checksum"]
        )

    elif node == "prepare":

        result["canonicalData"] = sha256_utf8(
            inputs["canonicalData"]
        )

        result["prepareCode"] = sha256_utf8(
            inputs["prepareCode"]
        )

        result["prepareConfig"] = sha256_utf8(
            inputs["prepareConfig"]
        )

        if parent_artifact is not None:
            result[
                "verifyDataArtifact"
            ] = parent_artifact

    elif node == "train":

        if parent_artifact is not None:
            result[
                "prepareArtifact"
            ] = parent_artifact

        result["trainCode"] = sha256_utf8(
            inputs["trainCode"]
        )

        result["trainConfig"] = sha256_utf8(
            inputs["trainConfig"]
        )

        result["runtime"] = sha256_utf8(
            inputs["runtime"]
        )

    elif node == "evaluate":

        if parent_artifact is not None:
            result[
                "trainArtifact"
            ] = parent_artifact

        result[
            "canonicalData"
        ] = sha256_utf8(
            inputs["canonicalData"]
        )

        result[
            "evaluateCode"
        ] = sha256_utf8(
            inputs["evaluateCode"]
        )

        result[
            "evaluateConfig"
        ] = sha256_utf8(
            inputs["evaluateConfig"]
        )

    elif node == "register":

        if parent_artifact is not None:
            result[
                "evaluateArtifact"
            ] = parent_artifact

        result[
            "schemaDigest"
        ] = sha256_utf8(
            inputs["schemaDigest"]
        )

    elif node == "publish":

        if parent_artifact is not None:
            result[
                "registerArtifact"
            ] = parent_artifact

        result[
            "publishConfig"
        ] = sha256_utf8(
            inputs["publishConfig"]
        )

    if cache_key is not None:
        result["cacheKey"] = cache_key

    return result


# ============================================================
# EVENT TRANSITION PROCESSING
# ============================================================

def process_event(
    state: Dict[str, Any],
    event: Dict[str, Any],
    accepted: List[str],
    ignored: List[str],
) -> None:

    event_id = event["eventId"]

    event_json = compact_json(event)

    # --------------------------------------------------------
    # Global event ID
    # --------------------------------------------------------

    if event_id in state["events"]:

        previous = state[
            "events"
        ][event_id]

        if previous == event_json:
            ignored.append(event_id)
            return

        raise ValueError(
            "EVENT_ID_CONFLICT"
        )

    # --------------------------------------------------------
    # Wrong revision
    # --------------------------------------------------------

    if event["revision"] != state[
        "revision"
    ]:
        ignored.append(event_id)
        return

    node = event["node"]

    inputs = state["inputs"]

    # --------------------------------------------------------
    # Parent must be reusable
    # --------------------------------------------------------

    parent_artifact = (
        reusable_artifact_for_node(
            state,
            node,
            inputs,
        )
    )

    key = calculate_cache_key(
        node,
        inputs,
        parent_artifact,
    )

    # Parent unavailable or wrong key.
    if (
        key is None
        or event["key"] != key
    ):
        ignored.append(event_id)
        return

    # --------------------------------------------------------
    # Immutable cache
    # --------------------------------------------------------

    cache_entry = state[
        "cache"
    ].get(key)

    if cache_entry is not None:

        if (
            event["status"]
            == "succeeded"
        ):

            if (
                event["artifactDigest"]
                != cache_entry[
                    "artifactDigest"
                ]
            ):
                raise ValueError(
                    "EVIDENCE_CONFLICT"
                )

        # Successful/current cache means
        # no new state transition is allowed.
        raise ValueError(
            "STATUS_CONFLICT"
        )

    # --------------------------------------------------------
    # Current node state
    # --------------------------------------------------------

    node_state = state[
        "nodes"
    ].get(node)

    # --------------------------------------------------------
    # No state yet
    # --------------------------------------------------------

    if node_state is None:

        # Only started(1) is accepted.
        if (
            event["status"]
            == "started"
            and event["attempt"] == 1
        ):

            state["nodes"][node] = {
                "key": key,
                "status": "started",
                "attempt": 1,
                "startEventId": event_id,
                "successEventId": None,
                "terminalEventId": None,
                "artifactDigest": None,
            }

            state["events"][
                event_id
            ] = event_json

            state[
                "eventObjects"
            ][event_id] = copy.deepcopy(
                event
            )

            accepted.append(event_id)
            return

        # Completion without start,
        # or attempt > 1 without retry state,
        # is ignored.
        ignored.append(event_id)
        return

    # Wrong current key means stale event.
    if node_state.get("key") != key:
        ignored.append(event_id)
        return

    current_status = node_state[
        "status"
    ]

    current_attempt = node_state[
        "attempt"
    ]

    # --------------------------------------------------------
    # Lower attempt is ignored
    # --------------------------------------------------------

    if event["attempt"] < current_attempt:
        ignored.append(event_id)
        return

    # --------------------------------------------------------
    # STARTED(n)
    # --------------------------------------------------------

    if current_status == "started":

        # Completion for exactly same attempt.
        if (
            event["attempt"]
            == current_attempt
            and event["status"]
            in {
                "succeeded",
                "retryable_failed",
                "terminal_failed",
            }
        ):

            node_state[
                "status"
            ] = event["status"]

            if (
                event["status"]
                == "succeeded"
            ):

                artifact = event[
                    "artifactDigest"
                ]

                node_state[
                    "artifactDigest"
                ] = artifact

                node_state[
                    "successEventId"
                ] = event_id

                # Immutable first evidence.
                state[
                    "cache"
                ][key] = {
                    "artifactDigest":
                        artifact,
                    "eventId":
                        event_id,
                    "node":
                        node,
                }

            elif (
                event["status"]
                == "terminal_failed"
            ):

                node_state[
                    "terminalEventId"
                ] = event_id

            state["events"][
                event_id
            ] = event_json

            state[
                "eventObjects"
            ][event_id] = copy.deepcopy(
                event
            )

            accepted.append(event_id)
            return

        # Same attempt but another transition.
        if (
            event["attempt"]
            == current_attempt
        ):
            raise ValueError(
                "STATUS_CONFLICT"
            )

        # Higher attempt from started state
        # is not a valid direct transition.
        raise ValueError(
            "STATUS_CONFLICT"
        )

    # --------------------------------------------------------
    # RETRYABLE_FAILED(n)
    # --------------------------------------------------------

    if (
        current_status
        == "retryable_failed"
    ):

        # Lower already handled above.

        if (
            event["status"]
            == "started"
            and event["attempt"]
            == current_attempt + 1
        ):

            node_state[
                "status"
            ] = "started"

            node_state[
                "attempt"
            ] = event["attempt"]

            node_state[
                "startEventId"
            ] = event_id

            state["events"][
                event_id
            ] = event_json

            state[
                "eventObjects"
            ][event_id] = copy.deepcopy(
                event
            )

            accepted.append(event_id)
            return

        raise ValueError(
            "STATUS_CONFLICT"
        )

    # --------------------------------------------------------
    # TERMINAL_FAILED
    # --------------------------------------------------------

    if (
        current_status
        == "terminal_failed"
    ):
        raise ValueError(
            "STATUS_CONFLICT"
        )

    # --------------------------------------------------------
    # SUCCEEDED
    # --------------------------------------------------------

    if current_status == "succeeded":

        if (
            event["status"]
            == "succeeded"
            and event["attempt"]
            == current_attempt
        ):

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


# ============================================================
# RESPONSE STATE
# ============================================================

def build_response_nodes(
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:

    inputs = state["inputs"]

    result = []

    upstream_terminal = False
    upstream_pending = False

    for node in NODES:

        parent = parent_node(node)

        parent_artifact = (
            reusable_artifact_for_node(
                state,
                node,
                inputs,
            )
        )

        key = calculate_cache_key(
            node,
            inputs,
            parent_artifact,
        )

        dependencies = (
            build_dependency_digests(
                node,
                inputs,
                key,
                parent_artifact,
            )
        )

        current = state[
            "nodes"
        ].get(node)

        cache_entry = (
            state["cache"].get(key)
            if key is not None
            else None
        )

        action = None
        reason = None
        triggering = []

        # --------------------------------------------------------
        # Descendant of terminal node
        # --------------------------------------------------------

        if upstream_terminal:

            action = "block"
            reason = "UPSTREAM_TERMINAL"

        # --------------------------------------------------------
        # Descendant of pending/running node
        # --------------------------------------------------------

        elif upstream_pending:

            action = "block"
            reason = "UPSTREAM_PENDING"

        # --------------------------------------------------------
        # Immutable cache hit
        # --------------------------------------------------------

        elif cache_entry is not None:

            action = "reuse"
            reason = "CACHE_HIT"

            triggering = [
                cache_entry["eventId"]
            ]

        # --------------------------------------------------------
        # Current terminal failure
        # --------------------------------------------------------

        elif (
            current is not None
            and current.get("key") == key
            and current.get("status")
            == "terminal_failed"
        ):

            action = "block"
            reason = "TERMINAL_FAILURE"

            terminal_id = current.get(
                "terminalEventId"
            )

            if terminal_id:
                triggering = [
                    terminal_id
                ]

        # --------------------------------------------------------
        # Currently running
        # --------------------------------------------------------

        elif (
            current is not None
            and current.get("key") == key
            and current.get("status")
            == "started"
        ):

            action = "block"
            reason = "RUNNING"

            start_id = current.get(
                "startEventId"
            )

            if start_id:
                triggering = [
                    start_id
                ]

        # --------------------------------------------------------
        # Retryable failure
        # --------------------------------------------------------

        elif (
            current is not None
            and current.get("key") == key
            and current.get("status")
            == "retryable_failed"
        ):

            action = "rerun"
            reason = "RETRYABLE_FAILURE"

        # --------------------------------------------------------
        # Ready but no cache
        # --------------------------------------------------------

        else:

            if (
                parent is not None
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

        # --------------------------------------------------------
        # Propagate blocking state to descendants
        # --------------------------------------------------------

        if (
            reason == "TERMINAL_FAILURE"
        ):
            upstream_terminal = True
            upstream_pending = False

        elif reason in {
            "RUNNING",
            "UPSTREAM_PENDING",
        }:
            upstream_pending = True

        elif action == "reuse":
            # A reusable node clears pending status
            # for its descendants.
            upstream_pending = False

    return result


# ============================================================
# ERROR RESPONSE
# ============================================================

def conflict_error(
    code: str,
):
    return JSONResponse(
        status_code=409,
        content={
            "error": code
        },
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "pipeline-controller",
        "ok": True,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    try:

        test_key = (
            "pipeline:health:"
            + hashlib.sha256(
                b"pipeline-health"
            ).hexdigest()
        )

        redis_client.set(
            test_key,
            "ok",
            ex=60,
        )

        value = redis_client.get(
            test_key
        )

        return {
            "ok": value == "ok",
            "redis": "connected",
        }

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "redis": "error",
                "error": str(exc),
            },
        )


# ============================================================
# POST /pipeline
# ============================================================

@app.post("/pipeline")
async def pipeline(
    request: Request,
):

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:
        body = await request.json()

    except Exception:
        return conflict_error(
            "INVALID_REQUEST"
        )

    # --------------------------------------------------------
    # Validate request
    # --------------------------------------------------------

    if not validate_request(body):

        return conflict_error(
            "INVALID_REQUEST"
        )

    session = body[
        "session"
    ]

    revision = body[
        "revision"
    ]

    inputs = body[
        "inputs"
    ]

    incoming_events = body[
        "events"
    ]

    # --------------------------------------------------------
    # Read persistent session
    # --------------------------------------------------------

    try:

        state = load_state(
            session
        )

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content={
                "error":
                    "REDIS_ERROR",
                "detail":
                    str(exc),
            },
        )

    # --------------------------------------------------------
    # Canonical input identity
    #
    # Extra metadata is included because the entire inputs
    # object is canonicalized.
    # --------------------------------------------------------

    input_canonical = compact_json(
        inputs
    )

    # ========================================================
    # REVISION HANDLING
    # ========================================================

    # --------------------------------------------------------
    # First revision
    # --------------------------------------------------------

    if state["revision"] is None:

        state["revision"] = revision

        state[
            "inputCanonical"
        ] = input_canonical

        state[
            "inputs"
        ] = copy.deepcopy(
            inputs
        )

    # --------------------------------------------------------
    # Older revision
    # --------------------------------------------------------

    elif revision < state[
        "revision"
    ]:

        # Do not modify persistent state.
        # Events from the older revision are ignored.
        pass

    # --------------------------------------------------------
    # Same revision
    # --------------------------------------------------------

    elif revision == state[
        "revision"
    ]:

        if (
            state[
                "inputCanonical"
            ]
            != input_canonical
        ):

            return conflict_error(
                "REVISION_CONFLICT"
            )

    # --------------------------------------------------------
    # New revision
    # --------------------------------------------------------

    else:

        state["revision"] = revision

        state[
            "inputCanonical"
        ] = input_canonical

        state[
            "inputs"
        ] = copy.deepcopy(
            inputs
        )

        # Clear current attempt/terminal state.
        #
        # Successful immutable cache remains.
        state["nodes"] = {
            node: None
            for node in NODES
        }

    # ========================================================
    # EVENT PROCESSING
    # ========================================================

    accepted_ids = []
    ignored_ids = []

    # --------------------------------------------------------
    # Older revision request
    # --------------------------------------------------------

    if revision != state[
        "revision"
    ]:

        for event in incoming_events:

            if (
                isinstance(event, dict)
                and nonempty_string(
                    event.get("eventId")
                )
            ):

                ignored_ids.append(
                    event["eventId"]
                )

    else:

        # ----------------------------------------------------
        # ATOMIC COPY
        #
        # Work on a complete deep copy.
        # Nothing reaches Redis until every event succeeds.
        # ----------------------------------------------------

        working = copy.deepcopy(
            state
        )

        try:

            for event in incoming_events:

                # Structural event problem.
                if not validate_event_shape(
                    event
                ):
                    raise ValueError(
                        "INVALID_EVENT"
                    )

                # Semantic-invalid events are ignored.
                if not event_semantically_valid(
                    event
                ):
                    ignored_ids.append(
                        event["eventId"]
                    )
                    continue

                process_event(
                    working,
                    event,
                    accepted_ids,
                    ignored_ids,
                )

        except ValueError as exc:

            # Entire batch rolls back.
            return conflict_error(
                str(exc)
            )

        # ----------------------------------------------------
        # Commit the complete batch.
        # ----------------------------------------------------

        state = working

    # ========================================================
    # PERSIST
    # ========================================================

    try:

        save_state(state)

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content={
                "error":
                    "REDIS_ERROR",
                "detail":
                    str(exc),
            },
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    try:

        nodes = build_response_nodes(
            state
        )

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content={
                "error":
                    "STATE_ERROR",
                "detail":
                    str(exc),
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "revision":
                state["revision"],

            "acceptedEventIds":
                accepted_ids,

            "ignoredEventIds":
                ignored_ids,

            "nodes":
                nodes,
        },
    )
