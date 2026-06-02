"""Central runtime limits for LLM calls and generated-code execution."""
import os
from pathlib import Path

from dotenv import load_dotenv


# Load the backend configuration even when a module is imported outside app.py.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}.")
    return value


# LLM generation limits. Complex analysis agents need room for reasoning, code,
# and reports. Provider-specific caps are applied in app.build_lm when known.
LLM_MAX_TOKENS = _positive_int("DATAPILOT_LLM_MAX_TOKENS", 32768)
LLM_REQUEST_TIMEOUT_SECONDS = _positive_int("DATAPILOT_LLM_REQUEST_TIMEOUT_SECONDS", 1800)

# Per-agent wall-clock limits. Keep these slightly above the provider timeout
# so an SDK-level timeout can finish and release its worker thread cleanly.
DATASET_DESCRIPTION_TIMEOUT_SECONDS = _positive_int(
    "DATAPILOT_DATASET_DESCRIPTION_TIMEOUT_SECONDS", 1860
)
CHANCELLOR_TIMEOUT_SECONDS = _positive_int("DATAPILOT_CHANCELLOR_TIMEOUT_SECONDS", 1860)
COMMANDER_TIMEOUT_SECONDS = _positive_int("DATAPILOT_COMMANDER_TIMEOUT_SECONDS", 1860)
EXECUTOR_AGENT_TIMEOUT_SECONDS = _positive_int("DATAPILOT_EXECUTOR_AGENT_TIMEOUT_SECONDS", 1860)
CENSOR_TIMEOUT_SECONDS = _positive_int("DATAPILOT_CENSOR_TIMEOUT_SECONDS", 1860)
HELPER_AGENT_TIMEOUT_SECONDS = _positive_int("DATAPILOT_HELPER_AGENT_TIMEOUT_SECONDS", 1860)
LEGACY_AGENT_TIMEOUT_SECONDS = _positive_int("DATAPILOT_LEGACY_AGENT_TIMEOUT_SECONDS", 1860)
LEGACY_PLANNER_TIMEOUT_SECONDS = _positive_int("DATAPILOT_LEGACY_PLANNER_TIMEOUT_SECONDS", 1860)

# Generated Python executes in a killable child process. The outer asyncio
# timeout must be larger than the subprocess timeout so the child can clean up.
CODE_EXECUTION_TIMEOUT_SECONDS = _positive_int("DATAPILOT_CODE_EXECUTION_TIMEOUT_SECONDS", 600)
CODE_EXECUTION_GRACE_SECONDS = _positive_int("DATAPILOT_CODE_EXECUTION_GRACE_SECONDS", 60)
CODE_EXECUTION_OUTER_TIMEOUT_SECONDS = (
    CODE_EXECUTION_TIMEOUT_SECONDS + CODE_EXECUTION_GRACE_SECONDS
)

PLOTLY_EXPORT_TIMEOUT_SECONDS = _positive_int("DATAPILOT_PLOTLY_EXPORT_TIMEOUT_SECONDS", 120)
