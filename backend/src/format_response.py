"""
Format agent responses to markdown and execute generated code.
Captures matplotlib figures as base64 PNG and plotly charts as embedded images.
"""
# MUST be set before importing matplotlib to avoid Tkinter main-thread errors in sub-threads.
import matplotlib
matplotlib.use('Agg')

import re
import json
import io
import base64
import traceback
import contextlib
import functools
import threading
import os
import pickle
import subprocess
import tempfile
import types
import hashlib
import pandas as pd
import sys

from src.runtime_config import CODE_EXECUTION_TIMEOUT_SECONDS, PLOTLY_EXPORT_TIMEOUT_SECONDS


_EXECUTION_LOCK = threading.RLock()


def _serialized_execution(fn):
    """Serialize generated-code execution because plotting hooks are process-global."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _EXECUTION_LOCK:
            return fn(*args, **kwargs)
    return wrapper


def execution_succeeded(output: str) -> bool:
    """Return whether generated code completed without a captured runtime error."""
    normalized = (output or "").lstrip()
    return bool(normalized) and not normalized.startswith(("Error:", "No dataset loaded."))


def extract_code_blocks(text: str) -> list:
    """Extract Python code blocks from markdown text."""
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        pattern = r'```\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
    return matches


def _fig_to_base64(fig) -> str | None:
    """Convert a matplotlib figure to a base64 PNG data URI."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    except Exception:
        return None


def _plotly_fig_to_base64(fig) -> str | None:
    """Convert a plotly figure to a base64 PNG data URI via kaleido."""
    import threading
    result = {"value": None, "exception": None, "done": False}

    def _worker():
        try:
            import plotly.io as pio
            buf = io.BytesIO()
            pio.write_image(fig, buf, format='png', width=800, height=500, scale=2)
            buf.seek(0)
            result["value"] = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
        except Exception as e:
            result["exception"] = e
        finally:
            result["done"] = True

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=PLOTLY_EXPORT_TIMEOUT_SECONDS)
    if not result["done"]:
        print(
            f"Warning: Plotly to base64 timed out ({PLOTLY_EXPORT_TIMEOUT_SECONDS}s), "
            "skipping chart..."
        )
        return None
    if result["exception"]:
        print(f"Warning: Plotly to base64 failed: {result['exception']}, skipping chart...")
        return None
    return result["value"]


def _make_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    return obj


def _plotly_fig_to_json_marker(fig) -> str | None:
    """Convert a Plotly figure to a JSON marker string for frontend rendering.

    Returns a string like: <<<PLOTLY_JSON>>>...<<<END_PLOTLY_JSON>>>
    which the frontend parses to render interactive charts with react-plotly.js.
    """
    try:
        fig_dict = _make_json_serializable(fig.to_dict())
        fig_json = json.dumps(fig_dict, ensure_ascii=False)
        return f"<<<PLOTLY_JSON>>>\n{fig_json}\n<<<END_PLOTLY_JSON>>>"
    except Exception as e:
        print(f"Warning: Plotly to JSON failed: {e}")
        return None


def _plotly_fig_fingerprint(fig) -> str | None:
    """Hash plotted data so cosmetic layout changes do not duplicate a chart."""
    try:
        fig_dict = _make_json_serializable(fig.to_dict())
        semantic_payload = {"data": fig_dict.get("data", [])}
        if not semantic_payload["data"]:
            semantic_payload["annotations"] = fig_dict.get("layout", {}).get("annotations", [])
        canonical = json.dumps(semantic_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception:
        return None


_CONTEXT_HELPERS = {"pd", "np", "plt", "go", "px", "sm", "sklearn", "json", "__builtins__"}
_MAX_CONTEXT_ITEM_BYTES = int(os.getenv("DATAPILOT_CONTEXT_ITEM_MB", "64")) * 1024 * 1024
_MAX_CONTEXT_BYTES = int(os.getenv("DATAPILOT_CONTEXT_TOTAL_MB", "256")) * 1024 * 1024


def _copy_execution_value(value):
    """Copy mutable tabular inputs while allowing model objects in the context."""
    try:
        return value.copy()
    except (AttributeError, TypeError):
        return value


def _export_execution_context(context: dict) -> dict:
    """Return bounded, pickle-safe variables for the next isolated execution."""
    exported = {}
    total_bytes = 0
    for key, value in context.items():
        if key.startswith("_") or key in _CONTEXT_HELPERS:
            continue
        if isinstance(value, types.ModuleType) or callable(value) or _is_plotly_figure(value):
            continue
        try:
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            continue
        if len(serialized) > _MAX_CONTEXT_ITEM_BYTES or total_bytes + len(serialized) > _MAX_CONTEXT_BYTES:
            continue
        exported[key] = value
        total_bytes += len(serialized)
    return exported


def _execute_code_from_markdown_impl(code_text: str, variables: dict, return_context: bool = False):
    """Execute Python code with provided variables and optionally export updated state."""
    if not variables:
        output = "No dataset loaded."
        return (output, {}) if return_context else output

    code_blocks = extract_code_blocks(code_text)
    code_to_run = "\n\n".join(code_blocks) if code_blocks else code_text

    if not code_to_run.strip():
        return ("", {}) if return_context else ""

    # Build execution context
    context: dict = {}
    for name, value in variables.items():
        context[name] = _copy_execution_value(value)

    text_output = ""
    image_md_blocks = []
    plotly_json_blocks = []
    plotly_json_seen = set()

    def _append_plotly_fig(fig):
        marker = _plotly_fig_to_json_marker(fig)
        if not marker:
            return
        fingerprint = _plotly_fig_fingerprint(fig) or marker
        if fingerprint in plotly_json_seen:
            return
        plotly_json_seen.add(fingerprint)
        plotly_json_blocks.append(marker)

    _orig_show = None
    _orig_plotly_show = None

    try:
        import matplotlib.pyplot as plt
        _orig_show = plt.show

        def _patched_show(*a, **kw):
            fig = plt.gcf()
            if fig.get_axes():
                b64 = _fig_to_base64(fig)
                if b64:
                    image_md_blocks.append(f"![chart]({b64})")
            plt.close('all')

        plt.show = _patched_show
    except Exception:
        pass

    try:
        import plotly.graph_objects as go
        _orig_plotly_show = go.Figure.show

        def _patched_plotly_show(self, *a, **kw):
            _append_plotly_fig(self)

        go.Figure.show = _patched_plotly_show
    except Exception:
        pass

    import numpy as np
    context.update({
        "pd": pd,
        "np": np,
        "plt": __import__("matplotlib.pyplot", fromlist=["pyplot"]),
        "go": __import__("plotly.graph_objects", fromlist=["graph_objects"]),
        "px": __import__("plotly.express", fromlist=["express"]),
        "sm": __import__("statsmodels.api", fromlist=["api"]),
        "sklearn": __import__("sklearn"),
        "json": json,
    })

    try:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code_to_run, context)

        text_output = buf.getvalue()

        try:
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            if fig.get_axes():
                b64 = _fig_to_base64(fig)
                if b64:
                    image_md_blocks.append(f"![chart]({b64})")
            plt.close('all')
        except Exception:
            pass

        # Fallback: capture one plotly figure from scope if user did not call fig.show().
        if not plotly_json_blocks:
            for key, val in context.items():
                if key.startswith('_'):
                    continue
                if _is_plotly_figure(val):
                    _append_plotly_fig(val)
                    break

        parts = []
        if text_output.strip():
            cleaned = text_output[:4000]
            if len(text_output) > 4000:
                cleaned += "\n... (output truncated)"
            parts.append(f"```\n{cleaned}\n```")

        if image_md_blocks:
            parts.extend(image_md_blocks)

        if plotly_json_blocks:
            parts.extend(plotly_json_blocks)

        for key, value in context.items():
            if isinstance(value, pd.DataFrame) and key not in variables and not key.startswith('_'):
                parts.append(f"\n**{key}** ({len(value)} rows x {len(value.columns)} cols):\n")
                parts.append(f"```\n{value.head(10).to_string()}\n```")

        output = "\n\n".join(parts) if parts else "_Code executed, no output._"
        if return_context:
            return output, _export_execution_context(context)
        return output

    except Exception as e:
        error_msg = f"Error: {type(e).__name__}: {str(e)}"
        tb = traceback.format_exc().splitlines()[-5:]
        error_msg += "\n" + "\n".join(tb)
        if return_context:
            return error_msg, {}
        return error_msg
    finally:
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_rows')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        try:
            import matplotlib.pyplot as plt
            if _orig_show:
                plt.show = _orig_show
        except Exception:
            pass
        try:
            import plotly.graph_objects as go
            if _orig_plotly_show:
                go.Figure.show = _orig_plotly_show
        except Exception:
            pass


_WORKER_SCRIPT = """
import pickle
import sys
from src.format_response import _execute_code_from_markdown_impl

code_text, variables, return_context = pickle.load(sys.stdin.buffer)
result = _execute_code_from_markdown_impl(code_text, variables, return_context)
with open(sys.argv[1], "wb") as result_file:
    pickle.dump(result, result_file, protocol=pickle.HIGHEST_PROTOCOL)
"""


def _run_code_worker(code_text: str, variables: dict, timeout: int, return_context: bool):
    """Execute generated code in a killable child process."""
    backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pickle") as result_file:
            result_path = result_file.name

        completed = subprocess.run(
            [sys.executable, "-c", _WORKER_SCRIPT, result_path],
            input=pickle.dumps((code_text, variables, return_context), protocol=pickle.HIGHEST_PROTOCOL),
            cwd=backend_root,
            capture_output=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            error = f"Error: RuntimeError: Generated code worker failed: {stderr or completed.returncode}"
            return (error, {}) if return_context else error

        with open(result_path, "rb") as result_file:
            return pickle.load(result_file)
    except subprocess.TimeoutExpired:
        error = f"Error: TimeoutError: Generated code exceeded the {timeout}-second limit."
        return (error, {}) if return_context else error
    except Exception as e:
        error = f"Error: RuntimeError: Generated code worker failed: {str(e)}"
        return (error, {}) if return_context else error
    finally:
        if result_path:
            try:
                os.unlink(result_path)
            except OSError:
                pass


@_serialized_execution
def execute_code_from_markdown(
    code_text: str,
    datasets: dict,
    timeout: int = CODE_EXECUTION_TIMEOUT_SECONDS,
) -> str:
    """Execute generated code and return its formatted output."""
    return _run_code_worker(code_text, datasets, timeout, return_context=False)


@_serialized_execution
def execute_code_with_state(
    code_text: str,
    variables: dict,
    timeout: int = CODE_EXECUTION_TIMEOUT_SECONDS,
) -> tuple[str, dict]:
    """Execute generated code and export pickle-safe variables for the next agent."""
    return _run_code_worker(code_text, variables, timeout, return_context=True)


def _is_plotly_figure(obj) -> bool:
    try:
        import plotly.graph_objects as go
        return isinstance(obj, go.Figure)
    except Exception:
        return False


def format_response_to_markdown(response: dict, datasets: dict) -> str:
    """Format agent response dict into markdown for display.

    Response is expected to be {agent_name: {code: ..., summary: ...}} or similar.
    Executes code and embeds any generated charts as markdown images.
    """
    if not response:
        return "No response generated."

    parts = []
    for agent_name, result in response.items():
        if not isinstance(result, dict):
            parts.append(str(result))
            continue

        # Handle planner responses
        if "plan" in result and "plan_instructions" in result:
            plan = result["plan"]
            instructions = result["plan_instructions"]
            if isinstance(instructions, str):
                try:
                    instructions = json.loads(instructions)
                except json.JSONDecodeError:
                    pass

            formatted = f"### 📋 Analysis Plan\n\n**Plan**: `{plan}`\n\n"
            if isinstance(instructions, dict):
                for ag, instr in instructions.items():
                    if isinstance(instr, dict):
                        formatted += f"**{ag}**:\n"
                        if "create" in instr:
                            formatted += f"- Create: {instr['create']}\n"
                        if "use" in instr:
                            formatted += f"- Use: {instr['use']}\n"
                        if "instruction" in instr:
                            formatted += f"- Task: {instr['instruction']}\n"
                        formatted += "\n"
                    else:
                        formatted += f"- {instr}\n"
            else:
                formatted += str(instructions)

            parts.append(formatted)
            continue

        # Handle agent results (code + summary)
        code = result.get("code", "")
        summary = result.get("summary", "")

        section = f"### 🤖 {agent_name.replace('_', ' ').title()}\n\n"

        if summary:
            section += f"{summary}\n\n"

        if code:
            section += f"```python\n{code}\n```\n\n"

            # Execute the code and capture output + images
            exec_result = execute_code_from_markdown(code, datasets)
            if exec_result and exec_result.strip():
                section += f"{exec_result}\n\n"

        parts.append(section)

    return "\n---\n\n".join(parts)
