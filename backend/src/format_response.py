"""
Format agent responses to markdown and execute generated code.
Captures matplotlib figures as base64 PNG and plotly charts as embedded images.
"""
# MUST be set before importing matplotlib — avoids Tkinter main-thread errors in sub-threads
import matplotlib
matplotlib.use('Agg')

import re
import json
import io
import base64
import traceback
import contextlib
import pandas as pd
import sys


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
    t.join(timeout=500)  # 15秒超时

    if not result["done"]:
        print("Warning: Plotly to base64 timed out (15s), skipping chart...")
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


def execute_code_from_markdown(code_text: str, datasets: dict, timeout: int = 30) -> str:
    """Execute Python code with session datasets in context.
    Returns markdown string with text output and embedded images.
    """
    if not datasets:
        return "No dataset loaded."

    code_blocks = extract_code_blocks(code_text)
    code_to_run = "\n\n".join(code_blocks) if code_blocks else code_text

    if not code_to_run.strip():
        return ""

    # Build execution context
    context: dict = {}
    for name, df in datasets.items():
        context[name] = df.copy()

    # Collected outputs
    text_output = ""
    image_md_blocks = []   # markdown image strings (matplotlib only)
    plotly_json_blocks = []  # plotly JSON markers for interactive rendering

    # ---- Patch plt.show to capture matplotlib figures ----
    _captured_figs = []
    _orig_show = None
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        _orig_show = plt.show
        # Instead of showing, capture current figure
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

    # ---- Patch plotly Figure.show to capture as interactive JSON ----
    _orig_plotly_show = None
    try:
        import plotly.graph_objects as go
        _orig_plotly_show = go.Figure.show
        def _patched_plotly_show(self, *a, **kw):
            # 忽略 renderer 参数，始终捕获为 JSON 用于前端交互式渲染
            # 如果指定了 renderer='json'，我们仍然捕获 JSON 标记
            marker = _plotly_fig_to_json_marker(self)
            if marker:
                plotly_json_blocks.append(marker)
        go.Figure.show = _patched_plotly_show
    except Exception:
        pass

    # Common imports for user code
    import numpy as np
    context.update({
        "pd": pd,
        "np": np,
        "plt": __import__("matplotlib.pyplot"),
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

        # Also capture any figures that were created but not shown
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

        # Also capture any plotly figures left in scope (even if fig.show() was not called)
        # 强制捕获最后一个 plotly figure，确保即使没有 fig.show() 也能显示图表
        last_fig = None
        for key, val in context.items():
            if key.startswith('_'):
                continue
            if _is_plotly_figure(val):
                last_fig = val  # 记录最后一个图表
        
        # 如果通过 fig.show() 没有捕获到任何图表，但找到了 plotly figure，则捕获最后一个
        if not plotly_json_blocks and last_fig:
            marker = _plotly_fig_to_json_marker(last_fig)
            if marker:
                plotly_json_blocks.append(marker)
        # 即使已经通过 fig.show() 捕获了图表，也尝试添加额外的图表（最多5个）
        elif last_fig:
            count = 0
            for key, val in context.items():
                if count >= 5:
                    break
                if key.startswith('_'):
                    continue
                if _is_plotly_figure(val) and val != last_fig:
                    marker = _plotly_fig_to_json_marker(val)
                    if marker:
                        plotly_json_blocks.append(marker)
                        count += 1

        # Build final markdown
        parts = []
        if text_output.strip():
            # Truncate very long output
            cleaned = text_output[:4000]
            if len(text_output) > 4000:
                cleaned += "\n... (output truncated)"
            parts.append(f"```\n{cleaned}\n```")
        if image_md_blocks:
            parts.extend(image_md_blocks)
        # Also capture any plotly figure left in scope
        # 扫描context中的所有Plotly图表（无条件执行，确保捕获所有图表）
        for key, val in context.items():
            if key.startswith('_'):
                continue
            try:
                if _is_plotly_figure(val):
                    marker = _plotly_fig_to_json_marker(val)
                    if marker:
                        plotly_json_blocks.append(marker)
                        break  # 一个代码块一个图表就足够了
            except Exception:
                continue
        
        # Embed Plotly JSON markers for frontend interactive rendering
        if plotly_json_blocks:
            parts.extend(plotly_json_blocks)

        # Show any new DataFrames
        for key, value in context.items():
            if isinstance(value, pd.DataFrame) and key not in datasets and not key.startswith('_'):
                parts.append(f"\n**{key}** ({len(value)} rows × {len(value.columns)} cols):\n")
                parts.append(f"```\n{value.head(10).to_string()}\n```")

        return "\n\n".join(parts) if parts else "_Code executed, no output._"

    except Exception as e:
        error_msg = f"Error: {type(e).__name__}: {str(e)}"
        tb = traceback.format_exc().splitlines()[-5:]
        error_msg += "\n" + "\n".join(tb)
        return error_msg
    finally:
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_rows')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        # Restore patched functions
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
