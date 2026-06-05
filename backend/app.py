"""
DataPilot Backend - AI Data Analyst
秦朝官职编排架构：秦始皇（用户） -> 丞相 -> 太尉 -> 执行智能体 -> 御史大夫
"""
import os
import json
import asyncio
import logging
import re
import threading
from contextlib import suppress
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional
from enum import Enum
from uuid import UUID, uuid4

import pandas as pd
import dspy
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from src.agents.agents import (
    preprocessing_agent, statistical_analytics_agent,
    sk_learn_agent, data_viz_agent,
    planner_module, code_combiner_agent, code_fix, code_edit,
    dataset_description_agent, chat_history_name_agent,
    chancellor_agent, censor_agent, commander_agent,
    qin_dynasty_orchestrator, AgentTaskState,
    _run_sync,
)
from src.format_response import execution_succeeded, format_response_to_markdown, execute_code_from_markdown
from src.runtime_config import (
    CODE_EXECUTION_OUTER_TIMEOUT_SECONDS,
    CODE_EXECUTION_TIMEOUT_SECONDS,
    DATASET_DESCRIPTION_TIMEOUT_SECONDS,
    HELPER_AGENT_TIMEOUT_SECONDS,
    LEGACY_AGENT_TIMEOUT_SECONDS,
    LEGACY_PLANNER_TIMEOUT_SECONDS,
    LLM_MAX_TOKENS,
    LLM_REQUEST_TIMEOUT_SECONDS,
)

load_dotenv()

# -- Logging ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("datapilot")

# -- Styling instructions for visualization ----------------------------
STYLING_INSTRUCTIONS = [
    str({"category": "line_charts", "description": "Trends over time", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
    str({"category": "bar_charts", "description": "Comparing categories", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
    str({"category": "scatter_charts", "description": "Relationships between variables", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
    str({"category": "pie_charts", "description": "Composition of whole", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
    str({"category": "histograms", "description": "Distribution of data", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
    str({"category": "heat_maps", "description": "Data density/intensity", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
    str({"category": "generic", "description": "General charts", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
]

# -- Session state (in-memory, per-session) -----------------------------
sessions: dict = {}
task_states: dict = {}  # session_id -> AgentTaskState
active_chat_runs: dict = {}  # session_id -> {run_id, cancel_event, task_state}
session_locks: dict = {}  # session_id -> threading.Lock for thread-safe parallel uploads
SESSION_STORAGE_DIR = Path(
    os.getenv("DATAPILOT_SESSION_DIR", Path(__file__).resolve().parent / ".datapilot_sessions")
).resolve()
RESERVED_DATASET_NAMES = {"df", "raw_df", "df_clean", "df_cleaned"}

def get_session_lock(session_id: str) -> threading.Lock:
    """Get or create a lock for a session to ensure thread-safe operations."""
    if session_id not in session_locks:
        session_locks[session_id] = threading.Lock()
    return session_locks[session_id]


def _new_session() -> dict:
    return {
        "datasets": {},        # name -> DataFrame
        "dataset_files": {},   # name -> original filename
        "dataset_descriptions": {},
        "primary_dataset": "",
        "description": "",     # dataset description JSON
        "dataset_filename": "",
        "chat_history": [],    # list of {role, content}
        "model_config": {
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        },
    }


def _session_storage_paths(session_id: str):
    """Return controlled storage paths for UUID sessions only."""
    try:
        safe_session_id = str(UUID(session_id))
    except (ValueError, AttributeError):
        return None, None
    return (
        SESSION_STORAGE_DIR / f"{safe_session_id}.json",
        SESSION_STORAGE_DIR / f"{safe_session_id}.pkl",
    )


def _safe_dataset_name(filename: str, existing_names) -> str:
    """Create a stable Python identifier while preserving the original filename separately."""
    stem = Path(filename or "dataset").stem
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", stem).strip("_").lower()
    if not normalized or not normalized.isidentifier():
        normalized = "dataset"
    if normalized in RESERVED_DATASET_NAMES:
        normalized = f"{normalized}_data"
    candidate = normalized
    suffix = 2
    while candidate in existing_names:
        candidate = f"{normalized}_{suffix}"
        suffix += 1
    return candidate


def _repair_dataset_metadata(session: dict):
    """Fill metadata for legacy snapshots and keep the primary dataset valid."""
    datasets = session.setdefault("datasets", {})
    dataset_files = session.setdefault("dataset_files", {})
    dataset_descriptions = session.setdefault("dataset_descriptions", {})
    primary = session.get("primary_dataset", "")

    # Older snapshots used dataframe aliases as dataset keys. Rename those keys
    # once so agents only see stable variables derived from uploaded filenames.
    for old_name in list(datasets):
        if old_name not in RESERVED_DATASET_NAMES:
            continue
        new_name = _safe_dataset_name(f"{old_name}_data.csv", datasets)
        datasets[new_name] = datasets.pop(old_name)
        if old_name in dataset_files:
            dataset_files[new_name] = dataset_files.pop(old_name)
        if old_name in dataset_descriptions:
            dataset_descriptions[new_name] = dataset_descriptions.pop(old_name)
        if primary == old_name:
            primary = new_name

    for name in datasets:
        dataset_files.setdefault(name, session.get("dataset_filename") or f"{name}.csv")
    if primary not in datasets:
        primary = next(iter(datasets), "")
    session["primary_dataset"] = primary
    session["dataset_filename"] = dataset_files.get(primary, "")


async def _generate_dataset_description(session: dict, dataset_name: str, df: pd.DataFrame, filename: str, description: str = ""):
    """Generate description for a single dataset."""
    session_lm = get_session_lm(session)
    try:
        with dspy.context(lm=session_lm):
            desc_agent = dspy.Predict(dataset_description_agent)
            buf = StringIO()
            df.info(buf=buf)
            info_str = buf.getvalue()
            sample = df.head(5).to_string()
            dataset_view = f"File: {filename}\nShape: {df.shape}\n\n{info_str}\n\nSample:\n{sample}"
            # Only use user-provided description; otherwise empty string to prevent
            # the LLM from "enhancing" old dataset descriptions.
            existing_desc = description if description else ""
            result = await asyncio.wait_for(
                _run_sync(
                    desc_agent,
                    dataset=dataset_view,
                    existing_description=existing_desc,
                ),
                timeout=DATASET_DESCRIPTION_TIMEOUT_SECONDS,
            )
            session["dataset_descriptions"][dataset_name] = result.description
    except Exception as e:
        logger.warning(f"Dataset description generation failed: {e}; using fallback info().")
        buf = StringIO()
        df.info(buf=buf)
        session["dataset_descriptions"][dataset_name] = buf.getvalue()


def _refresh_dataset_description(session: dict):
    """Build a multi-dataset context whose leading index always lists every file."""
    descriptions = session.get("dataset_descriptions", {})
    files = session.get("dataset_files", {})
    primary = session.get("primary_dataset", "")
    index_lines = ["DATASET INDEX (always complete):"]
    sections = []
    for name, df in session.get("datasets", {}).items():
        filename = files.get(name, name)
        columns = [str(column) for column in df.columns]
        visible_columns = columns[:32]
        remaining_columns = len(columns) - len(visible_columns)
        column_summary = ", ".join(visible_columns)
        if remaining_columns > 0:
            column_summary += f", ... (+{remaining_columns} more)"
        index_lines.append(
            f"- variable={name}; file={filename}; shape={df.shape}; "
            f"primary={name == primary}; columns=[{column_summary}]"
        )
        detail = descriptions.get(name, "")
        sections.append(
            f"Dataset variable: {name}\n"
            f"Original file: {filename}\n"
            f"Shape: {df.shape}\n"
            f"Columns: {columns}\n"
            f"{detail}"
        )
    index = "\n".join(index_lines)
    details = "\n\n--- DATASET DETAILS ---\n\n" + "\n\n---\n\n".join(sections) if sections else ""
    session["description"] = (index + details).strip()


def _execution_datasets(session: dict) -> dict:
    """Return uploaded datasets under their stable filename-derived variables."""
    return dict(session.get("datasets", {}))


def _dataset_summaries(session: dict) -> list[dict]:
    """Return lightweight dataset metadata for the sidebar."""
    primary = session.get("primary_dataset", "")
    files = session.get("dataset_files", {})
    return [
        {
            "name": name,
            "filename": files.get(name, name),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "primary": name == primary,
        }
        for name, df in session.get("datasets", {}).items()
    ]


def persist_session(session_id: str, session: dict):
    """Persist local session metadata and uploaded DataFrames without API keys."""
    metadata_path, dataframe_path = _session_storage_paths(session_id)
    if metadata_path is None:
        return

    SESSION_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    datasets = session.get("datasets", {})
    if datasets:
        dataframe_tmp = dataframe_path.with_suffix(".pkl.tmp")
        pd.to_pickle(datasets, dataframe_tmp)
        dataframe_tmp.replace(dataframe_path)
    else:
        dataframe_path.unlink(missing_ok=True)

    model_config = {
        key: value
        for key, value in session.get("model_config", {}).items()
        if key != "api_key"
    }
    metadata = {
        "description": session.get("description", ""),
        "dataset_filename": session.get("dataset_filename", ""),
        "dataset_files": session.get("dataset_files", {}),
        "dataset_descriptions": session.get("dataset_descriptions", {}),
        "primary_dataset": session.get("primary_dataset", ""),
        "chat_history": session.get("chat_history", [])[-20:],
        "model_config": model_config,
    }
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    metadata_tmp.replace(metadata_path)


def restore_session(session_id: str) -> dict:
    """Restore a previously uploaded local dataset after a backend restart."""
    session = _new_session()
    metadata_path, dataframe_path = _session_storage_paths(session_id)
    if metadata_path is None or not metadata_path.exists():
        return session

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if dataframe_path.exists():
            restored = pd.read_pickle(dataframe_path)
            session["datasets"] = restored if isinstance(restored, dict) else {"df": restored}
        session["description"] = str(metadata.get("description", ""))
        session["dataset_filename"] = str(metadata.get("dataset_filename", ""))
        session["dataset_files"] = dict(metadata.get("dataset_files", {}))
        session["dataset_descriptions"] = dict(metadata.get("dataset_descriptions", {}))
        session["primary_dataset"] = str(metadata.get("primary_dataset", ""))
        session["chat_history"] = list(metadata.get("chat_history", []))[-20:]
        session["model_config"].update(metadata.get("model_config", {}))
    except Exception as e:
        logger.warning(f"Failed to restore session {session_id}: {e}")
    _repair_dataset_metadata(session)
    _refresh_dataset_description(session)
    return session


def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = restore_session(session_id)
    return sessions[session_id]


def get_task_state(session_id: str) -> AgentTaskState:
    """Get or create task state for a session."""
    if session_id not in task_states:
        task_states[session_id] = AgentTaskState()
    return task_states[session_id]


# -- DSPy LM helpers ----------------------------------------------------
def build_lm(provider: str, model: str, api_key: Optional[str] = None, **kwargs):
    """Build a dspy.LM from provider/model/api_key."""
    provider_prefix = {
        "openai": "openai",
        "anthropic": "anthropic",
        "groq": "groq",
        "gemini": "gemini",
        "deepseek": "deepseek",
    }.get(provider.lower(), provider.lower())
    
    full_model = model if "/" in model else f"{provider_prefix}/{model}"
    requested_max_tokens = kwargs.get("max_tokens", LLM_MAX_TOKENS)
    # Some selectable legacy OpenAI models expose a smaller output window.
    # Clamp only well-known caps; configurable or newer models keep the
    # deployment-level DATAPILOT_LLM_MAX_TOKENS value.
    known_output_caps = {
        "openai/gpt-4o": 16384,
        "openai/gpt-4o-mini": 16384,
    }
    effective_max_tokens = min(
        requested_max_tokens,
        known_output_caps.get(full_model.lower(), requested_max_tokens),
    )
    
    lm_kwargs = {
        "model": full_model,
        "max_tokens": effective_max_tokens,
        "temperature": kwargs.get("temperature", 0.7),
        "timeout": kwargs.get("timeout", LLM_REQUEST_TIMEOUT_SECONDS),
    }
    if api_key:
        lm_kwargs["api_key"] = api_key
    
    return dspy.LM(**lm_kwargs)


REMOTE_LLM_PROVIDERS = {"openai", "anthropic", "groq", "gemini", "deepseek"}


def resolve_api_key(provider: str, configured_key: str = "") -> str:
    """Resolve a key without leaking credentials across providers."""
    return configured_key.strip() or os.getenv(f"{provider.upper()}_API_KEY", "").strip()


def require_api_key(provider: str, configured_key: str = "") -> str:
    """Return the effective key or fail early with an actionable message."""
    api_key = resolve_api_key(provider, configured_key)
    if provider.lower() in REMOTE_LLM_PROVIDERS and not api_key:
        raise HTTPException(
            400,
            f"Missing API key for provider '{provider}'. Configure {provider.upper()}_API_KEY in backend/.env or enter it in model settings.",
        )
    return api_key


def get_session_lm(session: dict) -> dspy.LM:
    cfg = session.get("model_config", {})
    provider = cfg.get("provider", os.getenv("LLM_PROVIDER", "openai"))
    model = cfg.get("model", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    api_key = require_api_key(provider, cfg.get("api_key", ""))
    return build_lm(provider, model, api_key)


# -- Default LM for startup --------------------------------------------
default_provider = os.getenv("LLM_PROVIDER", "openai")
default_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
default_api_key = os.getenv(f"{default_provider.upper()}_API_KEY", "")
try:
    default_lm = build_lm(default_provider, default_model, default_api_key)
    dspy.configure(lm=default_lm)
except Exception as e:
    logger.warning(f"Failed to configure default LLM: {e}; using session-level LLM instead.")


# -- FastAPI app --------------------------------------------------------
app = FastAPI(title="DataPilot API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Request models -----------------------------------------------------
class QueryRequest(BaseModel):
    query: str


class ModelConfigRequest(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""


class CodeFixRequest(BaseModel):
    code: str
    error: str


class CodeExecuteRequest(BaseModel):
    code: str


class CodeEditRequest(BaseModel):
    code: str
    prompt: str


class ReviewRequest(BaseModel):
    """Manual review payload."""
    approved: bool
    target: str  # "丞相" | "太尉" | agent_name
    comments: str = ""
    severity: str = "medium"  # "low" | "medium" | "high"


# -- Endpoints ----------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0", "architecture": "qin_dynasty"}


@app.post("/session")
async def create_session():
    import uuid
    session_id = str(uuid.uuid4())
    get_session(session_id)
    get_task_state(session_id)  # initialize task state for the new session
    return {"session_id": session_id}


@app.get("/session/{session_id}/model")
async def get_model_config(session_id: str):
    session = get_session(session_id)
    cfg = session.get("model_config", {})
    # Don't expose full API key
    safe_cfg = {k: v for k, v in cfg.items() if k != "api_key"}
    safe_cfg["has_api_key"] = bool(resolve_api_key(cfg.get("provider", "openai"), cfg.get("api_key", "")))
    return safe_cfg


@app.post("/session/{session_id}/model")
async def set_model_config(session_id: str, req: ModelConfigRequest):
    session = get_session(session_id)
    previous_config = session.get("model_config", {})
    previous_api_key = previous_config.get("api_key", "")
    submitted_api_key = req.api_key.strip()
    is_masked_key = submitted_api_key and all(char in {"*", "•", " "} for char in submitted_api_key)
    preserve_api_key = req.provider == previous_config.get("provider") and (not submitted_api_key or is_masked_key)
    next_api_key = previous_api_key if preserve_api_key else ("" if is_masked_key else submitted_api_key)
    require_api_key(req.provider, next_api_key)
    session["model_config"] = {
        "provider": req.provider,
        "model": req.model,
        "api_key": next_api_key,
    }
    persist_session(session_id, session)
    return {"status": "updated", "provider": req.provider, "model": req.model}


@app.post("/session/{session_id}/upload")
async def upload_dataset(session_id: str, file: UploadFile = File(...), description: str = Form("")):
    session_lock = get_session_lock(session_id)
    
    # Acquire lock for thread-safe session access during parallel uploads
    with session_lock:
        session = get_session(session_id)
        filename = file.filename
        ext = filename.rsplit(".", 1)[-1].lower()
        
        # Clear history when the available data collection changes so the LLM
        # does not answer with stale assumptions from earlier files.
        if session["datasets"]:
            session["chat_history"] = []
            task_states[session_id] = AgentTaskState()
        
        try:
            content = await file.read()
            if ext == "csv":
                try:
                    decoded_content = content.decode("utf-8-sig")
                except UnicodeDecodeError:
                    decoded_content = content.decode("gb18030")
                df = pd.read_csv(StringIO(decoded_content))
                dataset_name = _safe_dataset_name(filename, session["datasets"])
                session["datasets"][dataset_name] = df
                session["dataset_files"][dataset_name] = filename
                if not session.get("primary_dataset"):
                    session["primary_dataset"] = dataset_name
                session["dataset_filename"] = filename
                
                # Generate description for single dataset
                await _generate_dataset_description(session, dataset_name, df, filename, description)
                
                _refresh_dataset_description(session)
                persist_session(session_id, session)
                
                return {
                    "name": dataset_name,
                    "filename": filename,
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "description": session["description"],
                    "datasets": _dataset_summaries(session),
                }
                
            elif ext in ("xlsx", "xls"):
                # Read all sheets from Excel file
                excel_file = pd.ExcelFile(BytesIO(content))
                sheet_names = excel_file.sheet_names
                
                if len(sheet_names) == 1:
                    # Single sheet: use original logic
                    df = pd.read_excel(BytesIO(content))
                    dataset_name = _safe_dataset_name(filename, session["datasets"])
                    session["datasets"][dataset_name] = df
                    session["dataset_files"][dataset_name] = filename
                    if not session.get("primary_dataset"):
                        session["primary_dataset"] = dataset_name
                    session["dataset_filename"] = filename
                    
                    # Generate description
                    await _generate_dataset_description(session, dataset_name, df, filename, description)
                    
                    _refresh_dataset_description(session)
                    persist_session(session_id, session)
                    
                    return {
                        "name": dataset_name,
                        "filename": filename,
                        "shape": list(df.shape),
                        "columns": list(df.columns),
                        "description": session["description"],
                        "datasets": _dataset_summaries(session),
                    }
                else:
                    # Multiple sheets: create separate dataset for each sheet - processed in parallel
                    sheet_info = []
                    
                    # Read all sheets in parallel
                    async def read_sheet(sheet_name: str, content_bytes: bytes):
                        df = await asyncio.to_thread(pd.read_excel, BytesIO(content_bytes), sheet_name=sheet_name)
                        sheet_safe_name = re.sub(r"[^0-9a-zA-Z_]+", "_", sheet_name).strip("_") or f"sheet_{sheet_names.index(sheet_name)}"
                        sheet_dataset_name = _safe_dataset_name(f"{Path(filename).stem}_{sheet_safe_name}", session["datasets"])
                        return {
                            "name": sheet_dataset_name,
                            "sheet_name": sheet_name,
                            "df": df,
                        }
                    
                    # Parallel read all sheets
                    read_tasks = [read_sheet(sheet_name, content) for sheet_name in sheet_names]
                    sheet_info = await asyncio.gather(*read_tasks)
                    
                    # Store all datasets
                    primary_df = None
                    primary_dataset_name = None
                    for info in sheet_info:
                        session["datasets"][info["name"]] = info["df"]
                        session["dataset_files"][info["name"]] = f"{filename}::{info['sheet_name']}"
                        if primary_df is None:
                            primary_df = info["df"]
                            primary_dataset_name = info["name"]
                    
                    if not session.get("primary_dataset") and primary_dataset_name:
                        session["primary_dataset"] = primary_dataset_name
                    session["dataset_filename"] = filename
                    
                    # Generate descriptions in parallel
                    desc_tasks = [
                        _generate_dataset_description(session, info["name"], info["df"], f"{filename}::{info['sheet_name']}", description)
                        for info in sheet_info
                    ]
                    await asyncio.gather(*desc_tasks)
                    
                    _refresh_dataset_description(session)
                    persist_session(session_id, session)
                    
                    return {
                        "name": primary_dataset_name,
                        "filename": filename,
                        "shape": list(primary_df.shape),
                        "columns": list(primary_df.columns),
                        "description": session["description"],
                        "datasets": _dataset_summaries(session),
                    }
            else:
                raise HTTPException(400, f"不支持的文件格式: .{ext}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"文件读取失败: {e}")


@app.get("/session/{session_id}/dataset")
async def get_dataset_info(session_id: str):
    session = get_session(session_id)
    if not session["datasets"]:
        return {"loaded": False, "datasets": []}
    name = session.get("primary_dataset") or list(session["datasets"].keys())[0]
    df = session["datasets"][name]
    return {
        "loaded": True,
        "name": name,
        "filename": session.get("dataset_files", {}).get(name, ""),
        "shape": list(df.shape),
        "columns": list(df.columns),
        "description": session["description"],
        "datasets": _dataset_summaries(session),
    }


@app.delete("/session/{session_id}/dataset/{dataset_name}")
async def delete_dataset(session_id: str, dataset_name: str):
    session = get_session(session_id)
    if dataset_name not in session["datasets"]:
        raise HTTPException(404, f"Dataset not found: {dataset_name}")

    session["datasets"].pop(dataset_name, None)
    session.get("dataset_files", {}).pop(dataset_name, None)
    session.get("dataset_descriptions", {}).pop(dataset_name, None)
    if session.get("primary_dataset") == dataset_name:
        session["primary_dataset"] = next(iter(session["datasets"]), "")
    _repair_dataset_metadata(session)
    _refresh_dataset_description(session)
    session["chat_history"] = []
    task_states[session_id] = AgentTaskState()
    persist_session(session_id, session)
    return {
        "status": "deleted",
        "loaded": bool(session["datasets"]),
        "datasets": _dataset_summaries(session),
    }


@app.post("/session/{session_id}/upload/batch")
async def upload_dataset_batch(session_id: str, files: list[UploadFile] = File(...)):
    """批量上传多个文件，并行处理"""
    session_lock = get_session_lock(session_id)
    
    with session_lock:
        session = get_session(session_id)
        
        # Clear history when the available data collection changes
        if session["datasets"]:
            session["chat_history"] = []
            task_states[session_id] = AgentTaskState()
    
    results = []
    errors = []
    
    # 定义单个文件处理函数
    async def process_file(file: UploadFile):
        filename = file.filename
        ext = filename.rsplit(".", 1)[-1].lower() if filename else ""
        
        try:
            content = await file.read()
            
            if ext == "csv":
                try:
                    decoded_content = content.decode("utf-8-sig")
                except UnicodeDecodeError:
                    decoded_content = content.decode("gb18030")
                df = pd.read_csv(StringIO(decoded_content))
                
                dataset_name = _safe_dataset_name(filename, {})
                return {
                    "type": "single",
                    "filename": filename,
                    "dataset_name": dataset_name,
                    "df": df,
                }
            
            elif ext in ("xlsx", "xls"):
                excel_file = pd.ExcelFile(BytesIO(content))
                sheet_names = excel_file.sheet_names
                
                if len(sheet_names) == 1:
                    df = pd.read_excel(BytesIO(content))
                    dataset_name = _safe_dataset_name(filename, {})
                    return {
                        "type": "single",
                        "filename": filename,
                        "dataset_name": dataset_name,
                        "df": df,
                    }
                else:
                    sheets = []
                    for sheet_name in sheet_names:
                        df = pd.read_excel(BytesIO(content), sheet_name=sheet_name)
                        sheet_safe_name = re.sub(r"[^0-9a-zA-Z_]+", "_", sheet_name).strip("_") or f"sheet_{sheet_names.index(sheet_name)}"
                        sheet_dataset_name = _safe_dataset_name(f"{Path(filename).stem}_{sheet_safe_name}", {})
                        sheets.append({
                            "sheet_name": sheet_name,
                            "dataset_name": sheet_dataset_name,
                            "df": df,
                        })
                    return {
                        "type": "multi",
                        "filename": filename,
                        "sheets": sheets,
                    }
            else:
                return {"type": "error", "filename": filename, "error": f"不支持的文件格式: .{ext}"}
                
        except Exception as e:
            return {"type": "error", "filename": filename, "error": f"文件读取失败: {e}"}
    
    # 并行读取所有文件
    process_tasks = [process_file(file) for file in files]
    processed_results = await asyncio.gather(*process_tasks)
    
    # 获取所有数据集名称用于安全命名
    with session_lock:
        existing_names = set(session["datasets"].keys())
    
    # 收集所有要存储的数据集
    all_datasets = []
    primary_df = None
    primary_dataset_name = None
    
    for result in processed_results:
        if result["type"] == "error":
            errors.append({"filename": result["filename"], "error": result["error"]})
        elif result["type"] == "single":
            safe_name = _safe_dataset_name(result["filename"], existing_names)
            existing_names.add(safe_name)
            all_datasets.append({
                "name": safe_name,
                "filename": result["filename"],
                "df": result["df"],
            })
            if primary_df is None:
                primary_df = result["df"]
                primary_dataset_name = safe_name
        elif result["type"] == "multi":
            for sheet in result["sheets"]:
                safe_name = _safe_dataset_name(f"{Path(result['filename']).stem}_{sheet['sheet_name']}", existing_names)
                existing_names.add(safe_name)
                all_datasets.append({
                    "name": safe_name,
                    "filename": f"{result['filename']}::{sheet['sheet_name']}",
                    "df": sheet["df"],
                })
                if primary_df is None:
                    primary_df = sheet["df"]
                    primary_dataset_name = safe_name
    
    # 存储所有数据集
    with session_lock:
        for ds in all_datasets:
            session["datasets"][ds["name"]] = ds["df"]
            session["dataset_files"][ds["name"]] = ds["filename"]
        
        if not session.get("primary_dataset") and primary_dataset_name:
            session["primary_dataset"] = primary_dataset_name
        if all_datasets:
            session["dataset_filename"] = all_datasets[0]["filename"]
        
        # 并行生成所有描述
        desc_tasks = [
            _generate_dataset_description(session, ds["name"], ds["df"], ds["filename"], "")
            for ds in all_datasets
        ]
        await asyncio.gather(*desc_tasks)
        
        _refresh_dataset_description(session)
        persist_session(session_id, session)
        
        results = [{"name": ds["name"], "filename": ds["filename"], "shape": list(ds["df"].shape)} for ds in all_datasets]
    
    return {
        "results": results,
        "errors": errors,
        "datasets": _dataset_summaries(session),
        "description": session["description"],
    }


@app.post("/session/{session_id}/describe")
async def describe_dataset(session_id: str, description: str = Form("")):
    """Re-generate dataset description with user input."""
    session = get_session(session_id)
    if not session["datasets"]:
        raise HTTPException(400, "No dataset loaded.")
    
    session_lm = get_session_lm(session)
    name = session.get("primary_dataset") or list(session["datasets"].keys())[0]
    df = session["datasets"][name]
    
    try:
        with dspy.context(lm=session_lm):
            desc_agent = dspy.Predict(dataset_description_agent)
            buf = StringIO()
            df.info(buf=buf)
            result = await asyncio.wait_for(
                _run_sync(
                    desc_agent,
                    dataset=f"Shape: {df.shape}\n{buf.getvalue()}\n{df.head(5).to_string()}",
                    existing_description=description,
                ),
                timeout=DATASET_DESCRIPTION_TIMEOUT_SECONDS,
            )
            session["dataset_descriptions"][name] = result.description
    except Exception as e:
        logger.warning(f"描述生成失败: {e}")
        session["dataset_descriptions"][name] = description

    _refresh_dataset_description(session)
    persist_session(session_id, session)
    
    return {"description": session["description"]}


@app.get("/agents")
async def list_agents():
    return {
        "agents": [
            {"name": "preprocessing_agent", "display": "Preprocessing", "icon": "database", "desc": "Clean and prepare data with Pandas/NumPy.", "role": "executor"},
            {"name": "statistical_analytics_agent", "display": "Statistical Analysis", "icon": "bar-chart", "desc": "Run regression and statistical analysis.", "role": "executor"},
            {"name": "sk_learn_agent", "display": "Machine Learning", "icon": "brain", "desc": "Handle classification, regression, and clustering tasks.", "role": "executor"},
            {"name": "data_viz_agent", "display": "Visualization", "icon": "line-chart", "desc": "Create interactive charts with Plotly.", "role": "executor"},
            {"name": "chancellor_agent", "display": "Chancellor", "icon": "user-check", "desc": "Interpret user goals and refine tasks.", "role": "coordinator"},
            {"name": "commander_agent", "display": "Commander", "icon": "workflow", "desc": "Plan, split, and dispatch subtasks.", "role": "coordinator"},
            {"name": "censor_agent", "display": "Censor", "icon": "shield-check", "desc": "Review outputs and request rework when needed.", "role": "reviewer"},
        ]
    }


@app.post("/chat/{agent_name}")
async def chat_with_agent(session_id: str, agent_name: str, req: QueryRequest):
    """Chat with a specific agent (legacy endpoint, kept for compatibility)."""
    session = get_session(session_id)
    if not session["datasets"]:
        raise HTTPException(400, "No dataset loaded. Please upload a dataset first.")
    
    session_lm = get_session_lm(session)
    
    # Map agent name to signature
    agent_map = {
        "preprocessing_agent": preprocessing_agent,
        "statistical_analytics_agent": statistical_analytics_agent,
        "sk_learn_agent": sk_learn_agent,
        "data_viz_agent": data_viz_agent,
    }
    
    if agent_name not in agent_map:
        raise HTTPException(400, f"Unknown agent: {agent_name}. Allowed: {list(agent_map.keys())}")
    
    sig = agent_map[agent_name]
    
    try:
        with dspy.context(lm=session_lm):
            import functools
            cot = dspy.ChainOfThought(sig)
            loop = asyncio.get_running_loop()
            
            async def process_agent():
                response = await loop.run_in_executor(None, functools.partial(cot, **{
                    "goal": req.query, "dataset": session["description"], "plan_instructions": "",
                    **({"styling_index": " | ".join(STYLING_INSTRUCTIONS[:3])} if agent_name == "data_viz_agent" else {})
                }))

                result_dict = dict(response)

                formatted = format_response_to_markdown(
                    {agent_name: result_dict}, _execution_datasets(session)
                )

                return formatted
            
            formatted = await asyncio.wait_for(process_agent(), timeout=LEGACY_AGENT_TIMEOUT_SECONDS)
            
            session["chat_history"].append({"role": "user", "content": req.query})
            session["chat_history"].append({"role": "assistant", "content": formatted})
            
            return {"agent_name": agent_name, "query": req.query, "response": formatted}
    except asyncio.TimeoutError:
        raise HTTPException(504, "Request timed out. Please try a simpler query.")
    except Exception as e:
        logger.error(f"Agent 错误: {e}")
        raise HTTPException(500, f"Agent 错误: {str(e)}")


@app.post("/session/{session_id}/chat")
async def chat_with_qin_dynasty(session_id: str, req: QueryRequest):
    """Run orchestrated chat and stream SSE events back to the client."""
    session = get_session(session_id)
    session_lm = get_session_lm(session)

    # Each request owns an independent cancellation token and task state. A
    # session-level boolean is unsafe because starting a new turn would revive
    # an older stream by resetting the shared flag.
    previous_run = active_chat_runs.get(session_id)
    if previous_run:
        previous_run["cancel_event"].set()
        previous_run["task_state"].stop_task()

    run_id = uuid4().hex
    cancel_event = asyncio.Event()
    task_state = AgentTaskState()
    task_states[session_id] = task_state
    active_chat_runs[session_id] = {
        "run_id": run_id,
        "cancel_event": cancel_event,
        "task_state": task_state,
    }
    
    # Build retrievers
    from src.simple_retriever import SimpleRetriever
    retrievers = {
        "dataframe_index": session["description"],
        "style_index": SimpleRetriever(STYLING_INSTRUCTIONS),
    }
    
    # Create orchestrator
    orchestrator = qin_dynasty_orchestrator(retrievers=retrievers)

    def summarize_for_history(content) -> str:
        """Keep enough prior context for follow-up chat without storing large chart payloads."""
        if not isinstance(content, dict):
            return str(content)[:3000]
        if content.get("mode") == "chat":
            return str(content.get("response", ""))[:3000]
        if content.get("mode") == "report":
            return str(content.get("content", ""))[:5000]
        summaries = []
        for agent_name, result in content.items():
            if isinstance(result, dict) and result.get("summary"):
                summaries.append(f"{agent_name}: {result['summary']}")
        return "\n".join(summaries)[:5000]
    
    async def stream():
        assistant_history = ""
        orchestration = orchestrator.execute_user_query(
            query=req.query,
            session_lm=session_lm,
            task_state=task_state,
            datasets=_execution_datasets(session),
            chat_history=session["chat_history"],
            stop_flag=cancel_event.is_set,
        )
        try:
            while True:
                next_event_task = asyncio.create_task(anext(orchestration))
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, _ = await asyncio.wait(
                    {next_event_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if cancel_task in done:
                    next_event_task.cancel()
                    with suppress(asyncio.CancelledError, StopAsyncIteration):
                        await next_event_task
                    task_state.stop_task()
                    payload = {
                        "type": "stopped",
                        "content": "任务已被用户停止",
                        "task_state": task_state.get_state_snapshot(),
                    }
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                    return

                cancel_task.cancel()
                with suppress(asyncio.CancelledError):
                    await cancel_task
                try:
                    event = next_event_task.result()
                except StopAsyncIteration:
                    break

                agent_name, status, content = event

                if agent_name == "final":
                    if status in ("done", "success"):
                        assistant_history = summarize_for_history(content)
                    payload = {
                        "type": "final",
                        "content": content,
                        "status": "success" if status in ("done", "success") else status,
                        "task_state": task_state.get_state_snapshot(),
                    }
                    # 保存结构化的执行结果到 chat_history
                    if isinstance(content, dict):
                        chat_history_entry = {
                            "role": "assistant",
                            "mode": content.get("mode", "execute"),
                            "executor_results": {},
                        }
                        if content.get("mode") == "report":
                            chat_history_entry["report"] = content.get("content", "")
                        # 获取执行结果（可能是 content 本身或者是 content.executor_results）
                        executor_results = content.get("executor_results", content) if content.get("mode") != "report" else content.get("executor_results", {})
                        if content.get("mode") != "report" and not executor_results:
                            executor_results = content
                        # 保存执行智能体的完整结果
                        for agent_name_res, result in executor_results.items():
                            if isinstance(result, dict):
                                chat_history_entry["executor_results"][agent_name_res] = {
                                    "summary": result.get("summary", ""),
                                    "result": result.get("result", ""),
                                    "code": result.get("code", ""),
                                    "code_executed": result.get("code_executed", False),
                                }
                        session["chat_history"].append(chat_history_entry)
                else:
                    payload = {
                        "type": "agent_status",
                        "agent": agent_name,
                        "status": status,
                        "content": str(content),
                        "task_state": task_state.get_state_snapshot(),
                    }

                yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

            if cancel_event.is_set():
                return
            session["chat_history"].append({"role": "user", "content": req.query})
            if assistant_history:
                session["chat_history"].append({"role": "assistant", "content": assistant_history})
            persist_session(session_id, session)

        except asyncio.CancelledError:
            cancel_event.set()
            task_state.stop_task()
            raise
        except Exception as e:
            logger.error(f"秦朝官职编排错误: {e}")
            error_payload = {
                "type": "error",
                "content": f"处理出错: {str(e)}",
                "task_state": task_state.get_state_snapshot(),
            }
            yield "data: " + json.dumps(error_payload, ensure_ascii=False) + "\n\n"
        finally:
            with suppress(Exception):
                await orchestration.aclose()
            active_run = active_chat_runs.get(session_id)
            if active_run and active_run["run_id"] == run_id:
                active_chat_runs.pop(session_id, None)
    
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# -- Task state and agent status endpoints ------------------------------

@app.get("/session/{session_id}/task-state")
async def get_task_state_endpoint(session_id: str):
    """Get task state snapshot for the session."""
    task_state = get_task_state(session_id)
    return task_state.get_state_snapshot()


@app.get("/session/{session_id}/agents-status")
async def get_agents_status(session_id: str):
    """Get latest agent status, messages, and task history."""
    task_state = get_task_state(session_id)
    return {
        "agents": task_state.states,
        "messages": task_state.messages[-50:],
        "history": task_state.task_history[-100:],
    }


@app.post("/session/{session_id}/stop")
async def stop_chat(session_id: str):
    """Stop current chat task for this session."""
    get_session(session_id)
    active_run = active_chat_runs.get(session_id)
    if active_run:
        active_run["cancel_event"].set()
        active_run["task_state"].stop_task()
    else:
        get_task_state(session_id).stop_task()
    return {"status": "stopped", "message": "Chat task stopped."}


@app.post("/session/{session_id}/review")
async def submit_review(session_id: str, req: ReviewRequest):
    """Allow manual review feedback to be injected into task state."""
    task_state = get_task_state(session_id)

    review_result = {
        "approved": req.approved,
        "target": req.target,
        "comments": req.comments,
        "severity": req.severity,
        "reviewer": "human",
    }

    task_state.add_message(
        from_agent="human_reviewer",
        to_agent=req.target,
        content=json.dumps(review_result, ensure_ascii=False),
        message_type="review_result",
    )
    task_state.add_history(
        agent_name="human_reviewer",
        action=f"review {'approved' if req.approved else 'rejected'}: {req.comments}",
        result=review_result,
    )

    return {"status": "review_submitted", "approved": req.approved}


# Legacy planner endpoint (kept for compatibility)

@app.post("/chat-legacy")
async def chat_with_planner_legacy(session_id: str, req: QueryRequest):
    """Legacy planner endpoint (kept for compatibility)."""
    session = get_session(session_id)
    if not session["datasets"]:
        raise HTTPException(400, "No dataset loaded. Please upload a dataset first.")

    session_lm = get_session_lm(session)

    agent_desc = str([
        {"preprocessing_agent": "data preprocessing"},
        {"statistical_analytics_agent": "statistical analysis"},
        {"sk_learn_agent": "machine learning"},
        {"data_viz_agent": "data visualization"},
    ])

    from src.agents.agents import auto_analyst

    retrievers = {
        "dataframe_index": session["description"],
        "style_index": None,
    }

    try:
        ai_system = auto_analyst(agents=[], retrievers=retrievers)
    except Exception as e:
        logger.warning(f"Could not create auto_analyst: {e}")
        ai_system = None

    async def stream():
        try:
            with dspy.context(lm=session_lm):
                planner = planner_module()
                plan_response = await asyncio.wait_for(
                    planner.forward(
                        goal=req.query,
                        dataset=session["description"],
                        Agent_desc=agent_desc,
                    ),
                    timeout=LEGACY_PLANNER_TIMEOUT_SECONDS,
                )

            plan_desc = format_response_to_markdown(
                {"analytical_planner": plan_response}, _execution_datasets(session)
            )

            yield json.dumps({"type": "agent_status", "agent": "Analytical Planner", "content": plan_desc, "status": "success"}) + "\n"

            if ai_system:
                with dspy.context(lm=session_lm):
                    async for agent_name, inputs, response in ai_system.execute_plan(req.query, plan_response):
                        if agent_name in ("plan_not_found", "plan_not_formatted_correctly"):
                            yield json.dumps({"type": "error", "agent": "Planner", "content": f"**Error**: {agent_name}", "status": "error"}) + "\n"
                            return

                        formatted = format_response_to_markdown(
                            {agent_name: response}, _execution_datasets(session)
                        )
                        yield json.dumps({
                            "type": "agent_status",
                            "agent": agent_name.split("__")[0] if "__" in agent_name else agent_name,
                            "content": formatted,
                            "status": "success" if response else "error",
                        }) + "\n"

            session["chat_history"].append({"role": "user", "content": req.query})

        except asyncio.TimeoutError:
            yield json.dumps({"type": "error", "agent": "Planner", "content": "Request timed out.", "status": "error"}) + "\n"
        except Exception as e:
            logger.error(f"Planner stream error: {e}")
            yield json.dumps({"type": "error", "agent": "Planner", "content": f"Error: {str(e)}", "status": "error"}) + "\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Code execution & editing endpoints

@app.post("/session/{session_id}/execute-code")
async def execute_code(session_id: str, req: CodeExecuteRequest):
    """Execute code and return results."""
    session = get_session(session_id)
    if not session["datasets"]:
        raise HTTPException(400, "No dataset loaded.")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                execute_code_from_markdown,
                req.code,
                _execution_datasets(session),
                CODE_EXECUTION_TIMEOUT_SECONDS,
            ),
            timeout=CODE_EXECUTION_OUTER_TIMEOUT_SECONDS,
        )
        return {"status": "success" if execution_succeeded(result) else "error", "output": result}
    except Exception as e:
        return {"status": "error", "output": str(e)}


@app.post("/session/{session_id}/fix-code")
async def fix_code(session_id: str, req: CodeFixRequest):
    """Fix broken code using LLM."""
    session = get_session(session_id)
    session_lm = get_session_lm(session)

    try:
        with dspy.context(lm=session_lm):
            fixer = dspy.Predict(code_fix)
            result = await asyncio.wait_for(
                _run_sync(
                    fixer,
                    dataset_context=session["description"],
                    faulty_code=req.code,
                    error=req.error,
                ),
                timeout=HELPER_AGENT_TIMEOUT_SECONDS,
            )
            return {"fixed_code": result.fixed_code}
    except Exception as e:
        raise HTTPException(500, f"Code fix error: {str(e)}")


@app.post("/session/{session_id}/edit-code")
async def edit_code(session_id: str, req: CodeEditRequest):
    """Edit code using LLM."""
    session = get_session(session_id)
    session_lm = get_session_lm(session)

    try:
        with dspy.context(lm=session_lm):
            editor = dspy.Predict(code_edit)
            result = await asyncio.wait_for(
                _run_sync(
                    editor,
                    dataset_context=session["description"],
                    original_code=req.code,
                    user_prompt=req.prompt,
                ),
                timeout=HELPER_AGENT_TIMEOUT_SECONDS,
            )
            return {"edited_code": result.edited_code}
    except Exception as e:
        raise HTTPException(500, f"Code edit error: {str(e)}")


@app.post("/session/{session_id}/chat-name")
async def chat_name(session_id: str, req: QueryRequest):
    """Generate a short name for a chat query."""
    session = get_session(session_id)
    session_lm = get_session_lm(session)
    try:
        with dspy.context(lm=session_lm):
            namer = dspy.Predict(chat_history_name_agent)
            result = await asyncio.wait_for(
                _run_sync(namer, query=req.query),
                timeout=HELPER_AGENT_TIMEOUT_SECONDS,
            )
            return {"name": result.name}
    except Exception:
        return {"name": "New Chat"}


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    session = get_session(session_id)
    return {"history": session.get("chat_history", [])}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    task_states.pop(session_id, None)
    return {"status": "deleted"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host=host, port=port)
