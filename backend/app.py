"""
DataPilot Backend - AI Data Analyst
秦朝官职编排架构：秦始皇（用户） → 丞相 → 太尉 → 执行智能体 → 御史大夫
"""
import os
import json
import asyncio
import logging
from io import BytesIO, StringIO
from typing import Optional
from enum import Enum

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
)
from src.format_response import format_response_to_markdown, execute_code_from_markdown

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("datapilot")

# ── Styling instructions for visualization ───────────────────────────────
STYLING_INSTRUCTIONS = [
    str({"category": "line_charts", "description": "Trends over time", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
    str({"category": "bar_charts", "description": "Comparing categories", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
    str({"category": "scatter_charts", "description": "Relationships between variables", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
    str({"category": "pie_charts", "description": "Composition of whole", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
    str({"category": "histograms", "description": "Distribution of data", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
    str({"category": "heat_maps", "description": "Data density/intensity", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
    str({"category": "generic", "description": "General charts", "styling": {"template": "plotly_white", "default_size": {"height": 800, "width": 900}}}),
]

# ── Session state (in-memory, per-session) ──────────────────────────────
sessions: dict = {}
task_states: dict = {}  # session_id -> AgentTaskState


def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "datasets": {},        # name -> DataFrame
            "description": "",     # dataset description JSON
            "chat_history": [],    # list of {role, content}
            "model_config": {
                "provider": os.getenv("LLM_PROVIDER", "openai"),
                "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            },
            "stop_flag": False,    # 停止聊天标志
        }
    return sessions[session_id]


def get_task_state(session_id: str) -> AgentTaskState:
    """Get or create task state for a session."""
    if session_id not in task_states:
        task_states[session_id] = AgentTaskState()
    return task_states[session_id]


# ── DSPy LM helpers ─────────────────────────────────────────────────────
def build_lm(provider: str, model: str, api_key: Optional[str] = None, **kwargs):
    """Build a dspy.LM from provider/model/api_key."""
    provider_prefix = {
        "openai": "openai",
        "anthropic": "anthropic",
        "groq": "groq",
        "gemini": "gemini",
        "deepseek": "deepseek",
    }.get(provider.lower(), provider.lower())
    
    full_model = f"{provider_prefix}/{model}"
    
    lm_kwargs = {"model": full_model, "max_tokens": kwargs.get("max_tokens", 6000), "temperature": kwargs.get("temperature", 0.7)}
    if api_key:
        lm_kwargs["api_key"] = api_key
    
    return dspy.LM(**lm_kwargs)


def get_session_lm(session: dict) -> dspy.LM:
    cfg = session.get("model_config", {})
    provider = cfg.get("provider", os.getenv("LLM_PROVIDER", "openai"))
    model = cfg.get("model", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    api_key = cfg.get("api_key") or os.getenv(f"{provider.upper()}_API_KEY", "")
    return build_lm(provider, model, api_key)


# ── Default LM for startup ──────────────────────────────────────────────
default_provider = os.getenv("LLM_PROVIDER", "openai")
default_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
default_api_key = os.getenv("OPENAI_API_KEY", "")
try:
    default_lm = build_lm(default_provider, default_model, default_api_key)
    dspy.configure(lm=default_lm)
except Exception as e:
    logger.warning(f"无法配置默认 LLM: {e}，将使用会话级 LLM。")


# ── FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(title="DataPilot API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ──────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str


class ModelConfigRequest(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""


class CodeFixRequest(BaseModel):
    code: str
    error: str


class CodeEditRequest(BaseModel):
    code: str
    prompt: str


class ReviewRequest(BaseModel):
    """人工审查请求（可选，用于人工介入打回）。"""
    approved: bool
    target: str  # "丞相" | "太尉" | agent_name
    comments: str = ""
    severity: str = "medium"  # "low" | "medium" | "high"


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0", "architecture": "qin_dynasty"}


@app.post("/session")
async def create_session():
    import uuid
    session_id = str(uuid.uuid4())
    get_session(session_id)
    get_task_state(session_id)  # 同时初始化任务状态
    return {"session_id": session_id}


@app.get("/session/{session_id}/model")
async def get_model_config(session_id: str):
    session = get_session(session_id)
    cfg = session.get("model_config", {})
    # Don't expose full API key
    safe_cfg = {k: v for k, v in cfg.items() if k != "api_key"}
    safe_cfg["has_api_key"] = bool(cfg.get("api_key") or os.getenv(f"{cfg.get('provider', 'openai').upper()}_API_KEY"))
    return safe_cfg


@app.post("/session/{session_id}/model")
async def set_model_config(session_id: str, req: ModelConfigRequest):
    session = get_session(session_id)
    session["model_config"] = {
        "provider": req.provider,
        "model": req.model,
        "api_key": req.api_key,
    }
    return {"status": "updated", "provider": req.provider, "model": req.model}


@app.post("/session/{session_id}/upload")
async def upload_dataset(session_id: str, file: UploadFile = File(...), description: str = Form("")):
    session = get_session(session_id)
    filename = file.filename
    ext = filename.rsplit(".", 1)[-1].lower()
    
    try:
        content = await file.read()
        if ext == "csv":
            df = pd.read_csv(StringIO(content.decode("utf-8")))
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(BytesIO(content))
        else:
            raise HTTPException(400, f"不支持的文件格式: .{ext}")
    except Exception as e:
        raise HTTPException(400, f"文件读取失败: {e}")
    
    # Store dataset (overwrites if same name)
    dataset_name = "df"
    # Clear history when switching datasets so the LLM doesn't echo old answers
    if session["datasets"]:
        session["history"] = []
    session["datasets"][dataset_name] = df
    
    # Generate description
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
            result = desc_agent(
                dataset=dataset_view,
                existing_description=existing_desc,
            )
            session["description"] = result.description
    except Exception as e:
        logger.warning(f"描述生成失败: {e}，使用基本信息")
        buf = StringIO()
        df.info(buf=buf)
        session["description"] = buf.getvalue()
    
    return {
        "filename": filename,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "description": session["description"],
    }


@app.get("/session/{session_id}/dataset")
async def get_dataset_info(session_id: str):
    session = get_session(session_id)
    if not session["datasets"]:
        return {"loaded": False}
    name = list(session["datasets"].keys())[0]
    df = session["datasets"][name]
    return {
        "loaded": True,
        "name": name,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "description": session["description"],
    }


@app.post("/session/{session_id}/describe")
async def describe_dataset(session_id: str, description: str = Form("")):
    """Re-generate dataset description with user input."""
    session = get_session(session_id)
    if not session["datasets"]:
        raise HTTPException(400, "尚未加载数据集")
    
    session_lm = get_session_lm(session)
    name = list(session["datasets"].keys())[0]
    df = session["datasets"][name]
    
    try:
        with dspy.context(lm=session_lm):
            desc_agent = dspy.Predict(dataset_description_agent)
            buf = StringIO()
            df.info(buf=buf)
            result = desc_agent(
                dataset=f"Shape: {df.shape}\n{buf.getvalue()}\n{df.head(5).to_string()}",
                existing_description=description,
            )
            session["description"] = result.description
    except Exception as e:
        logger.warning(f"描述生成失败: {e}")
        session["description"] = description
    
    return {"description": session["description"]}


@app.get("/agents")
async def list_agents():
    return {
        "agents": [
            {"name": "preprocessing_agent", "display": "数据预处理", "icon": "🧹", "desc": "使用 Pandas/NumPy 清洗和准备数据", "role": "执行智能体"},
            {"name": "statistical_analytics_agent", "display": "统计分析", "icon": "📊", "desc": "回归分析、方差分析、统计建模", "role": "执行智能体"},
            {"name": "sk_learn_agent", "display": "机器学习", "icon": "🤖", "desc": "分类、回归、聚类等机器学习任务", "role": "执行智能体"},
            {"name": "data_viz_agent", "display": "数据可视化", "icon": "📈", "desc": "使用 Plotly 创建交互式图表", "role": "执行智能体"},
            {"name": "chancellor_agent", "display": "丞相", "icon": "👨💼", "desc": "接收用户指令，细化任务", "role": "丞相"},
            {"name": "commander_agent", "display": "太尉", "icon": "🎖️", "desc": "规划拆解，分发子任务", "role": "太尉"},
            {"name": "censor_agent", "display": "御史大夫", "icon": "🔍", "desc": "审查所有智能体工作，可打回", "role": "御史大夫"},
        ]
    }


@app.post("/chat/{agent_name}")
async def chat_with_agent(session_id: str, agent_name: str, req: QueryRequest):
    """Chat with a specific agent (legacy endpoint, kept for compatibility)."""
    session = get_session(session_id)
    if not session["datasets"]:
        raise HTTPException(400, "尚未加载数据集，请先上传数据文件。")
    
    session_lm = get_session_lm(session)
    
    # Map agent name to signature
    agent_map = {
        "preprocessing_agent": preprocessing_agent,
        "statistical_analytics_agent": statistical_analytics_agent,
        "sk_learn_agent": sk_learn_agent,
        "data_viz_agent": data_viz_agent,
    }
    
    if agent_name not in agent_map:
        raise HTTPException(400, f"未知 Agent: {agent_name}，可选: {list(agent_map.keys())}")
    
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
                    {agent_name: result_dict}, session["datasets"]
                )

                return formatted
            
            formatted = await asyncio.wait_for(process_agent(), timeout=180)
            
            session["chat_history"].append({"role": "user", "content": req.query})
            session["chat_history"].append({"role": "assistant", "content": formatted})
            
            return {"agent_name": agent_name, "query": req.query, "response": formatted}
    except asyncio.TimeoutError:
        raise HTTPException(504, "请求超时，请尝试更简单的查询。")
    except Exception as e:
        logger.error(f"Agent 错误: {e}")
        raise HTTPException(500, f"Agent 错误: {str(e)}")


@app.post("/session/{session_id}/chat")
async def chat_with_qin_dynasty(session_id: str, req: QueryRequest):
    """使用秦朝官职架构处理用户查询（SSE 流式返回）。
    
    SSE 事件格式：
    - {"type": "agent_status", "agent": "丞相", "status": "thinking"}
    - {"type": "message", "from": "丞相", "to": "太尉", "content": "...", "task_id": "..."}
    - {"type": "result", "agent": "太尉", "content": "...", "status": "success"}
    - {"type": "review_result", "agent": "御史大夫", "approved": false, "comments": "...", "task_id": "..."}
    - {"type": "final", "content": "...", "status": "success"}
    """
    session = get_session(session_id)
    if not session["datasets"]:
        raise HTTPException(400, "尚未加载数据集，请先上传数据文件。")
    
    session_lm = get_session_lm(session)
    task_state = get_task_state(session_id)
    
    # Build retrievers
    from src.simple_retriever import SimpleRetriever
    retrievers = {
        "dataframe_index": session["description"],
        "style_index": SimpleRetriever(STYLING_INSTRUCTIONS),
    }
    
    # Create orchestrator
    orchestrator = qin_dynasty_orchestrator(retrievers=retrievers)
    
    async def stream():
        try:
            # 使用新的秦朝官职编排架构
            async for event in orchestrator.execute_user_query(
                query=req.query,
                session_lm=session_lm,
                task_state=task_state,
                datasets=session["datasets"],
                stop_flag=lambda: session.get("stop_flag", False)
            ):
                # event 是 (agent_name, status, content) 元组
                agent_name, status, content = event
                
                # 构造 SSE 事件（必须前缀 "data: "）
                if agent_name == "final":
                    # 最终结果
                    yield f"data: {json.dumps({
                        'type': 'final',
                        'content': content,
                        'status': 'success',
                        'task_state': task_state.get_state_snapshot()
                    }, ensure_ascii=False)}\n\n"
                else:
                    # 智能体状态更新（不限制内容长度，确保完整显示代码和结果）
                    yield f"data: {json.dumps({
                        'type': 'agent_status',
                        'agent': agent_name,
                        'status': status,
                        'content': str(content),
                        'task_state': task_state.get_state_snapshot()
                    }, ensure_ascii=False)}\n\n"
            
            # 保存聊天历史
            session["chat_history"].append({"role": "user", "content": req.query})
            
        except Exception as e:
            logger.error(f"秦朝官职编排错误: {e}")
            yield f"data: {json.dumps({
                'type': 'error',
                'content': f'处理出错: {str(e)}',
                'task_state': task_state.get_state_snapshot()
            }, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── 任务状态和智能体消息端点 ────────────────────────────────────────────

@app.get("/session/{session_id}/task-state")
async def get_task_state_endpoint(session_id: str):
    """获取当前会话的任务状态和智能体消息流。"""
    task_state = get_task_state(session_id)
    return task_state.get_state_snapshot()


@app.get("/session/{session_id}/agents-status")
async def get_agents_status(session_id: str):
    """获取所有智能体的当前状态（用于前端实时展示）。"""
    task_state = get_task_state(session_id)
    return {
        "agents": task_state.states,
        "messages": task_state.messages[-50:],  # 最近 50 条消息
        "history": task_state.task_history[-100:]  # 最近 100 条历史
    }


@app.post("/session/{session_id}/stop")
async def stop_chat(session_id: str):
    """停止当前会话的聊天任务。"""
    session = get_session(session_id)
    session["stop_flag"] = True
    return {"status": "stopped", "message": "停止聊天任务"}


@app.post("/session/{session_id}/review")
async def submit_review(session_id: str, req: ReviewRequest):
    """人工介入审查（可选，用于人工打回）。"""
    task_state = get_task_state(session_id)
    
    review_result = {
        "approved": req.approved,
        "target": req.target,
        "comments": req.comments,
        "severity": req.severity,
        "reviewer": "human"
    }
    
    task_state.add_message(
        from_agent="人类审查员",
        to_agent=req.target,
        content=json.dumps(review_result, ensure_ascii=False),
        message_type="review_result"
    )
    task_state.add_history(
        agent_name="人类审查员",
        action=f"审查{'通过' if req.approved else '打回'}：{req.comments}",
        result=review_result,
    )
    
    return {"status": "review_submitted", "approved": req.approved}


# ── Legacy planner endpoint (保留兼容） ─────────────────────────────────

@app.post("/chat-legacy")
async def chat_with_planner_legacy(session_id: str, req: QueryRequest):
    """Legacy planner endpoint (保留兼容）。"""
    session = get_session(session_id)
    if not session["datasets"]:
        raise HTTPException(400, "尚未加载数据集，请先上传数据文件。")
    
    session_lm = get_session_lm(session)
    
    agent_desc = str([
        {"preprocessing_agent": "数据预处理"},
        {"statistical_analytics_agent": "统计分析"},
        {"sk_learn_agent": "机器学习"},
        {"data_viz_agent": "数据可视化"},
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
                    timeout=90,
                )
            
            plan_desc = format_response_to_markdown(
                {"analytical_planner": plan_response}, session["datasets"]
            )
            
            yield json.dumps({"agent": "Analytical Planner", "content": plan_desc, "status": "success"}) + "\n"
            
            if ai_system:
                with dspy.context(lm=session_lm):
                    async for agent_name, inputs, response in ai_system.execute_plan(req.query, plan_response):
                        if agent_name in ("plan_not_found", "plan_not_formatted_correctly"):
                            yield json.dumps({"agent": "Planner", "content": f"**Error**: {agent_name}", "status": "error"}) + "\n"
                            return
                        
                        formatted = format_response_to_markdown(
                            {agent_name: response}, session["datasets"]
                        )
                        yield json.dumps({
                            "agent": agent_name.split("__")[0] if "__" in agent_name else agent_name,
                            "content": formatted,
                            "status": "success" if response else "error",
                        }) + "\n"
            
            session["chat_history"].append({"role": "user", "content": req.query})
            
        except asyncio.TimeoutError:
            yield json.dumps({"agent": "Planner", "content": "请求超时。", "status": "error"}) + "\n"
        except Exception as e:
            logger.error(f"Planner stream error: {e}")
            yield json.dumps({"agent": "Planner", "content": f"错误: {str(e)}", "status": "error"}) + "\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Code execution & editing endpoints ───────────────────────────────────

@app.post("/session/{session_id}/execute-code")
async def execute_code(session_id: str, req: CodeFixRequest):
    """Execute code and return results."""
    session = get_session(session_id)
    if not session["datasets"]:
        raise HTTPException(400, "尚未加载数据集")
    
    try:
        result = execute_code_from_markdown(req.code, session["datasets"])
        return {"status": "success", "output": result}
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
            result = fixer(
                dataset_context=session["description"],
                faulty_code=req.code,
                error=req.error,
            )
            return {"fixed_code": result.fixed_code}
    except Exception as e:
        raise HTTPException(500, f"代码修复错误: {str(e)}")


@app.post("/session/{session_id}/edit-code")
async def edit_code(session_id: str, req: CodeEditRequest):
    """Edit code using LLM."""
    session = get_session(session_id)
    session_lm = get_session_lm(session)
    
    try:
        with dspy.context(lm=session_lm):
            editor = dspy.Predict(code_edit)
            result = editor(
                dataset_context=session["description"],
                original_code=req.code,
                user_prompt=req.prompt,
            )
            return {"edited_code": result.edited_code}
    except Exception as e:
        raise HTTPException(500, f"代码编辑错误: {str(e)}")


@app.post("/session/{session_id}/chat-name")
async def chat_name(session_id: str, req: QueryRequest):
    """Generate a short name for a chat query."""
    session = get_session(session_id)
    session_lm = get_session_lm(session)
    try:
        with dspy.context(lm=session_lm):
            namer = dspy.Predict(chat_history_name_agent)
            result = namer(query=req.query)
            return {"name": result.name}
    except Exception:
        return {"name": "新对话"}


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    session = get_session(session_id)
    return {"history": session.get("chat_history", [])}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    task_states.pop(session_id, None)  # 同时清理任务状态
    return {"status": "deleted"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host=host, port=port)
