"""
Agent definitions - Core DSPy Signatures for DataPilot.

秦朝官职编排架构：
- 丞相（chancellor_agent）：接收用户指令，细化任务
- 太尉（commander_agent）：规划拆解，分发子任务
- 4个执行智能体：独立执行具体任务
- 御史大夫（censor_agent）：审查所有智能体工作，可打回
"""
import dspy
import asyncio
import json
import logging
import functools
import contextvars
import types
import uuid
import re
import time

from src.runtime_config import (
    CENSOR_TIMEOUT_SECONDS,
    CHANCELLOR_TIMEOUT_SECONDS,
    CODE_EXECUTION_OUTER_TIMEOUT_SECONDS,
    CODE_EXECUTION_TIMEOUT_SECONDS,
    COMMANDER_TIMEOUT_SECONDS,
    EXECUTOR_AGENT_TIMEOUT_SECONDS,
)

logger = logging.getLogger("datapilot")


def current_timestamp_ms():
    """Return a wall-clock timestamp that frontend clients can display directly."""
    return time.time_ns() // 1_000_000


def parse_json_object(text, source="structured agent output"):
    """Extract the first JSON object while tolerating DSPy completion markers."""
    if isinstance(text, dict):
        return text

    raw_text = str(text or "").strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw_text):
        candidate = raw_text[match.start():]
        try:
            value, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        trailing_text = candidate[end:].strip()
        if trailing_text:
            logger.info(
                "Ignored trailing text after %s JSON object: %s",
                source,
                trailing_text[:160],
            )
        return value

    raise json.JSONDecodeError(
        f"No JSON object found in {source}",
        raw_text,
        0,
    )


# ── DSPy async compatibility ────────────────────────────────────────────
async def _run_sync(fn, *args, **kwargs):
    """Run a sync DSPy call in a thread so it doesn't block the event loop."""
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    return await loop.run_in_executor(None, functools.partial(context.run, fn, *args, **kwargs))


def _create_and_call_predict(signature, **kwargs):
    """Create and call DSPy Predict object in the same context to avoid contextvar issues."""
    predictor = dspy.Predict(signature)
    return predictor(**kwargs)


def _create_and_call_cot(signature, **kwargs):
    """Create and call DSPy ChainOfThought object in the same context to avoid contextvar issues."""
    cot = dspy.ChainOfThought(signature)
    return cot(**kwargs)


def asyncify_predict(signature):
    """Return an async callable for dspy.Predict(signature)."""
    async def call(**kwargs):
        return await _run_sync(_create_and_call_predict, signature, **kwargs)
    return call


def asyncify_cot(signature):
    """Return an async callable for dspy.ChainOfThought(signature)."""
    async def call(**kwargs):
        return await _run_sync(_create_and_call_cot, signature, **kwargs)
    return call


# ── Dataset Description Agent ─────────────────────────────────────────────
class dataset_description_agent(dspy.Signature):
    """Generate a structured dataset context/description from headers and sample data.
    Output a JSON-like description including:
    - Dataset name and description
    - Column names with type, description, preprocessing hints
    - Usage notes for analysis agents
    """
    dataset = dspy.InputField(desc="The dataset info including headers, sample data, null counts, and data types.")
    existing_description = dspy.InputField(desc="User-provided description to enhance.", default="")
    description = dspy.OutputField(desc="Comprehensive dataset context with business context and technical guidance for analysis agents.")


# ── Chat History Name Agent ──────────────────────────────────────────────
class chat_history_name_agent(dspy.Signature):
    """You are an agent that takes a query and returns a short name for the chat history."""
    query = dspy.InputField(desc="The query to make a name for")
    name = dspy.OutputField(desc="A name for the chat history (max 3 words)")


# ── 秦朝官职智能体 ─────────────────────────────────────────────────────

class chancellor_agent(dspy.Signature):
    """你是丞相，负责接收秦始皇（用户）的指令，理解其意图，并将任务细化为明确的可执行计划。

职责：
1. 理解用户的真实意图（可能是模糊的、口语化的）
2. 结合数据集信息，明确任务目标
3. 将任务细化为结构化的执行计划（包含子任务描述）
4. 指定每个子任务需要的执行智能体类型
5. 以清晰的中文输出细化后的任务

**重要要求**：
- 如果需要数据可视化，必须明确要求数据可视化智能体（data_viz_agent）使用Plotly库
- 如果需要数据可视化，必须强调数据可视化智能体必须使用Plotly，不能使用Matplotlib
- 如果需要数据可视化，数据可视化智能体的代码必须调用fig.show()或fig.show(renderer='json')

输出格式（JSON）：
{
  "task_id": "唯一任务ID",
  "user_goal": "用户的原始指令",
  "refined_goal": "细化后的任务描述",
  "subtasks": [
    {"agent": "preprocessing_agent", "instruction": "..."},
    {"agent": "data_viz_agent", "instruction": "...（必须明确要求使用Plotly）"}
  ]
}

注意：你只负责细化任务，不执行任何代码。
    """
    user_instruction = dspy.InputField(desc="秦始皇（用户）的原始指令")
    dataset_description = dspy.InputField(desc="数据集描述信息")
    conversation_history = dspy.InputField(desc="Recent conversation and task context")
    refined_task = dspy.OutputField(desc="细化后的结构化任务（JSON格式）")


class censor_agent(dspy.Signature):
    """你是御史大夫，负责审查所有智能体的工作输出。

职责：
1. 审查丞相的任务细化结果是否合理
2. 审查太尉的规划拆解是否完整
3. 审查各执行智能体生成的代码是否正确、有无错误
4. 如发现错误、遗漏或逻辑问题，打回并要求重做
5. 审查通过后，任务结果返回给用户

### 严格审查标准：
1. 代码完整性：检查代码是否完整（不能是空的或不完整的代码片段）
2. 代码可执行性：检查代码是否有语法错误或明显的逻辑问题
3. 结果有效性：检查执行结果是否有效（对于数据可视化，必须有图表输出）
4. 任务完成度：检查是否完成了用户要求的任务
5. **代码必须被执行**：检查每个执行智能体的"运行结果"是否为空，如果为空则必须打回
6. **执行状态标记**：检查每个智能体的"状态"是否标记为"完整"

### 打回条件（满足任一条件必须打回）：
- 代码为空或不完整
- 代码有明显的语法错误
- **执行智能体的运行结果为空**（这表明代码没有被执行）
- **执行智能体的状态标记为"不完整"或包含"⚠️代码未执行"**
- 数据可视化智能体没有生成图表（没有fig对象或没有调用fig.show()）
- 执行结果为空或错误
- 代码逻辑明显不符合任务要求
- 任何执行智能体的状态标记为"不完整"

打回格式（JSON）：
{
  "approved": false,
  "target": "丞相" | "太尉" | "执行智能体名称",
  "comments": "具体的问题描述和改进建议",
  "severity": "low" | "medium" | "high"
}

通过格式（JSON）：
{
  "approved": true,
  "summary": "审查通过，结果可信"
}
    """
    agent_name = dspy.InputField(desc="被审查的智能体名称")
    agent_output = dspy.InputField(desc="该智能体的输出内容（代码、摘要等）")
    task_context = dspy.InputField(desc="任务上下文（用户指令、数据集信息等）")
    review_result = dspy.OutputField(desc="审查结果（JSON格式，必须包含 approved 字段，false表示打回，true表示通过）")


class commander_agent(dspy.Signature):
    """你是太尉，负责接收丞相细化的任务，进行规划拆解，并分发给执行智能体。

职责：
1. 接收丞相的细化任务（JSON格式）
2. 将任务拆解为可独立执行的子任务序列
3. 确定子任务之间的依赖关系（哪些可以并行，哪些必须顺序执行）
4. 将子任务分发给对应的执行智能体
5. 收集执行结果，汇总后提交给御史大夫审查
6. 如收到御史大夫的打回，重新规划或重新分发

### 重要要求：
- 如果子任务包含数据可视化智能体（data_viz_agent），必须在其instruction中明确强调：
  * 必须使用Plotly库
  * 不能使用Matplotlib
  * 必须调用fig.show()或fig.show(renderer='json')来输出图表
  * 使用数据集描述和执行上下文中列出的真实变量名，不要自行创建通用数据框别名
  * 目标列必须读取`target_col`，不能假定始终为`price`

### 输出格式（必须严格遵循）：
```json
{
  "subtasks": [
    {"agent": "执行智能体名称", "instruction": "具体任务指令"},
    ...
  ]
}
```

### 重要说明：
- 输出必须是一个完整的、有效 JSON 对象
- subtasks 数组不能为空，至少包含一个子任务
- 每个子任务必须指定 agent 和 instruction 字段
- 可用的执行智能体：preprocessing_agent, statistical_analytics_agent, sk_learn_agent, data_viz_agent
- 不要输出任何其他格式的内容，只输出 JSON

注意：
- 每个执行智能体的上下文是独立的，不共享状态
- 你负责维护任务状态的跟踪
- 子任务结果按顺序汇总，最终生成完整报告
    """
    refined_task = dspy.InputField(desc="丞相细化的任务（JSON格式）")
    dataset_description = dspy.InputField(desc="数据集描述信息")
    execution_plan = dspy.OutputField(desc="执行计划，包含子任务序列（JSON格式，必须包含subtasks数组且不能为空）")


# ── Planner Agents (保留原规划器作为太尉的内部组件） ──────────────────

# Keep the core orchestration instructions readable and deterministic.
chancellor_agent.instructions = """
You are the chancellor and the first-turn router. Read the user's request, dataset
description, recent conversation history, and the complete compacted record of prior
agent activity. Return one JSON object only.

For ordinary conversation, follow-up questions about prior work, explanations, greetings,
or requests that do not require new data computation, return:
{"mode":"chat","response":"answer in the user's language","subtasks":[]}
Answer from the provided context. Do not dispatch executors just to restate or explain prior work.

For requests that require new data computation, cleaning, modeling, or visualization, return:
{"mode":"execute","task_id":"...","user_goal":"...","refined_goal":"...","subtasks":[...]}
Each execution subtask must contain agent and instruction. Use the fewest necessary executors:
preprocessing_agent, statistical_analytics_agent, sk_learn_agent, or data_viz_agent.
Choose data_viz_agent only when a chart is requested or materially useful. For a
visualization, require Plotly, the filename-derived dataset variables listed in the
dataset description, and fig.show(). Do not execute code.
"""

censor_agent.instructions = """
You are the censor. Review an execution attempt consistently and proportionately.
Return one JSON object only. Approve when the requested analysis is complete and the
reported execution result is valid. Reject only for an objective defect: missing required
work, empty output, runtime error, invalid code, or a requested visualization without a
rendered Plotly chart. Do not reject correct work for optional enhancements or stylistic
preferences. When rejecting, return approved=false, target, comments, and severity.
When approving, return approved=true and a short summary.
"""

commander_agent.instructions = """
You are the commander. Convert the refined task into the smallest executable JSON plan.
Return one JSON object only with a non-empty subtasks array. Each subtask must contain
agent and instruction. Valid executors are preprocessing_agent, statistical_analytics_agent,
sk_learn_agent, and data_viz_agent. Preserve required dependencies. Use data_viz_agent only
for visualization work; require Plotly and fig.show().
Uploaded datasets are already loaded under the filename-derived variables listed in the
dataset description; never ask an executor to read a CSV file. Tell executors to use those
exact variables. Preprocessing outputs must use a descriptive `<source>_cleaned` name.
Do not create or mention generic dataframe aliases. `target_col` is the actual target
column name, and ML outputs are `model`, `y_test`, and `y_pred`. Do not assume that the
target is always named `price`.
"""


class advanced_query_planner(dspy.Signature):
    """You are an advanced data analytics planner. Generate the most efficient plan using the fewest necessary agents to achieve the user's goal.

**Inputs**: Datasets, Agent descriptions, User-defined goal
**Responsibilities**:
1. Confirm the goal is achievable with the provided data and agents.
2. Use the smallest set of agents and variables.
3. For each agent, define: create (output variables), use (input variables), instruction (what to do).
4. Keep instructions precise and minimal.

### Output Format:
Example: 1 agent use
  goal: "Generate a bar plot showing sales by category"
Output:
  plan: data_viz_agent
  plan_instructions:
  {"data_viz_agent": {"create": ["sales_cleaned: DataFrame"], "use": ["sales: DataFrame"], "instruction": "Clean sales and generate a bar plot showing sales by category."}}

Example 3 agents:
  plan: preprocessing_agent -> statistical_analytics_agent -> data_viz_agent
  plan_instructions: (JSON with create/use/instruction per agent)

Respond in the user's language for all explanations but keep code, variable names, agent names in English.
    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system")
    Agent_desc = dspy.InputField(desc="The agents available in the system")
    goal = dspy.InputField(desc="The user defined goal")
    plan = dspy.OutputField(desc="The plan to achieve the goal", prefix='Plan:')
    plan_instructions = dspy.OutputField(desc="Detailed variable-level instructions per agent for the plan")


class basic_query_planner(dspy.Signature):
    """You are the basic query planner. You pick one agent to answer the user's goal.

Example: Visualize height and salary?
plan: data_viz_agent
plan_instructions: {"data_viz_agent": {"create": ["scatter_plot"], "use": ["employees"], "instruction": "Create scatter plot of height & salary using plotly"}}

Respond in the user's language for all explanations but keep code, variable names, agent names in English.
    """
    dataset = dspy.InputField(desc="Available datasets")
    Agent_desc = dspy.InputField(desc="Agents available")
    goal = dspy.InputField(desc="User defined goal")
    plan = dspy.OutputField(desc="The plan", prefix='Plan:')
    plan_instructions = dspy.OutputField(desc="Instructions for the agent")


class intermediate_query_planner(dspy.Signature):
    """You are an intermediate data analytics planner. You pick 1-2 agents.

Output format:
plan: Agent1->Agent2
plan_instructions: JSON with create/use/instruction per agent

Keep instructions minimal. Use no more than 2 agents unless completely necessary.

Respond in the user's language for all explanations but keep code, variable names, agent names in English.
    """
    dataset = dspy.InputField(desc="Available datasets")
    Agent_desc = dspy.InputField(desc="Agents available")
    goal = dspy.InputField(desc="User defined goal")
    plan = dspy.OutputField(desc="The plan", prefix='Plan:')
    plan_instructions = dspy.OutputField(desc="Instructions from the planner")


class planner_module(dspy.Module):
    """Routes queries to appropriate planner complexity level."""
    
    def __init__(self):
        self.planners = {
            "advanced": asyncify_predict(advanced_query_planner),
            "intermediate": asyncify_predict(intermediate_query_planner),
            "basic": asyncify_predict(basic_query_planner),
        }
        self.allocator = asyncify_predict(
            "user_query,dataset->exact_word_complexity:Literal['basic','intermediate','advanced','unrelated'],analysis_query:bool"
        )
    
    async def forward(self, goal, dataset, Agent_desc):
        if not Agent_desc or Agent_desc == "[]":
            return {
                "complexity": "no_agents",
                "plan": "no_agents",
                "plan_instructions": {"message": "No agents available."}
            }
        
        try:
            # Determine complexity
            try:
                complexity = await self.allocator(user_query=goal, dataset=str(dataset)[:2000])
                comp = complexity.exact_word_complexity.strip().lower()
            except Exception:
                comp = "basic"
            
            # If unrelated but analysis-related, downgrade to basic
            if comp == "unrelated":
                try:
                    if complexity.analysis_query:
                        comp = "basic"
                    else:
                        return {
                            "complexity": "unrelated",
                            "plan": "basic_qa_agent",
                            "plan_instructions": "Not a data-related query."
                        }
                except Exception:
                    comp = "basic"
            
            # Get plan
            planner = self.planners.get(comp, self.planners["basic"])
            plan = await planner(goal=goal, dataset=dataset, Agent_desc=Agent_desc)
            
            if not plan or not hasattr(plan, 'plan'):
                return {
                    "complexity": comp,
                    "plan": "error",
                    "plan_instructions": {"error": "Planning failed. Please try again."}
                }
            
            return {
                "complexity": comp,
                "plan": plan.plan,
                "plan_instructions": plan.plan_instructions,
            }
        except Exception as e:
            logger.error(f"Planner 错误: {e}")
            return {
                "complexity": "error",
                "plan": "error",
                "plan_instructions": {"error": str(e)}
            }


# ── Core Analysis Agents ────────────────────────────────────────────────

class preprocessing_agent(dspy.Signature):
    """You are a data preprocessing agent. You clean and prepare DataFrames using Pandas and NumPy.

### Your Responsibilities:
- If plan_instructions are provided, follow them. Otherwise, perform standard preprocessing.
- Handle missing values (impute numeric with median, categorical with mode)
- Detect and convert date columns to datetime
- Separate numeric and categorical columns
- Do NOT create fake data or modify DataFrame index
- Do NOT generate plots or visualizations
- Uploaded datasets are already loaded under the filename-derived variables listed in the execution context. Do NOT write code to load files.
- Keep each uploaded source variable unchanged. Copy the source before cleaning.
- Assign each cleaned DataFrame to a descriptive `<source>_cleaned` variable. Do NOT create generic dataframe aliases.
- Store the actual target column name in the top-level string variable `target_col`. If you create `log_price`, set `target_col = 'log_price'`; otherwise preserve the real target name.

### Output:
1. code: Python code for preprocessing (do NOT include dataset loading code)
2. summary: Brief explanation of what was done

Respond in the user's language for all summary but keep the code in English.
    """
    dataset = dspy.InputField(desc="Dataset info with filename-derived execution variables")
    goal = dspy.InputField(desc="User-defined goal for the analysis")
    plan_instructions = dspy.InputField(desc="Agent-level instructions (optional)", default="")
    code = dspy.OutputField(desc="Generated Python code for preprocessing")
    summary = dspy.OutputField(desc="Explanation of what was done and why")


class statistical_analytics_agent(dspy.Signature):
    """You are a statistical analytics agent. Perform statistical analysis using statsmodels.

### Guidelines:
- Handle strings as categorical variables using C(column) in formulas
- Always add constant with sm.add_constant()
- Convert X and y to float before fitting
- Handle missing values before modeling
- Do NOT generate visualizations
- Use print() for output
- Uploaded datasets and preprocessing outputs are already loaded under the exact variables listed in the execution context. Do NOT write code to load files.
- Prefer a descriptive `<source>_cleaned` variable when preprocessing output is available. Use the original filename-derived source variable when the unmodified data is required.
- Use the top-level `target_col` variable when the target name matters. Do not assume that it is always `price`.

### Output:
1. code: Python code for statistical modeling (do NOT include dataset loading code)
2. summary: Brief explanation of results

Respond in the user's language for all summary but keep the code in English.
    """
    dataset = dspy.InputField(desc="Dataset info with filename-derived and cleaned variables")
    goal = dspy.InputField(desc="User's statistical analysis goal")
    plan_instructions = dspy.InputField(desc="Instructions (optional)", default="")
    code = dspy.OutputField(desc="Python code for statistical modeling")
    summary = dspy.OutputField(desc="Concise summary of the analysis and key findings")


class sk_learn_agent(dspy.Signature):
    """You are a machine learning agent. Train and evaluate models using scikit-learn.

### Guidelines:
- Always split data into train/test sets
- Set random_state=42 for reproducibility
- Use print() for all outputs
- Do NOT generate visualizations
- Do NOT create variables not in plan_instructions
- Uploaded datasets and preprocessing outputs are already loaded under the exact variables listed in the execution context. Do NOT write code to load files.
- Prefer a descriptive `<source>_cleaned` variable when it is available. Read the target name from the top-level `target_col` variable; do not assume that it is always `price`.
- Keep reusable outputs in top-level variables. Name the trained estimator `model`, predictions `y_pred`, and test targets `y_test` so downstream visualization tasks can use them.

### Output:
1. code: Python code for ML pipeline (do NOT include dataset loading code)
2. summary: Brief explanation of model and results

Respond in the user's language for all summary but keep the code in English.
    """
    dataset = dspy.InputField(desc="Input dataset, often cleaned")
    goal = dspy.InputField(desc="User's ML goal")
    plan_instructions = dspy.InputField(desc="Instructions (optional)", default="")
    code = dspy.OutputField(desc="Scikit-learn based machine learning code")
    summary = dspy.OutputField(desc="Explanation of the ML approach and evaluation")


class data_viz_agent(dspy.Signature):
    """You are a data visualization agent. Create interactive visualizations using Plotly.

### Guidelines:
- If the selected DataFrame has more than 50000 rows, sample to 5000 rows first
- Each visualization must be a separate go.Figure() assigned to a variable named `fig`
- Apply update_layout with clean titles, axis labels, and proper formatting
- Every cartesian chart MUST define readable x-axis and y-axis titles with
  `fig.update_xaxes(title_text=...)` and `fig.update_yaxes(title_text=...)`.
- Inspect the actual plotted series before drawing. Do NOT hard-code arbitrary
  axis ranges. Let Plotly autorange from the real data unless the user
  explicitly requests a fixed range.
- For histograms and bar charts, keep the value/count axis anchored at zero
  using `fig.update_yaxes(rangemode='tozero')`.
- For highly skewed data with extreme outliers, keep the complete data visible
  by default. Only use a quantile-based zoom when the user requests it, and
  clearly state the chosen quantile range in the summary.
- Use `automargin=True` for both axes so labels and tick values are not clipped.
- Use low opacity (0.4-0.7) where appropriate
- Use distinct colors for different categories
- Use only one number format consistently (K, M, or comma-separated)
- Add trendlines only if explicitly requested
- Never include dataset or styling_index in output
- Uploaded datasets and preprocessing outputs are already loaded under the exact variables listed in the execution context. Do NOT write code to load files or invent generic dataframe aliases.
- Only use variables explicitly listed as available in the execution context. The canonical trained-estimator variable is `model`; do not invent aliases such as `rf_model`.
- Do not generate placeholder charts for missing upstream variables. If a required variable is unavailable, explain the missing dependency instead of creating an empty figure.

### CRITICAL - Chart Display:
- Each chart MUST call fig.show(renderer='json') at the end
- This converts the Plotly figure to interactive JSON that renders in the chat interface
- Do NOT use matplotlib, plt.show(), or IPython.display
- The figure variable MUST be named `fig`
- If you create multiple charts, each one must end with fig.show(renderer='json')

### Example:
```python
fig = go.Figure()
fig.add_trace(...)
fig.update_layout(title="Sales by Category")
fig.show(renderer='json')
```

### Output:
1. code: Plotly Python code for visualization (do NOT include dataset loading code)
2. summary: Brief description of what is being visualized

Respond in the user's language for all summary but keep the code in English.
    """
    goal = dspy.InputField(desc="User-defined chart goal")
    dataset = dspy.InputField(desc="Details of the dataframe and its columns")
    styling_index = dspy.InputField(desc="Instructions for plot styling")
    plan_instructions = dspy.InputField(desc="Instructions (optional)", default="")
    code = dspy.OutputField(desc="Plotly Python code")
    summary = dspy.OutputField(desc="Summary of what is being visualized")


# ── Helper Agents ────────────────────────────────────────────────────────

class code_combiner_agent(dspy.Signature):
    """You combine Python code from multiple agents into one executable script.
Fix any errors. Copy the selected filename-derived source variable before mutation. Add fig.show() for Plotly charts.
Double check column names and data types against dataset.

Respond in the user's language for all summary but keep the code in English.
    """
    dataset = dspy.InputField(desc="Dataset context for validation")
    agent_code_list = dspy.InputField(desc="List of code from each agent")
    refined_complete_code = dspy.OutputField(desc="Refined complete code")
    summary = dspy.OutputField(desc="4 bullet-point summary of integration")


class code_fix(dspy.Signature):
    """You fix broken Python code for data analytics.

1. Examine faulty_code and error message
2. Identify the exact cause
3. Modify only necessary parts using dataset_context
4. Preserve intended behavior
5. Ensure output is runnable and error-free

Strict: Don't modify working parts. Don't add explanations. Output only the fixed code.

Respond in the user's language for all summary but keep the code in English.
    """
    dataset_context = dspy.InputField(desc="Dataset context")
    faulty_code = dspy.InputField(desc="The broken code")
    error = dspy.InputField(desc="The error message")
    fixed_code = dspy.OutputField(desc="The corrected code")


class code_edit(dspy.Signature):
    """You edit existing data analytics code based on user requests.

1. Analyze original_code, user_prompt, and dataset_context
2. Modify only relevant parts
3. Leave unrelated code unchanged
4. Ensure changes maintain correctness

Respond in the user's language for all summary but keep the code in English.
    """
    dataset_context = dspy.InputField(desc="Dataset context")
    original_code = dspy.InputField(desc="Original code")
    user_prompt = dspy.InputField(desc="Desired change")
    edited_code = dspy.OutputField(desc="Updated code")


# ── Task State Management ────────────────────────────────────────────────

class AgentTaskState:
    """跟踪每个智能体的任务状态（内存，每个会话独立）。"""
    
    STATUS = ["idle", "thinking", "working", "reviewing", "done", "error", "stopped"]
    
    def __init__(self):
        self.states = {
            "chancellor_agent": {"status": "idle", "last_active": None, "current_task": None},
            "commander_agent": {"status": "idle", "last_active": None, "current_task": None},
            "censor_agent": {"status": "idle", "last_active": None, "current_task": None},
            "preprocessing_agent": {"status": "idle", "last_active": None, "current_task": None},
            "statistical_analytics_agent": {"status": "idle", "last_active": None, "current_task": None},
            "sk_learn_agent": {"status": "idle", "last_active": None, "current_task": None},
            "data_viz_agent": {"status": "idle", "last_active": None, "current_task": None},
        }
        self.messages = []
        self.task_history = []
    
    def set_status(self, agent_name, status, task_id=None):
        if agent_name in self.states:
            self.states[agent_name]["status"] = status
            self.states[agent_name]["last_active"] = current_timestamp_ms()
            if task_id:
                self.states[agent_name]["current_task"] = task_id

    def begin_task(self, task_id):
        """Reset visible state so a new turn does not inherit stale agent statuses."""
        for state in self.states.values():
            state["status"] = "idle"
            state["last_active"] = None
            state["current_task"] = task_id

    def stop_task(self):
        """Mark any in-flight agents as stopped for an accurate UI snapshot."""
        for state in self.states.values():
            if state["status"] not in {"idle", "done", "error"}:
                state["status"] = "stopped"
                state["last_active"] = current_timestamp_ms()
    
    def add_message(self, from_agent, to_agent, content, task_id=None, message_type="task"):
        msg = {
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "task_id": task_id,
            "type": message_type,
            "timestamp": current_timestamp_ms()
        }
        self.messages.append(msg)
        return msg
    
    def add_history(self, agent_name, action, result, task_id=None):
        entry = {
            "agent": agent_name,
            "action": action,
            "result": result,
            "task_id": task_id,
            "timestamp": current_timestamp_ms()
        }
        self.task_history.append(entry)
        return entry
    
    def get_state_snapshot(self):
        return {
            "states": dict(self.states),
            "messages": list(self.messages),
            "history": list(self.task_history)
        }


# ── New Orchestration: 秦朝官职编排 ─────────────────────────────────────

class qin_dynasty_orchestrator(dspy.Module):
    """秦朝官职编排器：丞相 → 太尉 → 执行智能体 → 御史大夫
    
    上下文隔离：
    - 丞相、太尉、执行智能体各自有独立的 dspy.context
    - 智能体之间只通过消息传递通信
    - 御史大夫可以查看所有智能体的工作输出
    """
    
    def __init__(self, retrievers):
        self.retrievers = retrievers
        self.dataset_description = retrievers.get("dataframe_index", "")
        
        # 初始化所有智能体
        self._chancellor_predict = asyncify_predict(chancellor_agent)
        self.chancellor = self._route_with_chancellor
        self.commander = asyncify_predict(commander_agent)
        self.censor = asyncify_predict(censor_agent)
        
        # 4个执行智能体（独立上下文）
        self.executors = {
            "preprocessing_agent": asyncify_cot(preprocessing_agent),
            "statistical_analytics_agent": asyncify_cot(statistical_analytics_agent),
            "sk_learn_agent": asyncify_cot(sk_learn_agent),
            "data_viz_agent": asyncify_cot(data_viz_agent),
        }
        
        # 执行智能体的输入字段
        self.executor_inputs = {
            "preprocessing_agent": {"goal", "dataset", "plan_instructions"},
            "statistical_analytics_agent": {"goal", "dataset", "plan_instructions"},
            "sk_learn_agent": {"goal", "dataset", "plan_instructions"},
            "data_viz_agent": {"goal", "dataset", "styling_index", "plan_instructions"},
        }
        
        # 辅助智能体
        self.code_fixer = asyncify_predict(code_fix)
        self.code_editor = asyncify_predict(code_edit)

    def _default_subtasks(self, query):
        return self._infer_subtasks(query)

    def _looks_like_execution_request(self, query):
        """Fast-path explicit analytics requests without spending an LLM routing call."""
        normalized = query.lower()
        follow_up_terms = (
            "刚才", "刚刚", "之前", "上一次", "你做了什么", "解释一下", "为什么",
            "什么是", "是什么", "怎么理解", "介绍一下", "不要执行", "不用执行",
            "不需要分析", "不要分析", "聊天", "对话", "what did", "previous",
            "what is", "explain", "just chat", "do not execute",
        )
        if any(term in normalized for term in follow_up_terms):
            return False
        execution_terms = (
            "预处理", "清洗", "缺失值", "异常值", "统计", "相关性", "描述性", "分析",
            "建模", "模型", "机器学习", "预测", "回归", "分类", "聚类", "可视化",
            "画图", "绘图", "图表", "plot", "chart", "graph", "visual", "analyze",
            "analysis", "clean", "preprocess", "model", "predict", "forecast", "regression",
            "classification", "cluster", "correlation", "statistics",
        )
        return any(term in normalized for term in execution_terms)

    def _infer_subtasks(self, query):
        """Build a conservative ordered fallback plan for explicit analytics requests."""
        normalized = query.lower()
        groups = [
            (
                "preprocessing_agent",
                ("预处理", "清洗", "缺失值", "异常值", "clean", "preprocess", "missing", "outlier"),
                "使用执行上下文中列出的真实数据集变量完成清洗和编码。保持上传变量不变，将清洗结果赋给描述性的 <source>_cleaned 变量，并设置实际目标列名 target_col。不要创建通用数据框别名。",
            ),
            (
                "statistical_analytics_agent",
                ("统计", "相关性", "描述性", "分析", "statistics", "correlation", "analysis", "analyze"),
                "使用执行上下文中列出的真实数据集变量和描述性的 <source>_cleaned 变量完成用户要求的统计分析。需要目标列时读取 target_col。",
            ),
            (
                "sk_learn_agent",
                ("建模", "模型", "机器学习", "预测", "回归", "分类", "聚类", "model", "predict", "forecast", "regression", "classification", "cluster"),
                "使用执行上下文中列出的描述性 <source>_cleaned 变量训练并评估合适的模型。读取 target_col，并输出 model、y_test、y_pred。",
            ),
            (
                "data_viz_agent",
                ("可视化", "画图", "绘图", "图表", "plot", "chart", "graph", "visual", "figure"),
                "使用 Plotly 基于现有上下文变量完成用户要求的可视化，每张图调用 fig.show(renderer='json')。不要生成占位图。",
            ),
        ]
        subtasks = [
            {"agent": agent, "instruction": instruction}
            for agent, terms, instruction in groups
            if any(term in normalized for term in terms)
        ]
        return subtasks or [{
            "agent": "statistical_analytics_agent",
            "instruction": "使用执行上下文中列出的真实数据集变量完成用户要求的数据分析。",
        }]

    async def _route_with_chancellor(self, **kwargs):
        """Use deterministic routing for explicit work and the LLM for conversation."""
        instruction = str(kwargs.get("user_instruction", ""))
        if self._looks_like_execution_request(instruction):
            display_instruction = instruction.partition(" | ts=")[0]
            task = {
                "mode": "execute",
                "user_goal": display_instruction,
                "refined_goal": display_instruction,
                "subtasks": self._infer_subtasks(display_instruction),
            }
            return types.SimpleNamespace(refined_task=json.dumps(task, ensure_ascii=False))
        return await self._chancellor_predict(**kwargs)

    def _normalize_subtasks(self, subtasks, query):
        """Validate planner output and merge duplicate assignments to the same executor."""
        if not isinstance(subtasks, list):
            return self._default_subtasks(query)

        merged = {}
        for subtask in subtasks:
            if not isinstance(subtask, dict):
                continue
            agent = subtask.get("agent", "")
            instruction = subtask.get("instruction", subtask.get("task", ""))
            if agent not in self.executors or not isinstance(instruction, str) or not instruction.strip():
                continue
            if agent in merged:
                existing = merged[agent]["instruction"]
                if instruction.strip() not in existing:
                    merged[agent]["instruction"] += f"\n\nAdditional requirement: {instruction.strip()}"
            else:
                merged[agent] = {"agent": agent, "instruction": instruction.strip()}

        return list(merged.values()) or self._default_subtasks(query)

    def _compact_display_text(self, text, fallback=""):
        """Remove timestamps and repeated LLM paragraphs from user-facing summaries."""
        value = str(text or fallback).partition(" | ts=")[0].strip()
        paragraphs = [
            part.strip()
            for part in re.split(r"\n+|(?<=[。！？!?])\s*", value)
            if part.strip()
        ]
        unique_paragraphs = []
        seen = set()
        for paragraph in paragraphs:
            normalized = re.sub(r"\s+", " ", paragraph)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_paragraphs.append(paragraph)
        return "\n".join(unique_paragraphs)[:1200]

    def _format_chancellor_summary(self, refined_task, query):
        """Show a concise responsibility-level plan without leaking repeated prompts."""
        agent_summaries = {
            "preprocessing_agent": ("数据预处理", "清洗、编码并发布统一的数据集变量。"),
            "statistical_analytics_agent": ("统计分析", "完成描述性统计、相关性或用户指定的统计分析。"),
            "sk_learn_agent": ("机器学习", "训练并评估模型，输出可供下游使用的预测结果。"),
            "data_viz_agent": ("数据可视化", "使用 Plotly 展示需要的分析结果。"),
        }
        lines = []
        seen_agents = set()
        for subtask in refined_task.get("subtasks", []):
            agent = subtask.get("agent", "")
            if agent in seen_agents or agent not in agent_summaries:
                continue
            seen_agents.add(agent)
            label, summary = agent_summaries[agent]
            lines.append(f"{len(lines) + 1}. **{label}**：{summary}")

        goal = self._compact_display_text(refined_task.get("refined_goal"), query)
        task_lines = "\n".join(lines) or "1. **统计分析**：基于现有数据完成用户要求的分析。"
        return f"## 丞相任务细化结果\n\n**任务目标**：{goal}\n\n**拟调用智能体**：\n{task_lines}"

    def _get_styling_index(self, query):
        retriever = self.retrievers.get("style_index")
        if hasattr(retriever, "retrieve"):
            return " | ".join(retriever.retrieve(query, k=3))
        return str(retriever or "")

    def _describe_execution_context(self, variables):
        """Summarize variables exported by prior executor code."""
        items = []
        for name, value in sorted(variables.items()):
            if hasattr(value, "columns"):
                columns = ", ".join(str(column) for column in list(value.columns)[:30])
                if len(value.columns) > 30:
                    columns += ", ..."
                items.append(f"{name}: {type(value).__name__} columns=[{columns}]")
            elif name == "target_col":
                items.append(f"{name}: {value!r}")
            else:
                items.append(f"{name}: {type(value).__name__}")
        return "Available execution variables: " + ", ".join(items)

    def _normalize_preprocessing_context(self, execution_context, source_datasets):
        """Keep uploaded variables immutable and publish descriptive cleaned names."""
        legacy_aliases = ("df", "raw_df", "df_clean", "df_cleaned")
        cleaned_name = ""
        cleaned_df = None

        for name, value in execution_context.items():
            if name not in legacy_aliases and name.endswith("_cleaned") and hasattr(value, "columns"):
                cleaned_name = name
                cleaned_df = value
                break

        if cleaned_df is None:
            for alias in ("df_cleaned", "df_clean", "df"):
                value = execution_context.get(alias)
                if hasattr(value, "columns"):
                    cleaned_df = value
                    break

        if cleaned_df is None:
            for name, original in source_datasets.items():
                value = execution_context.get(name)
                if not hasattr(value, "columns"):
                    continue
                try:
                    changed = not value.equals(original)
                except Exception:
                    changed = True
                if changed:
                    cleaned_df = value
                    break

        if cleaned_df is None and source_datasets:
            cleaned_df = next(iter(source_datasets.values())).copy()

        if cleaned_df is not None and not cleaned_name:
            source_name = next(iter(source_datasets), "dataset")
            cleaned_name = f"{source_name}_cleaned"
            suffix = 2
            while cleaned_name in execution_context:
                cleaned_name = f"{source_name}_cleaned_{suffix}"
                suffix += 1
            execution_context[cleaned_name] = cleaned_df

        for name, original in source_datasets.items():
            execution_context[name] = original.copy()
        for alias in legacy_aliases:
            execution_context.pop(alias, None)

        if cleaned_df is None:
            return

        target_col = execution_context.get("target_col")
        if not isinstance(target_col, str) or target_col not in cleaned_df.columns:
            if "log_price" in cleaned_df.columns:
                target_col = "log_price"
            elif "price" in cleaned_df.columns:
                target_col = "price"
            else:
                target_col = ""
        execution_context["target_col"] = target_col

    def _missing_visualization_dependencies(self, subtasks, index, execution_context):
        """Block model-dependent charts when an upstream ML step did not publish outputs."""
        prior_agents = [item.get("agent", "") for item in subtasks[:index]]
        if "sk_learn_agent" not in prior_agents:
            return []
        required = ("model", "y_test", "y_pred")
        return [name for name in required if name not in execution_context]

    def _retry_start_index(self, subtasks, failed_agents, review):
        """Return the earliest failed executor index so only it and dependents rerun."""
        indexes = {
            subtask.get("agent", ""): index
            for index, subtask in enumerate(subtasks)
        }
        candidates = [indexes[name] for name in failed_agents if name in indexes]
        target = str(review.get("target", ""))
        if target in indexes:
            candidates.append(indexes[target])
        return min(candidates) if candidates else 0

    def _summarize_failed_results(self, executor_results):
        """Keep final SSE errors small and useful instead of returning chart payloads."""
        failures = {}
        for agent_name, result in executor_results.items():
            if not isinstance(result, dict):
                continue
            if result.get("code_executed", False):
                continue
            failures[agent_name] = str(
                result.get("execution_error") or result.get("result") or "No successful execution result."
            )[:1000]
        return failures

    def _summarize_result_for_review(self, result):
        """Keep chart payloads out of the censor prompt while preserving render evidence."""
        text = str(result or "")
        chart_count = text.count("<<<PLOTLY_JSON>>>")
        text = re.sub(
            r"<<<PLOTLY_JSON>>>.*?<<<END_PLOTLY_JSON>>>",
            "[Rendered interactive Plotly chart]",
            text,
            flags=re.DOTALL,
        )
        if chart_count:
            text += f"\n[Rendered Plotly chart count: {chart_count}]"
        return text[:5000]

    def _build_conversation_context(self, chat_history, task_state):
        """Give the chancellor every agent record while compacting bulky payloads."""
        def compact(value, limit):
            text = re.sub(
                r"<<<PLOTLY_JSON>>>.*?<<<END_PLOTLY_JSON>>>",
                "[Rendered interactive Plotly chart]",
                str(value or ""),
                flags=re.DOTALL,
            )
            return text[:limit]

        recent_chat = [
            {"role": item.get("role", ""), "content": str(item.get("content", ""))[:1000]}
            for item in (chat_history or [])[-6:]
        ]
        all_tasks = [
            {
                "agent": item.get("agent", ""),
                "action": item.get("action", ""),
                "result": compact(item.get("result", ""), 1000),
                "task_id": item.get("task_id"),
            }
            for item in task_state.task_history
        ]
        all_agent_messages = [
            {
                "from": item.get("from", ""),
                "to": item.get("to", ""),
                "type": item.get("type", ""),
                "content": compact(item.get("content", ""), 1000),
                "task_id": item.get("task_id"),
            }
            for item in task_state.messages
        ]
        context = json.dumps(
            {
                "recent_conversation": recent_chat,
                "agent_states": task_state.states,
                "all_task_history": all_tasks,
                "all_agent_messages": all_agent_messages,
            },
            ensure_ascii=False,
        )
        return context

    def _get_routing_dataset_description(self):
        """Keep routing compact while ensuring every uploaded dataset is visible."""
        description = str(self.dataset_description)
        details_marker = "\n\n--- DATASET DETAILS ---\n\n"
        if details_marker not in description:
            return description[:12000]
        dataset_index, details = description.split(details_marker, 1)
        details_budget = max(0, 12000 - len(dataset_index) - len(details_marker))
        return dataset_index + details_marker + details[:details_budget]
    
    async def execute_user_query(self, query, session_lm, task_state: AgentTaskState, datasets: dict, chat_history=None, stop_flag=None):
        """执行用户查询的完整流程（SSE 生成器）。
        
        流程：
        1. 用户（秦始皇）发送指令
        2. 丞相接收并细化任务
        3. 太尉规划拆解子任务
        4. 太尉分发给执行智能体
        5. 执行智能体执行（独立上下文）
        6. 太尉汇总结果
        7. 御史大夫审查
        8. 通过则返回用户，不通过则打回
        
        stop_flag: 可选的回调节函数，如果返回True则停止执行
        """
        # 检查是否停止
        def should_stop():
            if callable(stop_flag):
                stopped = stop_flag()
                if stopped:
                    task_state.stop_task()
                return stopped
            return False
        
        if should_stop():
            yield ("system", "stopped", "任务已停止")
            return
        # 添加时间戳到用户查询
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        query_with_timestamp = f"{query} | ts={timestamp}"
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        conversation_context = self._build_conversation_context(chat_history, task_state)
        task_state.begin_task(task_id)
        
        task_state.set_status("chancellor_agent", "thinking", task_id)
        yield ("chancellor_agent", "thinking", "正在理解您的指令并判断是否需要分发任务...")
        
        try:
            with dspy.context(lm=session_lm):
                chancellor_result = await asyncio.wait_for(
                    self.chancellor(
                        user_instruction=query_with_timestamp,  # 使用带时间戳的查询
                        dataset_description=self._get_routing_dataset_description(),
                        conversation_history=conversation_context,
                    ),
                    timeout=CHANCELLOR_TIMEOUT_SECONDS
                )
            refined_task_str = chancellor_result.refined_task
            
            try:
                refined_task = parse_json_object(refined_task_str, "chancellor")
                if not isinstance(refined_task, dict):
                    raise TypeError("chancellor output must be a JSON object")
                if str(refined_task.get("mode", "execute")).lower() == "chat":
                    direct_response = str(refined_task.get("response", "")).strip()
                    if not direct_response:
                        raise TypeError("chat response must not be empty")
                    task_state.set_status("chancellor_agent", "done", task_id)
                    task_state.add_message("秦始皇", "chancellor_agent", query, task_id)
                    task_state.add_message("chancellor_agent", "秦始皇", direct_response, task_id, "direct_response")
                    task_state.add_history("chancellor_agent", "直接对话回复", direct_response, task_id)
                    yield ("chancellor_agent", "done", direct_response)
                    yield ("final", "done", {"mode": "chat", "response": direct_response})
                    return
                # 确保 subtasks 存在，如果不存在则创建一个
                if 'subtasks' not in refined_task:
                    refined_task['subtasks'] = []
                # 如果 subtasks 为空，添加一个默认任务
                if len(refined_task.get('subtasks', [])) == 0:
                    logger.warning(f"丞相生成的 subtasks 为空，添加默认任务")
                    refined_task['subtasks'] = self._default_subtasks(query_with_timestamp)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"丞相返回的细化任务解析失败: {e}, 原始输出: {refined_task_str[:500]}")
                refined_task = {
                    "task_id": task_id,
                    "user_goal": query_with_timestamp,  # 使用带时间戳的查询
                    "refined_goal": refined_task_str,
                    "subtasks": self._default_subtasks(query_with_timestamp)
                }

            refined_task["subtasks"] = self._normalize_subtasks(refined_task.get("subtasks"), query_with_timestamp)
            refined_task_str = json.dumps(refined_task, ensure_ascii=False)
            
            task_state.set_status("chancellor_agent", "done", task_id)
            task_state.add_message("秦始皇", "chancellor_agent", query_with_timestamp, task_id)  # 使用带时间戳的查询
            task_state.add_message("chancellor_agent", "commander_agent", refined_task_str, task_id)
            task_state.add_history("chancellor_agent", "任务细化完成", refined_task_str, task_id)
            
            # Show a compact responsibility-level summary. The full task JSON remains
            # internal for the commander and executors.
            yield ("chancellor_agent", "done", self._format_chancellor_summary(refined_task, query))
            
            # 检查是否停止
            if should_stop():
                yield ("system", "stopped", "任务已被用户停止")
                return

            if not datasets:
                task_state.set_status("chancellor_agent", "error", task_id)
                yield ("chancellor_agent", "error", "该请求需要执行数据分析，请先上传数据集。")
                yield ("final", "error", "该请求需要执行数据分析，请先上传数据集。")
                return
            
        except asyncio.TimeoutError:
            task_state.set_status("chancellor_agent", "error", task_id)
            yield ("chancellor_agent", "error", "丞相处理超时")
            yield ("final", "error", "丞相超时，任务未能完成")
            return
        except Exception as e:
            task_state.set_status("chancellor_agent", "error", task_id)
            logger.error(f"丞相错误: {e}")
            yield ("chancellor_agent", "error", f"丞相处理出错：{str(e)}")
            yield ("final", "error", f"丞相出错：{str(e)}")
            return
        
        task_state.set_status("commander_agent", "thinking", task_id)
        yield ("commander_agent", "thinking", "正在规划执行计划...")
        
        try:
            commander_fallback_reason = ""
            try:
                with dspy.context(lm=session_lm):
                    commander_result = await asyncio.wait_for(
                        self.commander(
                            refined_task=refined_task_str,
                            dataset_description=self._get_routing_dataset_description(),
                        ),
                        timeout=COMMANDER_TIMEOUT_SECONDS,
                    )
                execution_plan_str = commander_result.execution_plan
            except asyncio.TimeoutError:
                commander_fallback_reason = "太尉规划超时，已使用丞相的子任务继续执行。"
                logger.warning(commander_fallback_reason)
                execution_plan_str = json.dumps(
                    {"subtasks": refined_task.get("subtasks", [])},
                    ensure_ascii=False,
                )
            except Exception as e:
                commander_fallback_reason = f"太尉规划异常，已使用丞相的子任务继续执行：{str(e)}"
                logger.warning(commander_fallback_reason)
                execution_plan_str = json.dumps(
                    {"subtasks": refined_task.get("subtasks", [])},
                    ensure_ascii=False,
                )
            
            try:
                execution_plan = parse_json_object(execution_plan_str, "commander")
                if not isinstance(execution_plan, dict):
                    raise TypeError("commander output must be a JSON object")
                # 确保 subtasks 不为空，如果为空则使用丞相的 subtasks
                chancellor_subtasks = refined_task.get("subtasks", [])
                if not execution_plan.get('subtasks') or len(execution_plan.get('subtasks', [])) == 0:
                    logger.warning(f"太尉生成的 subtasks 为空，使用丞相的 subtasks")
                    execution_plan['subtasks'] = chancellor_subtasks
                
                # 如果太尉和丞相的subtasks都为空，添加一个默认任务
                if not execution_plan.get('subtasks') or len(execution_plan.get('subtasks', [])) == 0:
                    logger.warning(f"太尉和丞相的 subtasks 都为空，使用默认任务")
                    execution_plan['subtasks'] = self._default_subtasks(query_with_timestamp)
                    
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"太尉返回的执行计划解析失败: {e}, 原始输出: {execution_plan_str[:500]}")
                chancellor_subtasks = refined_task.get("subtasks", [])
                execution_plan = {"subtasks": chancellor_subtasks or self._default_subtasks(query_with_timestamp)}

            execution_plan["subtasks"] = self._normalize_subtasks(execution_plan.get("subtasks"), query_with_timestamp)
            execution_plan_str = json.dumps(execution_plan, ensure_ascii=False)
            
            task_state.set_status("commander_agent", "working", task_id)
            task_state.add_message("commander_agent", "执行智能体", execution_plan_str, task_id)
            task_state.add_history("commander_agent", "规划拆解完成", execution_plan_str, task_id)
            if commander_fallback_reason:
                yield ("commander_agent", "working", commander_fallback_reason)
            
            # 显示完整的执行计划和子任务内容
            subtasks_list = execution_plan.get('subtasks', [])
            plan_details = f"## 太尉执行计划\n\n**总任务数**: {len(subtasks_list)}\n\n**子任务详情**:\n\n"
            for i, subtask in enumerate(subtasks_list):
                agent = subtask.get("agent", "")
                instruction = subtask.get("instruction", subtask.get("task", ""))
                plan_details += f"### 子任务 {i+1}: {agent}\n\n**任务指令**:\n{instruction}\n\n"
            
            yield ("commander_agent", "working", plan_details)
            
            # 最终检查：确保至少有一个子任务
            if len(subtasks_list) == 0:
                logger.error(f"严重错误：太尉和丞相的 subtasks 都为空，但保护逻辑未生效")
                yield ("system", "error", "任务规划失败：无法生成有效的子任务")
                yield ("final", "error", "任务规划失败")
                return
            
            # 检查是否停止
            if should_stop():
                yield ("system", "stopped", "任务已被用户停止")
                return
            
        except asyncio.TimeoutError:
            task_state.set_status("commander_agent", "error", task_id)
            yield ("commander_agent", "error", "太尉规划超时")
            yield ("final", "error", "太尉规划超时，任务未能完成")
            return
        except Exception as e:
            task_state.set_status("commander_agent", "error", task_id)
            logger.error(f"太尉错误: {e}")
            yield ("commander_agent", "error", f"太尉规划出错：{str(e)}")
            yield ("final", "error", f"太尉出错：{str(e)}")
            return
        
        
        # ── 执行智能体 + 御史大夫审查循环 ───────────────
        MAX_ATTEMPTS = 3
        censor_feedback = ""
        executor_results = {}
        approved = False
        subtasks = execution_plan.get("subtasks", []) or refined_task.get("subtasks", [])
        base_execution_context = dict(datasets)
        execution_context = dict(base_execution_context)
        context_snapshots = {-1: dict(execution_context)}
        detail_messages = {}
        retry_start_index = 0

        for attempt in range(1, MAX_ATTEMPTS + 1):
            if retry_start_index == 0:
                execution_context = dict(base_execution_context)
                executor_results = {}
                detail_messages = {}
                context_snapshots = {-1: dict(execution_context)}
            else:
                execution_context = dict(context_snapshots.get(retry_start_index - 1, base_execution_context))
                for stale_index in range(retry_start_index, len(subtasks)):
                    stale_agent = subtasks[stale_index].get("agent", "")
                    executor_results.pop(stale_agent, None)
                    detail_messages.pop(stale_agent, None)
                    context_snapshots.pop(stale_index, None)

            attempt_messages = []
            upstream_failed = False

            # ── 执行每个子任务 ──
            for i, subtask in enumerate(subtasks):
                if i < retry_start_index:
                    continue

                agent_name = subtask.get("agent", "")
                instruction = subtask.get("instruction", subtask.get("task", ""))
                # Append censor feedback if retrying
                if censor_feedback:
                    instruction += f"\n\n请根据御史大夫的反馈改进代码：{censor_feedback}"

                if agent_name not in self.executors:
                    yield (agent_name, "error", f"未找到执行智能体: {agent_name}")
                    upstream_failed = True
                    continue

                if upstream_failed:
                    skip_reason = "Skipped because an upstream executor failed. Fix the earlier failure first."
                    executor_results[agent_name] = {
                        "summary": skip_reason,
                        "result": skip_reason,
                        "execution_error": skip_reason,
                        "code_executed": False,
                    }
                    task_state.set_status(agent_name, "error", task_id)
                    yield (agent_name, "error", skip_reason)
                    continue

                missing_dependencies = self._missing_visualization_dependencies(subtasks, i, execution_context)
                if agent_name == "data_viz_agent" and missing_dependencies:
                    skip_reason = (
                        "Skipped visualization because upstream ML outputs are missing: "
                        + ", ".join(missing_dependencies)
                    )
                    executor_results[agent_name] = {
                        "summary": skip_reason,
                        "result": skip_reason,
                        "execution_error": skip_reason,
                        "code_executed": False,
                    }
                    task_state.set_status(agent_name, "error", task_id)
                    yield (agent_name, "error", skip_reason)
                    upstream_failed = True
                    continue

                yield ("commander_agent", "working", f"分发子任务 {i+1}/{len(subtasks)} 给 {agent_name}")

                task_state.set_status(agent_name, "working", task_id)
                yield (agent_name, "working", f"正在执行: {instruction[:80]}...")

                try:
                    # 1. 智能体生成代码阶段 - 需要 DSPy 上下文
                    with dspy.context(lm=session_lm):
                        # 构建plan_instructions，确保不重复添加反馈
                        plan_instructions = json.dumps(subtask, ensure_ascii=False)
                        if censor_feedback:
                            plan_instructions += f"\n\n御史大夫反馈: {censor_feedback}"
                        
                        inputs = {
                            "goal": instruction,
                            "dataset": f"{self.dataset_description}\n\n{self._describe_execution_context(execution_context)}",
                            "plan_instructions": plan_instructions,
                            "styling_index": self._get_styling_index(instruction),
                        }

                        required_keys = self.executor_inputs.get(agent_name, set())
                        filtered_inputs = {k: v for k, v in inputs.items() if k in required_keys}

                        result = await asyncio.wait_for(
                            self.executors[agent_name](**filtered_inputs),
                            timeout=EXECUTOR_AGENT_TIMEOUT_SECONDS
                        )
                        
                        # 调试：记录智能体返回的完整结果
                        logger.info(f"{agent_name} 返回结果类型: {type(result)}, 内容: {str(result)[:500]}")
                        result_dict = dict(result)
                        executor_results[agent_name] = result_dict
                        
                        # 调试：记录转换后的字典
                        logger.info(f"{agent_name} 转换后的字典 keys: {list(result_dict.keys())}")
                    
                    # 2. 代码执行阶段 - 移出 DSPy 上下文，避免 contextvars 传递到子进程
                    exec_result = ""
                    code_to_exec = result_dict.get('code', '')
                    code_executed_successfully = False
                    logger.info(f"{agent_name} 提取的代码: '{code_to_exec[:200]}...', datasets长度: {len(datasets) if datasets else 0}")
                    
                    if code_to_exec and datasets is not None and len(datasets) > 0:
                        try:
                            from src.format_response import execute_code_with_state, execution_succeeded
                            logger.info(f"{agent_name} 开始执行代码, context keys: {list(execution_context.keys())}")
                            logger.info(f"{agent_name} 代码内容前500字符: {code_to_exec[:500]}")
                            # 直接调用，不使用 _run_sync，因为代码执行使用子进程，不需要 contextvars
                            exec_result, updated_context = await asyncio.wait_for(
                                asyncio.to_thread(
                                    execute_code_with_state,
                                    code_to_exec,
                                    execution_context,
                                    CODE_EXECUTION_TIMEOUT_SECONDS,
                                ),
                                timeout=CODE_EXECUTION_OUTER_TIMEOUT_SECONDS,
                            )
                            code_executed_successfully = execution_succeeded(exec_result)
                            if code_executed_successfully:
                                execution_context.update(updated_context)
                                if agent_name == "preprocessing_agent":
                                    self._normalize_preprocessing_context(execution_context, datasets)
                                if agent_name == "sk_learn_agent":
                                    missing_ml_outputs = [
                                        name for name in ("model", "y_test", "y_pred")
                                        if name not in execution_context
                                    ]
                                    if missing_ml_outputs:
                                        code_executed_successfully = False
                                        exec_result = (
                                            "Error: ML executor did not publish required outputs: "
                                            + ", ".join(missing_ml_outputs)
                                        )
                                    else:
                                        execution_context.setdefault("rf_model", execution_context["model"])
                                if code_executed_successfully:
                                    context_snapshots[i] = dict(execution_context)
                            logger.info(f"{agent_name} 代码执行成功，结果长度: {len(exec_result)}")
                            logger.info(f"{agent_name} 执行结果前200字符: {exec_result[:200]}")
                        except Exception as exec_e:
                            exec_result = f"代码执行错误: {str(exec_e)}"
                            logger.error(f"{agent_name} 代码执行失败: {exec_e}", exc_info=True)
                    else:
                        logger.warning(f"{agent_name} 未执行代码: code_to_exec长度={len(code_to_exec) if code_to_exec else 0}, datasets长度={len(datasets) if datasets else 0}")
                    
                    # 将执行结果添加到result_dict
                    result_dict['result'] = exec_result
                    result_dict['code_executed'] = code_executed_successfully
                    if not code_executed_successfully:
                        result_dict['execution_error'] = exec_result or "Generated code did not execute successfully."
                        upstream_failed = True

                    # 生成详细的执行结果消息，包含思考、代码、结果
                    detail_message = f"## {agent_name} 执行结果\n\n"
                    if result_dict.get('summary'):
                        detail_message += f"### 分析思考\n{result_dict['summary']}\n\n"
                    if result_dict.get('code'):
                        detail_message += f"### 程序代码\n```python\n{result_dict['code']}\n```\n\n"
                    if exec_result:
                        detail_message += f"### 运行结果\n{exec_result}\n"
                    
                    # 如果代码执行失败，在消息中明确标记
                    if not code_executed_successfully and code_to_exec:
                        detail_message += f"⚠️ **警告：代码执行失败，需要重新生成**\n"
                    
                    task_state.set_status(agent_name, "done" if code_executed_successfully else "error", task_id)
                    # 立即返回执行结果，让用户在御史大夫审查前就能看到
                    yield (agent_name, "done", detail_message)
                    attempt_messages.append((agent_name, detail_message))
                    detail_messages[agent_name] = detail_message

                except asyncio.TimeoutError:
                    task_state.set_status(agent_name, "error", task_id)
                    executor_results[agent_name] = {
                        "result": "Executor timed out.",
                        "execution_error": "Executor timed out.",
                        "code_executed": False,
                    }
                    upstream_failed = True
                    yield (agent_name, "error", "执行超时")
                    continue
                except Exception as e:
                    task_state.set_status(agent_name, "error", task_id)
                    executor_results[agent_name] = {
                        "result": f"Executor error: {str(e)}",
                        "execution_error": f"Executor error: {str(e)}",
                        "code_executed": False,
                    }
                    upstream_failed = True
                    yield (agent_name, "error", f"执行错误: {str(e)}")
                    continue

            task_state.set_status("commander_agent", "done", task_id)
            task_state.set_status("censor_agent", "reviewing", task_id)
            censor_reviewing_message = "御史大夫正在审查所有智能体的工作..."
            task_state.add_message("censor_agent", "秦始皇", censor_reviewing_message, task_id)
            yield ("censor_agent", "reviewing", censor_reviewing_message)
            
            # 检查是否停止
            if should_stop():
                yield ("system", "stopped", "任务已被用户停止")
                return

            review_context = f"用户指令：{query_with_timestamp}\n\n"
            review_context += f"丞相细化任务：{refined_task_str}\n"
            review_context += f"太尉执行计划：{execution_plan_str}\n\n"
            review_context += "执行结果详情：\n\n"
            
            # 检查是否有任何执行智能体返回了结果
            has_any_results = False
            all_code_executed = True
            for ag, res in executor_results.items():
                review_context += f"=== {ag} ===\n"
                if isinstance(res, dict):
                    has_code = bool(res.get('code'))
                    has_summary = bool(res.get('summary'))
                    has_result = bool(res.get('result'))
                    code_executed = res.get('code_executed', False)
                    
                    if has_code and not code_executed:
                        all_code_executed = False
                    
                    if has_summary:
                        review_context += f"分析思考：{res['summary']}\n\n"
                        has_any_results = True
                    if has_code:
                        review_context += f"代码：\n{res['code']}\n\n"
                        has_any_results = True
                    if has_result:
                        review_context += f"运行结果：{self._summarize_result_for_review(res['result'])}\n"
                        has_any_results = True
                    else:
                        review_context += f"运行结果：无（代码可能未执行）\n"
                    
                    # 添加状态标记
                    status = "完整" if (has_code and has_result and code_executed) else "不完整"
                    if not code_executed and has_code:
                        status += " ⚠️代码未执行"
                    review_context += f"状态：{status}\n"
                else:
                    review_context += f"{json.dumps(res, ensure_ascii=False)}\n"
                review_context += "\n"
            
            # 如果所有执行智能体都没有返回有效结果，记录警告
            if not has_any_results:
                logger.warning(f"警告：所有执行智能体都没有返回有效结果！")
            
            # 如果有代码未执行，添加到审查上下文
            if not all_code_executed:
                review_context += "\n⚠️ 严重问题：有执行智能体的代码未被执行！必须打回重做！\n"

            try:
                with dspy.context(lm=session_lm):
                    censor_result = await asyncio.wait_for(
                        self.censor(
                            agent_name="all_agents",
                            agent_output=review_context,
                            task_context=f"user_query: {query_with_timestamp}"
                        ),
                        timeout=CENSOR_TIMEOUT_SECONDS
                    )
                review_str = censor_result.review_result
                try:
                    review = parse_json_object(review_str, "censor")
                except Exception:
                    logger.warning(f"Censor output is not valid JSON: {review_str[:200]}")
                    review = {"approved": False, "summary": "Invalid review format", "comments": review_str}
            except asyncio.TimeoutError:
                review = {"approved": False, "comments": "Censor review timeout; retry required."}
            except Exception as e:
                logger.error(f"Censor review failed: {e}")
                review = {"approved": False, "comments": f"Censor runtime error: {str(e)}"}

            # Deterministic safety guard to reduce random strict/lenient behavior.
            expected_agents = [s.get("agent", "") for s in subtasks if s.get("agent", "") in self.executors]
            missing_results = [ag for ag in expected_agents if ag not in executor_results]
            code_not_executed = [
                ag for ag, res in executor_results.items()
                if isinstance(res, dict) and res.get("code") and not res.get("code_executed", False)
            ]
            execution_errors = [
                ag for ag, res in executor_results.items()
                if isinstance(res, dict) and res.get("execution_error")
            ]
            missing_visualizations = [
                ag for ag, res in executor_results.items()
                if ag == "data_viz_agent"
                and isinstance(res, dict)
                and "<<<PLOTLY_JSON>>>" not in str(res.get("result", ""))
            ]
            no_results = (not has_any_results) or (len(executor_results) == 0)

            guard_reasons = []
            if no_results:
                guard_reasons.append("No usable executor results were produced.")
            if missing_results:
                guard_reasons.append(f"Missing results from: {', '.join(missing_results)}")
            if code_not_executed:
                guard_reasons.append(f"Code not executed successfully for: {', '.join(code_not_executed)}")
            if execution_errors:
                guard_reasons.append(f"Executor failures: {', '.join(execution_errors)}")
            if missing_visualizations:
                guard_reasons.append("Visualization task did not produce an interactive Plotly chart.")

            approved = bool(review.get("approved", False))
            comments = str(review.get("comments", ""))
            if guard_reasons:
                approved = False
                guard_text = " | ".join(guard_reasons)
                comments = f"{comments} | Guard: {guard_text}".strip(" |")

            if approved:
                censor_feedback = ""
                # 执行智能体的结果已经在执行时立即返回给用户了
                # 这里只需要记录任务状态，不需要再次返回结果
                for subtask in subtasks:
                    agent_name = subtask.get("agent", "")
                    detail_message = detail_messages.get(agent_name)
                    if detail_message:
                        task_state.add_message(agent_name, "commander_agent", detail_message, task_id)
                censor_approved_message = "审查通过：执行结果完整，允许向用户回奏。"
                task_state.add_message("censor_agent", "秦始皇", censor_approved_message, task_id)
                task_state.set_status("censor_agent", "done", task_id)
                yield ("censor_agent", "done", censor_approved_message)
                break
            else:
                failure_details = self._summarize_failed_results(executor_results)
                failure_text = " | ".join(
                    f"{agent}: {reason}"
                    for agent, reason in failure_details.items()
                )
                censor_feedback = comments
                if failure_text:
                    censor_feedback += f"\nFailure details: {failure_text}"
                task_state.set_status("censor_agent", "working", task_id)
                censor_rejected_message = f"审查未通过：{censor_feedback}"
                task_state.add_message("censor_agent", "秦始皇", censor_rejected_message, task_id)
                yield ("censor_agent", "working", censor_rejected_message)
                if attempt >= MAX_ATTEMPTS:
                    task_state.set_status("censor_agent", "error", task_id)
                    censor_error_message = f"审查终止：达到最大重试次数 ({MAX_ATTEMPTS})。"
                    task_state.add_message("censor_agent", "秦始皇", censor_error_message, task_id)
                    yield ("censor_agent", "error", censor_error_message)
                    break
                failed_agents = set(missing_results + code_not_executed + execution_errors + missing_visualizations)
                retry_start_index = self._retry_start_index(subtasks, failed_agents, review)
                retry_agent = subtasks[retry_start_index].get("agent", "") if subtasks else ""
                yield (
                    "commander_agent",
                    "working",
                    f"仅重试失败节点及其下游任务，从 {retry_agent or '第一个执行智能体'} 开始。",
                )

        if not approved:
            yield (
                "final",
                "error",
                {
                    "message": "Review failed after maximum retries.",
                    "failed_agents": self._summarize_failed_results(executor_results),
                    "review_comments": censor_feedback[:2000],
                },
            )
            return

        final_result = executor_results
        yield ("final", "done", final_result)



    async def get_plan(self, query):
        """Get plan from planner (保留原接口兼容）。"""
        planner = planner_module()
        return await planner.forward(
            goal=query,
            dataset=self.dataset_description,
            Agent_desc=str([
                {"preprocessing_agent": "数据预处理"},
                {"statistical_analytics_agent": "统计分析"},
                {"sk_learn_agent": "机器学习"},
                {"data_viz_agent": "数据可视化"},
            ])
        )
    
    async def execute_plan(self, query, plan_response):
        """Execute plan (保留原接口兼容）。"""
        # 为了兼容旧的 API，这里调用原 auto_analyst 的逻辑
        # 新编排架构请使用 execute_user_query
        yield ("plan_not_found", {}, {"error": "请使用新的秦朝官职编排架构"})
