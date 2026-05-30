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
import uuid

logger = logging.getLogger("datapilot")


# ── DSPy async compatibility ────────────────────────────────────────────
async def _run_sync(fn, *args, **kwargs):
    """Run a sync DSPy call in a thread so it doesn't block the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


def asyncify_predict(signature):
    """Return an async callable for dspy.Predict(signature)."""
    predictor = dspy.Predict(signature)
    async def call(**kwargs):
        return await _run_sync(predictor, **kwargs)
    return call


def asyncify_cot(signature):
    """Return an async callable for dspy.ChainOfThought(signature)."""
    cot = dspy.ChainOfThought(signature)
    async def call(**kwargs):
        return await _run_sync(cot, **kwargs)
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
  * 代码中使用的数据框变量必须是`df`（不是df_cleaned或其他）

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
  {"data_viz_agent": {"create": ["cleaned_data: DataFrame"], "use": ["df: DataFrame"], "instruction": "Clean df and generate a bar plot showing sales by category."}}

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
plan_instructions: {"data_viz_agent": {"create": ["scatter_plot"], "use": ["df"], "instruction": "Create scatter plot of height & salary using plotly"}}

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
- **IMPORTANT**: The dataset is already loaded as `df` in the execution context. Do NOT write code to load the dataset (like `df = pd.read_csv(...)`). Just use `df` directly.

### Output:
1. code: Python code for preprocessing (do NOT include dataset loading code)
2. summary: Brief explanation of what was done

Respond in the user's language for all summary but keep the code in English.
    """
    dataset = dspy.InputField(desc="The dataset info, preloaded as df")
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
- **IMPORTANT**: The dataset is already loaded as `df` in the execution context. Do NOT write code to load the dataset (like `df = pd.read_csv(...)`). Just use `df` directly.

### Output:
1. code: Python code for statistical modeling (do NOT include dataset loading code)
2. summary: Brief explanation of results

Respond in the user's language for all summary but keep the code in English.
    """
    dataset = dspy.InputField(desc="Dataset info, often df_cleaned")
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
- **IMPORTANT**: The dataset is already loaded as `df` in the execution context. Do NOT write code to load the dataset (like `df = pd.read_csv(...)`). Just use `df` directly.

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
- If len(df) > 50000, sample to 5000 rows first
- Each visualization must be a separate go.Figure() assigned to a variable named `fig`
- Apply update_layout with clean titles, axis labels, and proper formatting
- Use low opacity (0.4-0.7) where appropriate
- Use distinct colors for different categories
- Use only one number format consistently (K, M, or comma-separated)
- Add trendlines only if explicitly requested
- Never include dataset or styling_index in output
- **IMPORTANT**: The dataset is already loaded as `df` in the execution context. Do NOT write code to load the dataset (like `df = pd.read_csv(...)`). Just use `df` directly.

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
Fix any errors. Add df = df.copy() at start. Add fig.show() for Plotly charts.
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
    
    STATUS = ["idle", "thinking", "working", "reviewing", "done", "error"]
    
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
            self.states[agent_name]["last_active"] = asyncio.get_event_loop().time()
            if task_id:
                self.states[agent_name]["current_task"] = task_id
    
    def add_message(self, from_agent, to_agent, content, task_id=None, message_type="task"):
        msg = {
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "task_id": task_id,
            "type": message_type,
            "timestamp": asyncio.get_event_loop().time()
        }
        self.messages.append(msg)
        return msg
    
    def add_history(self, agent_name, action, result, task_id=None):
        entry = {
            "agent": agent_name,
            "action": action,
            "result": result,
            "task_id": task_id,
            "timestamp": asyncio.get_event_loop().time()
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
        self.chancellor = asyncify_cot(chancellor_agent)
        self.commander = asyncify_cot(commander_agent)
        self.censor = asyncify_cot(censor_agent)
        
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
    
    async def execute_user_query(self, query, session_lm, task_state: AgentTaskState, datasets: dict, stop_flag=None):
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
                return stop_flag()
            return False
        
        if should_stop():
            yield ("system", "stopped", "任务已停止")
            return
        # 添加时间戳到用户查询
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        query_with_timestamp = f"{query}——{timestamp}"
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task_state.set_status("chancellor_agent", "thinking", task_id)
        yield ("chancellor_agent", "thinking", f"正在理解您的指令：{query}")
        
        try:
            with dspy.context(lm=session_lm):
                chancellor_result = await asyncio.wait_for(
                    self.chancellor(
                        user_instruction=query_with_timestamp,  # 使用带时间戳的查询
                        dataset_description=self.dataset_description
                    ),
                    timeout=60
                )
            refined_task_str = chancellor_result.refined_task
            
            try:
                refined_task = json.loads(refined_task_str)
                # 确保 subtasks 存在，如果不存在则创建一个
                if 'subtasks' not in refined_task:
                    refined_task['subtasks'] = []
                # 如果 subtasks 为空，添加一个默认任务
                if len(refined_task.get('subtasks', [])) == 0:
                    logger.warning(f"丞相生成的 subtasks 为空，添加默认任务")
                    refined_task['subtasks'] = [{
                        'agent': 'data_viz_agent',
                        'instruction': query_with_timestamp
                    }]
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"丞相返回的细化任务解析失败: {e}, 原始输出: {refined_task_str[:500]}")
                refined_task = {
                    "task_id": task_id,
                    "user_goal": query_with_timestamp,  # 使用带时间戳的查询
                    "refined_goal": refined_task_str,
                    "subtasks": [{
                        'agent': 'data_viz_agent',
                        'instruction': query_with_timestamp
                    }]
                }
            
            task_state.set_status("chancellor_agent", "done", task_id)
            task_state.add_message("秦始皇", "chancellor_agent", query_with_timestamp, task_id)  # 使用带时间戳的查询
            task_state.add_message("chancellor_agent", "commander_agent", refined_task_str, task_id)
            task_state.add_history("chancellor_agent", "任务细化完成", refined_task_str, task_id)
            
            # 显示完整的细化任务
            subtasks = refined_task.get('subtasks', [])
            subtask_list = '\n'.join([f"{i+1}. {subtask.get('instruction', subtask.get('task', ''))}" for i, subtask in enumerate(subtasks)])
            yield ("chancellor_agent", "done", f"## 丞相任务细化结果\n\n**细化目标**: {refined_task.get('refined_goal', '')}\n\n**子任务列表**:\n{subtask_list}")
            
            # 检查是否停止
            if should_stop():
                yield ("system", "stopped", "任务已被用户停止")
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
            with dspy.context(lm=session_lm):
                commander_result = await asyncio.wait_for(
                    self.commander(
                        refined_task=refined_task_str,
                        dataset_description=self.dataset_description
                    ),
                    timeout=60
                )
            execution_plan_str = commander_result.execution_plan
            
            try:
                execution_plan = json.loads(execution_plan_str)
                # 确保 subtasks 不为空，如果为空则使用丞相的 subtasks
                chancellor_subtasks = refined_task.get("subtasks", [])
                if not execution_plan.get('subtasks') or len(execution_plan.get('subtasks', [])) == 0:
                    logger.warning(f"太尉生成的 subtasks 为空，使用丞相的 subtasks")
                    execution_plan['subtasks'] = chancellor_subtasks if chancellor_subtasks else [{"agent": "data_viz_agent", "instruction": query_with_timestamp}]
                
                # 如果太尉和丞相的subtasks都为空，添加一个默认任务
                if not execution_plan.get('subtasks') or len(execution_plan.get('subtasks', [])) == 0:
                    logger.warning(f"太尉和丞相的 subtasks 都为空，使用默认任务")
                    execution_plan['subtasks'] = [{"agent": "data_viz_agent", "instruction": query_with_timestamp}]
                    
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"太尉返回的执行计划解析失败: {e}, 原始输出: {execution_plan_str[:500]}")
                chancellor_subtasks = refined_task.get("subtasks", [])
                execution_plan = {"subtasks": chancellor_subtasks if chancellor_subtasks else [{"agent": "data_viz_agent", "instruction": query_with_timestamp}]}
            
            task_state.set_status("commander_agent", "working", task_id)
            task_state.add_message("commander_agent", "执行智能体", execution_plan_str, task_id)
            task_state.add_history("commander_agent", "规划拆解完成", execution_plan_str, task_id)
            
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
        MAX_ATTEMPTS = 10
        censor_feedback = ""
        executor_results = {}
        approved = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            subtasks = execution_plan.get("subtasks", [])
            if not subtasks:
                subtasks = refined_task.get("subtasks", [])

            executor_results = {}

            # ── 执行每个子任务 ──
            for i, subtask in enumerate(subtasks):
                agent_name = subtask.get("agent", "")
                instruction = subtask.get("instruction", subtask.get("task", ""))
                # Append censor feedback if retrying
                if censor_feedback:
                    instruction += f"\n\n请根据御史大夫的反馈改进代码：{censor_feedback}"

                if agent_name not in self.executors:
                    yield (agent_name, "error", f"未找到执行智能体: {agent_name}")
                    continue

                yield ("commander_agent", "working", f"分发子任务 {i+1}/{len(subtasks)} 给 {agent_name}")

                task_state.set_status(agent_name, "working", task_id)
                yield (agent_name, "working", f"正在执行: {instruction[:80]}...")

                try:
                    with dspy.context(lm=session_lm):
                        # 构建plan_instructions，确保不重复添加反馈
                        plan_instructions = json.dumps(subtask, ensure_ascii=False)
                        if censor_feedback:
                            plan_instructions += f"\n\n御史大夫反馈: {censor_feedback}"
                        
                        inputs = {
                            "goal": instruction,
                            "dataset": self.dataset_description,
                            "plan_instructions": plan_instructions
                        }

                        required_keys = self.executor_inputs.get(agent_name, set())
                        filtered_inputs = {k: v for k, v in inputs.items() if k in required_keys}

                        result = await asyncio.wait_for(
                            self.executors[agent_name](**filtered_inputs),
                            timeout=90
                        )
                        
                        # 调试：记录智能体返回的完整结果
                        logger.info(f"{agent_name} 返回结果类型: {type(result)}, 内容: {str(result)[:500]}")
                        result_dict = dict(result)
                        executor_results[agent_name] = result_dict
                        
                        # 调试：记录转换后的字典
                        logger.info(f"{agent_name} 转换后的字典 keys: {list(result_dict.keys())}")
                        
                        # 执行代码并获取运行结果
                        exec_result = ""
                        code_to_exec = result_dict.get('code', '')
                        code_executed_successfully = False
                        logger.info(f"{agent_name} 提取的代码: '{code_to_exec[:200]}...', datasets长度: {len(datasets) if datasets else 0}")
                        
                        if code_to_exec and datasets is not None and len(datasets) > 0:
                            try:
                                from src.format_response import execute_code_from_markdown
                                logger.info(f"{agent_name} 开始执行代码, datasets keys: {list(datasets.keys())}")
                                logger.info(f"{agent_name} 代码内容前500字符: {code_to_exec[:500]}")
                                exec_result = execute_code_from_markdown(code_to_exec, datasets)
                                code_executed_successfully = True
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
                        
                        task_state.add_message(agent_name, "commander_agent", detail_message, task_id)
                        task_state.set_status(agent_name, "done", task_id)
                        # 返回完整的详细结果，用于实时显示
                        yield (agent_name, "done", detail_message)

                except asyncio.TimeoutError:
                    task_state.set_status(agent_name, "error", task_id)
                    yield (agent_name, "error", "执行超时")
                    continue
                except Exception as e:
                    task_state.set_status(agent_name, "error", task_id)
                    yield (agent_name, "error", f"执行错误: {str(e)}")
                    continue

            task_state.set_status("censor_agent", "reviewing", task_id)
            yield ("censor_agent", "reviewing", "御史大夫正在审查所有智能体的工作...")
            
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
                    
                    if not code_executed:
                        all_code_executed = False
                    
                    if has_summary:
                        review_context += f"分析思考：{res['summary']}\n\n"
                        has_any_results = True
                    if has_code:
                        review_context += f"代码：\n{res['code']}\n\n"
                        has_any_results = True
                    if has_result:
                        review_context += f"运行结果：{res['result']}\n"
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

            with dspy.context(lm=session_lm):
                censor_result = await asyncio.wait_for(
                    self.censor(
                        agent_name="所有智能体",
                        agent_output=review_context,
                        task_context=f"用户指令：{query_with_timestamp}"
                    ),
                    timeout=60
                )

            review_str = censor_result.review_result
            try:
                review = json.loads(review_str)
            except:
                # JSON解析失败时不默认通过，而是要求查看原始输出
                logger.warning(f"御史大夫返回的审查结果解析失败: {review_str[:200]}")
                review = {"approved": False, "summary": "审查结果格式错误，请重新审查", "comments": review_str}

            approved = review.get("approved", True)
            comments = review.get("comments", "")

            if approved:
                censor_feedback = ""
                task_state.add_message("censor_agent", "秦始皇", "审查通过", task_id)
                task_state.set_status("censor_agent", "done", task_id)
                yield ("censor_agent", "done", "审查通过")
                break
            else:
                censor_feedback = comments
                yield ("censor_agent", "rejected", f"审查未通过：{comments}")
                if attempt >= MAX_ATTEMPTS:
                    yield ("censor_agent", "done", f"达到最大重试次数 ({MAX_ATTEMPTS})")
                    break

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
