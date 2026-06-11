# DataPilot - AI 数据分析助手

一个本地化的智能数据分析平台，采用**秦朝官职编排架构**的多智能体系统。通过自然语言与数据对话，AI 自动规划分析流程并生成可执行代码。

## ✨ 特性

- 🔓 **无需登录** — 本地部署，开箱即用
- 🏛️ **秦朝官职编排** — 丞相→太尉→执行智能体→御史大夫的四层架构
- 🤖 **多模型支持** — 支持 OpenAI / Anthropic / DeepSeek / Gemini / Groq
- 📊 **四大执行智能体** — 数据预处理、统计分析、机器学习、数据可视化
- 📁 **多格式支持** — 上传 CSV / XLS / XLSX 文件即可开始分析
- 💻 **代码执行** — 生成 Python 代码并实时执行，支持编辑与修复
- 📈 **交互式图表** — Plotly 图表在界面中直接渲染
- 📕 **报告下载** — 可下载 Markdown 报告及独立图片文件
- 🔄 **实时状态展示** — 侧边栏实时显示各智能体工作状态
- 🔒 **会话隔离** — 不同对话框数据完全隔离，互不干扰

## 🚀 快速开始

### 1. 后端启动（Windows）

```bash
cd backend

# 一键启动（自动安装依赖 + 启动服务）
start.bat
```

后端默认运行在 **http://localhost:8001**

> 💡 首次运行会自动创建虚拟环境并安装依赖
>
> 🔧 如需手动配置：
> ```bash
> pip install -r requirements.txt
> copy .env.template .env
> # 编辑 .env 填入 API Key
> python app.py
> ```

### 2. 前端启动

```bash
cd frontend

# 安装依赖
npm install

# 创建本地前端配置
copy .env.example .env.local

# 前端启动
npm run dev
```

前端默认运行在 **http://localhost:3000**

> ⚙️ 前端配置文件：
> - `frontend/.env.local` — API 地址（默认 `http://localhost:8001`）
> - `frontend/.env.example` — 可提交到仓库的配置模板
> - `frontend/src/app/globals.css` — 主题配色
> - `frontend/src/components/ThemeProvider.tsx` — 深色/浅色模式切换

### 3. 使用

1. 打开浏览器访问 http://localhost:3000
2. **配置 API**（点击右上角 ⚙️ 图标）：
   - 选择模型提供商（DeepSeek / OpenAI / Anthropic / Gemini / Groq）
   - 填入 API Key
   - 点击「保存」
3. **上传数据**（点击右上角 📁 图标）：
   - 支持 CSV / XLSX / Parquet 格式
   - 上传后左侧边栏显示数据集信息
4. **开始对话**：
   - 在底部输入框输入分析需求
   - 按 Enter 发送（Shift+Enter 换行）
   - 实时查看各智能体工作状态（左侧边栏）
5. **查看结果**：
   - Markdown 格式的分析回答
   - 可执行的 Python 代码
   - 交互式的 Plotly 图表
6. **下载报告**：
   - 点击报告消息的下载按钮
   - 自动下载 Markdown 文件和所有图表图片
   - 将文件放在同一目录即可正常查看图片

## ⚠️ 安全提示

- DataPilot 会在本机执行模型生成的 Python 代码。建议先检查代码，并在隔离的虚拟环境或容器中运行。
- 默认配置适用于本地使用。不要在未增加身份认证、权限控制和执行沙箱的情况下直接暴露到公网。
- 不要上传包含隐私、商业机密或其他敏感信息的数据文件。
- 不要提交 `backend/.env` 或 `frontend/.env.local`。API Key 仅应保存在本地配置中。

> 🌙 点击左侧边栏 ⚡ DataPilot 旁的 ☀️/🌙 图标切换深色/浅色主题

## ⚙️ 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 提供商 | `openai` |
| `LLM_MODEL` | 模型名称 | `gpt-4o-mini` |
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `ANTHROPIC_API_KEY` | Anthropic API Key | - |
| `GEMINI_API_KEY` | Gemini API Key | - |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | - |
| `GROQ_API_KEY` | Groq API Key | - |
| `HOST` | 后端监听地址 | `0.0.0.0` |
| `PORT` | 后端监听端口 | `8001` |

> 💡 也可以在前端设置面板中动态切换模型和 API Key

## 🏗️ 项目架构

```
DataPilot/
├── .vscode/                    # VS Code 配置
│   └── settings.json           # IDE 配置
├── ReportExamples/             # 报告示例
│   ├── datapilot-analysis-report.md  # 示例报告
│   └── chart_*.png             # 示例图表
├── backend/                    # Python FastAPI 后端
│   ├── app.py                  # 主应用入口（FastAPI + SSE 流式）
│   ├── requirements.txt        # Python 依赖
│   ├── .env.template           # 环境变量模板
│   ├── start.bat              # Windows 启动脚本
│   ├── sample_data.csv         # 示例数据集
│   └── src/
│       ├── agents/
│       │   ├── __init__.py
│       │   └── agents.py       # DSPy 智能体签名定义与编排逻辑
│       ├── __init__.py
│       ├── format_response.py  # 响应格式化与代码执行
│       ├── runtime_config.py   # 运行时配置参数
│       └── simple_retriever.py # 简易检索器
│
└── frontend/                   # Next.js 前端
    ├── app/
    │   ├── page.tsx            # 页面入口
    │   ├── layout.tsx          # 布局（含 ThemeProvider）
    │   └── globals.css         # 全局样式（深色/浅色主题）
    ├── components/
    │   ├── chat/
    │   │   └── ChatPage.tsx    # 聊天界面组件（含 SSE 流式解析）
    │   ├── Providers.tsx       # 应用提供者
    │   └── ThemeProvider.tsx   # 主题切换上下文
    ├── lib/
    │   └── api.ts              # API 调用封装（含类型定义）
    ├── types/
    │   └── plotly-dist.d.ts    # Plotly 类型定义
    ├── next.config.js          # Next.js 配置
    ├── postcss.config.js       # PostCSS 配置
    ├── tailwind.config.js      # Tailwind CSS 配置
    ├── tsconfig.json           # TypeScript 配置
    └── package.json
```

## 🏛️ 秦朝官职编排架构

系统采用四层智能体架构，模拟秦朝中央集权制度：

```
用户（秦始皇）
    ↓
丞相（chancellor_agent）— 接收指令，细化任务
    ↓
太尉（commander_agent）— 规划拆解，分发子任务
    ↓
执行智能体                      ↓
- 预处理智能体                  ↑
- 统计分析智能体    协作执行     ↑
- 机器学习智能体                ↑
- 数据可视化智能体              ↑
    ↓
御史大夫（censor_agent）— 审查所有工作，可打回重做
```

### 🤖 智能体列表

| 官职 | 智能体名称 | 功能 | 技术栈 |
|------|------------|------|--------|
| 丞相 | `chancellor_agent` | 接收用户指令，细化任务目标，生成报告 | DSPy + LLM |
| 太尉 | `commander_agent` | 规划拆解，分发子任务 | DSPy + LLM |
| 执行者 | `preprocessing_agent` | 缺失值处理、类型转换 | Pandas / NumPy |
| 执行者 | `statistical_analytics_agent` | 回归分析、方差分析、描述统计 | statsmodels / scipy |
| 执行者 | `sk_learn_agent` | 分类、回归、聚类 | scikit-learn / XGBoost / LightGBM |
| 执行者 | `data_viz_agent` | 交互式图表生成 | Plotly / Matplotlib / Seaborn |
| 御史大夫 | `censor_agent` | 审查所有工作，可打回重做 | DSPy + LLM |
| 意图判定 | `intent_classifier_agent` | 分析用户意图，判定是否需要执行、是否需要报告等 | DSPy + LLM |

### 🎯 意图判定机制

系统采用 **LLM 动态意图判定**，取代传统的关键词匹配方式：

- **意图分类器**：通过大模型分析用户查询，自动识别以下意图：
  - 是否为纯对话请求（解释、追问等）
  - 是否需要执行数据分析
  - 是否请求生成报告
  - 是否为纯报告请求
  - 是否请求重新执行任务
  - 需要调用哪些执行智能体

- **缓存机制**：意图判定结果会被缓存，避免重复调用 LLM

- **降级策略**：当 LLM 判定失败时，自动回退到关键词判定作为备用方案

这种方式大大提升了系统的灵活性，可以理解更复杂的语义表达。

## 🔄 工作流程

1. **用户输入** → 意图分类器分析用户意图
2. **意图判定** → 识别是否为纯对话、是否需要执行、是否请求报告等
3. **任务细化** → 丞相接收并细化任务目标
4. **太尉规划** → 拆解为子任务并分发给执行智能体
5. **执行协作** → 各执行智能体并行/串行完成子任务
6. **御史审查** → 审查结果质量，决定是否打回（报告请求跳过审查）
7. **报告生成** → 丞相整合所有结果生成综合报告
8. **最终输出** → 返回给用户

前端通过 **SSE（Server-Sent Events）** 实时展示各智能体状态：
- 🔵 思考中（thinking）
- 🟡 工作中（working）
- 🟣 审查中（reviewing）
- 🟢 完成（done）
- 🔴 错误（error）

## 📡 API 端点

| 类别 | 方法 | 路径 | 功能 |
|------|------|------|------|
| 会话管理 | POST/GET/DELETE | `/session` / `/session/{id}` | 创建/获取/删除会话 |
| 模型配置 | GET/POST | `/session/{id}/model` | 获取/设置模型配置 |
| 数据上传 | POST | `/session/{id}/upload` / `/batch` | 上传数据文件 |
| 数据管理 | GET/DELETE/POST | `/session/{id}/dataset` / `/describe` | 数据集信息与描述 |
| 核心对话 | POST | `/session/{id}/chat` | 与智能体系统对话（SSE 流式）|
| 代码操作 | POST | `/session/{id}/execute-code` / `/fix-code` / `/edit-code` | 代码执行与修复 |
| 任务控制 | GET/POST | `/session/{id}/task-state` / `/stop` / `/review` | 任务状态与控制 |
| 智能体 | GET/POST | `/agents` / `/chat/{agent_name}` | 获取列表与单独对话 |

> **SSE 事件类型**：`agent_status`（智能体状态）、`final`（最终结果）、`error`（错误）

## 🛠️ 技术栈

**后端**
- Python 3.10+
- FastAPI — Web 框架
- DSPy 3.1.3 — LLM 编排框架
- LiteLLM 1.82.3 — LLM 统一接口
- Pandas 2.2.3 / NumPy 2.2.2 — 数据处理
- statsmodels 0.14.4 — 统计分析
- scikit-learn 1.6.1 — 机器学习
- XGBoost 3.0.0 / LightGBM 4.6.0 — 梯度提升
- Plotly 5.24.1 — 可视化
- Matplotlib 3.10.0 / Seaborn 0.13.2 — 绘图

**前端**
- Next.js 14 — React 框架
- TypeScript
- Tailwind CSS — 样式
- Plotly.js — 图表渲染
- react-markdown — Markdown 渲染
- react-syntax-highlighter — 代码高亮
- file-saver — 文件下载

## 📝 许可证

本项目为 2026 年研电赛 AI 智能体专项赛道作品，仅供学习研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 报告下载格式

下载报告时，系统会生成以下文件：
- `datapilot-analysis-report.md` — Markdown 报告文件
- `chart_1.png`, `chart_2.png`, ... — 图表图片文件

将这些文件放在同一目录下，Markdown 即可正确显示图片。