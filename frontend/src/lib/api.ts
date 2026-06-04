export const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

// ── 类型定义 ──────────────────────────────────────────────────

export interface Agent {
  name: string;
  display: string;
  icon: string;
  desc: string;
  role?: string;  // "丞相" | "太尉" | "执行智能体" | "御史大夫"
}

export interface ModelConfig {
  provider: string;
  model: string;
  api_key: string;
  has_api_key?: boolean;
}

export interface DatasetInfo {
  loaded: boolean;
  name?: string;
  filename?: string;
  shape?: number[];
  columns?: string[];
  description?: string;
  datasets?: DatasetSummary[];
}

export interface DatasetSummary {
  name: string;
  filename: string;
  shape: number[];
  columns: string[];
  primary: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  agent?: string;
  agentEventKey?: string;
  messageKind?: 'report';
  timestamp: number;
}

// ── 新架构：智能体状态 & 消息流类型 ─────────────────────

export type AgentStatus = 'idle' | 'thinking' | 'working' | 'reviewing' | 'done' | 'error' | 'stopped';

export interface AgentStateSnapshot {
  [agentName: string]: {
    status: AgentStatus;
    last_active: number | null;
    current_task: string | null;
  };
}

export interface AgentMessage {
  from: string;
  to: string;
  content: string;
  task_id?: string;
  type: string;
  timestamp: number;
}

export interface AgentHistoryEntry {
  agent: string;
  action: string;
  result: any;
  task_id?: string;
  timestamp: number;
}

export interface TaskStateSnapshot {
  states: AgentStateSnapshot;
  messages: AgentMessage[];
  history: AgentHistoryEntry[];
}

// SSE 事件类型
export interface SseEventAgentStatus {
  type: 'agent_status';
  agent: string;
  status: AgentStatus;
  content?: string;
  task_state?: TaskStateSnapshot;
}

export interface SseEventMessage {
  type: 'message';
  from: string;
  to: string;
  content: string;
  task_id?: string;
}

export interface SseEventResult {
  type: 'result';
  agent: string;
  content: string;
  status: 'success' | 'error';
  task_state?: TaskStateSnapshot;
}

export interface SseEventReview {
  type: 'review_result';
  agent: string;
  approved: boolean;
  comments?: string;
  target?: string;
  task_id?: string;
}

export interface SseEventFinal {
  type: 'final';
  content: any;
  status: string;
  task_state?: TaskStateSnapshot;
}

export interface SseEventError {
  type: 'error';
  content: string;
  task_state?: TaskStateSnapshot;
}

export interface SseEventStopped {
  type: 'stopped';
  content: string;
  task_state?: TaskStateSnapshot;
}

export type SseEvent =
  | SseEventAgentStatus
  | SseEventMessage
  | SseEventResult
  | SseEventReview
  | SseEventFinal
  | SseEventError
  | SseEventStopped;

// ── API 调用 ──────────────────────────────────────────────────

export async function createSession(): Promise<string> {
  const res = await fetch(`${API_BASE}/session`, { method: 'POST' });
  const data = await res.json();
  return data.session_id;
}

export interface UploadResult {
  name: string;
  filename: string;
  shape: number[];
  columns: string[];
  description: string;
  datasets: DatasetSummary[];
}

export interface BatchUploadResult {
  results: Array<{
    name: string;
    filename: string;
    shape: number[];
  }>;
  errors: Array<{
    filename: string;
    error: string;
  }>;
  datasets: DatasetSummary[];
  description: string;
}

export async function uploadDataset(sessionId: string, file: File, description: string = ''): Promise<UploadResult> {
  const form = new FormData();
  form.append('file', file);
  if (description) form.append('description', description);
  
  const res = await fetch(`${API_BASE}/session/${sessionId}/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '上传失败' }));
    throw new Error(err.detail || '上传失败');
  }
  return res.json();
}

export async function uploadDatasetsBatch(sessionId: string, files: File[]): Promise<BatchUploadResult> {
  const form = new FormData();
  files.forEach(file => {
    form.append('files', file);
  });
  
  const res = await fetch(`${API_BASE}/session/${sessionId}/upload/batch`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '批量上传失败' }));
    throw new Error(err.detail || '批量上传失败');
  }
  return res.json();
}

export async function getDatasetInfo(sessionId: string): Promise<DatasetInfo> {
  const res = await fetch(`${API_BASE}/session/${sessionId}/dataset`);
  return res.json();
}

export async function deleteDataset(sessionId: string, datasetName: string): Promise<void> {
  const res = await fetch(`${API_BASE}/session/${sessionId}/dataset/${encodeURIComponent(datasetName)}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '删除失败' }));
    throw new Error(err.detail || '删除失败');
  }
}

export async function getModelConfig(sessionId: string): Promise<ModelConfig> {
  const res = await fetch(`${API_BASE}/session/${sessionId}/model`);
  return res.json();
}

export async function setModelConfig(sessionId: string, config: ModelConfig): Promise<void> {
  const res = await fetch(`${API_BASE}/session/${sessionId}/model`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to save model settings' }));
    throw new Error(err.detail || 'Failed to save model settings');
  }
}

export async function getAgents(): Promise<Agent[]> {
  const res = await fetch(`${API_BASE}/agents`);
  const data = await res.json();
  return data.agents;
}

// ── 新架构：聊天（SSE 流式，含智能体状态事件）─────────────

export async function chatWithPlanner(
  sessionId: string,
  query: string,
  onEvent: (event: SseEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_BASE}/session/${sessionId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
    signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '请求失败' }));
    throw new Error(err.detail || '请求失败');
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim() || !line.startsWith('data: ')) continue;
      try {
        const jsonStr = line.slice(6);
        const event: SseEvent = JSON.parse(jsonStr);
        onEvent(event);
      } catch {
      }
    }
  }

  if (buffer.trim() && buffer.startsWith('data: ')) {
    try {
      const event: SseEvent = JSON.parse(buffer.slice(6));
      onEvent(event);
    } catch {}
  }
}

// ── 兼容旧接口：与非编排智能体对话（非流式）─────────────

export async function chatWithAgent(sessionId: string, agentName: string, query: string, signal?: AbortSignal): Promise<any> {
  const res = await fetch(`${API_BASE}/chat/${agentName}?session_id=${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
    signal,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '请求失败' }));
    throw new Error(err.detail || '请求失败');
  }
  return res.json();
}

// ── 新架构：获取任务状态 & 智能体消息流 ─────────────────

export async function getTaskState(sessionId: string): Promise<TaskStateSnapshot> {
  const res = await fetch(`${API_BASE}/session/${sessionId}/task-state`);
  return res.json();
}

export async function getAgentsStatus(sessionId: string): Promise<{
  agents: AgentStateSnapshot;
  messages: AgentMessage[];
  history: AgentHistoryEntry[];
}> {
  const res = await fetch(`${API_BASE}/session/${sessionId}/agents-status`);
  return res.json();
}

export async function submitReview(
  sessionId: string,
  approved: boolean,
  target: string,
  comments: string = '',
  severity: 'low' | 'medium' | 'high' = 'medium'
): Promise<void> {
  await fetch(`${API_BASE}/session/${sessionId}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approved, target, comments, severity }),
  });
}

export async function stopChat(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/session/${sessionId}/stop`, {
    method: 'POST',
  });
}

// ── 代码执行 & 修复 ─────────────────────────────────────────

export async function fixCode(sessionId: string, code: string, error: string): Promise<string> {
  const res = await fetch(`${API_BASE}/session/${sessionId}/fix-code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, error }),
  });
  const data = await res.json();
  return data.fixed_code;
}

export async function executeCode(sessionId: string, code: string): Promise<{ status: string; output: string }> {
  const res = await fetch(`${API_BASE}/session/${sessionId}/execute-code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  });
  const data = await res.json();
  return data;
}

// ── 兼容旧版 planner（可选）─────────────────────────────────

export async function chatWithPlannerLegacy(
  sessionId: string,
  query: string,
  onEvent: (event: any) => void
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat-legacy?session_id=${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '请求失败' }));
    throw new Error(err.detail || '请求失败');
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line);
        onEvent(event);
      } catch {}
    }
  }
}
