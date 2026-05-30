'use client'

import React, { useState, useEffect, useRef, useCallback } from 'react'
import dynamic from 'next/dynamic'
import ReactMarkdown from 'react-markdown'
import { useTheme } from '@/components/ThemeProvider'

// Dynamically import react-plotly.js to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })
import {
  createSession, uploadDataset, getDatasetInfo,
  getModelConfig, setModelConfig, getAgents, getAgentsStatus,
  chatWithAgent, chatWithPlanner, fixCode, executeCode, stopChat,
  type Agent, type ChatMessage, type ModelConfig, type DatasetInfo,
} from '@/lib/api'

// Helper to extract code blocks from markdown
function extractCodeFromMarkdown(markdown: string): string | null {
  const codeBlockMatch = markdown.match(/```python\s*\n([\s\S]*?)```/);
  if (codeBlockMatch) return codeBlockMatch[1].trim();
  const genericCodeMatch = markdown.match(/```\s*\n([\s\S]*?)```/);
  if (genericCodeMatch) return genericCodeMatch[1].trim();
  return null;
}

// Parse Plotly JSON markers from text
// Format: <<<PLOTLY_JSON>>>\n{json}\n<<<END_PLOTLY_JSON>>>
interface ParsedSegment {
  type: 'markdown' | 'plotly'
  content: string
}

function parsePlotlyMarkers(text: string): ParsedSegment[] {
  const segments: ParsedSegment[] = []
  const regex = /<<<PLOTLY_JSON>>>\n([\s\S]*?)\n<<<END_PLOTLY_JSON>>>/g
  let lastIndex = 0
  let match
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ type: 'markdown', content: text.slice(lastIndex, match.index) })
    }
    segments.push({ type: 'plotly', content: match[1].trim() })
    lastIndex = match.index + match[0].length
  }
  if (lastIndex < text.length) {
    segments.push({ type: 'markdown', content: text.slice(lastIndex) })
  }
  if (segments.length === 0) {
    segments.push({ type: 'markdown', content: text })
  }
  return segments
}

// Plotly chart component
function PlotlyChart({ jsonData }: { jsonData: string }) {
  const [figData, setFigData] = React.useState<any>(null)
  const [error, setError] = React.useState<string | null>(null)
  useEffect(() => {
    try {
      setFigData(JSON.parse(jsonData))
    } catch {
      setError('Failed to parse chart data')
    }
  }, [jsonData])
  if (error) return <div className="text-red-400 text-sm p-2">⚠️ {error}</div>
  if (!figData) return <div className="text-[var(--text-secondary)] text-sm p-2">⏳ 加载图表中...</div>
  return (
    <div className="my-3 rounded-lg overflow-hidden border border-[var(--border)]">
      <Plot
        data={figData.data || []}
        layout={{
          ...(figData.layout || {}),
          autosize: true,
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { color: '#ccc', ...(figData.layout?.font || {}) },
          margin: { t: 40, r: 20, b: 40, l: 50, ...(figData.layout?.margin || {}) },
        }}
        config={{ responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }}
        style={{ width: '100%', height: '500px' }}
        useResizeHandler
      />
    </div>
  )
}

// Render message content with embedded Plotly charts
function MessageContent({ content }: { content: string }) {
  const segments = parsePlotlyMarkers(content)
  return (
    <div className="prose prose-invert prose-sm max-w-none">
      {segments.map((seg, i) => {
        if (seg.type === 'plotly') {
          return <PlotlyChart key={i} jsonData={seg.content} />
        }
        return (
          <ReactMarkdown key={i}>{seg.content}</ReactMarkdown>
        )
      })}
    </div>
  )
}

interface ChatSession {
  id: string
  title: string
  messages: ChatMessage[]
}

const AGENT_LIST = [
  { name: "chancellor_agent", display: "丞相", emoji: "📜" },
  { name: "commander_agent", display: "太尉", emoji: "⚔️" },
  { name: "censor_agent", display: "御史大夫", emoji: "⚖️" },
  { name: "preprocessing_agent", display: "数据预处理", emoji: "🔧" },
  { name: "statistical_analytics_agent", display: "统计分析", emoji: "📊" },
  { name: "sk_learn_agent", display: "机器学习", emoji: "🧠" },
  { name: "data_viz_agent", display: "数据可视化", emoji: "📈" },
]

// ── Main Chat Page ──────────────────────────────────────────────────────
export default function ChatPage() {
  const { theme, toggleTheme } = useTheme()
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string>('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [dataset, setDataset] = useState<DatasetInfo>({ loaded: false })
  const [agents, setAgents] = useState<Agent[]>([])
  const [agentStatus, setAgentStatus] = useState<Record<string, { status: string; last_active: number | null; current_task: string | null }>>({})
  const [showSettings, setShowSettings] = useState(false)
  const [showUpload, setShowUpload] = useState(false)
  const [modelConfig, setModelConfigState] = useState<ModelConfig>({
    provider: 'deepseek', model: 'deepseek-chat', api_key: '',
  })
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  // Code editing state
  const [editingCode, setEditingCode] = useState<string | null>(null)
  const [editedCode, setEditedCode] = useState<string>('')
  const [codeOutput, setCodeOutput] = useState<string>('')
  const [runningCode, setRunningCode] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null)
  const [editSessionName, setEditSessionName] = useState('')

  useEffect(() => {
    const saved = localStorage.getItem('datapilot-sessions')
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        // 过滤掉空对话（没有消息的会话）
        const validSessions = parsed.filter((s: ChatSession) => s.messages && s.messages.length > 0)
        setSessions(validSessions)
        // 默认进入新对话，不加载历史会话
        initSession()
      } catch {
        initSession()
      }
    } else {
      initSession()
    }
    getAgents().then(setAgents)
  }, [])

  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem('datapilot-sessions', JSON.stringify(sessions))
    }
  }, [sessions])

  useEffect(() => {
    if (currentSessionId) {
      getDatasetInfo(currentSessionId).then(setDataset)
      getModelConfig(currentSessionId).then(cfg => setModelConfigState({
        provider: cfg.provider || 'deepseek',
        model: cfg.model || 'deepseek-chat',
        api_key: cfg.has_api_key ? '••••••••' : '',
      }))
    }
  }, [currentSessionId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    const interval = setInterval(fetchAgentStatus, 2000)
    return () => clearInterval(interval)
  }, [currentSessionId])

  const initSession = async () => {
    try {
      const sid = await createSession()
      setCurrentSessionId(sid)
      setMessages([])
      setDataset({ loaded: false })
      // 新建会话时不添加到历史记录，只有在有消息后才保存到历史记录
    } catch (err) {
      console.error('初始化失败:', err)
    }
  }

  const fetchAgentStatus = async () => {
    if (!currentSessionId) return
    try {
      const result = await getAgentsStatus(currentSessionId)
      if (result && result.agents) {
        setAgentStatus(result.agents)
      }
      // 消息现在通过SSE流式获取，这里不再重复添加
      // 如果消息数组为空（如刚打开历史会话），则加载历史消息
      if (result && result.messages && result.messages.length > 0) {
        setMessages(prev => {
          // 只有当没有任何assistant消息时才加载历史消息
          const hasAssistantMessages = prev.some(m => m.role === 'assistant')
          if (hasAssistantMessages) return prev
          
          const newAgentMessages: ChatMessage[] = []
          result.messages.forEach((msg: any) => {
            const fromAgent = msg.from || msg.from_agent
            if (fromAgent && fromAgent !== '人类审查员' && fromAgent !== '秦始皇' && msg.content) {
              newAgentMessages.push({
                id: `agent-${fromAgent}-${msg.timestamp || Date.now()}`,
                role: 'assistant',
                content: msg.content,
                agent: fromAgent,
                timestamp: msg.timestamp || Date.now(),
              })
            }
          })
          
          if (newAgentMessages.length > 0) {
            const updated = [...prev, ...newAgentMessages]
            setSessions(prevSessions =>
              prevSessions.map(s =>
                s.id === currentSessionId ? { ...s, messages: updated } : s
              )
            )
            return updated
          }
          return prev
        })
      }
    } catch (err) {
      console.error('获取智能体状态失败:', err)
    }
  }

  const handleSelectSession = (sessionId: string) => {
    const session = sessions.find(s => s.id === sessionId)
    if (session) {
      setCurrentSessionId(sessionId)
      setMessages(session.messages || [])
      // 清除loading状态，避免显示"分析中..."
      setLoading(false)
      // 清除智能体状态，避免显示不正确的工作状态
      setAgentStatus({})
    }
  }

  const handleDeleteSession = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation()
    setSessions(prev => prev.filter(s => s.id !== sessionId))
    if (currentSessionId === sessionId) {
      const next = sessions.find(s => s.id !== sessionId)
      if (next) {
        setCurrentSessionId(next.id)
        setMessages(next.messages || [])
      } else {
        initSession()
      }
    }
  }

  const startEditSessionName = (e: React.MouseEvent, session: ChatSession) => {
    e.stopPropagation()
    setEditingSessionId(session.id)
    setEditSessionName(session.title)
  }

  const saveSessionName = () => {
    if (editingSessionId && editSessionName.trim()) {
      setSessions(prev =>
        prev.map(s =>
          s.id === editingSessionId
            ? { ...s, title: editSessionName.trim() }
            : s
        )
      )
      setEditingSessionId(null)
      setEditSessionName('')
    }
  }

  const cancelEditSessionName = () => {
    setEditingSessionId(null)
    setEditSessionName('')
  }

  const sendMessage = useCallback(async () => {
    if (!input.trim() || loading || !currentSessionId) return
    
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: Date.now(),
    }
    setMessages(prev => {
      const updated = [...prev, userMsg]
      setSessions(prevSessions => {
        const existingSession = prevSessions.find(s => s.id === currentSessionId)
        if (existingSession) {
          return prevSessions.map(s =>
            s.id === currentSessionId
              ? { ...s, messages: updated, title: prev.length === 0 ? input.trim().slice(0, 20) + (input.length > 20 ? '...' : '') : s.title }
              : s
          )
        } else {
          return [...prevSessions, { id: currentSessionId, title: input.trim().slice(0, 20) + (input.length > 20 ? '...' : ''), messages: updated }]
        }
      })
      return updated
    })
    setInput('')
    setLoading(true)

    try {
      await chatWithPlanner(currentSessionId, userMsg.content, (event) => {
        if (event.type === 'agent_status') {
          setAgentStatus(prev => {
            const newStatus = { ...prev }
            newStatus[event.agent] = {
              status: event.status,
              last_active: Date.now(),
              current_task: null
            }
            return newStatus
          })
          
          // 实时显示智能体消息（流式输出）
          if (event.content && event.content.length > 0) {
            // 过滤掉简短的状态消息，只显示有意义的内容
            if (event.content.length > 20 || event.status === 'done') {
              setMessages(prev => {
                // 检查是否已有相同消息
                const exists = prev.some(
                  m => m.agent === event.agent && 
                       m.content.startsWith(event.content?.substring(0, 50) || '')
                )
                if (exists) return prev
                
                const assistantMsg: ChatMessage = {
                  id: `agent-${event.agent}-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
                  role: 'assistant',
                  content: event.content || '',
                  agent: event.agent,
                  timestamp: Date.now(),
                }
                const updated = [...prev, assistantMsg]
                setSessions(prevSessions =>
                  prevSessions.map(s =>
                    s.id === currentSessionId ? { ...s, messages: updated } : s
                  )
                )
                return updated
              })
            }
          }
        }
      })
      await fetchAgentStatus()
    } catch (err: any) {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'system',
        content: `Error: ${err.message}`,
        timestamp: Date.now(),
      }])
    } finally {
      setLoading(false)
    }
  }, [input, loading, currentSessionId, messages])

  const handleStopChat = useCallback(async () => {
    if (!currentSessionId) return
    try {
      await stopChat(currentSessionId)
      setLoading(false)
    } catch (err) {
      console.error('停止聊天失败:', err)
    }
  }, [currentSessionId])

  const sendToAgent = useCallback(async (agentName: string) => {
    if (!input.trim() || loading || !currentSessionId) return
    
    const query = input.trim()
    setInput('')
    setLoading(true)
    
    setMessages(prev => {
      const userMsg: ChatMessage = {
        id: Date.now().toString(),
        role: 'user',
        content: `@${agentName} ${query}`,
        timestamp: Date.now(),
      }
      const updated = [...prev, userMsg]
      setSessions(prevSessions =>
        prevSessions.map(s =>
          s.id === currentSessionId
            ? { ...s, messages: updated, title: prev.length === 0 ? query.slice(0, 20) + (query.length > 20 ? '...' : '') : s.title }
            : s
        )
      )
      return updated
    })

    try {
      const result = await chatWithAgent(currentSessionId, agentName, query)
      setMessages(prev => {
        const assistantMsg: ChatMessage = {
          id: Date.now().toString(),
          role: 'assistant',
          content: result.response,
          agent: agentName,
          timestamp: Date.now(),
        }
        const updated = [...prev, assistantMsg]
        setSessions(prevSessions =>
          prevSessions.map(s =>
            s.id === currentSessionId ? { ...s, messages: updated } : s
          )
        )
        return updated
      })
    } catch (err: any) {
      setMessages(prev => {
        const systemMsg: ChatMessage = {
          id: Date.now().toString(),
          role: 'system',
          content: `Error: ${err.message}`,
          timestamp: Date.now(),
        }
        return [...prev, systemMsg]
      })
    } finally {
      setLoading(false)
    }
  }, [input, loading, currentSessionId])

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!currentSessionId || !files || !files.length) return
    const file = files[0]
    
    try {
      const info = await uploadDataset(currentSessionId, file)
      setDataset({ loaded: true, name: 'df', shape: info.shape, columns: info.columns, description: info.description })
      setShowUpload(false)
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'system',
        content: `📁 Uploaded **${file.name}** (${info.shape?.[0]} rows × ${info.shape?.[1]} cols)`,
        timestamp: Date.now(),
      }])
    } catch (err: any) {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'system',
        content: `Upload failed: ${err.message}`,
        timestamp: Date.now(),
      }])
    }
    // Reset input so same file can be re-selected
    e.target.value = ''
  }, [currentSessionId])

  // Drag & drop handlers for the upload modal area
  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragOver(true) }, [])
  const handleDragLeave = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragOver(false) }, [])
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const files = e.dataTransfer.files
    if (!currentSessionId || !files.length) return
    const file = files[0]
    
    try {
      const info = await uploadDataset(currentSessionId, file)
      setDataset({ loaded: true, name: 'df', shape: info.shape, columns: info.columns, description: info.description })
      setShowUpload(false)
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'system',
        content: `📁 Uploaded **${file.name}** (${info.shape?.[0]} rows × ${info.shape?.[1]} cols)`,
        timestamp: Date.now(),
      }])
    } catch (err: any) {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'system',
        content: `Upload failed: ${err.message}`,
        timestamp: Date.now(),
      }])
    }
  }, [currentSessionId])

  const saveModelConfig = useCallback(async () => {
    if (!currentSessionId) return
    try {
      await setModelConfig(currentSessionId, modelConfig)
      localStorage.setItem('datapilot-model', JSON.stringify({ provider: modelConfig.provider, model: modelConfig.model }))
      setShowSettings(false)
    } catch {}
  }, [currentSessionId, modelConfig])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // Code editing handlers
  const startEditCode = useCallback((code: string) => {
    setEditingCode(code)
    setEditedCode(code)
    setCodeOutput('')
  }, [])

  const cancelEditCode = useCallback(() => {
    setEditingCode(null)
    setEditedCode('')
    setCodeOutput('')
  }, [])

  const runEditedCode = useCallback(async () => {
    if (!editedCode.trim() || !currentSessionId || runningCode) return
    setRunningCode(true)
    setCodeOutput('⏳ Running...')

    try {
      const result = await executeCode(currentSessionId, editedCode)
      if (result.status === 'success') {
        setCodeOutput(result.output)
      } else {
        setCodeOutput(`❌ Error: ${result.output}`)
      }
    } catch (err: any) {
      setCodeOutput(`❌ Error: ${err.message}`)
    } finally {
      setRunningCode(false)
    }
  }, [editedCode, currentSessionId, runningCode])

  useEffect(() => {
    const saved = localStorage.getItem('datapilot-model')
    if (saved) {
      try {
        const { provider, model } = JSON.parse(saved)
        if (provider) setModelConfigState(prev => ({ ...prev, provider }))
        if (model) setModelConfigState(prev => ({ ...prev, model }))
      } catch {}
    }
  }, [])

  return (
    <div className="flex h-screen">
      {/* ── Sidebar ─────────────────────────────────────────── */}
      <aside className="w-64 flex-shrink-0 bg-[var(--bg-secondary)] border-r border-[var(--border)] flex flex-col">
        <div className="p-4 border-b border-[var(--border)]">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-lg font-bold text-brand-500">⚡ DataPilot</h1>
              <p className="text-xs text-[var(--text-secondary)] mt-1">AI 数据分析助手</p>
            </div>
            <button onClick={toggleTheme} className="p-1.5 rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]" title={theme === 'dark' ? '浅色模式' : '深色模式'}>
              {theme === 'dark' ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
        
        {/* Dataset info */}
        <div className="p-4 border-b border-[var(--border)]">
          <h3 className="text-sm font-semibold mb-2 text-[var(--text-secondary)]">数据集</h3>
          {dataset.loaded ? (
            <div className="text-xs space-y-1">
              <p className="text-green-400">✅ {dataset.name}</p>
              <p className="text-[var(--text-secondary)]">{dataset.shape?.[0]} rows × {dataset.shape?.[1]} cols</p>
              <details className="mt-2">
                <summary className="cursor-pointer text-[var(--text-secondary)] hover:text-[var(--text-primary)]">列名</summary>
                <div className="mt-1 max-h-32 overflow-y-auto">
                  {dataset.columns?.map(col => (
                    <p key={col} className="text-[var(--text-secondary)]">{col}</p>
                  ))}
                </div>
              </details>
            </div>
          ) : (
            <p className="text-xs text-[var(--text-secondary)]">未加载数据集</p>
          )}
          <button
            onClick={() => setShowUpload(true)}
            className="mt-2 w-full text-xs bg-[var(--bg-tertiary)] hover:bg-brand-600 text-[var(--text-primary)] py-1.5 rounded transition-colors"
          >
            📎 上传文件
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-3 border-b border-[var(--border)]">
          <button onClick={() => initSession()} className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white rounded-lg font-medium text-sm transition-colors">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            新建对话
          </button>
        </div>

        {/* History */}
        <div className="border-b border-[var(--border)] px-3 py-2">
          <div className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider px-2 py-2">历史记录</div>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {sessions.map(session => (
              <div key={session.id} onClick={() => handleSelectSession(session.id)} className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors ${session.id === currentSessionId ? 'bg-[var(--bg-tertiary)] text-[var(--text-primary)]' : 'text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)]'}`}>
                <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                {editingSessionId === session.id ? (
                  <input
                    type="text"
                    value={editSessionName}
                    onChange={(e) => setEditSessionName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') saveSessionName()
                      if (e.key === 'Escape') cancelEditSessionName()
                    }}
                    onBlur={saveSessionName}
                    className="flex-1 text-sm bg-[var(--bg-primary)] border border-[var(--border)] rounded px-2 py-1 focus:outline-none focus:border-brand-500"
                    autoFocus
                  />
                ) : (
                  <span className="flex-1 text-sm truncate">{session.title || '新对话'}</span>
                )}
                {editingSessionId !== session.id && (
                  <>
                    <button onClick={(e) => startEditSessionName(e, session)} className="opacity-0 group-hover:opacity-100 p-1 hover:bg-[var(--bg-primary)] rounded">
                      <svg className="w-3 h-3 text-[var(--text-secondary)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                      </svg>
                    </button>
                    <button onClick={(e) => handleDeleteSession(e, session.id)} className="opacity-0 group-hover:opacity-100 p-1 hover:bg-[var(--bg-primary)] rounded">
                      <svg className="w-3 h-3 text-[var(--text-secondary)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Agent Status */}
        <div className="p-4 border-b border-[var(--border)] flex-1 overflow-y-auto">
          <h3 className="text-sm font-semibold mb-2 text-[var(--text-secondary)]">智能体状态</h3>
          <div className="space-y-1.5">
            {AGENT_LIST.map(agent => {
              const agentInfo = agentStatus[agent.name]
              const status = agentInfo?.status || 'idle'
              const statusText = status === 'idle' ? '空闲' : '工作中'
              return (
                <div key={agent.name} className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-[var(--bg-tertiary)]">
                  <span className="text-sm">{agent.emoji}</span>
                  <span className="flex-1 text-xs text-[var(--text-secondary)] truncate">{agent.display}</span>
                  <span className={`w-2 h-2 rounded-full ${status === 'idle' ? 'bg-gray-400' : status === 'thinking' ? 'bg-blue-500' : status === 'working' ? 'bg-yellow-500' : status === 'reviewing' ? 'bg-purple-500' : status === 'done' ? 'bg-green-500' : 'bg-red-500'} ${status !== 'idle' ? 'animate-pulse' : ''}`} title={statusText} />
                </div>
              )
            })}
          </div>
        </div>
        
        {/* Settings button */}
        <div className="p-4">
          <button
            onClick={() => setShowSettings(true)}
            className="w-full text-xs bg-[var(--bg-tertiary)] hover:bg-[var(--border)] py-2 rounded transition-colors text-[var(--text-secondary)]"
          >
            ⚙️ 设置
          </button>
        </div>
      </aside>

      {/* ── Main Chat Area ──────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center space-y-4">
                <h2 className="text-2xl font-bold text-[var(--text-primary)]">
                  欢迎使用 DataPilot
                </h2>
                <p className="text-[var(--text-secondary)] max-w-md">
                  上传 CSV 或 XLSX 文件，然后对数据进行提问。AI 规划器将自动把您的问题路由到合适的智能体。
                </p>
                {/* 智能体分为两行显示 */}
                <div className="space-y-2">
                  {/* 第一行：丞相、太尉、御史大夫 */}
                  <div className="flex flex-wrap gap-2 justify-center">
                    {agents.filter(a => ['chancellor_agent', 'commander_agent', 'censor_agent'].includes(a.name)).map(agent => (
                      <span key={agent.name} className="text-sm px-3 py-1 bg-[var(--bg-secondary)] rounded-full border border-[var(--border)]">
                        {agent.icon} {agent.display}
                      </span>
                    ))}
                  </div>
                  {/* 第二行：4个执行智能体 */}
                  <div className="flex flex-wrap gap-2 justify-center">
                    {agents.filter(a => !['chancellor_agent', 'commander_agent', 'censor_agent'].includes(a.name)).map(agent => (
                      <span key={agent.name} className="text-sm px-3 py-1 bg-[var(--bg-secondary)] rounded-full border border-[var(--border)]">
                        {agent.icon} {agent.display}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {messages.map(msg => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`flex items-start gap-3 max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 text-sm">
                  {msg.role === 'user' ? (
                    <span className="bg-brand-600 text-white">👤</span>
                  ) : msg.agent ? (
                    <span className="bg-[var(--bg-tertiary)] text-[var(--text-primary)]">
                      {AGENT_LIST.find(a => a.name === msg.agent)?.emoji || '🤖'}
                    </span>
                  ) : (
                    <span className="bg-[var(--bg-tertiary)] text-[var(--text-primary)]">📢</span>
                  )}
                </div>
                <div className={`rounded-2xl p-4 ${
                  msg.role === 'user'
                    ? 'bg-brand-600 text-white rounded-tr-sm'
                    : msg.role === 'system'
                    ? 'bg-[var(--bg-secondary)] border border-[var(--border)] rounded-tl-sm'
                    : 'bg-[var(--bg-secondary)] border border-[var(--border)] rounded-tl-sm'
                }`}>
                  {msg.agent && (
                    <p className="text-xs text-brand-500 font-semibold mb-1">{AGENT_LIST.find(a => a.name === msg.agent)?.display || msg.agent}</p>
                  )}
                  <MessageContent content={msg.content} />
                  {msg.role === 'assistant' && extractCodeFromMarkdown(msg.content) && (
                    <button
                      onClick={() => startEditCode(extractCodeFromMarkdown(msg.content)!)}
                      className="mt-3 text-xs bg-[var(--bg-tertiary)] hover:bg-brand-600 text-[var(--text-primary)] hover:text-white px-3 py-1.5 rounded transition-colors"
                    >
                      ✏️ 编辑并运行代码
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="flex justify-start">
              <div className="flex items-start gap-3 max-w-[80%]">
                <div className="w-8 h-8 rounded-full bg-[var(--bg-tertiary)] text-[var(--text-primary)] flex items-center justify-center flex-shrink-0 text-sm">🤖</div>
                <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-2xl rounded-tl-sm p-4">
                  <div className="flex items-center gap-2 text-[var(--text-secondary)]">
                    <div className="animate-spin h-4 w-4 border-2 border-brand-500 border-t-transparent rounded-full" />
                    <span className="text-sm">分析中...</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        {/* Input Area */}
        <div className="border-t border-[var(--border)] p-4">
          <div className="flex gap-3 max-w-4xl mx-auto">
            <button onClick={() => setShowUpload(true)} className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)] text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] transition-colors flex-shrink-0">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </button>
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={dataset.loaded ? "输入关于数据的问题..." : "请先上传数据集..."}
              disabled={loading || !dataset.loaded}
              className="chat-input flex-1 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg px-4 py-3 text-sm text-[var(--text-primary)] placeholder-[var(--text-secondary)] disabled:opacity-50"
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim() || !dataset.loaded}
              className="bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white px-6 py-3 rounded-lg text-sm font-medium transition-colors"
            >
              发送
            </button>
            {loading && (
              <button
                onClick={handleStopChat}
                className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg text-sm font-medium transition-colors"
              >
                停止
              </button>
            )}
          </div>
        </div>
      </main>

      {/* ── Upload Modal ────────────────────────────────────── */}
      {showUpload && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowUpload(false)}>
          <div className="bg-[var(--bg-secondary)] rounded-xl p-6 w-[480px] border border-[var(--border)]" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-semibold mb-4">上传数据集</h3>
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragOver ? 'border-brand-500 bg-brand-50/5' : 'border-[var(--border)] hover:border-brand-500'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileSelect}
                className="hidden"
              />
              <div className="text-4xl mb-3">📁</div>
              <p className="text-[var(--text-primary)] font-medium">点击或拖拽文件到此处</p>
              <p className="text-[var(--text-secondary)] text-sm mt-1">支持 CSV, XLSX, XLS 格式</p>
            </div>
            <button
              onClick={() => setShowUpload(false)}
              className="mt-4 w-full py-2 rounded-lg border border-[var(--border)] text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] transition-colors"
            >
              取消
            </button>
          </div>
        </div>
      )}

      {/* ── Settings Modal ──────────────────────────────────── */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowSettings(false)}>
          <div className="bg-[var(--bg-secondary)] rounded-xl p-6 w-[480px] border border-[var(--border)]" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-semibold mb-4">⚙️ 模型设置</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-[var(--text-secondary)] mb-1">提供商</label>
                <select
                  value={modelConfig.provider}
                  onChange={e => setModelConfigState(prev => ({ ...prev, provider: e.target.value }))}
                  className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)]"
                >
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                  <option value="groq">Groq</option>
                  <option value="gemini">Gemini</option>
                  <option value="deepseek">DeepSeek</option>
                  <option value="custom">自定义</option>
                </select>
              </div>
              
              {modelConfig.provider === 'custom' && (
                <div>
                  <label className="block text-sm text-[var(--text-secondary)] mb-1">自定义提供商名称</label>
                  <input
                    type="text"
                    value={modelConfig.model.split('/')[0] || ''}
                    onChange={e => setModelConfigState(prev => {
                      const parts = prev.model.split('/')
                      const modelName = parts.slice(1).join('/')
                      return { ...prev, model: `${e.target.value}/${modelName}` }
                    })}
                    className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)]"
                    placeholder="例如: ollama"
                  />
                </div>
              )}
              
              <div>
                <label className="block text-sm text-[var(--text-secondary)] mb-1">模型</label>
                <input
                  type="text"
                  value={modelConfig.provider === 'custom' ? modelConfig.model.split('/').slice(1).join('/') : modelConfig.model}
                  onChange={e => setModelConfigState(prev => {
                    if (prev.provider === 'custom') {
                      const customProvider = prev.model.split('/')[0] || ''
                      return { ...prev, model: `${customProvider}/${e.target.value}` }
                    }
                    return { ...prev, model: e.target.value }
                  })}
                  className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)]"
                  placeholder="例如: gpt-4o-mini"
                />
              </div>
              
              <div>
                <label className="block text-sm text-[var(--text-secondary)] mb-1">API 密钥</label>
                <input
                  type="password"
                  value={modelConfig.api_key}
                  onChange={e => setModelConfigState(prev => ({ ...prev, api_key: e.target.value }))}
                  className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)]"
                  placeholder="sk-..."
                />
              </div>
              
              <p className="text-xs text-[var(--text-secondary)]">
                API 密钥仅存储在当前浏览器会话中，仅在后端调用 LLM 时发送。
              </p>
            </div>
            
            <div className="flex gap-3 mt-6">
              <button
                onClick={saveModelConfig}
                className="flex-1 bg-brand-600 hover:bg-brand-700 text-white py-2 rounded-lg text-sm font-medium transition-colors"
              >
                保存
              </button>
              <button
                onClick={() => setShowSettings(false)}
                className="flex-1 bg-[var(--bg-tertiary)] hover:bg-[var(--border)] py-2 rounded-lg text-sm transition-colors text-[var(--text-secondary)]"
              >
                取消
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Code Editor Modal ─────────────────────────────────────── */}
      {editingCode !== null && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={cancelEditCode}>
          <div 
            className="bg-[var(--bg-secondary)] rounded-xl p-6 w-[90vw] max-w-4xl h-[80vh] border border-[var(--border)] flex flex-col" 
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">✏️ 编辑并运行代码</h3>
              <button 
                onClick={cancelEditCode}
                className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] text-2xl leading-none"
              >
                ×
              </button>
            </div>
            
            {/* Code Editor */}
            <textarea
              value={editedCode}
              onChange={e => setEditedCode(e.target.value)}
              className="flex-1 min-h-[200px] w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg p-4 text-sm font-mono text-[var(--text-primary)] resize-none"
              placeholder="# 在此处输入 Python 代码..."
              spellCheck={false}
            />
            
            {/* Action Buttons */}
            <div className="flex gap-3 mt-4">
              <button
                onClick={runEditedCode}
                disabled={runningCode || !editedCode.trim()}
                className="flex-1 bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white py-2 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
              >
                {runningCode ? (
                  <>
                    <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                    运行中...
                  </>
                ) : (
                  <>▶️ 运行代码</>
                )}
              </button>
              <button
                onClick={cancelEditCode}
                className="flex-1 bg-[var(--bg-tertiary)] hover:bg-[var(--border)] py-2 rounded-lg text-sm transition-colors text-[var(--text-secondary)]"
              >
                取消
              </button>
            </div>
            
            {/* Code Output */}
            {codeOutput && (
              <div className="mt-4 border-t border-[var(--border)] pt-4">
                <h4 className="text-sm font-semibold mb-2">📤 输出</h4>
                <div className="bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg p-4 max-h-[200px] overflow-y-auto text-sm">
                  <ReactMarkdown>{codeOutput}</ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
