import { useState, useRef, useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import {
    Send,
    Bot,
    User,
    Clock,
    Target,
    Cpu,
    Trash2,
    Settings2,
    BookOpen,
    X,
} from 'lucide-react';
import { apiService } from '../services/api';
import type { AskRequest } from '../services/api';
import { useLogStore, useNotificationStore } from '../stores';
import './Ask.css';

interface ChatMessage {
    id: string;
    type: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    metadata?: {
        confidence?: number;
        strategy?: string;
        duration?: number;
        context?: string[];
    };
}

export function Ask() {
    const [question, setQuestion] = useState('');
    const [strategy, setStrategy] = useState<'hybrid' | 'search' | 'reason' | 'auto'>('hybrid');
    const [strictMode, setStrictMode] = useState(true);
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [showSettings, setShowSettings] = useState(false);
    const [selectedContext, setSelectedContext] = useState<any[] | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);
    const { addLog } = useLogStore();
    const { addNotification } = useNotificationStore();

    const askMutation = useMutation({
        mutationFn: async (request: AskRequest) => {
            addLog({
                level: 'info',
                source: 'ask',
                message: `Asking: "${request.question.slice(0, 50)}..." (strategy: ${request.strategy})`,
            });
            const response = await apiService.ask(request);
            return response.data;
        },
        onSuccess: (data) => {
            const assistantMessage: ChatMessage = {
                id: `${Date.now()}-assistant`,
                type: 'assistant',
                content: data.answer,
                timestamp: new Date(),
                metadata: {
                    confidence: data.confidence,
                    strategy: data.strategy_used,
                    duration: data.duration_seconds,
                    context: data.context_used,
                },
            };
            setMessages((prev) => [...prev, assistantMessage]);
            addLog({
                level: 'success',
                source: 'ask',
                message: `Answer received (${(data.confidence * 100).toFixed(1)}% confidence, ${data.duration_seconds.toFixed(2)}s)`,
            });
        },
        onError: (error) => {
            const message = error instanceof Error ? error.message : 'Unknown error';
            addLog({
                level: 'error',
                source: 'ask',
                message: `Ask failed: ${message}`,
            });
            addNotification({
                type: 'error',
                title: 'Ask Failed',
                message,
            });

            const errorMessage: ChatMessage = {
                id: `${Date.now()}-error`,
                type: 'assistant',
                content: `Error: ${message}`,
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errorMessage]);
        },
    });

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!question.trim() || askMutation.isPending) return;

        const userMessage: ChatMessage = {
            id: `${Date.now()}-user`,
            type: 'user',
            content: question.trim(),
            timestamp: new Date(),
        };
        setMessages((prev) => [...prev, userMessage]);

        askMutation.mutate({
            question: question.trim(),
            strategy,
            strict_mode: strictMode,
        });

        setQuestion('');
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    const clearChat = () => {
        setMessages([]);
        addLog({
            level: 'info',
            source: 'ask',
            message: 'Chat history cleared',
        });
    };

    const formatTime = (date: Date) => {
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    return (
        <div className="ask-page">
            <div className="page-header">
                <div>
                    <h1>Ask the Kernel</h1>
                    <p>Query the AGI Kernel using its full cognitive architecture</p>
                </div>
                <div className="header-actions">
                    <button
                        className="btn btn-ghost"
                        onClick={() => setShowSettings(!showSettings)}
                    >
                        <Settings2 size={16} />
                        Settings
                    </button>
                    <button className="btn btn-ghost" onClick={clearChat}>
                        <Trash2 size={16} />
                        Clear
                    </button>
                </div>
            </div>

            {/* Settings Panel */}
            {showSettings && (
                <div className="settings-panel">
                    <div className="setting-item">
                        <label className="form-label">Strategy</label>
                        <select
                            className="select"
                            value={strategy}
                            onChange={(e) => setStrategy(e.target.value as typeof strategy)}
                        >
                            <option value="hybrid">Hybrid (Vector + Graph)</option>
                            <option value="search">Fast Recall (Vector Search)</option>
                            <option value="reason">Causal Reasoning (Graph)</option>
                            <option value="auto">Auto (Let Kernel Decide)</option>
                        </select>
                    </div>
                    <div className="setting-item">
                        <label className="toggle-label">
                            <span className="toggle">
                                <input
                                    type="checkbox"
                                    checked={strictMode}
                                    onChange={(e) => setStrictMode(e.target.checked)}
                                />
                                <span className="toggle-slider"></span>
                            </span>
                            <span>Strict Mode</span>
                        </label>
                        <span className="setting-hint">
                            Only use data from Qdrant/Graph, no general knowledge
                        </span>
                    </div>
                </div>
            )}

            {/* Chat Container */}
            <div className="chat-container">
                <div className="messages-list">
                    {messages.length === 0 ? (
                        <div className="chat-empty">
                            <Bot size={64} className="empty-icon" />
                            <h3>Ready to Help</h3>
                            <p>Ask any question about the ingested knowledge base</p>
                            <div className="suggestion-chips">
                                <button
                                    className="chip"
                                    onClick={() => setQuestion('What topics are in the knowledge base?')}
                                >
                                    What topics are in the knowledge base?
                                </button>
                                <button
                                    className="chip"
                                    onClick={() => setQuestion('Summarize the main concepts')}
                                >
                                    Summarize the main concepts
                                </button>
                                <button
                                    className="chip"
                                    onClick={() => setQuestion('What are the key relationships between entities?')}
                                >
                                    Key entity relationships
                                </button>
                            </div>
                        </div>
                    ) : (
                        <>
                            {messages.map((msg) => (
                                <div
                                    key={msg.id}
                                    className={`message ${msg.type === 'user' ? 'user' : 'assistant'}`}
                                >
                                    <div className="message-avatar">
                                        {msg.type === 'user' ? (
                                            <User size={20} />
                                        ) : (
                                            <Bot size={20} />
                                        )}
                                    </div>
                                    <div className="message-content">
                                        <div className="message-header">
                                            <span className="message-author">
                                                {msg.type === 'user' ? 'You' : 'AGI Kernel'}
                                            </span>
                                            <span className="message-time">{formatTime(msg.timestamp)}</span>
                                        </div>
                                        <div className="message-text">{msg.content}</div>
                                        {msg.metadata && (
                                            <div className="message-meta">
                                                {msg.metadata.strategy && (
                                                    <span className="meta-item">
                                                        <Cpu size={12} />
                                                        {msg.metadata.strategy}
                                                    </span>
                                                )}
                                                {msg.metadata.confidence !== undefined && (
                                                    <span className="meta-item">
                                                        <Target size={12} />
                                                        {(msg.metadata.confidence * 100).toFixed(1)}%
                                                    </span>
                                                )}
                                                {msg.metadata.duration !== undefined && (
                                                    <span className="meta-item">
                                                        <Clock size={12} />
                                                        {msg.metadata.duration.toFixed(2)}s
                                                    </span>
                                                )}
                                                {msg.metadata.context && msg.metadata.context.length > 0 && (
                                                    <span
                                                        className="meta-item clickable"
                                                        onClick={() => setSelectedContext(msg.metadata.context || null)}
                                                        title="Click to view sources"
                                                    >
                                                        <BookOpen size={12} />
                                                        {msg.metadata.context.length} sources
                                                    </span>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                            {askMutation.isPending && (
                                <div className="message assistant typing">
                                    <div className="message-avatar">
                                        <Bot size={20} />
                                    </div>
                                    <div className="message-content">
                                        <div className="typing-indicator">
                                            <span></span>
                                            <span></span>
                                            <span></span>
                                        </div>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </>
                    )}
                </div>

                {/* Input Area */}
                <form className="chat-input" onSubmit={handleSubmit}>
                    <div className="input-wrapper">
                        <textarea
                            ref={inputRef}
                            className="chat-textarea"
                            placeholder="Ask a question..."
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            onKeyDown={handleKeyDown}
                            disabled={askMutation.isPending}
                            rows={1}
                        />
                        <button
                            type="submit"
                            className="btn btn-primary send-btn"
                            disabled={!question.trim() || askMutation.isPending}
                        >
                            <Send size={18} />
                        </button>
                    </div>
                    <div className="input-hint">
                        Press <kbd>Enter</kbd> to send, <kbd>Shift + Enter</kbd> for new line
                    </div>
                </form>
            </div>

            {/* Context Dialog */}
            {
                selectedContext && (
                    <div className="dialog-overlay" onClick={() => setSelectedContext(null)}>
                        <div className="dialog-content" onClick={e => e.stopPropagation()}>
                            <div className="dialog-header">
                                <h3>Context Used</h3>
                                <button className="btn btn-ghost close-btn" onClick={() => setSelectedContext(null)}>
                                    <X size={20} />
                                </button>
                            </div>
                            <div className="dialog-body">
                                {selectedContext.map((item, i) => (
                                    <div key={i} className="context-item">
                                        <div className="context-header">
                                            <span className="context-source">
                                                {item.source || 'Unknown Source'}
                                            </span>
                                            <span>{item.type || 'memory'}</span>
                                        </div>
                                        <div className="context-content">
                                            {typeof item.content === 'string'
                                                ? item.content
                                                : (item.content?.text || JSON.stringify(item.content))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )
            }
        </div >
    );
}

export default Ask;
