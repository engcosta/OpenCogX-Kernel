import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 120000, // 2 minutes for long operations
    headers: {
        'Content-Type': 'application/json',
    },
});

// Types
export interface KernelStatus {
    world: WorldStats;
    memory: MemoryStats;
    goals: GoalStats;
    reasoning: ReasoningStats;
    meta: MetaStats;
    learning_loop: LearningLoopStats;
    ingestion: IngestionStats;
}

export interface WorldStats {
    total_states: number;
    total_events: number;
    causal_chains: number;
}

export interface MemoryStats {
    episodic_count: number;
    semantic_count: number;
    recent_memories: number;
}

export interface GoalStats {
    active_goals: number;
    completed_goals: number;
    failed_goals: number;
    top_priority?: string;
}

export interface ReasoningStats {
    strategies_used: Record<string, number>;
    total_reasonings: number;
    avg_confidence: number;
}

export interface MetaStats {
    self_corrections: number;
    strategy_shifts: number;
    pending_changes: number;
}

export interface LearningLoopStats {
    total_iterations: number;
    passed: number;
    failed: number;
    pass_rate: number;
    average_confidence: number;
}

export interface IngestionStats {
    files_processed: number;
    chunks_created: number;
    entities_extracted: number;
    relations_created: number;
}

export interface IngestRequest {
    path: string;
    is_directory: boolean;
}

export interface IngestResponse {
    success: boolean;
    files_processed?: number;
    chunks?: number;
    entities?: number;
    relations?: number;
    error?: string;
}

export interface LearnRequest {
    iterations: number;
    interval_seconds: number;
}

export interface LearnResponse {
    success: boolean;
    total_iterations: number;
    passed: number;
    failed: number;
    pass_rate: number;
    average_confidence: number;
}

export interface StepResponse {
    iteration_id: number;
    verdict: string;
    question: string;
    answer: string;
    strategy: string;
    confidence: number;
    duration_ms: number;
}

export interface AskRequest {
    question: string;
    strategy?: 'hybrid' | 'search' | 'reason' | 'auto';
    strict_mode?: boolean;
}

export interface AskResponse {
    question: string;
    answer: string;
    confidence: number;
    strategy_used: string;
    duration_seconds: number;
    context_used: string[];
}

export interface MemoryQueryRequest {
    query: string;
    limit?: number;
    memory_types?: string[];
}

export interface MemoryResult {
    id: string;
    type: string;
    content: Record<string, unknown>;
}

export interface Goal {
    id: string;
    description: string;
    priority: number;
    state: string;
    created_at: string;
    progress?: number;
}

export interface GoalsResponse {
    active_goals: Goal[];
    completed_count: number;
    failed_count: number;
    stats: GoalStats;
}

export interface EvaluationReport {
    timestamp: string;
    summary: {
        total_iterations: number;
        pass_rate: number;
        avg_confidence: number;
    };
    metrics: Record<string, unknown>;
    recommendations: string[];
}

export interface MetricsExport {
    snapshots: Array<{
        timestamp: string;
        memory: MemoryStats;
        world: WorldStats;
        goals: GoalStats;
        reasoning: ReasoningStats;
    }>;
}

// API Functions
export const apiService = {
    // Health & Status
    getHealth: () => api.get<{ status: string; llm: boolean; vector: boolean; graph: boolean }>('/health'),
    getStatus: () => api.get<KernelStatus>('/status'),

    // Ingestion
    ingest: (request: IngestRequest) => api.post<IngestResponse>('/ingest', request),

    // Learning
    learn: (request: LearnRequest) => api.post<LearnResponse>('/learn', request),
    step: () => api.post<StepResponse>('/step'),

    // Ask
    ask: (request: AskRequest) => api.post<AskResponse>('/ask', request),

    // Evaluation
    evaluate: () => api.get<EvaluationReport>('/evaluate'),

    // Memory
    queryMemory: (request: MemoryQueryRequest) => api.post<{ query: string; count: number; results: MemoryResult[] }>('/memory/query', request),
    getMemoryStats: () => api.get<MemoryStats>('/memory/stats'),

    // Goals
    getGoals: () => api.get<GoalsResponse>('/goals'),

    // World
    getWorld: () => api.get<{
        stats: WorldStats;
        recent_states: Array<Record<string, unknown>>;
        recent_events: Array<Record<string, unknown>>;
    }>('/world'),

    // Reasoning
    getReasoning: () => api.get<ReasoningStats>('/reasoning'),

    // Meta
    getMeta: () => api.get<{
        self_knowledge: Record<string, unknown>;
        stats: MetaStats;
        pending_changes: Array<Record<string, unknown>>;
    }>('/meta'),

    // Metrics
    exportMetrics: () => api.get<MetricsExport>('/metrics/export'),
    collectSnapshot: () => api.post('/metrics/snapshot'),

    // Clean databases
    clean: (confirm: boolean) => api.post<{ status: string; message: string }>('/clean', { confirm }),
};

export default apiService;
