import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    Brain,
    Database,
    Target,
    Cpu,
    Sparkles,
    Activity,
    TrendingUp,
    MemoryStick,
    Network,
    CheckCircle,
    XCircle,
    RefreshCw,
} from 'lucide-react';
import { apiService } from '../services/api';
import { useLogStore, useNotificationStore } from '../stores';
import LogPanel from '../components/LogPanel';
import './Dashboard.css';

export function Dashboard() {
    const { addLog } = useLogStore();
    const { addNotification } = useNotificationStore();

    const {
        data: status,
        isLoading,
        error,
        refetch,
        dataUpdatedAt,
    } = useQuery({
        queryKey: ['status'],
        queryFn: async () => {
            addLog({ level: 'info', source: 'dashboard', message: 'Fetching kernel status...' });
            const res = await apiService.getStatus();
            addLog({ level: 'success', source: 'dashboard', message: 'Status fetched successfully' });
            return res.data;
        },
        refetchInterval: 5000,
    });

    const { data: health } = useQuery({
        queryKey: ['health'],
        queryFn: async () => {
            const res = await apiService.getHealth();
            return res.data;
        },
        refetchInterval: 10000,
    });

    useEffect(() => {
        if (error) {
            addNotification({
                type: 'error',
                title: 'Connection Error',
                message: 'Failed to connect to AGI Kernel API',
            });
            addLog({
                level: 'error',
                source: 'dashboard',
                message: `API Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
            });
        }
    }, [error]);

    const formatLastUpdate = () => {
        if (!dataUpdatedAt) return 'Never';
        const diff = Date.now() - dataUpdatedAt;
        if (diff < 1000) return 'Just now';
        if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
        return `${Math.floor(diff / 60000)}m ago`;
    };

    return (
        <div className="dashboard">
            <div className="page-header">
                <div>
                    <h1>Dashboard</h1>
                    <p>Overview of the AGI Kernel system status</p>
                </div>
                <div className="header-actions">
                    <span className="last-update">Updated {formatLastUpdate()}</span>
                    <button
                        className="btn btn-secondary"
                        onClick={() => refetch()}
                        disabled={isLoading}
                    >
                        <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
                        Refresh
                    </button>
                </div>
            </div>

            {/* Health Status */}
            <div className="health-status">
                <div className="health-item">
                    <span className={`health-dot ${health?.llm ? 'online' : 'offline'}`}></span>
                    <span>LLM Plugin</span>
                </div>
                <div className="health-item">
                    <span className={`health-dot ${health?.vector ? 'online' : 'offline'}`}></span>
                    <span>Vector DB</span>
                </div>
                <div className="health-item">
                    <span className={`health-dot ${health?.graph ? 'online' : 'offline'}`}></span>
                    <span>Graph DB</span>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="stats-grid">
                <StatCard
                    icon={<MemoryStick />}
                    label="Total Memories"
                    value={
                        (status?.memory.episodic_count ?? 0) +
                        (status?.memory.semantic_count ?? 0)
                    }
                    subValue={`${status?.memory.episodic_count ?? 0} episodic, ${status?.memory.semantic_count ?? 0} semantic`}
                    color="purple"
                />
                <StatCard
                    icon={<Network />}
                    label="World States"
                    value={status?.world.total_states ?? 0}
                    subValue={`${status?.world.total_events ?? 0} events, ${status?.world.causal_chains ?? 0} chains`}
                    color="blue"
                />
                <StatCard
                    icon={<Target />}
                    label="Active Goals"
                    value={status?.goals.active_goals ?? 0}
                    subValue={`${status?.goals.completed_goals ?? 0} completed, ${status?.goals.failed_goals ?? 0} failed`}
                    color="green"
                />
                <StatCard
                    icon={<Brain />}
                    label="Learning Pass Rate"
                    value={`${((status?.learning_loop.pass_rate ?? 0) * 100).toFixed(1)}%`}
                    subValue={`${status?.learning_loop.total_iterations ?? 0} total iterations`}
                    color="orange"
                />
            </div>

            {/* Two Column Layout */}
            <div className="dashboard-grid">
                {/* Reasoning Stats */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Cpu size={20} />
                            Reasoning Strategies
                        </h3>
                    </div>
                    <div className="strategy-stats">
                        {status?.reasoning.strategies_used &&
                            Object.entries(status.reasoning.strategies_used).map(([strategy, count]) => (
                                <div key={strategy} className="strategy-item">
                                    <span className="strategy-name">{strategy.replace(/_/g, ' ')}</span>
                                    <div className="strategy-bar">
                                        <div
                                            className="strategy-fill"
                                            style={{
                                                width: `${Math.min((count / (status?.reasoning.total_reasonings || 1)) * 100, 100)}%`,
                                            }}
                                        ></div>
                                    </div>
                                    <span className="strategy-count">{count}</span>
                                </div>
                            ))}
                        {(!status?.reasoning.strategies_used ||
                            Object.keys(status.reasoning.strategies_used).length === 0) && (
                                <div className="empty-state">
                                    <Activity size={32} />
                                    <p>No reasoning data yet</p>
                                </div>
                            )}
                    </div>
                </div>

                {/* Meta-Cognition */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Sparkles size={20} />
                            Meta-Cognition
                        </h3>
                    </div>
                    <div className="meta-stats">
                        <div className="meta-item">
                            <span className="meta-label">Self Corrections</span>
                            <span className="meta-value">{status?.meta.self_corrections ?? 0}</span>
                        </div>
                        <div className="meta-item">
                            <span className="meta-label">Strategy Shifts</span>
                            <span className="meta-value">{status?.meta.strategy_shifts ?? 0}</span>
                        </div>
                        <div className="meta-item">
                            <span className="meta-label">Pending Changes</span>
                            <span className="meta-value">{status?.meta.pending_changes ?? 0}</span>
                        </div>
                        <div className="meta-item">
                            <span className="meta-label">Avg Confidence</span>
                            <span className="meta-value">
                                {((status?.reasoning.avg_confidence ?? 0) * 100).toFixed(1)}%
                            </span>
                        </div>
                    </div>
                </div>

                {/* Learning Summary */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <TrendingUp size={20} />
                            Learning Summary
                        </h3>
                    </div>
                    <div className="learning-summary">
                        <div className="learning-chart">
                            <div className="learning-ring">
                                <svg viewBox="0 0 36 36" className="circular-chart">
                                    <path
                                        className="circle-bg"
                                        d="M18 2.0845
                       a 15.9155 15.9155 0 0 1 0 31.831
                       a 15.9155 15.9155 0 0 1 0 -31.831"
                                    />
                                    <path
                                        className="circle"
                                        strokeDasharray={`${(status?.learning_loop.pass_rate ?? 0) * 100}, 100`}
                                        d="M18 2.0845
                       a 15.9155 15.9155 0 0 1 0 31.831
                       a 15.9155 15.9155 0 0 1 0 -31.831"
                                    />
                                </svg>
                                <span className="ring-value">
                                    {((status?.learning_loop.pass_rate ?? 0) * 100).toFixed(0)}%
                                </span>
                            </div>
                        </div>
                        <div className="learning-details">
                            <div className="learning-stat">
                                <CheckCircle className="icon-success" size={16} />
                                <span>Passed: {status?.learning_loop.passed ?? 0}</span>
                            </div>
                            <div className="learning-stat">
                                <XCircle className="icon-error" size={16} />
                                <span>Failed: {status?.learning_loop.failed ?? 0}</span>
                            </div>
                            <div className="learning-stat">
                                <Activity size={16} />
                                <span>
                                    Avg Confidence: {((status?.learning_loop.average_confidence ?? 0) * 100).toFixed(1)}%
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Ingestion Stats */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Database size={20} />
                            Ingestion Stats
                        </h3>
                    </div>
                    <div className="ingestion-stats">
                        <div className="ingestion-item">
                            <span className="ingestion-value">{status?.ingestion.files_processed ?? 0}</span>
                            <span className="ingestion-label">Files</span>
                        </div>
                        <div className="ingestion-item">
                            <span className="ingestion-value">{status?.ingestion.chunks_created ?? 0}</span>
                            <span className="ingestion-label">Chunks</span>
                        </div>
                        <div className="ingestion-item">
                            <span className="ingestion-value">{status?.ingestion.entities_extracted ?? 0}</span>
                            <span className="ingestion-label">Entities</span>
                        </div>
                        <div className="ingestion-item">
                            <span className="ingestion-value">{status?.ingestion.relations_created ?? 0}</span>
                            <span className="ingestion-label">Relations</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Logs */}
            <div className="mt-lg">
                <LogPanel title="Dashboard Activity" maxHeight="300px" />
            </div>
        </div>
    );
}

interface StatCardProps {
    icon: React.ReactNode;
    label: string;
    value: number | string;
    subValue?: string;
    color?: 'purple' | 'blue' | 'green' | 'orange';
}

function StatCard({ icon, label, value, subValue, color = 'purple' }: StatCardProps) {
    return (
        <div className={`stat-card stat-${color}`}>
            <div className="stat-icon">{icon}</div>
            <div className="stat-content">
                <span className="stat-label">{label}</span>
                <span className="stat-value">{value}</span>
                {subValue && <span className="stat-sub">{subValue}</span>}
            </div>
        </div>
    );
}

export default Dashboard;
