import { useQuery } from '@tanstack/react-query';
import {
    Target,
    RefreshCw,
    CheckCircle,
    XCircle,
    Clock,
    TrendingUp,
    Flag,
} from 'lucide-react';
import { apiService } from '../services/api';
import { useLogStore } from '../stores';
import './Goals.css';

export function Goals() {
    const { addLog } = useLogStore();

    const { data, isLoading, refetch } = useQuery({
        queryKey: ['goals'],
        queryFn: async () => {
            addLog({
                level: 'info',
                source: 'goals',
                message: 'Fetching goals...',
            });
            const res = await apiService.getGoals();
            addLog({
                level: 'success',
                source: 'goals',
                message: `Found ${res.data.active_goals.length} active goals`,
            });
            return res.data;
        },
        refetchInterval: 10000,
    });

    const getPriorityColor = (priority: number) => {
        if (priority >= 0.8) return 'high';
        if (priority >= 0.5) return 'medium';
        return 'low';
    };

    const getStateIcon = (state: string) => {
        switch (state.toLowerCase()) {
            case 'completed':
                return <CheckCircle size={16} className="icon-success" />;
            case 'failed':
                return <XCircle size={16} className="icon-error" />;
            case 'in_progress':
                return <Clock size={16} className="icon-warning" />;
            default:
                return <Flag size={16} />;
        }
    };

    return (
        <div className="goals-page">
            <div className="page-header">
                <div>
                    <h1>Goal Engine</h1>
                    <p>View and monitor kernel goals and objectives</p>
                </div>
                <button
                    className="btn btn-secondary"
                    onClick={() => refetch()}
                    disabled={isLoading}
                >
                    <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
                    Refresh
                </button>
            </div>

            {/* Stats */}
            <div className="goals-stats">
                <div className="stat-card active">
                    <Target size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{data?.active_goals.length ?? 0}</span>
                        <span className="stat-label">Active Goals</span>
                    </div>
                </div>
                <div className="stat-card completed">
                    <CheckCircle size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{data?.completed_count ?? 0}</span>
                        <span className="stat-label">Completed</span>
                    </div>
                </div>
                <div className="stat-card failed">
                    <XCircle size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{data?.failed_count ?? 0}</span>
                        <span className="stat-label">Failed</span>
                    </div>
                </div>
            </div>

            {/* Active Goals */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">
                        <Target size={20} />
                        Active Goals
                    </h3>
                </div>

                <div className="goals-list">
                    {isLoading ? (
                        <div className="loading-state">
                            <div className="spinner"></div>
                            <p>Loading goals...</p>
                        </div>
                    ) : data?.active_goals && data.active_goals.length > 0 ? (
                        data.active_goals.map((goal) => (
                            <div key={goal.id} className={`goal-item priority-${getPriorityColor(goal.priority)}`}>
                                <div className="goal-header">
                                    <div className="goal-meta">
                                        {getStateIcon(goal.state)}
                                        <span className="goal-id">{goal.id.slice(0, 8)}</span>
                                        <span className={`badge badge-${goal.state === 'in_progress' ? 'warning' : 'neutral'}`}>
                                            {goal.state}
                                        </span>
                                    </div>
                                    <div className="goal-priority">
                                        <TrendingUp size={14} />
                                        Priority: {(goal.priority * 100).toFixed(0)}%
                                    </div>
                                </div>
                                <p className="goal-description">{goal.description}</p>
                                {goal.progress !== undefined && (
                                    <div className="goal-progress">
                                        <div className="progress">
                                            <div
                                                className="progress-bar"
                                                style={{ width: `${goal.progress * 100}%` }}
                                            ></div>
                                        </div>
                                        <span>{(goal.progress * 100).toFixed(0)}%</span>
                                    </div>
                                )}
                                <div className="goal-footer">
                                    <span className="goal-created">
                                        <Clock size={12} />
                                        Created: {new Date(goal.created_at).toLocaleString()}
                                    </span>
                                </div>
                            </div>
                        ))
                    ) : (
                        <div className="empty-state">
                            <Target size={48} />
                            <p>No active goals</p>
                            <span>Goals will be generated during learning iterations</span>
                        </div>
                    )}
                </div>
            </div>

            {/* Goal Stats */}
            {data?.stats && (
                <div className="card mt-lg">
                    <div className="card-header">
                        <h3 className="card-title">
                            <TrendingUp size={20} />
                            Goal Statistics
                        </h3>
                    </div>
                    <div className="goal-stats-grid">
                        <div className="goal-stat-item">
                            <span className="goal-stat-label">Active Goals</span>
                            <span className="goal-stat-value">{data.stats.active_goals}</span>
                        </div>
                        <div className="goal-stat-item">
                            <span className="goal-stat-label">Completed Goals</span>
                            <span className="goal-stat-value">{data.stats.completed_goals}</span>
                        </div>
                        <div className="goal-stat-item">
                            <span className="goal-stat-label">Failed Goals</span>
                            <span className="goal-stat-value">{data.stats.failed_goals}</span>
                        </div>
                        {data.stats.top_priority && (
                            <div className="goal-stat-item full-width">
                                <span className="goal-stat-label">Top Priority</span>
                                <span className="goal-stat-value text-sm">{data.stats.top_priority}</span>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

export default Goals;
