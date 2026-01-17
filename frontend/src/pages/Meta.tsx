import { useQuery } from '@tanstack/react-query';
import {
    Sparkles,
    RefreshCw,
    Brain,
    GitBranch,
    AlertCircle,
    Lightbulb,
    Settings,
} from 'lucide-react';
import { apiService } from '../services/api';
import { useLogStore } from '../stores';
import './Meta.css';

export function Meta() {
    const { addLog } = useLogStore();

    const { data, isLoading, refetch } = useQuery({
        queryKey: ['meta'],
        queryFn: async () => {
            addLog({
                level: 'info',
                source: 'meta',
                message: 'Fetching meta-cognition data...',
            });
            const res = await apiService.getMeta();
            addLog({
                level: 'success',
                source: 'meta',
                message: 'Meta-cognition data loaded',
            });
            return res.data;
        },
        refetchInterval: 15000,
    });

    return (
        <div className="meta-page">
            <div className="page-header">
                <div>
                    <h1>Meta-Cognition</h1>
                    <p>Self-monitoring and structural adaptation</p>
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
            <div className="meta-stats">
                <div className="stat-card">
                    <Brain size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{data?.stats.self_corrections ?? 0}</span>
                        <span className="stat-label">Self Corrections</span>
                    </div>
                </div>
                <div className="stat-card">
                    <GitBranch size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{data?.stats.strategy_shifts ?? 0}</span>
                        <span className="stat-label">Strategy Shifts</span>
                    </div>
                </div>
                <div className="stat-card">
                    <AlertCircle size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{data?.stats.pending_changes ?? 0}</span>
                        <span className="stat-label">Pending Changes</span>
                    </div>
                </div>
            </div>

            <div className="meta-grid">
                {/* Self Knowledge */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Lightbulb size={20} />
                            Self Knowledge
                        </h3>
                    </div>
                    <div className="knowledge-content">
                        {data?.self_knowledge && Object.keys(data.self_knowledge).length > 0 ? (
                            <div className="knowledge-list">
                                {Object.entries(data.self_knowledge).map(([key, value]) => (
                                    <div key={key} className="knowledge-item">
                                        <span className="knowledge-key">{key.replace(/_/g, ' ')}</span>
                                        <span className="knowledge-value">
                                            {typeof value === 'object'
                                                ? JSON.stringify(value, null, 2)
                                                : String(value)}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="empty-state">
                                <Brain size={32} />
                                <p>No self-knowledge recorded yet</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Pending Changes */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Settings size={20} />
                            Pending Changes
                        </h3>
                    </div>
                    <div className="changes-content">
                        {data?.pending_changes && data.pending_changes.length > 0 ? (
                            <div className="changes-list">
                                {data.pending_changes.map((change, idx) => (
                                    <div key={idx} className="change-item">
                                        <div className="change-header">
                                            <span className="change-type badge badge-warning">
                                                {(change as any).type ?? 'unknown'}
                                            </span>
                                        </div>
                                        <pre className="change-content">
                                            {JSON.stringify(change, null, 2)}
                                        </pre>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="empty-state">
                                <Settings size={32} />
                                <p>No pending changes</p>
                                <span>The kernel is stable</span>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Meta;
