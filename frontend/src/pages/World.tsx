import { useQuery } from '@tanstack/react-query';
import {
    Cpu,
    RefreshCw,
    Network,
    Activity,
    GitBranch,
    Clock,
    Zap,
} from 'lucide-react';
import { apiService } from '../services/api';
import { useLogStore } from '../stores';
import './World.css';

export function World() {
    const { addLog } = useLogStore();

    const { data, isLoading, refetch } = useQuery({
        queryKey: ['world'],
        queryFn: async () => {
            addLog({
                level: 'info',
                source: 'world',
                message: 'Fetching world model...',
            });
            const res = await apiService.getWorld();
            addLog({
                level: 'success',
                source: 'world',
                message: 'World model loaded',
            });
            return res.data;
        },
        refetchInterval: 15000,
    });

    return (
        <div className="world-page">
            <div className="page-header">
                <div>
                    <h1>World Model</h1>
                    <p>States, events, and causal relationships</p>
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
            <div className="world-stats">
                <div className="stat-card">
                    <Network size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{data?.stats.total_states ?? 0}</span>
                        <span className="stat-label">Total States</span>
                    </div>
                </div>
                <div className="stat-card">
                    <Activity size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{data?.stats.total_events ?? 0}</span>
                        <span className="stat-label">Total Events</span>
                    </div>
                </div>
                <div className="stat-card">
                    <GitBranch size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{data?.stats.causal_chains ?? 0}</span>
                        <span className="stat-label">Causal Chains</span>
                    </div>
                </div>
            </div>

            <div className="world-grid">
                {/* Recent States */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Cpu size={20} />
                            Recent States
                        </h3>
                    </div>
                    <div className="items-list">
                        {data?.recent_states && data.recent_states.length > 0 ? (
                            data.recent_states.map((state, idx) => (
                                <div key={idx} className="world-item">
                                    <div className="item-header">
                                        <span className="item-id">{(state as any).id?.slice(0, 8) ?? `state-${idx}`}</span>
                                        <span className="item-time">
                                            <Clock size={12} />
                                            {(state as any).timestamp
                                                ? new Date((state as any).timestamp).toLocaleTimeString()
                                                : 'N/A'}
                                        </span>
                                    </div>
                                    <pre className="item-content">
                                        {JSON.stringify(state, null, 2)}
                                    </pre>
                                </div>
                            ))
                        ) : (
                            <div className="empty-state">
                                <Network size={32} />
                                <p>No states recorded yet</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Recent Events */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Zap size={20} />
                            Recent Events
                        </h3>
                    </div>
                    <div className="items-list">
                        {data?.recent_events && data.recent_events.length > 0 ? (
                            data.recent_events.map((event, idx) => (
                                <div key={idx} className="world-item event">
                                    <div className="item-header">
                                        <span className="item-id">{(event as any).id?.slice(0, 8) ?? `event-${idx}`}</span>
                                        <span className="item-type badge badge-info">
                                            {(event as any).type ?? 'unknown'}
                                        </span>
                                    </div>
                                    <pre className="item-content">
                                        {JSON.stringify(event, null, 2)}
                                    </pre>
                                </div>
                            ))
                        ) : (
                            <div className="empty-state">
                                <Activity size={32} />
                                <p>No events recorded yet</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default World;
