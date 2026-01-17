import { useQuery } from '@tanstack/react-query';
import {
    BarChart3,
    TrendingUp,
    TrendingDown,
    Activity,
    RefreshCw,
    Download,
    CheckCircle,
    XCircle,
    Clock,
    Lightbulb,
} from 'lucide-react';
import { apiService } from '../services/api';
import { useLogStore } from '../stores';
import LogPanel from '../components/LogPanel';
import './Evaluate.css';

export function Evaluate() {
    const { addLog } = useLogStore();

    const { data: report, isLoading, refetch } = useQuery({
        queryKey: ['evaluate'],
        queryFn: async () => {
            addLog({
                level: 'info',
                source: 'evaluate',
                message: 'Generating evaluation report...',
            });
            const res = await apiService.evaluate();
            addLog({
                level: 'success',
                source: 'evaluate',
                message: 'Evaluation report generated',
            });
            return res.data;
        },
    });

    const { data: metrics } = useQuery({
        queryKey: ['metrics'],
        queryFn: async () => {
            const res = await apiService.exportMetrics();
            return res.data;
        },
    });

    const downloadReport = () => {
        if (!report) return;
        const content = JSON.stringify(report, null, 2);
        const blob = new Blob([content], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `evaluation-${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="evaluate-page">
            <div className="page-header">
                <div>
                    <h1>Evaluation Report</h1>
                    <p>Comprehensive analysis of kernel performance and learning</p>
                </div>
                <div className="header-actions">
                    <button
                        className="btn btn-secondary"
                        onClick={() => refetch()}
                        disabled={isLoading}
                    >
                        <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
                        Refresh
                    </button>
                    <button
                        className="btn btn-primary"
                        onClick={downloadReport}
                        disabled={!report}
                    >
                        <Download size={16} />
                        Export
                    </button>
                </div>
            </div>

            {isLoading ? (
                <div className="loading-state">
                    <div className="spinner spinner-lg"></div>
                    <p>Generating evaluation report...</p>
                </div>
            ) : report ? (
                <>
                    {/* Summary Cards */}
                    <div className="summary-grid">
                        <div className="summary-card">
                            <div className="summary-icon iterations">
                                <Activity size={24} />
                            </div>
                            <div className="summary-content">
                                <span className="summary-value">
                                    {report.summary?.total_iterations ?? 0}
                                </span>
                                <span className="summary-label">Total Iterations</span>
                            </div>
                        </div>

                        <div className="summary-card">
                            <div className="summary-icon pass-rate">
                                {(report.summary?.pass_rate ?? 0) >= 0.7 ? (
                                    <TrendingUp size={24} />
                                ) : (
                                    <TrendingDown size={24} />
                                )}
                            </div>
                            <div className="summary-content">
                                <span className="summary-value">
                                    {((report.summary?.pass_rate ?? 0) * 100).toFixed(1)}%
                                </span>
                                <span className="summary-label">Pass Rate</span>
                            </div>
                        </div>

                        <div className="summary-card">
                            <div className="summary-icon confidence">
                                <CheckCircle size={24} />
                            </div>
                            <div className="summary-content">
                                <span className="summary-value">
                                    {((report.summary?.avg_confidence ?? 0) * 100).toFixed(1)}%
                                </span>
                                <span className="summary-label">Avg Confidence</span>
                            </div>
                        </div>

                        <div className="summary-card">
                            <div className="summary-icon time">
                                <Clock size={24} />
                            </div>
                            <div className="summary-content">
                                <span className="summary-value">
                                    {report.timestamp ? new Date(report.timestamp).toLocaleDateString() : 'N/A'}
                                </span>
                                <span className="summary-label">Report Date</span>
                            </div>
                        </div>
                    </div>

                    <div className="evaluate-grid">
                        {/* Metrics Detail */}
                        <div className="card">
                            <div className="card-header">
                                <h3 className="card-title">
                                    <BarChart3 size={20} />
                                    Detailed Metrics
                                </h3>
                            </div>
                            <div className="metrics-list">
                                {report.metrics && Object.entries(report.metrics).map(([key, value]) => (
                                    <div key={key} className="metric-item">
                                        <span className="metric-key">{key.replace(/_/g, ' ')}</span>
                                        <span className="metric-value">
                                            {typeof value === 'number'
                                                ? value.toFixed(2)
                                                : JSON.stringify(value)}
                                        </span>
                                    </div>
                                ))}
                                {(!report.metrics || Object.keys(report.metrics).length === 0) && (
                                    <div className="empty-state">
                                        <Activity size={32} />
                                        <p>No metrics available</p>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Recommendations */}
                        <div className="card">
                            <div className="card-header">
                                <h3 className="card-title">
                                    <Lightbulb size={20} />
                                    Recommendations
                                </h3>
                            </div>
                            <div className="recommendations-list">
                                {report.recommendations && report.recommendations.length > 0 ? (
                                    report.recommendations.map((rec, idx) => (
                                        <div key={idx} className="recommendation-item">
                                            <Lightbulb size={16} />
                                            <span>{rec}</span>
                                        </div>
                                    ))
                                ) : (
                                    <div className="empty-state">
                                        <Lightbulb size={32} />
                                        <p>No recommendations yet</p>
                                        <span>Run more learning iterations to get insights</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Progress Over Time */}
                    {metrics?.snapshots && metrics.snapshots.length > 0 && (
                        <div className="card mt-lg">
                            <div className="card-header">
                                <h3 className="card-title">
                                    <TrendingUp size={20} />
                                    Progress Over Time
                                </h3>
                            </div>
                            <div className="timeline">
                                {metrics.snapshots.slice(-10).map((snapshot, idx) => (
                                    <div key={idx} className="timeline-item">
                                        <div className="timeline-marker"></div>
                                        <div className="timeline-content">
                                            <span className="timeline-time">
                                                {new Date(snapshot.timestamp).toLocaleTimeString()}
                                            </span>
                                            <div className="timeline-stats">
                                                <span>Memory: {snapshot.memory?.episodic_count ?? 0}</span>
                                                <span>Goals: {snapshot.goals?.active_goals ?? 0}</span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </>
            ) : (
                <div className="empty-state">
                    <BarChart3 size={64} />
                    <h3>No Evaluation Data</h3>
                    <p>Run learning iterations first to generate an evaluation report</p>
                </div>
            )}

            <div className="mt-lg">
                <LogPanel title="Evaluation Logs" maxHeight="250px" filter={['evaluate']} />
            </div>
        </div>
    );
}

export default Evaluate;
