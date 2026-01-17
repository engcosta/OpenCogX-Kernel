import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Trash2, AlertTriangle, RefreshCw, CheckCircle } from 'lucide-react';
import { apiService } from '../services/api';
import { useLogStore, useNotificationStore } from '../stores';
import './Clean.css';

export function Clean() {
    const [confirmed, setConfirmed] = useState(false);
    const { addLog, clearLogs } = useLogStore();
    const { addNotification } = useNotificationStore();

    const cleanMutation = useMutation({
        mutationFn: async () => {
            addLog({
                level: 'warning',
                source: 'clean',
                message: 'Starting database cleanup...',
            });
            const res = await apiService.clean(true);
            return res.data;
        },
        onSuccess: (data) => {
            addLog({
                level: 'success',
                source: 'clean',
                message: data.message,
            });
            addNotification({
                type: 'success',
                title: 'Databases Cleaned',
                message: 'All databases have been successfully cleared',
            });
            setConfirmed(false);
        },
        onError: (error) => {
            addLog({
                level: 'error',
                source: 'clean',
                message: `Cleanup failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
            });
            addNotification({
                type: 'error',
                title: 'Cleanup Failed',
                message: error instanceof Error ? error.message : 'Unknown error',
            });
        },
    });

    const handleClean = () => {
        if (confirmed) {
            cleanMutation.mutate();
        }
    };

    return (
        <div className="clean-page">
            <div className="page-header">
                <div>
                    <h1>Clean Databases</h1>
                    <p>Clear all data from Neo4j and Qdrant databases</p>
                </div>
            </div>

            <div className="clean-container">
                <div className="card clean-card">
                    <div className="warning-icon">
                        <AlertTriangle size={64} />
                    </div>
                    <h2>Danger Zone</h2>
                    <p>
                        This action will permanently delete all data from the Neo4j graph database
                        and Qdrant vector database. This includes:
                    </p>
                    <ul className="delete-list">
                        <li>All ingested documents and chunks</li>
                        <li>All extracted entities and relations</li>
                        <li>All vector embeddings</li>
                        <li>All knowledge graph data</li>
                    </ul>
                    <p className="warning-text">
                        <strong>This action cannot be undone.</strong>
                    </p>

                    <div className="confirm-section">
                        <label className="confirm-label">
                            <input
                                type="checkbox"
                                checked={confirmed}
                                onChange={(e) => setConfirmed(e.target.checked)}
                                disabled={cleanMutation.isPending}
                            />
                            <span>I understand that this action is irreversible</span>
                        </label>
                    </div>

                    <div className="action-buttons">
                        <button
                            className="btn btn-danger btn-lg"
                            onClick={handleClean}
                            disabled={!confirmed || cleanMutation.isPending}
                        >
                            {cleanMutation.isPending ? (
                                <>
                                    <RefreshCw size={18} className="animate-spin" />
                                    Cleaning...
                                </>
                            ) : (
                                <>
                                    <Trash2 size={18} />
                                    Clean All Databases
                                </>
                            )}
                        </button>
                    </div>

                    {cleanMutation.isSuccess && (
                        <div className="success-message">
                            <CheckCircle size={20} />
                            <span>Databases cleaned successfully!</span>
                        </div>
                    )}
                </div>

                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Trash2 size={20} />
                            Clear Local Data
                        </h3>
                    </div>
                    <div className="local-actions">
                        <button className="btn btn-secondary" onClick={clearLogs}>
                            Clear Logs
                        </button>
                        <button
                            className="btn btn-secondary"
                            onClick={() => {
                                localStorage.clear();
                                addNotification({
                                    type: 'success',
                                    title: 'Local Storage Cleared',
                                    message: 'All local settings have been reset',
                                });
                            }}
                        >
                            Clear Local Settings
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Clean;
