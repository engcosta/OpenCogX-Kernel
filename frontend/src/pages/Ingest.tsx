import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import {
    FileUp,
    Folder,
    File,
    Upload,
    CheckCircle,
    AlertCircle,
    Loader2,
    FileText,
    Network,
    Database,
    Layers,
} from 'lucide-react';
import { apiService } from '../services/api';
import type { IngestRequest, IngestResponse } from '../services/api';
import { useLogStore, useNotificationStore } from '../stores';
import LogPanel from '../components/LogPanel';
import './Ingest.css';

export function Ingest() {
    const [path, setPath] = useState('');
    const [isDirectory, setIsDirectory] = useState(false);
    const { addLog } = useLogStore();
    const { addNotification } = useNotificationStore();
    const [lastResult, setLastResult] = useState<IngestResponse | null>(null);

    const ingestMutation = useMutation({
        mutationFn: async (request: IngestRequest) => {
            addLog({
                level: 'info',
                source: 'ingest',
                message: `Starting ingestion: ${request.path} (${request.is_directory ? 'directory' : 'file'})`,
            });

            const response = await apiService.ingest(request);
            return response.data;
        },
        onSuccess: (data) => {
            setLastResult(data);
            if (data.success) {
                addLog({
                    level: 'success',
                    source: 'ingest',
                    message: `Ingestion completed: ${data.files_processed} files, ${data.chunks} chunks, ${data.entities} entities, ${data.relations} relations`,
                });
                addNotification({
                    type: 'success',
                    title: 'Ingestion Complete',
                    message: `Successfully processed ${data.files_processed} file(s)`,
                });
            } else {
                addLog({
                    level: 'error',
                    source: 'ingest',
                    message: `Ingestion failed: ${data.error}`,
                });
                addNotification({
                    type: 'error',
                    title: 'Ingestion Failed',
                    message: data.error,
                });
            }
        },
        onError: (error) => {
            addLog({
                level: 'error',
                source: 'ingest',
                message: `Ingestion error: ${error instanceof Error ? error.message : 'Unknown error'}`,
            });
            addNotification({
                type: 'error',
                title: 'Ingestion Error',
                message: error instanceof Error ? error.message : 'Unknown error occurred',
            });
        },
    });

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!path.trim()) {
            addNotification({
                type: 'warning',
                title: 'Path Required',
                message: 'Please enter a file or directory path',
            });
            return;
        }
        ingestMutation.mutate({ path: path.trim(), is_directory: isDirectory });
    };

    return (
        <div className="ingest-page">
            <div className="page-header">
                <div>
                    <h1>Document Ingestion</h1>
                    <p>Ingest documents into the knowledge base for learning</p>
                </div>
            </div>

            <div className="ingest-container">
                {/* Input Section */}
                <div className="card ingest-form-card">
                    <div className="card-header">
                        <h3 className="card-title">
                            <FileUp size={20} />
                            Ingest Documents
                        </h3>
                    </div>

                    <form onSubmit={handleSubmit} className="ingest-form">
                        <div className="form-group">
                            <label className="form-label">Path</label>
                            <div className="input-group">
                                <span className="input-icon">
                                    {isDirectory ? <Folder size={18} /> : <File size={18} />}
                                </span>
                                <input
                                    type="text"
                                    className="input input-with-icon"
                                    placeholder={isDirectory ? 'Enter directory path...' : 'Enter file path...'}
                                    value={path}
                                    onChange={(e) => setPath(e.target.value)}
                                    disabled={ingestMutation.isPending}
                                />
                            </div>
                            <span className="form-hint">
                                Example: {isDirectory ? './corpus' : './corpus/document.txt'}
                            </span>
                        </div>

                        <div className="form-group">
                            <label className="toggle-label">
                                <span className="toggle">
                                    <input
                                        type="checkbox"
                                        checked={isDirectory}
                                        onChange={(e) => setIsDirectory(e.target.checked)}
                                        disabled={ingestMutation.isPending}
                                    />
                                    <span className="toggle-slider"></span>
                                </span>
                                <span>Directory Mode</span>
                            </label>
                            <span className="form-hint">
                                Enable to process all files in a directory
                            </span>
                        </div>

                        <button
                            type="submit"
                            className="btn btn-primary btn-lg"
                            disabled={ingestMutation.isPending || !path.trim()}
                        >
                            {ingestMutation.isPending ? (
                                <>
                                    <Loader2 size={18} className="animate-spin" />
                                    Processing...
                                </>
                            ) : (
                                <>
                                    <Upload size={18} />
                                    Start Ingestion
                                </>
                            )}
                        </button>
                    </form>

                    {/* Pipeline Steps */}
                    <div className="pipeline-steps">
                        <div className={`pipeline-step ${ingestMutation.isPending ? 'active' : ''}`}>
                            <div className="step-icon">
                                <FileText size={18} />
                            </div>
                            <span>Parse & Chunk</span>
                        </div>
                        <div className="pipeline-arrow">→</div>
                        <div className={`pipeline-step ${ingestMutation.isPending ? 'active' : ''}`}>
                            <div className="step-icon">
                                <Database size={18} />
                            </div>
                            <span>Vectorize</span>
                        </div>
                        <div className="pipeline-arrow">→</div>
                        <div className={`pipeline-step ${ingestMutation.isPending ? 'active' : ''}`}>
                            <div className="step-icon">
                                <Layers size={18} />
                            </div>
                            <span>Extract Entities</span>
                        </div>
                        <div className="pipeline-arrow">→</div>
                        <div className={`pipeline-step ${ingestMutation.isPending ? 'active' : ''}`}>
                            <div className="step-icon">
                                <Network size={18} />
                            </div>
                            <span>Build Graph</span>
                        </div>
                    </div>
                </div>

                {/* Results Section */}
                <div className="card ingest-results-card">
                    <div className="card-header">
                        <h3 className="card-title">
                            {lastResult?.success ? (
                                <CheckCircle size={20} className="icon-success" />
                            ) : lastResult ? (
                                <AlertCircle size={20} className="icon-error" />
                            ) : (
                                <FileUp size={20} />
                            )}
                            Ingestion Results
                        </h3>
                    </div>

                    {lastResult ? (
                        <div className="results-content">
                            {lastResult.success ? (
                                <div className="results-grid">
                                    <div className="result-item">
                                        <span className="result-value">{lastResult.files_processed}</span>
                                        <span className="result-label">Files Processed</span>
                                    </div>
                                    <div className="result-item">
                                        <span className="result-value">{lastResult.chunks}</span>
                                        <span className="result-label">Chunks Created</span>
                                    </div>
                                    <div className="result-item">
                                        <span className="result-value">{lastResult.entities}</span>
                                        <span className="result-label">Entities Extracted</span>
                                    </div>
                                    <div className="result-item">
                                        <span className="result-value">{lastResult.relations}</span>
                                        <span className="result-label">Relations Built</span>
                                    </div>
                                </div>
                            ) : (
                                <div className="error-message">
                                    <AlertCircle size={24} />
                                    <p>{lastResult.error}</p>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="empty-state">
                            <Upload size={48} />
                            <p>No ingestion performed yet</p>
                            <span>Select a file or directory to begin</span>
                        </div>
                    )}
                </div>
            </div>

            {/* Logs */}
            <div className="mt-lg">
                <LogPanel title="Ingestion Logs" maxHeight="350px" filter={['ingest']} />
            </div>
        </div>
    );
}

export default Ingest;
