import { useEffect, useRef } from 'react';
import { Terminal, Trash2, Download, Pause, Play } from 'lucide-react';
import { useLogStore } from '../stores';
import type { LogEntry } from '../stores';
import './LogPanel.css';

interface LogPanelProps {
    title?: string;
    logs?: LogEntry[];
    maxHeight?: string;
    autoScroll?: boolean;
    showControls?: boolean;
    filter?: string[];
}

export function LogPanel({
    title = 'System Logs',
    logs: externalLogs,
    maxHeight = '400px',
    autoScroll = true,
    showControls = true,
    filter,
}: LogPanelProps) {
    const { logs: storeLogs, clearLogs } = useLogStore();
    const containerRef = useRef<HTMLDivElement>(null);
    const isAutoScrolling = useRef(true);

    const logs = externalLogs ?? storeLogs;
    const filteredLogs = filter
        ? logs.filter((log) => filter.includes(log.source))
        : logs;

    useEffect(() => {
        if (autoScroll && isAutoScrolling.current && containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [filteredLogs, autoScroll]);

    const handleScroll = () => {
        if (containerRef.current) {
            const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
            isAutoScrolling.current = scrollHeight - scrollTop - clientHeight < 50;
        }
    };

    const downloadLogs = () => {
        const content = filteredLogs
            .map(
                (log) =>
                    `[${log.timestamp.toISOString()}] [${log.level.toUpperCase()}] [${log.source}] ${log.message}`
            )
            .join('\n');
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `logs-${new Date().toISOString().slice(0, 10)}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const formatTime = (date: Date) => {
        return date.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            fractionalSecondDigits: 3,
        });
    };

    return (
        <div className="log-panel-container">
            <div className="log-panel-header">
                <div className="log-panel-title">
                    <Terminal size={18} />
                    <span>{title}</span>
                    <span className="log-count">{filteredLogs.length}</span>
                </div>
                {showControls && (
                    <div className="log-panel-actions">
                        <button
                            className="btn btn-ghost btn-sm"
                            onClick={() => {
                                isAutoScrolling.current = !isAutoScrolling.current;
                            }}
                            title={isAutoScrolling.current ? 'Pause auto-scroll' : 'Resume auto-scroll'}
                        >
                            {isAutoScrolling.current ? <Pause size={14} /> : <Play size={14} />}
                        </button>
                        <button
                            className="btn btn-ghost btn-sm"
                            onClick={downloadLogs}
                            title="Download logs"
                        >
                            <Download size={14} />
                        </button>
                        <button
                            className="btn btn-ghost btn-sm"
                            onClick={clearLogs}
                            title="Clear logs"
                        >
                            <Trash2 size={14} />
                        </button>
                    </div>
                )}
            </div>

            <div
                ref={containerRef}
                className="log-panel-content"
                style={{ maxHeight }}
                onScroll={handleScroll}
            >
                {filteredLogs.length === 0 ? (
                    <div className="log-empty">
                        <Terminal size={24} />
                        <span>No logs yet</span>
                    </div>
                ) : (
                    filteredLogs.map((log) => (
                        <div key={log.id} className={`log-entry log-${log.level}`}>
                            <span className="log-time">{formatTime(log.timestamp)}</span>
                            <span className={`log-level ${log.level}`}>{log.level}</span>
                            <span className="log-source">[{log.source}]</span>
                            <span className="log-message">{log.message}</span>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}

export default LogPanel;
