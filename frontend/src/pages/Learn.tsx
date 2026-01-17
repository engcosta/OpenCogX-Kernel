import { useState, useEffect, useRef } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import {
    GraduationCap,
    Play,
    Pause,
    RotateCcw,
    Settings2,
    CheckCircle,
    XCircle,
    Clock,
    Brain,
    MessageSquare,
    Target,
    Zap,
    Activity,
} from 'lucide-react';
import { apiService } from '../services/api';
import type { LearnRequest, StepResponse } from '../services/api';
import { useLogStore, useNotificationStore, useTerminalStore } from '../stores';
import LogPanel from '../components/LogPanel';
import './Learn.css';

export function Learn() {
    const [iterations, setIterations] = useState(10);
    const [intervalSeconds, setIntervalSeconds] = useState(5);
    const [stepHistory, setStepHistory] = useState<StepResponse[]>([]);
    const { addLog } = useLogStore();
    const { addNotification } = useNotificationStore();
    const { addEntry, setRunning, isRunning, currentOperation } = useTerminalStore();

    const learnMutation = useMutation({
        mutationFn: async (request: LearnRequest) => {
            setRunning(true, 'learning');
            addLog({
                level: 'info',
                source: 'learn',
                message: `Starting learning loop: ${request.iterations} iterations, ${request.interval_seconds}s interval`,
            });
            addEntry({
                type: 'input',
                content: `Starting learning loop with ${request.iterations} iterations...`,
            });

            const response = await apiService.learn(request);
            return response.data;
        },
        onSuccess: (data) => {
            setRunning(false);
            addLog({
                level: 'success',
                source: 'learn',
                message: `Learning completed: ${data.passed}/${data.total_iterations} passed (${(data.pass_rate * 100).toFixed(1)}%)`,
            });
            addEntry({
                type: 'output',
                content: `✓ Learning completed! Pass rate: ${(data.pass_rate * 100).toFixed(1)}%`,
            });
            addNotification({
                type: 'success',
                title: 'Learning Complete',
                message: `${data.passed} passed, ${data.failed} failed`,
            });
        },
        onError: (error) => {
            setRunning(false);
            const message = error instanceof Error ? error.message : 'Unknown error';
            addLog({
                level: 'error',
                source: 'learn',
                message: `Learning failed: ${message}`,
            });
            addEntry({
                type: 'error',
                content: `✗ Learning failed: ${message}`,
            });
            addNotification({
                type: 'error',
                title: 'Learning Failed',
                message,
            });
        },
    });

    const stepMutation = useMutation({
        mutationFn: async () => {
            addLog({
                level: 'info',
                source: 'learn',
                message: 'Executing single learning step...',
            });
            const response = await apiService.step();
            return response.data;
        },
        onSuccess: (data) => {
            setStepHistory((prev) => [data, ...prev].slice(0, 50));
            const isPassed = data.verdict.toLowerCase() === 'pass';
            addLog({
                level: isPassed ? 'success' : 'warning',
                source: 'learn',
                message: `Step #${data.iteration_id}: ${data.verdict} (${(data.confidence * 100).toFixed(1)}% confidence) - "${data.question.slice(0, 50)}..."`,
            });
            addEntry({
                type: 'output',
                content: `[${data.verdict}] Q: ${data.question}\nA: ${data.answer}`,
            });
        },
        onError: (error) => {
            const message = error instanceof Error ? error.message : 'Unknown error';
            addLog({
                level: 'error',
                source: 'learn',
                message: `Step failed: ${message}`,
            });
        },
    });

    const handleStartLearning = () => {
        learnMutation.mutate({
            iterations,
            interval_seconds: intervalSeconds,
        });
    };

    return (
        <div className="learn-page">
            <div className="page-header">
                <div>
                    <h1>Learning Loop</h1>
                    <p>Train the AGI Kernel through iterative self-questioning</p>
                </div>
            </div>

            <div className="learn-container">
                {/* Control Panel */}
                <div className="card learn-controls">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Settings2 size={20} />
                            Learning Configuration
                        </h3>
                    </div>

                    <div className="controls-form">
                        <div className="form-row">
                            <div className="form-group">
                                <label className="form-label">Iterations</label>
                                <input
                                    type="number"
                                    className="input"
                                    value={iterations}
                                    onChange={(e) => setIterations(Number(e.target.value))}
                                    min={1}
                                    max={1000}
                                    disabled={learnMutation.isPending}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Interval (seconds)</label>
                                <input
                                    type="number"
                                    className="input"
                                    value={intervalSeconds}
                                    onChange={(e) => setIntervalSeconds(Number(e.target.value))}
                                    min={0}
                                    max={60}
                                    step={0.5}
                                    disabled={learnMutation.isPending}
                                />
                            </div>
                        </div>

                        <div className="controls-actions">
                            <button
                                className="btn btn-primary btn-lg"
                                onClick={handleStartLearning}
                                disabled={learnMutation.isPending}
                            >
                                {learnMutation.isPending ? (
                                    <>
                                        <Activity size={18} className="animate-pulse" />
                                        Learning...
                                    </>
                                ) : (
                                    <>
                                        <Play size={18} />
                                        Start Learning
                                    </>
                                )}
                            </button>

                            <button
                                className="btn btn-secondary"
                                onClick={() => stepMutation.mutate()}
                                disabled={learnMutation.isPending || stepMutation.isPending}
                            >
                                <Zap size={18} />
                                Single Step
                            </button>
                        </div>
                    </div>

                    {/* Learning Stats */}
                    {learnMutation.data && (
                        <div className="learning-result">
                            <div className="result-stat">
                                <span className="result-number">
                                    {learnMutation.data.total_iterations}
                                </span>
                                <span className="result-text">Iterations</span>
                            </div>
                            <div className="result-stat success">
                                <span className="result-number">{learnMutation.data.passed}</span>
                                <span className="result-text">Passed</span>
                            </div>
                            <div className="result-stat error">
                                <span className="result-number">{learnMutation.data.failed}</span>
                                <span className="result-text">Failed</span>
                            </div>
                            <div className="result-stat">
                                <span className="result-number">
                                    {(learnMutation.data.pass_rate * 100).toFixed(1)}%
                                </span>
                                <span className="result-text">Pass Rate</span>
                            </div>
                        </div>
                    )}
                </div>

                {/* Step History */}
                <div className="card step-history">
                    <div className="card-header">
                        <h3 className="card-title">
                            <MessageSquare size={20} />
                            Step History
                        </h3>
                        <span className="badge badge-neutral">{stepHistory.length} steps</span>
                    </div>

                    <div className="history-list">
                        {stepHistory.length === 0 ? (
                            <div className="empty-state">
                                <Brain size={48} />
                                <p>No learning steps yet</p>
                                <span>Run a learning loop or execute single steps</span>
                            </div>
                        ) : (
                            stepHistory.map((step, index) => (
                                <div
                                    key={`${step.iteration_id}-${index}`}
                                    className={`history-item ${step.verdict.toLowerCase()}`}
                                >
                                    <div className="history-header">
                                        <div className="history-meta">
                                            <span className="history-id">#{step.iteration_id}</span>
                                            <span
                                                className={`badge ${step.verdict.toLowerCase() === 'pass'
                                                    ? 'badge-success'
                                                    : 'badge-error'
                                                    }`}
                                            >
                                                {step.verdict.toLowerCase() === 'pass' ? (
                                                    <CheckCircle size={12} />
                                                ) : (
                                                    <XCircle size={12} />
                                                )}
                                                {step.verdict}
                                            </span>
                                            <span className="badge badge-info">{step.strategy}</span>
                                        </div>
                                        <div className="history-stats">
                                            <span>
                                                <Target size={12} />
                                                {(step.confidence * 100).toFixed(1)}%
                                            </span>
                                            <span>
                                                <Clock size={12} />
                                                {step.duration_ms.toFixed(0)}ms
                                            </span>
                                        </div>
                                    </div>
                                    <div className="history-content">
                                        <div className="history-question">
                                            <strong>Q:</strong> {step.question}
                                        </div>
                                        <div className="history-answer">
                                            <strong>A:</strong> {step.answer}
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>

            {/* Logs */}
            <div className="mt-lg">
                <LogPanel title="Learning Logs" maxHeight="300px" filter={['learn']} />
            </div>
        </div>
    );
}

export default Learn;
