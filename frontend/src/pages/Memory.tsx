import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import {
    Database,
    Search,
    RefreshCw,
    Filter,
    Brain,
    Clock,
    BookOpen,
    Layers,
} from 'lucide-react';
import { apiService } from '../services/api';
import type { MemoryQueryRequest, MemoryResult } from '../services/api';
import { useLogStore } from '../stores';
import './Memory.css';

export function Memory() {
    const [query, setQuery] = useState('');
    const [limit, setLimit] = useState(10);
    const [memoryTypes, setMemoryTypes] = useState<string[]>([]);
    const [results, setResults] = useState<MemoryResult[]>([]);
    const { addLog } = useLogStore();

    const { data: stats, refetch: refetchStats } = useQuery({
        queryKey: ['memoryStats'],
        queryFn: async () => {
            const res = await apiService.getMemoryStats();
            return res.data;
        },
    });

    const searchMutation = useMutation({
        mutationFn: async (request: MemoryQueryRequest) => {
            addLog({
                level: 'info',
                source: 'memory',
                message: `Searching memory: "${request.query}"`,
            });
            const res = await apiService.queryMemory(request);
            return res.data;
        },
        onSuccess: (data) => {
            setResults(data.results);
            addLog({
                level: 'success',
                source: 'memory',
                message: `Found ${data.count} results`,
            });
        },
        onError: (error) => {
            addLog({
                level: 'error',
                source: 'memory',
                message: `Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
            });
        },
    });

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim()) return;

        searchMutation.mutate({
            query: query.trim(),
            limit,
            memory_types: memoryTypes.length > 0 ? memoryTypes : undefined,
        });
    };

    const toggleMemoryType = (type: string) => {
        setMemoryTypes((prev) =>
            prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
        );
    };

    return (
        <div className="memory-page">
            <div className="page-header">
                <div>
                    <h1>Memory System</h1>
                    <p>Query and explore the kernel's memory stores</p>
                </div>
                <button
                    className="btn btn-secondary"
                    onClick={() => refetchStats()}
                >
                    <RefreshCw size={16} />
                    Refresh Stats
                </button>
            </div>

            {/* Stats */}
            <div className="memory-stats">
                <div className="stat-card">
                    <Brain size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{stats?.episodic_count ?? 0}</span>
                        <span className="stat-label">Episodic Memories</span>
                    </div>
                </div>
                <div className="stat-card">
                    <BookOpen size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{stats?.semantic_count ?? 0}</span>
                        <span className="stat-label">Semantic Memories</span>
                    </div>
                </div>
                <div className="stat-card">
                    <Clock size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{stats?.recent_memories ?? 0}</span>
                        <span className="stat-label">Recent Memories</span>
                    </div>
                </div>
            </div>

            {/* Search */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">
                        <Search size={20} />
                        Query Memory
                    </h3>
                </div>

                <form onSubmit={handleSearch} className="search-form">
                    <div className="search-input-row">
                        <input
                            type="text"
                            className="input search-input"
                            placeholder="Enter search query..."
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                        />
                        <input
                            type="number"
                            className="input limit-input"
                            placeholder="Limit"
                            value={limit}
                            onChange={(e) => setLimit(Number(e.target.value))}
                            min={1}
                            max={100}
                        />
                        <button
                            type="submit"
                            className="btn btn-primary"
                            disabled={searchMutation.isPending || !query.trim()}
                        >
                            {searchMutation.isPending ? (
                                <RefreshCw size={16} className="animate-spin" />
                            ) : (
                                <Search size={16} />
                            )}
                            Search
                        </button>
                    </div>

                    <div className="filter-row">
                        <span className="filter-label">
                            <Filter size={14} />
                            Filter by type:
                        </span>
                        <div className="filter-chips">
                            {['semantic', 'episodic'].map((type) => (
                                <button
                                    key={type}
                                    type="button"
                                    className={`filter-chip ${memoryTypes.includes(type) ? 'active' : ''}`}
                                    onClick={() => toggleMemoryType(type)}
                                >
                                    {type}
                                </button>
                            ))}
                        </div>
                    </div>
                </form>
            </div>

            {/* Results */}
            <div className="card mt-lg">
                <div className="card-header">
                    <h3 className="card-title">
                        <Layers size={20} />
                        Results
                    </h3>
                    {results.length > 0 && (
                        <span className="badge badge-info">{results.length} found</span>
                    )}
                </div>

                <div className="results-list">
                    {results.length === 0 ? (
                        <div className="empty-state">
                            <Database size={48} />
                            <p>No results</p>
                            <span>Enter a search query to explore memories</span>
                        </div>
                    ) : (
                        results.map((result) => (
                            <div key={result.id} className="result-item">
                                <div className="result-header">
                                    <span className="result-id">{result.id}</span>
                                    <span className={`badge badge-${result.type === 'semantic' ? 'info' : 'success'}`}>
                                        {result.type}
                                    </span>
                                </div>
                                <div className="result-content">
                                    <pre>{JSON.stringify(result.content, null, 2)}</pre>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
}

export default Memory;
