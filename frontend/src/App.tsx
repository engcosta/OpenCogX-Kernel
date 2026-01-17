import { useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useLogStore } from './stores';
import { Sidebar, Toast } from './components';
import {
  Dashboard,
  Ingest,
  Learn,
  Ask,
  Evaluate,
  Memory,
  Goals,
  World,
  Meta,
  Settings,
  Clean,
} from './pages';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 5000,
    },
  },
});

function App() {
  const { addLog } = useLogStore();

  useEffect(() => {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/logs/socket';
    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout>;

    const connect = () => {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        addLog({
          level: 'success',
          source: 'system',
          message: 'Connected to real-time log stream',
        });
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          addLog({
            level: data.level,
            source: data.source,
            message: data.message,
            data: data.data,
            timestamp: data.timestamp,
          });
        } catch (error) {
          console.error('Failed to parse log message:', error);
        }
      };

      ws.onclose = () => {
        reconnectTimeout = setTimeout(connect, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        ws?.close();
      };
    };

    connect();

    return () => {
      if (ws) ws.close();
      clearTimeout(reconnectTimeout);
    };
  }, [addLog]);

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="app-container">
          <Sidebar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/ingest" element={<Ingest />} />
              <Route path="/learn" element={<Learn />} />
              <Route path="/ask" element={<Ask />} />
              <Route path="/evaluate" element={<Evaluate />} />
              <Route path="/memory" element={<Memory />} />
              <Route path="/goals" element={<Goals />} />
              <Route path="/world" element={<World />} />
              <Route path="/meta" element={<Meta />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/clean" element={<Clean />} />
            </Routes>
          </main>
          <Toast />
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
