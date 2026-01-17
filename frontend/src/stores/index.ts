import { create } from 'zustand';

export interface LogEntry {
    id: string;
    timestamp: Date;
    level: 'info' | 'warning' | 'error' | 'success' | 'debug';
    source: string;
    message: string;
    data?: Record<string, unknown>;
}

interface LogStore {
    logs: LogEntry[];
    maxLogs: number;
    addLog: (log: Omit<LogEntry, 'id' | 'timestamp'> & { timestamp?: string | Date }) => void;
    clearLogs: () => void;
    setMaxLogs: (max: number) => void;
}

export const useLogStore = create<LogStore>((set) => ({
    logs: [],
    maxLogs: 500,

    addLog: (log) =>
        set((state) => {
            const newLog: LogEntry = {
                ...log,
                id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
                timestamp: log.timestamp ? new Date(log.timestamp) : new Date(),
            };
            const newLogs = [newLog, ...state.logs].slice(0, state.maxLogs);
            return { logs: newLogs };
        }),

    clearLogs: () => set({ logs: [] }),

    setMaxLogs: (max) => set({ maxLogs: max }),
}));

// Notification types
export interface Notification {
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    title: string;
    message?: string;
    duration?: number;
}

interface NotificationStore {
    notifications: Notification[];
    addNotification: (notification: Omit<Notification, 'id'>) => void;
    removeNotification: (id: string) => void;
    clearNotifications: () => void;
}

export const useNotificationStore = create<NotificationStore>((set) => ({
    notifications: [],

    addNotification: (notification) =>
        set((state) => {
            const id = `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
            const newNotification = { ...notification, id };

            // Auto-remove after duration
            const duration = notification.duration ?? 5000;
            if (duration > 0) {
                setTimeout(() => {
                    set((s) => ({
                        notifications: s.notifications.filter((n) => n.id !== id),
                    }));
                }, duration);
            }

            return { notifications: [...state.notifications, newNotification] };
        }),

    removeNotification: (id) =>
        set((state) => ({
            notifications: state.notifications.filter((n) => n.id !== id),
        })),

    clearNotifications: () => set({ notifications: [] }),
}));

// App-wide state
interface AppState {
    isLoading: boolean;
    sidebarOpen: boolean;
    activeSection: string;
    setLoading: (loading: boolean) => void;
    toggleSidebar: () => void;
    setActiveSection: (section: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
    isLoading: false,
    sidebarOpen: true,
    activeSection: 'dashboard',

    setLoading: (loading) => set({ isLoading: loading }),
    toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
    setActiveSection: (section) => set({ activeSection: section }),
}));

// Terminal (learning/operation) state
interface TerminalEntry {
    id: string;
    timestamp: Date;
    type: 'input' | 'output' | 'error' | 'info';
    content: string;
}

interface TerminalStore {
    entries: TerminalEntry[];
    isRunning: boolean;
    currentOperation: string | null;
    addEntry: (entry: Omit<TerminalEntry, 'id' | 'timestamp'>) => void;
    clearEntries: () => void;
    setRunning: (running: boolean, operation?: string) => void;
}

export const useTerminalStore = create<TerminalStore>((set) => ({
    entries: [],
    isRunning: false,
    currentOperation: null,

    addEntry: (entry) =>
        set((state) => {
            const newEntry: TerminalEntry = {
                ...entry,
                id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
                timestamp: new Date(),
            };
            return { entries: [...state.entries, newEntry].slice(-1000) };
        }),

    clearEntries: () => set({ entries: [] }),

    setRunning: (running, operation) =>
        set({ isRunning: running, currentOperation: running ? operation ?? null : null }),
}));
