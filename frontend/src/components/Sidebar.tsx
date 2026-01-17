import { NavLink } from 'react-router-dom';
import {
    Brain,
    LayoutDashboard,
    FileUp,
    GraduationCap,
    MessageSquareText,
    BarChart3,
    Database,
    Target,
    Cpu,
    Sparkles,
    Settings,
    Trash2,
} from 'lucide-react';
import './Sidebar.css';

const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/ingest', label: 'Ingest', icon: FileUp },
    { path: '/learn', label: 'Learn', icon: GraduationCap },
    { path: '/ask', label: 'Ask', icon: MessageSquareText },
    { path: '/evaluate', label: 'Evaluate', icon: BarChart3 },
    { path: '/memory', label: 'Memory', icon: Database },
    { path: '/goals', label: 'Goals', icon: Target },
    { path: '/world', label: 'World Model', icon: Cpu },
    { path: '/meta', label: 'Meta-Cognition', icon: Sparkles },
];

const utilityItems = [
    { path: '/settings', label: 'Settings', icon: Settings },
    { path: '/clean', label: 'Clean', icon: Trash2 },
];

export function Sidebar() {
    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="sidebar-logo">
                    <Brain className="logo-icon" />
                    <div className="logo-text">
                        <span className="logo-title">AGI Kernel</span>
                        <span className="logo-subtitle">Control Panel</span>
                    </div>
                </div>
            </div>

            <nav className="sidebar-nav">
                <div className="nav-section">
                    <span className="nav-section-title">Main</span>
                    <ul className="nav-list">
                        {navItems.map((item) => (
                            <li key={item.path}>
                                <NavLink
                                    to={item.path}
                                    className={({ isActive }) =>
                                        `nav-link ${isActive ? 'active' : ''}`
                                    }
                                >
                                    <item.icon className="nav-icon" />
                                    <span>{item.label}</span>
                                </NavLink>
                            </li>
                        ))}
                    </ul>
                </div>

                <div className="nav-section">
                    <span className="nav-section-title">Utilities</span>
                    <ul className="nav-list">
                        {utilityItems.map((item) => (
                            <li key={item.path}>
                                <NavLink
                                    to={item.path}
                                    className={({ isActive }) =>
                                        `nav-link ${isActive ? 'active' : ''}`
                                    }
                                >
                                    <item.icon className="nav-icon" />
                                    <span>{item.label}</span>
                                </NavLink>
                            </li>
                        ))}
                    </ul>
                </div>
            </nav>

            <div className="sidebar-footer">
                <div className="status-indicator">
                    <span className="status-dot online"></span>
                    <span className="status-text">System Active</span>
                </div>
                <span className="version">v0.1.0</span>
            </div>
        </aside>
    );
}

export default Sidebar;
