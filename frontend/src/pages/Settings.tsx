import { useState } from 'react';
import { Moon, Bell, Cpu, Globe } from 'lucide-react';
import './Settings.css';

export function Settings() {
    const [apiUrl, setApiUrl] = useState(
        localStorage.getItem('apiUrl') || 'http://localhost:8000'
    );
    const [refreshInterval, setRefreshInterval] = useState(
        localStorage.getItem('refreshInterval') || '5'
    );
    const [darkMode, setDarkMode] = useState(true);
    const [notifications, setNotifications] = useState(true);

    const handleSave = () => {
        localStorage.setItem('apiUrl', apiUrl);
        localStorage.setItem('refreshInterval', refreshInterval);
    };

    return (
        <div className="settings-page">
            <div className="page-header">
                <div>
                    <h1>Settings</h1>
                    <p>Configure dashboard preferences</p>
                </div>
            </div>

            <div className="settings-grid">
                {/* API Configuration */}
                <div className="card settings-section">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Globe size={20} />
                            API Configuration
                        </h3>
                    </div>
                    <div className="settings-content">
                        <div className="form-group">
                            <label className="form-label">API Base URL</label>
                            <input
                                type="text"
                                className="input"
                                value={apiUrl}
                                onChange={(e) => setApiUrl(e.target.value)}
                                placeholder="http://localhost:8000"
                            />
                            <span className="form-hint">
                                The base URL of the AGI Kernel API server
                            </span>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Auto-refresh Interval (seconds)</label>
                            <select
                                className="select"
                                value={refreshInterval}
                                onChange={(e) => setRefreshInterval(e.target.value)}
                            >
                                <option value="5">5 seconds</option>
                                <option value="10">10 seconds</option>
                                <option value="30">30 seconds</option>
                                <option value="60">1 minute</option>
                                <option value="0">Disabled</option>
                            </select>
                        </div>
                        <button className="btn btn-primary" onClick={handleSave}>
                            Save Changes
                        </button>
                    </div>
                </div>

                {/* Appearance */}
                <div className="card settings-section">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Moon size={20} />
                            Appearance
                        </h3>
                    </div>
                    <div className="settings-content">
                        <div className="setting-row">
                            <div className="setting-info">
                                <span className="setting-title">Dark Mode</span>
                                <span className="setting-description">
                                    Use dark theme for the dashboard
                                </span>
                            </div>
                            <label className="toggle">
                                <input
                                    type="checkbox"
                                    checked={darkMode}
                                    onChange={(e) => setDarkMode(e.target.checked)}
                                />
                                <span className="toggle-slider"></span>
                            </label>
                        </div>
                    </div>
                </div>

                {/* Notifications */}
                <div className="card settings-section">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Bell size={20} />
                            Notifications
                        </h3>
                    </div>
                    <div className="settings-content">
                        <div className="setting-row">
                            <div className="setting-info">
                                <span className="setting-title">Enable Notifications</span>
                                <span className="setting-description">
                                    Show toast notifications for events
                                </span>
                            </div>
                            <label className="toggle">
                                <input
                                    type="checkbox"
                                    checked={notifications}
                                    onChange={(e) => setNotifications(e.target.checked)}
                                />
                                <span className="toggle-slider"></span>
                            </label>
                        </div>
                    </div>
                </div>

                {/* System Info */}
                <div className="card settings-section">
                    <div className="card-header">
                        <h3 className="card-title">
                            <Cpu size={20} />
                            System Information
                        </h3>
                    </div>
                    <div className="settings-content">
                        <div className="info-row">
                            <span className="info-label">Frontend Version</span>
                            <span className="info-value">0.1.0</span>
                        </div>
                        <div className="info-row">
                            <span className="info-label">API Version</span>
                            <span className="info-value">0.1.0</span>
                        </div>
                        <div className="info-row">
                            <span className="info-label">Environment</span>
                            <span className="info-value">Development</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Settings;
