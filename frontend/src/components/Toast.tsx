import { useEffect } from 'react';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react';
import { useNotificationStore } from '../stores';
import './Toast.css';

const icons = {
    success: CheckCircle,
    error: AlertCircle,
    warning: AlertTriangle,
    info: Info,
};

export function Toast() {
    const { notifications, removeNotification } = useNotificationStore();

    return (
        <div className="toast-container">
            {notifications.map((notification) => {
                const Icon = icons[notification.type];
                return (
                    <div
                        key={notification.id}
                        className={`toast toast-${notification.type}`}
                    >
                        <Icon className="toast-icon" />
                        <div className="toast-content">
                            <span className="toast-title">{notification.title}</span>
                            {notification.message && (
                                <span className="toast-message">{notification.message}</span>
                            )}
                        </div>
                        <button
                            className="toast-close"
                            onClick={() => removeNotification(notification.id)}
                        >
                            <X size={14} />
                        </button>
                    </div>
                );
            })}
        </div>
    );
}

export default Toast;
