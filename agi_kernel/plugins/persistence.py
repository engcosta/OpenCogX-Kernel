"""
💾 Persistence Plugin (SQLite/SQL)
==================================

Provides structured persistence for Goals, History, and Metrics.

Designed to be database-agnostic where possible, allowing future upgrade 
to PostgreSQL by simply swapping the adapter.

Currently uses: aiosqlite
Future path: asyncpg
"""

from __future__ import annotations

import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, Any, List
import structlog
import aiosqlite

logger = structlog.get_logger()


class PersistencePlugin:
    """
    Persistence layer using SQLite.
    
    Manages:
    - Persistent Goals
    - Learning History (Execution logs)
    - World Events
    """
    
    def __init__(self, db_path: str = "agi_state.db"):
        """
        Initialize the Persistence Plugin.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._initialized = False
        
        logger.info("persistence_plugin_created", db_path=db_path)

    async def initialize(self) -> bool:
        """
        Initialize the database connection and schema.
        """
        try:
            self._db = await aiosqlite.connect(self.db_path)
            # Enable dictionary row access
            self._db.row_factory = aiosqlite.Row
            
            await self._create_schema()
            
            self._initialized = True
            logger.info("persistence_plugin_initialized")
            return True
            
        except Exception as e:
            logger.error("persistence_init_failed", error=str(e))
            return False

    async def _create_schema(self):
        """Create the necessary tables if they don't exist."""
        if not self._db:
            return

        # 1. Goals Table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                description TEXT,
                priority REAL,
                expected_gain REAL,
                target_entity TEXT,
                target_relation TEXT,
                status TEXT,
                created_at TEXT,
                completed_at TEXT,
                actual_gain REAL,
                attempts INTEGER,
                max_attempts INTEGER,
                meta_data TEXT
            )
        """)

        # 2. Learning Iterations (History)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS learning_history (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                goal_id TEXT,
                question TEXT,
                question_type TEXT,
                answer TEXT,
                strategy TEXT,
                verdict TEXT,
                confidence REAL,
                duration_ms REAL,
                knowledge_gained INTEGER,
                gap_recorded INTEGER,
                issues TEXT       -- JSON list
            )
        """)
        
        # 3. World Events (Audit Log) - Optional
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS world_events (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                actor TEXT,
                action TEXT,
                context TEXT      -- JSON dict
            )
        """)

        await self._db.commit()

    # =========================================================================
    # GOAL MANAGEMENT
    # =========================================================================

    async def upsert_goal(self, goal_data: dict) -> bool:
        """
        Insert or Update a goal.
        """
        if not self._initialized or not self._db:
            return False
            
        try:
            query = """
                INSERT INTO goals (
                    id, type, description, priority, expected_gain, 
                    target_entity, target_relation, status, created_at, 
                    completed_at, actual_gain, attempts, max_attempts
                ) VALUES (
                    :id, :type, :description, :priority, :expected_gain, 
                    :target_entity, :target_relation, :status, :created_at, 
                    :completed_at, :actual_gain, :attempts, :max_attempts
                )
                ON CONFLICT(id) DO UPDATE SET
                    description=excluded.description,
                    priority=excluded.priority,
                    status=excluded.status,
                    completed_at=excluded.completed_at,
                    actual_gain=excluded.actual_gain,
                    attempts=excluded.attempts
            """
            
            await self._db.execute(query, goal_data)
            await self._db.commit()
            return True
            
        except Exception as e:
            logger.error("upsert_goal_failed", error=str(e), goal_id=goal_data.get('id'))
            return False

    async def get_active_goals(self) -> List[dict]:
        """Fetch all goals that are pending or active."""
        if not self._initialized or not self._db:
            return []
            
        async with self._db.execute(
            "SELECT * FROM goals WHERE status IN ('pending', 'active') ORDER BY priority DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_goal(self, goal_id: str) -> Optional[dict]:
        """Fetch a single goal by ID."""
        if not self._initialized or not self._db:
            return None
            
        async with self._db.execute(
            "SELECT * FROM goals WHERE id = ?", (goal_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    # =========================================================================
    # HISTORY LOGGING
    # =========================================================================

    async def log_learning_iteration(self, iteration_data: dict) -> bool:
        """Log a learning step result."""
        if not self._initialized or not self._db:
            return False
            
        try:
            # Prepare JSON fields
            issues_json = json.dumps(iteration_data.get("issues", []))
            
            query = """
                INSERT INTO learning_history (
                    timestamp, goal_id, question, question_type, 
                    answer, strategy, verdict, confidence, duration_ms, 
                    knowledge_gained, gap_recorded, issues
                ) VALUES (
                    :timestamp, :goal_id, :question, :question_type, 
                    :answer, :strategy, :verdict, :confidence, :duration_ms, 
                    :knowledge_gained, :gap_recorded, :issues
                )
            """
            
            data = iteration_data.copy()
            data['issues'] = issues_json
            
            await self._db.execute(query, data)
            await self._db.commit()
            return True
        except Exception as e:
            logger.error("log_history_failed", error=str(e))
            return False

    async def log_world_event(self, event_data: dict) -> bool:
        """Log a world event."""
        if not self._initialized or not self._db:
            return False
            
        try:
            context_json = json.dumps(event_data.get("context", {}))
            
            query = """
                INSERT INTO world_events (
                    id, timestamp, actor, action, context
                ) VALUES (
                    :id, :timestamp, :actor, :action, :context
                )
            """
            
            data = event_data.copy()
            data['context'] = context_json
            
            await self._db.execute(query, data)
            await self._db.commit()
            return True
        except Exception as e:
            logger.error("log_event_failed", error=str(e))
            return False

    async def get_stats(self) -> dict:
        """Get persistence statistics."""
        if not self._initialized or not self._db:
            return {"status": "uninitialized"}
            
        try:
            async with self._db.execute("SELECT COUNT(*) FROM goals") as c:
                goal_count = (await c.fetchone())[0]
            
            async with self._db.execute("SELECT COUNT(*) FROM learning_history") as c:
                history_count = (await c.fetchone())[0]

            return {
                "status": "connected",
                "engine": "sqlite",
                "total_goals": goal_count,
                "history_events": history_count
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def clear_all_data(self) -> bool:
        """Clear all data from the database."""
        if not self._initialized or not self._db:
            return False
            
        try:
            await self._db.execute("DELETE FROM goals")
            await self._db.execute("DELETE FROM learning_history")
            await self._db.execute("DELETE FROM world_events")
            await self._db.commit()
            logger.info("persistence_data_cleared")
            return True
        except Exception as e:
            logger.error("persistence_clear_failed", error=str(e))
            return False

    async def close(self):
        """Close the database connection."""
        if self._db:
            await self._db.close()
            logger.info("persistence_plugin_closed")
