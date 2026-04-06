"""
Monitoring service for runtime logging and metrics.
Supports SQLite (local) and Supabase (cloud) backends.
"""

import json
import sqlite3
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from config import config


@dataclass
class QueryLog:
    """Represents a single query log entry."""
    query_id: str
    timestamp: str
    query_text: str
    papers_fetched: int
    chunks_indexed: int
    chunks_retrieved: int
    retrieval_latency_ms: float
    total_latency_ms: float
    groq_tokens_used: int
    agent_calls: Dict[str, int]
    hallucination_flag: bool
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "query_text": self.query_text,
            "papers_fetched": self.papers_fetched,
            "chunks_indexed": self.chunks_indexed,
            "chunks_retrieved": self.chunks_retrieved,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "groq_tokens_used": self.groq_tokens_used,
            "agent_calls": self.agent_calls,
            "hallucination_flag": self.hallucination_flag,
            "success": self.success,
            "error_message": self.error_message
        }


class MonitoringBackend(ABC):
    """Abstract base class for monitoring backends."""
    
    @abstractmethod
    def log_query(self, log: QueryLog) -> None:
        pass
    
    @abstractmethod
    def get_recent_logs(self, limit: int) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_aggregate_stats(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_latency_trend(self, limit: int) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_failure_logs(self, limit: int) -> List[Dict[str, Any]]:
        pass


class SQLiteBackend(MonitoringBackend):
    """SQLite backend for local monitoring."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT UNIQUE,
                    timestamp TEXT,
                    query_text TEXT,
                    papers_fetched INTEGER,
                    chunks_indexed INTEGER,
                    chunks_retrieved INTEGER,
                    retrieval_latency_ms REAL,
                    total_latency_ms REAL,
                    groq_tokens_used INTEGER,
                    agent_calls TEXT,
                    hallucination_flag BOOLEAN,
                    success BOOLEAN,
                    error_message TEXT
                )
            """)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def log_query(self, log: QueryLog) -> None:
        """Log a query to SQLite."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO query_logs (
                    query_id, timestamp, query_text, papers_fetched,
                    chunks_indexed, chunks_retrieved, retrieval_latency_ms,
                    total_latency_ms, groq_tokens_used, agent_calls,
                    hallucination_flag, success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log.query_id,
                log.timestamp,
                log.query_text,
                log.papers_fetched,
                log.chunks_indexed,
                log.chunks_retrieved,
                log.retrieval_latency_ms,
                log.total_latency_ms,
                log.groq_tokens_used,
                json.dumps(log.agent_calls),
                log.hallucination_flag,
                log.success,
                log.error_message
            ))
            conn.commit()
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query logs."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM query_logs 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            logs = []
            for row in rows:
                log_dict = dict(row)
                log_dict["agent_calls"] = json.loads(log_dict["agent_calls"])
                logs.append(log_dict)
            
            return logs
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) as count FROM query_logs").fetchone()["count"]
            
            if total == 0:
                return self._empty_stats()
            
            stats = conn.execute("""
                SELECT 
                    AVG(total_latency_ms) as avg_latency,
                    AVG(papers_fetched) as avg_papers,
                    AVG(chunks_indexed) as avg_chunks,
                    SUM(CASE WHEN hallucination_flag THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as hallucination_rate,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    SUM(groq_tokens_used) as total_tokens
                FROM query_logs
            """).fetchone()
            
            return {
                "total_queries": total,
                "avg_latency_ms": round(stats["avg_latency"] or 0, 2),
                "avg_papers_fetched": round(stats["avg_papers"] or 0, 2),
                "avg_chunks_indexed": round(stats["avg_chunks"] or 0, 2),
                "hallucination_rate": round(stats["hallucination_rate"] or 0, 2),
                "success_rate": round(stats["success_rate"] or 0, 2),
                "total_tokens_used": stats["total_tokens"] or 0
            }
    
    def get_latency_trend(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get latency trend data."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT timestamp, total_latency_ms, retrieval_latency_ms
                FROM query_logs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_failure_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent failure logs."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM query_logs 
                WHERE success = 0
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def _empty_stats(self) -> Dict[str, Any]:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0,
            "avg_papers_fetched": 0,
            "avg_chunks_indexed": 0,
            "hallucination_rate": 0,
            "success_rate": 0,
            "total_tokens_used": 0
        }


class SupabaseBackend(MonitoringBackend):
    """Supabase backend for cloud monitoring."""
    
    TABLE_NAME = "query_logs"
    
    def __init__(self, url: str, key: str):
        try:
            from supabase import create_client
            self.client = create_client(url, key)
            self._ensure_table_exists()
        except ImportError:
            raise ImportError("supabase package not installed. Run: pip install supabase")
    
    def _ensure_table_exists(self):
        """
        Note: Table must be created in Supabase dashboard with this schema:
        
        CREATE TABLE query_logs (
            id SERIAL PRIMARY KEY,
            query_id TEXT UNIQUE,
            timestamp TIMESTAMPTZ,
            query_text TEXT,
            papers_fetched INTEGER,
            chunks_indexed INTEGER,
            chunks_retrieved INTEGER,
            retrieval_latency_ms FLOAT,
            total_latency_ms FLOAT,
            groq_tokens_used INTEGER,
            agent_calls JSONB,
            hallucination_flag BOOLEAN,
            success BOOLEAN,
            error_message TEXT
        );
        """
        pass  # Table must be created manually in Supabase
    
    def log_query(self, log: QueryLog) -> None:
        """Log a query to Supabase."""
        try:
            data = log.to_dict()
            self.client.table(self.TABLE_NAME).upsert(
                data,
                on_conflict="query_id"
            ).execute()
        except Exception as e:
            print(f"Supabase log error: {e}")
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query logs from Supabase."""
        try:
            response = self.client.table(self.TABLE_NAME)\
                .select("*")\
                .order("timestamp", desc=True)\
                .limit(limit)\
                .execute()
            return response.data or []
        except Exception as e:
            print(f"Supabase fetch error: {e}")
            return []
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics from Supabase."""
        try:
            # Fetch all logs for aggregation (Supabase doesn't support complex aggregates)
            response = self.client.table(self.TABLE_NAME)\
                .select("*")\
                .execute()
            
            logs = response.data or []
            
            if not logs:
                return self._empty_stats()
            
            total = len(logs)
            successful = [l for l in logs if l.get("success")]
            hallucinated = [l for l in logs if l.get("hallucination_flag")]
            
            avg_latency = sum(l.get("total_latency_ms", 0) for l in logs) / total
            avg_papers = sum(l.get("papers_fetched", 0) for l in logs) / total
            avg_chunks = sum(l.get("chunks_indexed", 0) for l in logs) / total
            total_tokens = sum(l.get("groq_tokens_used", 0) for l in logs)
            
            return {
                "total_queries": total,
                "avg_latency_ms": round(avg_latency, 2),
                "avg_papers_fetched": round(avg_papers, 2),
                "avg_chunks_indexed": round(avg_chunks, 2),
                "hallucination_rate": round(len(hallucinated) / total * 100, 2),
                "success_rate": round(len(successful) / total * 100, 2),
                "total_tokens_used": total_tokens
            }
        except Exception as e:
            print(f"Supabase stats error: {e}")
            return self._empty_stats()
    
    def get_latency_trend(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get latency trend data from Supabase."""
        try:
            response = self.client.table(self.TABLE_NAME)\
                .select("timestamp,total_latency_ms,retrieval_latency_ms")\
                .order("timestamp", desc=True)\
                .limit(limit)\
                .execute()
            return response.data or []
        except Exception as e:
            print(f"Supabase trend error: {e}")
            return []
    
    def get_failure_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent failure logs from Supabase."""
        try:
            response = self.client.table(self.TABLE_NAME)\
                .select("*")\
                .eq("success", False)\
                .order("timestamp", desc=True)\
                .limit(limit)\
                .execute()
            return response.data or []
        except Exception as e:
            print(f"Supabase failure logs error: {e}")
            return []
    
    def _empty_stats(self) -> Dict[str, Any]:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0,
            "avg_papers_fetched": 0,
            "avg_chunks_indexed": 0,
            "hallucination_rate": 0,
            "success_rate": 0,
            "total_tokens_used": 0
        }


class MonitoringService:
    """
    Unified monitoring service that auto-selects backend.
    
    - Uses Supabase if SUPABASE_URL and SUPABASE_KEY are set
    - Falls back to SQLite for local development
    """
    
    def __init__(self):
        """Initialize monitoring service with appropriate backend."""
        self.backend = self._create_backend()
        self.backend_type = "supabase" if isinstance(self.backend, SupabaseBackend) else "sqlite"
    
    def _create_backend(self) -> MonitoringBackend:
        """Create the appropriate backend based on configuration."""
        if config.SUPABASE_URL and config.SUPABASE_KEY:
            try:
                return SupabaseBackend(config.SUPABASE_URL, config.SUPABASE_KEY)
            except Exception as e:
                print(f"Failed to initialize Supabase, falling back to SQLite: {e}")
                return SQLiteBackend(config.MONITORING_DB)
        else:
            return SQLiteBackend(config.MONITORING_DB)
    
    def log_query(self, log: QueryLog) -> None:
        """Log a query."""
        self.backend.log_query(log)
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query logs."""
        return self.backend.get_recent_logs(limit)
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        stats = self.backend.get_aggregate_stats()
        stats["backend"] = self.backend_type
        return stats
    
    def get_latency_trend(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get latency trend data."""
        return self.backend.get_latency_trend(limit)
    
    def get_failure_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent failure logs."""
        return self.backend.get_failure_logs(limit)


class QueryTimer:
    """Context manager for timing query operations."""
    
    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.checkpoints: Dict[str, float] = {}
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
    
    def checkpoint(self, name: str):
        """Record a checkpoint."""
        self.checkpoints[name] = time.time() - self.start_time
    
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
    
    @property
    def total_ms(self) -> float:
        """Get total elapsed time in milliseconds."""
        return (self.end_time - self.start_time) * 1000
    
    def get_checkpoint_ms(self, name: str) -> float:
        """Get checkpoint time in milliseconds."""
        return self.checkpoints.get(name, 0) * 1000


# Global monitoring service instance
monitoring_service = MonitoringService()
