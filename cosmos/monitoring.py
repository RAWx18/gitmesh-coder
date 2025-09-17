"""
Comprehensive monitoring and logging system for Redis GitHub Integration.

This module provides detailed logging, performance monitoring, health checks,
and alerting capabilities for production deployment.
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError, TimeoutError
except ImportError:
    redis = None

try:
    from .config import get_config
except ImportError:
    from config import get_config


# Configure structured logging
class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        """Format log record with structured data."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }


@dataclass
class HealthCheckResult:
    """Health check result data structure."""
    
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'status': self.status,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'details': self.details or {}
        }


class PerformanceMonitor:
    """
    Performance monitoring system for Redis operations and virtual file system.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.metrics: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0.0,
            'min_duration': float('inf'),
            'max_duration': 0.0,
            'error_count': 0,
            'last_error': None
        })
        self._lock = threading.Lock()
        
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
    
    def record_operation(
        self, 
        operation_name: str, 
        duration_ms: float, 
        success: bool = True,
        error: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record operation performance metrics.
        
        Args:
            operation_name: Name of the operation
            duration_ms: Duration in milliseconds
            success: Whether operation was successful
            error: Error message if operation failed
            tags: Additional tags for the metric
        """
        with self._lock:
            # Record metric
            metric = PerformanceMetric(
                name=f"operation.{operation_name}.duration",
                value=duration_ms,
                unit="ms",
                tags=tags or {}
            )
            self.metrics.append(metric)
            
            # Update operation statistics
            stats = self.operation_stats[operation_name]
            stats['count'] += 1
            stats['total_duration'] += duration_ms
            stats['min_duration'] = min(stats['min_duration'], duration_ms)
            stats['max_duration'] = max(stats['max_duration'], duration_ms)
            
            if not success:
                stats['error_count'] += 1
                stats['last_error'] = error
            
            # Log performance data
            self.logger.info(
                f"Operation completed: {operation_name}",
                extra={
                    'extra_fields': {
                        'operation': operation_name,
                        'duration_ms': duration_ms,
                        'success': success,
                        'error': error,
                        'tags': tags or {}
                    }
                }
            )
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get operation statistics.
        
        Args:
            operation_name: Specific operation name, or None for all operations
            
        Returns:
            Dictionary with operation statistics
        """
        with self._lock:
            if operation_name:
                if operation_name not in self.operation_stats:
                    return {}
                
                stats = self.operation_stats[operation_name].copy()
                if stats['count'] > 0:
                    stats['avg_duration'] = stats['total_duration'] / stats['count']
                    stats['error_rate'] = stats['error_count'] / stats['count']
                else:
                    stats['avg_duration'] = 0.0
                    stats['error_rate'] = 0.0
                
                return {operation_name: stats}
            else:
                # Return all operation stats
                result = {}
                for op_name, stats in self.operation_stats.items():
                    op_stats = stats.copy()
                    if op_stats['count'] > 0:
                        op_stats['avg_duration'] = op_stats['total_duration'] / op_stats['count']
                        op_stats['error_rate'] = op_stats['error_count'] / op_stats['count']
                    else:
                        op_stats['avg_duration'] = 0.0
                        op_stats['error_rate'] = 0.0
                    
                    result[op_name] = op_stats
                
                return result
    
    def get_recent_metrics(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent metrics within specified time window.
        
        Args:
            minutes: Time window in minutes
            
        Returns:
            List of recent metrics
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_metrics = [
                metric.to_dict() 
                for metric in self.metrics 
                if metric.timestamp >= cutoff_time
            ]
        
        return recent_metrics
    
    def clear_stats(self) -> None:
        """Clear all performance statistics."""
        with self._lock:
            self.metrics.clear()
            self.operation_stats.clear()


class RedisMonitor:
    """
    Redis-specific monitoring and health checking.
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        """
        Initialize Redis monitor.
        
        Args:
            performance_monitor: Performance monitor instance
        """
        self.performance_monitor = performance_monitor
        self.logger = logging.getLogger(f"{__name__}.RedisMonitor")
        self._redis_client: Optional[redis.Redis] = None
        
        # Initialize Redis client
        self._initialize_redis_client()
    
    def _initialize_redis_client(self) -> None:
        """Initialize Redis client for monitoring."""
        try:
            config = get_config()
            redis_kwargs = config.get_redis_connection_kwargs()
            self._redis_client = redis.Redis(**redis_kwargs)
            
            self.logger.info("Redis monitor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis monitor: {e}")
            self._redis_client = None
    
    def check_redis_health(self) -> HealthCheckResult:
        """
        Perform comprehensive Redis health check.
        
        Returns:
            HealthCheckResult with Redis health status
        """
        start_time = time.time()
        
        if not self._redis_client:
            return HealthCheckResult(
                name="redis_health",
                status="unhealthy",
                message="Redis client not initialized",
                duration_ms=0
            )
        
        try:
            # Test basic connectivity
            ping_result = self._redis_client.ping()
            
            # Get Redis info
            info = self._redis_client.info()
            
            # Check memory usage
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            memory_usage_pct = (used_memory / max_memory * 100) if max_memory > 0 else 0
            
            # Check connected clients
            connected_clients = info.get('connected_clients', 0)
            max_clients = info.get('maxclients', 10000)
            client_usage_pct = (connected_clients / max_clients * 100) if max_clients > 0 else 0
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine health status
            status = "healthy"
            issues = []
            
            if memory_usage_pct > 90:
                status = "degraded"
                issues.append(f"High memory usage: {memory_usage_pct:.1f}%")
            elif memory_usage_pct > 95:
                status = "unhealthy"
                issues.append(f"Critical memory usage: {memory_usage_pct:.1f}%")
            
            if client_usage_pct > 80:
                status = "degraded"
                issues.append(f"High client usage: {client_usage_pct:.1f}%")
            
            message = "Redis is healthy" if not issues else f"Issues: {', '.join(issues)}"
            
            # Record performance metric
            self.performance_monitor.record_operation(
                "redis_health_check",
                duration_ms,
                success=True,
                tags={"status": status}
            )
            
            return HealthCheckResult(
                name="redis_health",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details={
                    "ping_result": ping_result,
                    "redis_version": info.get("redis_version"),
                    "memory_usage_pct": memory_usage_pct,
                    "client_usage_pct": client_usage_pct,
                    "connected_clients": connected_clients,
                    "used_memory_human": info.get("used_memory_human"),
                    "uptime_seconds": info.get("uptime_in_seconds")
                }
            )
            
        except (ConnectionError, TimeoutError) as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.performance_monitor.record_operation(
                "redis_health_check",
                duration_ms,
                success=False,
                error=str(e)
            )
            
            self.logger.error(f"Redis health check failed: {e}")
            
            return HealthCheckResult(
                name="redis_health",
                status="unhealthy",
                message=f"Redis connection failed: {e}",
                duration_ms=duration_ms
            )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.performance_monitor.record_operation(
                "redis_health_check",
                duration_ms,
                success=False,
                error=str(e)
            )
            
            self.logger.error(f"Redis health check error: {e}")
            
            return HealthCheckResult(
                name="redis_health",
                status="unhealthy",
                message=f"Redis health check error: {e}",
                duration_ms=duration_ms
            )
    
    def get_redis_metrics(self) -> Dict[str, Any]:
        """
        Get detailed Redis metrics.
        
        Returns:
            Dictionary with Redis metrics
        """
        if not self._redis_client:
            return {"error": "Redis client not initialized"}
        
        try:
            info = self._redis_client.info()
            
            return {
                "server": {
                    "redis_version": info.get("redis_version"),
                    "uptime_seconds": info.get("uptime_in_seconds"),
                    "process_id": info.get("process_id")
                },
                "memory": {
                    "used_memory": info.get("used_memory"),
                    "used_memory_human": info.get("used_memory_human"),
                    "used_memory_peak": info.get("used_memory_peak"),
                    "used_memory_peak_human": info.get("used_memory_peak_human"),
                    "maxmemory": info.get("maxmemory"),
                    "maxmemory_human": info.get("maxmemory_human")
                },
                "clients": {
                    "connected_clients": info.get("connected_clients"),
                    "client_recent_max_input_buffer": info.get("client_recent_max_input_buffer"),
                    "client_recent_max_output_buffer": info.get("client_recent_max_output_buffer")
                },
                "stats": {
                    "total_connections_received": info.get("total_connections_received"),
                    "total_commands_processed": info.get("total_commands_processed"),
                    "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec"),
                    "total_net_input_bytes": info.get("total_net_input_bytes"),
                    "total_net_output_bytes": info.get("total_net_output_bytes"),
                    "rejected_connections": info.get("rejected_connections"),
                    "expired_keys": info.get("expired_keys"),
                    "evicted_keys": info.get("evicted_keys")
                },
                "keyspace": {
                    db: {
                        "keys": info.get(f"{db}_keys", 0),
                        "expires": info.get(f"{db}_expires", 0),
                        "avg_ttl": info.get(f"{db}_avg_ttl", 0)
                    }
                    for db in ["db0", "db1", "db2", "db3", "db4"]
                    if f"{db}_keys" in info
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get Redis metrics: {e}")
            return {"error": str(e)}


class SystemMonitor:
    """
    System-wide monitoring and alerting.
    """
    
    def __init__(self):
        """Initialize system monitor."""
        self.performance_monitor = PerformanceMonitor()
        self.redis_monitor = RedisMonitor(self.performance_monitor)
        self.logger = logging.getLogger(f"{__name__}.SystemMonitor")
        
        # Health check registry
        self.health_checks: Dict[str, Callable[[], HealthCheckResult]] = {
            "redis": self.redis_monitor.check_redis_health,
            "configuration": self._check_configuration_health,
            "tier_system": self._check_tier_system_health
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "redis_response_time_ms": 1000,
            "redis_error_rate": 0.05,  # 5%
            "memory_usage_pct": 90,
            "client_usage_pct": 80
        }
    
    def _check_configuration_health(self) -> HealthCheckResult:
        """Check configuration health."""
        start_time = time.time()
        
        try:
            config = get_config()
            summary = config.get_configuration_summary()
            
            duration_ms = (time.time() - start_time) * 1000
            
            if summary.get("status") == "validated":
                return HealthCheckResult(
                    name="configuration",
                    status="healthy",
                    message="Configuration is valid",
                    duration_ms=duration_ms,
                    details=summary
                )
            else:
                return HealthCheckResult(
                    name="configuration",
                    status="unhealthy",
                    message="Configuration validation failed",
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="configuration",
                status="unhealthy",
                message=f"Configuration check failed: {e}",
                duration_ms=duration_ms
            )
    
    def _check_tier_system_health(self) -> HealthCheckResult:
        """Check tier system health."""
        start_time = time.time()
        
        try:
            config = get_config()
            tier_limits = config.tier_limits
            current_tier = config.system.user_tier
            
            # Validate tier system
            is_valid = tier_limits.validate_tier(current_tier)
            current_limit = tier_limits.get_limit(current_tier)
            
            duration_ms = (time.time() - start_time) * 1000
            
            if is_valid:
                return HealthCheckResult(
                    name="tier_system",
                    status="healthy",
                    message="Tier system is operational",
                    duration_ms=duration_ms,
                    details={
                        "current_tier": current_tier,
                        "current_limit": current_limit,
                        "tier_valid": is_valid
                    }
                )
            else:
                return HealthCheckResult(
                    name="tier_system",
                    status="unhealthy",
                    message=f"Invalid tier configuration: {current_tier}",
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="tier_system",
                status="unhealthy",
                message=f"Tier system check failed: {e}",
                duration_ms=duration_ms
            )
    
    def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary mapping check names to results
        """
        results = {}
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[check_name] = result
                
                self.logger.info(
                    f"Health check completed: {check_name}",
                    extra={
                        'extra_fields': {
                            'check_name': check_name,
                            'status': result.status,
                            'duration_ms': result.duration_ms
                        }
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Health check failed: {check_name}: {e}")
                
                results[check_name] = HealthCheckResult(
                    name=check_name,
                    status="unhealthy",
                    message=f"Health check exception: {e}"
                )
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary with system status information
        """
        # Run health checks
        health_results = self.run_all_health_checks()
        
        # Get performance stats
        performance_stats = self.performance_monitor.get_operation_stats()
        
        # Get Redis metrics
        redis_metrics = self.redis_monitor.get_redis_metrics()
        
        # Determine overall status
        statuses = [result.status for result in health_results.values()]
        if "unhealthy" in statuses:
            overall_status = "unhealthy"
        elif "degraded" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status,
            "health_checks": {
                name: result.to_dict() 
                for name, result in health_results.items()
            },
            "performance_stats": performance_stats,
            "redis_metrics": redis_metrics,
            "alert_thresholds": self.alert_thresholds
        }
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for alert conditions.
        
        Returns:
            List of active alerts
        """
        alerts = []
        
        # Check Redis performance alerts
        redis_stats = self.performance_monitor.get_operation_stats()
        
        for operation, stats in redis_stats.items():
            if "redis" in operation.lower():
                # Check response time
                avg_duration = stats.get('avg_duration', 0)
                if avg_duration > self.alert_thresholds['redis_response_time_ms']:
                    alerts.append({
                        "type": "performance",
                        "severity": "warning",
                        "message": f"High Redis response time: {avg_duration:.1f}ms",
                        "operation": operation,
                        "threshold": self.alert_thresholds['redis_response_time_ms'],
                        "current_value": avg_duration
                    })
                
                # Check error rate
                error_rate = stats.get('error_rate', 0)
                if error_rate > self.alert_thresholds['redis_error_rate']:
                    alerts.append({
                        "type": "error_rate",
                        "severity": "critical",
                        "message": f"High Redis error rate: {error_rate:.1%}",
                        "operation": operation,
                        "threshold": self.alert_thresholds['redis_error_rate'],
                        "current_value": error_rate
                    })
        
        # Check Redis resource alerts
        redis_metrics = self.redis_monitor.get_redis_metrics()
        if "memory" in redis_metrics:
            memory_info = redis_metrics["memory"]
            used_memory = memory_info.get("used_memory", 0)
            max_memory = memory_info.get("maxmemory", 0)
            
            if max_memory > 0:
                memory_usage_pct = (used_memory / max_memory) * 100
                if memory_usage_pct > self.alert_thresholds['memory_usage_pct']:
                    alerts.append({
                        "type": "resource",
                        "severity": "critical" if memory_usage_pct > 95 else "warning",
                        "message": f"High Redis memory usage: {memory_usage_pct:.1f}%",
                        "threshold": self.alert_thresholds['memory_usage_pct'],
                        "current_value": memory_usage_pct
                    })
        
        return alerts


# Global monitoring instance
_system_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """
    Get the global system monitor instance.
    
    Returns:
        SystemMonitor instance
    """
    global _system_monitor
    
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    
    return _system_monitor


def setup_monitoring_logging(log_level: str = "INFO") -> None:
    """
    Setup monitoring and structured logging.
    
    Args:
        log_level: Logging level
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add structured logging handler
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(handler)
    
    # Configure specific loggers
    monitoring_logger = logging.getLogger(__name__)
    monitoring_logger.setLevel(getattr(logging, log_level.upper()))
    
    monitoring_logger.info("Monitoring and logging system initialized")


# Context manager for operation monitoring
class MonitoredOperation:
    """Context manager for monitoring operations."""
    
    def __init__(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Initialize monitored operation.
        
        Args:
            operation_name: Name of the operation
            tags: Additional tags for monitoring
        """
        self.operation_name = operation_name
        self.tags = tags or {}
        self.start_time = None
        self.monitor = get_system_monitor().performance_monitor
        self.logger = logging.getLogger(f"{__name__}.MonitoredOperation")
    
    def __enter__(self):
        """Start monitoring operation."""
        self.start_time = time.time()
        
        self.logger.info(
            f"Operation started: {self.operation_name}",
            extra={
                'extra_fields': {
                    'operation': self.operation_name,
                    'tags': self.tags,
                    'phase': 'start'
                }
            }
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End monitoring operation."""
        if self.start_time is None:
            return
        
        duration_ms = (time.time() - self.start_time) * 1000
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        
        # Record performance metric
        self.monitor.record_operation(
            self.operation_name,
            duration_ms,
            success=success,
            error=error,
            tags=self.tags
        )
        
        self.logger.info(
            f"Operation completed: {self.operation_name}",
            extra={
                'extra_fields': {
                    'operation': self.operation_name,
                    'duration_ms': duration_ms,
                    'success': success,
                    'error': error,
                    'tags': self.tags,
                    'phase': 'end'
                }
            }
        )