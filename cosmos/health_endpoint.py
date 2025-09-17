"""
Health check endpoint for Redis GitHub Integration.

This module provides HTTP health check endpoints for monitoring
and alerting in production deployments.
"""

import json
import time
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import logging

from .monitoring import get_system_monitor, setup_monitoring_logging


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints."""
    
    def __init__(self, *args, **kwargs):
        """Initialize handler."""
        self.system_monitor = get_system_monitor()
        self.logger = logging.getLogger(f"{__name__}.HealthCheckHandler")
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        try:
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            query_params = parse_qs(parsed_url.query)
            
            if path == "/health":
                self._handle_health_check(query_params)
            elif path == "/health/redis":
                self._handle_redis_health()
            elif path == "/health/detailed":
                self._handle_detailed_health()
            elif path == "/metrics":
                self._handle_metrics()
            elif path == "/alerts":
                self._handle_alerts()
            else:
                self._send_error(404, "Not Found")
                
        except Exception as e:
            self.logger.error(f"Health check handler error: {e}")
            self._send_error(500, f"Internal Server Error: {e}")
    
    def _handle_health_check(self, query_params: Dict[str, list]):
        """Handle basic health check."""
        try:
            # Run health checks
            health_results = self.system_monitor.run_all_health_checks()
            
            # Determine overall status
            statuses = [result.status for result in health_results.values()]
            
            if "unhealthy" in statuses:
                overall_status = "unhealthy"
                http_status = 503  # Service Unavailable
            elif "degraded" in statuses:
                overall_status = "degraded"
                http_status = 200  # OK but with warnings
            else:
                overall_status = "healthy"
                http_status = 200  # OK
            
            response_data = {
                "status": overall_status,
                "timestamp": time.time(),
                "checks": {
                    name: {
                        "status": result.status,
                        "message": result.message,
                        "duration_ms": result.duration_ms
                    }
                    for name, result in health_results.items()
                }
            }
            
            self._send_json_response(response_data, http_status)
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._send_error(500, f"Health check failed: {e}")
    
    def _handle_redis_health(self):
        """Handle Redis-specific health check."""
        try:
            redis_health = self.system_monitor.redis_monitor.check_redis_health()
            
            http_status = 200 if redis_health.status == "healthy" else 503
            
            response_data = redis_health.to_dict()
            self._send_json_response(response_data, http_status)
            
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            self._send_error(500, f"Redis health check failed: {e}")
    
    def _handle_detailed_health(self):
        """Handle detailed system health check."""
        try:
            system_status = self.system_monitor.get_system_status()
            
            overall_status = system_status.get("overall_status", "unknown")
            http_status = 200 if overall_status == "healthy" else 503
            
            self._send_json_response(system_status, http_status)
            
        except Exception as e:
            self.logger.error(f"Detailed health check failed: {e}")
            self._send_error(500, f"Detailed health check failed: {e}")
    
    def _handle_metrics(self):
        """Handle metrics endpoint."""
        try:
            performance_stats = self.system_monitor.performance_monitor.get_operation_stats()
            redis_metrics = self.system_monitor.redis_monitor.get_redis_metrics()
            
            response_data = {
                "timestamp": time.time(),
                "performance_stats": performance_stats,
                "redis_metrics": redis_metrics
            }
            
            self._send_json_response(response_data, 200)
            
        except Exception as e:
            self.logger.error(f"Metrics endpoint failed: {e}")
            self._send_error(500, f"Metrics endpoint failed: {e}")
    
    def _handle_alerts(self):
        """Handle alerts endpoint."""
        try:
            alerts = self.system_monitor.check_alerts()
            
            response_data = {
                "timestamp": time.time(),
                "alert_count": len(alerts),
                "alerts": alerts
            }
            
            # Return 200 for no alerts, 503 for critical alerts
            has_critical = any(alert.get("severity") == "critical" for alert in alerts)
            http_status = 503 if has_critical else 200
            
            self._send_json_response(response_data, http_status)
            
        except Exception as e:
            self.logger.error(f"Alerts endpoint failed: {e}")
            self._send_error(500, f"Alerts endpoint failed: {e}")
    
    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response."""
        response_body = json.dumps(data, indent=2).encode('utf-8')
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_body)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        
        self.wfile.write(response_body)
    
    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        error_data = {
            "error": message,
            "status_code": status_code,
            "timestamp": time.time()
        }
        
        self._send_json_response(error_data, status_code)
    
    def log_message(self, format, *args):
        """Override log message to use our logger."""
        self.logger.info(f"{self.address_string()} - {format % args}")


class HealthCheckServer:
    """
    HTTP server for health check endpoints.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Initialize health check server.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(f"{__name__}.HealthCheckServer")
    
    def start(self) -> None:
        """Start the health check server."""
        try:
            self.server = HTTPServer((self.host, self.port), HealthCheckHandler)
            
            self.server_thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True
            )
            self.server_thread.start()
            
            self.logger.info(f"Health check server started on {self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start health check server: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the health check server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            
            if self.server_thread:
                self.server_thread.join(timeout=5)
            
            self.logger.info("Health check server stopped")
    
    def is_running(self) -> bool:
        """Check if server is running."""
        return (self.server is not None and 
                self.server_thread is not None and 
                self.server_thread.is_alive())


def start_health_server(host: str = "0.0.0.0", port: int = 8080) -> HealthCheckServer:
    """
    Start health check server.
    
    Args:
        host: Server host
        port: Server port
        
    Returns:
        HealthCheckServer instance
    """
    server = HealthCheckServer(host, port)
    server.start()
    return server


if __name__ == "__main__":
    """Run health check server standalone."""
    import sys
    import signal
    
    # Setup logging
    setup_monitoring_logging()
    
    # Parse command line arguments
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    
    # Start server
    server = start_health_server(host, port)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"Health check server running on http://{host}:{port}")
    print("Available endpoints:")
    print("  GET /health          - Basic health check")
    print("  GET /health/redis    - Redis health check")
    print("  GET /health/detailed - Detailed system health")
    print("  GET /metrics         - Performance metrics")
    print("  GET /alerts          - Active alerts")
    print("\nPress Ctrl+C to stop...")
    
    # Keep main thread alive
    try:
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()