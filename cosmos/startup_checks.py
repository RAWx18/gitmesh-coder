"""
Startup validation and health checks for Redis GitHub Integration.

This module provides comprehensive startup checks to ensure the system
is properly configured and ready for production deployment.
"""

import sys
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse

try:
    import redis
    from redis.exceptions import ConnectionError, TimeoutError, RedisError
except ImportError:
    redis = None

from .config import initialize_configuration, ConfigurationError, SecurityError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    
    name: str
    status: str  # "pass", "fail", "warn"
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None


class StartupValidator:
    """
    Comprehensive startup validation and health checking system.
    """
    
    def __init__(self):
        """Initialize startup validator."""
        self.config = None
        self.results: List[HealthCheckResult] = []
    
    def run_all_checks(self) -> bool:
        """
        Run all startup validation checks.
        
        Returns:
            True if all critical checks pass, False otherwise
        """
        logger.info("Starting comprehensive startup validation...")
        
        self.results.clear()
        
        # Configuration checks
        config_ok = self._check_configuration()
        
        # Redis connectivity checks
        redis_ok = self._check_redis_connectivity() if config_ok else False
        
        # Redis operations checks
        redis_ops_ok = self._check_redis_operations() if redis_ok else False
        
        # Tier system checks
        tier_ok = self._check_tier_system() if config_ok else False
        
        # GitHub API checks
        github_ok = self._check_github_connectivity() if config_ok else False
        
        # Security validation
        security_ok = self._check_security_configuration() if config_ok else False
        
        # Performance checks
        performance_ok = self._check_performance_settings() if config_ok else False
        
        # Determine overall status
        critical_checks = [config_ok, redis_ok, redis_ops_ok, tier_ok, security_ok]
        all_critical_pass = all(critical_checks)
        
        # Log summary
        self._log_validation_summary(all_critical_pass)
        
        return all_critical_pass
    
    def _check_configuration(self) -> bool:
        """Check configuration loading and validation."""
        start_time = time.time()
        
        try:
            logger.info("Validating configuration...")
            
            self.config = initialize_configuration()
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.append(HealthCheckResult(
                name="configuration_loading",
                status="pass",
                message="Configuration loaded and validated successfully",
                duration_ms=duration_ms
            ))
            
            return True
            
        except (ConfigurationError, SecurityError) as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.append(HealthCheckResult(
                name="configuration_loading",
                status="fail",
                message=f"Configuration validation failed: {e}",
                duration_ms=duration_ms
            ))
            
            logger.error(f"Configuration validation failed: {e}")
            return False
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.append(HealthCheckResult(
                name="configuration_loading",
                status="fail",
                message=f"Unexpected configuration error: {e}",
                duration_ms=duration_ms
            ))
            
            logger.error(f"Unexpected configuration error: {e}")
            return False
    
    def _check_redis_connectivity(self) -> bool:
        """Check Redis Cloud connection and basic connectivity."""
        if not redis:
            self.results.append(HealthCheckResult(
                name="redis_cloud_connectivity",
                status="fail",
                message="Redis package not installed"
            ))
            return False
        
        start_time = time.time()
        
        try:
            logger.info("Testing Redis Cloud connectivity...")
            
            # Validate Redis Cloud configuration first
            redis_config = self.config.redis
            parsed_url = urlparse(redis_config.url)
            
            # Ensure it's not localhost
            if parsed_url.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
                duration_ms = (time.time() - start_time) * 1000
                self.results.append(HealthCheckResult(
                    name="redis_cloud_connectivity",
                    status="fail",
                    message="Local Redis detected - Redis Cloud required",
                    duration_ms=duration_ms
                ))
                return False
            
            # Get connection parameters
            conn_kwargs = self.config.get_redis_connection_kwargs()
            
            # Create test connection
            client = redis.Redis(**conn_kwargs)
            
            # Test basic connectivity
            ping_result = client.ping()
            
            # Get Redis info
            info = client.info()
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Validate it's actually Redis Cloud
            redis_version = info.get("redis_version", "")
            server_info = info.get("redis_mode", "standalone")
            
            self.results.append(HealthCheckResult(
                name="redis_cloud_connectivity",
                status="pass",
                message="Redis Cloud connection successful",
                details={
                    "ping_result": ping_result,
                    "redis_version": redis_version,
                    "server_mode": server_info,
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "ssl_enabled": redis_config.ssl_enabled,
                    "hostname": parsed_url.hostname
                },
                duration_ms=duration_ms
            ))
            
            # Close test connection
            client.close()
            
            return True
            
        except (ConnectionError, TimeoutError) as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.append(HealthCheckResult(
                name="redis_cloud_connectivity",
                status="fail",
                message=f"Redis Cloud connection failed: {e}",
                details={"hostname": parsed_url.hostname if 'parsed_url' in locals() else "unknown"},
                duration_ms=duration_ms
            ))
            
            logger.error(f"Redis Cloud connection failed: {e}")
            return False
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.append(HealthCheckResult(
                name="redis_cloud_connectivity",
                status="fail",
                message=f"Unexpected Redis Cloud error: {e}",
                duration_ms=duration_ms
            ))
            
            logger.error(f"Unexpected Redis Cloud error: {e}")
            return False
    
    def _check_redis_operations(self) -> bool:
        """Check Redis operations and performance."""
        start_time = time.time()
        
        try:
            logger.info("Testing Redis operations...")
            
            # Get connection parameters
            conn_kwargs = self.config.get_redis_connection_kwargs()
            client = redis.Redis(**conn_kwargs)
            
            # Test basic operations
            test_key = "startup_check:test"
            test_value = "test_value_123"
            
            # Test SET operation
            set_result = client.set(test_key, test_value, ex=60)  # 60 second expiry
            
            # Test GET operation
            get_result = client.get(test_key)
            
            # Test pipeline operations
            pipe = client.pipeline()
            pipe.set(f"{test_key}:1", "value1")
            pipe.set(f"{test_key}:2", "value2")
            pipe.get(f"{test_key}:1")
            pipe.get(f"{test_key}:2")
            pipeline_results = pipe.execute()
            
            # Test hash operations
            hash_key = f"{test_key}:hash"
            client.hset(hash_key, mapping={"field1": "value1", "field2": "value2"})
            hash_result = client.hgetall(hash_key)
            
            # Cleanup test keys
            client.delete(test_key, f"{test_key}:1", f"{test_key}:2", hash_key)
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Validate results
            if (set_result and 
                get_result.decode('utf-8') == test_value and 
                len(pipeline_results) == 4 and
                len(hash_result) == 2):
                
                self.results.append(HealthCheckResult(
                    name="redis_operations",
                    status="pass",
                    message="Redis operations test successful",
                    details={
                        "set_operation": "pass",
                        "get_operation": "pass",
                        "pipeline_operation": "pass",
                        "hash_operation": "pass"
                    },
                    duration_ms=duration_ms
                ))
                
                client.close()
                return True
            else:
                self.results.append(HealthCheckResult(
                    name="redis_operations",
                    status="fail",
                    message="Redis operations validation failed",
                    duration_ms=duration_ms
                ))
                
                client.close()
                return False
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.append(HealthCheckResult(
                name="redis_operations",
                status="fail",
                message=f"Redis operations test failed: {e}",
                duration_ms=duration_ms
            ))
            
            logger.error(f"Redis operations test failed: {e}")
            return False
    
    def _check_tier_system(self) -> bool:
        """Check tier system configuration and validation."""
        start_time = time.time()
        
        try:
            logger.info("Validating tier system...")
            
            # Test tier validation
            tier_limits = self.config.tier_limits
            
            # Test all tier types
            test_results = {}
            for tier in ['free', 'pro', 'enterprise']:
                try:
                    limit = tier_limits.get_limit(tier)
                    is_valid = tier_limits.validate_tier(tier)
                    test_results[tier] = {"limit": limit, "valid": is_valid}
                except Exception as e:
                    test_results[tier] = {"error": str(e)}
            
            # Test current user tier
            current_tier = self.config.system.user_tier
            current_limit = tier_limits.get_limit(current_tier)
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Validate all tiers work
            all_valid = all(
                result.get("valid", False) for result in test_results.values()
            )
            
            if all_valid:
                self.results.append(HealthCheckResult(
                    name="tier_system",
                    status="pass",
                    message="Tier system validation successful",
                    details={
                        "current_tier": current_tier,
                        "current_limit": current_limit,
                        "tier_tests": test_results
                    },
                    duration_ms=duration_ms
                ))
                
                return True
            else:
                self.results.append(HealthCheckResult(
                    name="tier_system",
                    status="fail",
                    message="Tier system validation failed",
                    details={"tier_tests": test_results},
                    duration_ms=duration_ms
                ))
                
                return False
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.append(HealthCheckResult(
                name="tier_system",
                status="fail",
                message=f"Tier system check failed: {e}",
                duration_ms=duration_ms
            ))
            
            logger.error(f"Tier system check failed: {e}")
            return False
    
    def _check_github_connectivity(self) -> bool:
        """Check GitHub API connectivity (optional)."""
        start_time = time.time()
        
        try:
            logger.info("Testing GitHub API connectivity...")
            
            github_config = self.config.github
            
            # If no token, mark as warning but not failure
            if not github_config.token:
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.append(HealthCheckResult(
                    name="github_connectivity",
                    status="warn",
                    message="GitHub token not configured - API rate limits will apply",
                    duration_ms=duration_ms
                ))
                
                return True
            
            # Test basic GitHub API connectivity (simple approach)
            import urllib.request
            import urllib.error
            
            try:
                headers = {
                    'Authorization': f'token {github_config.token}',
                    'User-Agent': 'cosmos-redis-integration/1.0'
                }
                
                req = urllib.request.Request(
                    f"{github_config.api_base_url}/user",
                    headers=headers
                )
                
                with urllib.request.urlopen(req, timeout=github_config.timeout) as response:
                    if response.status == 200:
                        duration_ms = (time.time() - start_time) * 1000
                        
                        self.results.append(HealthCheckResult(
                            name="github_connectivity",
                            status="pass",
                            message="GitHub API connectivity successful",
                            duration_ms=duration_ms
                        ))
                        
                        return True
                    else:
                        duration_ms = (time.time() - start_time) * 1000
                        
                        self.results.append(HealthCheckResult(
                            name="github_connectivity",
                            status="warn",
                            message=f"GitHub API returned status {response.status}",
                            duration_ms=duration_ms
                        ))
                        
                        return True  # Non-critical
                        
            except urllib.error.URLError as e:
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.append(HealthCheckResult(
                    name="github_connectivity",
                    status="warn",
                    message=f"GitHub API connectivity issue: {e}",
                    duration_ms=duration_ms
                ))
                
                return True  # Non-critical
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.append(HealthCheckResult(
                name="github_connectivity",
                status="warn",
                message=f"GitHub connectivity check failed: {e}",
                duration_ms=duration_ms
            ))
            
            logger.warning(f"GitHub connectivity check failed: {e}")
            return True  # Non-critical
    
    def _check_security_configuration(self) -> bool:
        """Check security configuration and best practices."""
        start_time = time.time()
        
        try:
            logger.info("Validating security configuration...")
            
            security_issues = []
            security_warnings = []
            
            # Check Redis security
            redis_config = self.config.redis
            
            if not redis_config.ssl_enabled and not self.config.system.debug_mode:
                security_warnings.append("Redis SSL not enabled")
            
            if not redis_config.password and not self.config.system.debug_mode:
                security_warnings.append("Redis password not set")
            
            if redis_config.ssl_enabled and redis_config.ssl_cert_reqs == 'none':
                security_issues.append("SSL certificate verification disabled")
            
            # Check GitHub token security
            github_config = self.config.github
            if github_config.token and len(github_config.token) < 20:
                security_warnings.append("GitHub token seems too short")
            
            duration_ms = (time.time() - start_time) * 1000
            
            if security_issues:
                self.results.append(HealthCheckResult(
                    name="security_configuration",
                    status="fail",
                    message="Security configuration issues found",
                    details={
                        "issues": security_issues,
                        "warnings": security_warnings
                    },
                    duration_ms=duration_ms
                ))
                
                return False
            elif security_warnings:
                self.results.append(HealthCheckResult(
                    name="security_configuration",
                    status="warn",
                    message="Security configuration warnings",
                    details={"warnings": security_warnings},
                    duration_ms=duration_ms
                ))
                
                return True
            else:
                self.results.append(HealthCheckResult(
                    name="security_configuration",
                    status="pass",
                    message="Security configuration validated",
                    duration_ms=duration_ms
                ))
                
                return True
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.append(HealthCheckResult(
                name="security_configuration",
                status="fail",
                message=f"Security validation failed: {e}",
                duration_ms=duration_ms
            ))
            
            logger.error(f"Security validation failed: {e}")
            return False
    
    def _check_performance_settings(self) -> bool:
        """Check performance-related configuration settings."""
        start_time = time.time()
        
        try:
            logger.info("Validating performance settings...")
            
            performance_warnings = []
            
            redis_config = self.config.redis
            
            # Check connection pool size
            if redis_config.max_connections < 10:
                performance_warnings.append("Redis max_connections is low (< 10)")
            elif redis_config.max_connections > 100:
                performance_warnings.append("Redis max_connections is very high (> 100)")
            
            # Check timeout settings
            if redis_config.socket_timeout < 1.0:
                performance_warnings.append("Redis socket_timeout is very low (< 1s)")
            elif redis_config.socket_timeout > 30.0:
                performance_warnings.append("Redis socket_timeout is very high (> 30s)")
            
            # Check health check interval
            if redis_config.health_check_interval < 10:
                performance_warnings.append("Redis health_check_interval is very low (< 10s)")
            
            duration_ms = (time.time() - start_time) * 1000
            
            if performance_warnings:
                self.results.append(HealthCheckResult(
                    name="performance_settings",
                    status="warn",
                    message="Performance configuration warnings",
                    details={"warnings": performance_warnings},
                    duration_ms=duration_ms
                ))
            else:
                self.results.append(HealthCheckResult(
                    name="performance_settings",
                    status="pass",
                    message="Performance settings validated",
                    duration_ms=duration_ms
                ))
            
            return True
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.append(HealthCheckResult(
                name="performance_settings",
                status="warn",
                message=f"Performance validation failed: {e}",
                duration_ms=duration_ms
            ))
            
            logger.warning(f"Performance validation failed: {e}")
            return True  # Non-critical
    
    def _log_validation_summary(self, overall_success: bool) -> None:
        """Log comprehensive validation summary."""
        logger.info("=" * 60)
        logger.info("STARTUP VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        # Count results by status
        pass_count = sum(1 for r in self.results if r.status == "pass")
        warn_count = sum(1 for r in self.results if r.status == "warn")
        fail_count = sum(1 for r in self.results if r.status == "fail")
        
        logger.info(f"Total checks: {len(self.results)}")
        logger.info(f"Passed: {pass_count}")
        logger.info(f"Warnings: {warn_count}")
        logger.info(f"Failed: {fail_count}")
        logger.info(f"Overall status: {'SUCCESS' if overall_success else 'FAILURE'}")
        
        # Log individual results
        for result in self.results:
            status_symbol = {
                "pass": "✓",
                "warn": "⚠",
                "fail": "✗"
            }.get(result.status, "?")
            
            duration_str = f" ({result.duration_ms:.1f}ms)" if result.duration_ms else ""
            logger.info(f"{status_symbol} {result.name}: {result.message}{duration_str}")
            
            if result.details and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  Details: {result.details}")
        
        logger.info("=" * 60)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status for monitoring/API endpoints.
        
        Returns:
            Dictionary with health status information
        """
        if not self.results:
            return {"status": "unknown", "message": "No health checks performed"}
        
        # Determine overall status
        has_failures = any(r.status == "fail" for r in self.results)
        has_warnings = any(r.status == "warn" for r in self.results)
        
        if has_failures:
            overall_status = "unhealthy"
        elif has_warnings:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "checks": {
                result.name: {
                    "status": result.status,
                    "message": result.message,
                    "duration_ms": result.duration_ms
                }
                for result in self.results
            },
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.status == "pass"),
                "warnings": sum(1 for r in self.results if r.status == "warn"),
                "failed": sum(1 for r in self.results if r.status == "fail")
            }
        }


def run_startup_validation() -> bool:
    """
    Run comprehensive startup validation.
    
    Returns:
        True if all critical checks pass, False otherwise
    """
    validator = StartupValidator()
    return validator.run_all_checks()


def get_system_health() -> Dict[str, Any]:
    """
    Get current system health status.
    
    Returns:
        Dictionary with health status
    """
    validator = StartupValidator()
    validator.run_all_checks()
    return validator.get_health_status()


if __name__ == "__main__":
    """Command-line startup validation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    success = run_startup_validation()
    
    if success:
        print("✓ All startup validation checks passed")
        sys.exit(0)
    else:
        print("✗ Startup validation failed")
        sys.exit(1)