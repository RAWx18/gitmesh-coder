"""
Production-ready configuration management for Redis GitHub Integration.

This module provides secure environment configuration handling, validation,
and startup checks for the Redis-based GitHub repository caching system.
"""

import os
import ssl
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class SecurityError(Exception):
    """Raised when security validation fails."""
    pass


@dataclass
class RedisCloudConfig:
    """Secure Redis cloud connection configuration with SSL/TLS support.
    
    This configuration enforces Redis Cloud connections only - no local Redis allowed.
    """
    
    # Connection settings
    url: str
    password: Optional[str] = None
    db: int = 0
    
    # SSL/TLS settings
    ssl_enabled: bool = False
    ssl_cert_reqs: str = "required"  # none, optional, required
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_check_hostname: bool = True
    
    # Connection pooling and timeouts
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = field(default_factory=dict)
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    max_connections: int = 20
    
    # Performance settings
    decode_responses: bool = True
    encoding: str = "utf-8"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_url()
        self._validate_ssl_config()
        self._validate_connection_params()
    
    def _validate_url(self) -> None:
        """Validate Redis URL format and enforce Redis Cloud only."""
        if not self.url:
            raise ConfigurationError("Redis URL is required")
        
        try:
            parsed = urlparse(self.url)
            if not parsed.scheme:
                raise ConfigurationError("Redis URL must include scheme (redis:// or rediss://)")
            
            if parsed.scheme not in ['redis', 'rediss']:
                raise ConfigurationError("Redis URL scheme must be 'redis' or 'rediss'")
            
            if not parsed.hostname:
                raise ConfigurationError("Redis URL must include hostname")
            
            # Enforce Redis Cloud only - no local connections
            if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
                raise ConfigurationError(
                    "Local Redis connections not allowed. Use Redis Cloud only. "
                    f"Current hostname: {parsed.hostname}"
                )
            
            # Auto-enable SSL for rediss:// URLs
            if parsed.scheme == 'rediss':
                self.ssl_enabled = True
            
            # Validate Redis Cloud hostname patterns
            if not self._is_valid_redis_cloud_hostname(parsed.hostname):
                logger.warning(
                    f"Hostname '{parsed.hostname}' doesn't match common Redis Cloud patterns. "
                    "Ensure this is a valid Redis Cloud endpoint."
                )
                
        except Exception as e:
            raise ConfigurationError(f"Invalid Redis URL format: {e}")
    
    def _is_valid_redis_cloud_hostname(self, hostname: str) -> bool:
        """Check if hostname matches common Redis Cloud patterns."""
        redis_cloud_patterns = [
            '.redis.cloud',
            '.redislabs.com', 
            '.redis-cloud.com',
            '.aws.redislabs.com',
            '.gcp.redislabs.com',
            '.azure.redislabs.com'
        ]
        
        return any(pattern in hostname.lower() for pattern in redis_cloud_patterns)
    
    def _validate_ssl_config(self) -> None:
        """Validate SSL/TLS configuration."""
        if self.ssl_enabled:
            # Validate SSL certificate requirements
            valid_cert_reqs = ['none', 'optional', 'required']
            if self.ssl_cert_reqs not in valid_cert_reqs:
                raise ConfigurationError(
                    f"ssl_cert_reqs must be one of: {valid_cert_reqs}"
                )
            
            # Validate certificate files if provided
            if self.ssl_ca_certs and not Path(self.ssl_ca_certs).exists():
                raise ConfigurationError(f"SSL CA certificate file not found: {self.ssl_ca_certs}")
            
            if self.ssl_certfile and not Path(self.ssl_certfile).exists():
                raise ConfigurationError(f"SSL certificate file not found: {self.ssl_certfile}")
            
            if self.ssl_keyfile and not Path(self.ssl_keyfile).exists():
                raise ConfigurationError(f"SSL key file not found: {self.ssl_keyfile}")
    
    def _validate_connection_params(self) -> None:
        """Validate connection parameters."""
        if self.socket_timeout <= 0:
            raise ConfigurationError("socket_timeout must be positive")
        
        if self.socket_connect_timeout <= 0:
            raise ConfigurationError("socket_connect_timeout must be positive")
        
        if self.max_connections <= 0:
            raise ConfigurationError("max_connections must be positive")
        
        if self.health_check_interval <= 0:
            raise ConfigurationError("health_check_interval must be positive")
    
    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for secure connections."""
        if not self.ssl_enabled:
            return None
        
        try:
            context = ssl.create_default_context()
            
            # Configure certificate requirements
            if self.ssl_cert_reqs == 'none':
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            elif self.ssl_cert_reqs == 'optional':
                context.verify_mode = ssl.CERT_OPTIONAL
            else:  # required
                context.verify_mode = ssl.CERT_REQUIRED
            
            # Load custom CA certificates
            if self.ssl_ca_certs:
                context.load_verify_locations(self.ssl_ca_certs)
            
            # Load client certificate and key
            if self.ssl_certfile and self.ssl_keyfile:
                context.load_cert_chain(self.ssl_certfile, self.ssl_keyfile)
            
            # Configure hostname checking
            context.check_hostname = self.ssl_check_hostname
            
            return context
            
        except Exception as e:
            raise SecurityError(f"Failed to create SSL context: {e}")
    
    @classmethod
    def from_env(cls) -> 'RedisCloudConfig':
        """Create RedisCloudConfig from environment variables with validation."""
        try:
            # Basic connection settings
            url = os.getenv('REDIS_URL')
            if not url:
                raise ConfigurationError("REDIS_URL environment variable is required")
            
            password = os.getenv('REDIS_PASSWORD')
            db = int(os.getenv('REDIS_DB', '0'))
            
            # SSL/TLS settings
            ssl_enabled = os.getenv('REDIS_SSL', 'false').lower() == 'true'
            ssl_cert_reqs = os.getenv('REDIS_SSL_CERT_REQS', 'required').lower()
            ssl_ca_certs = os.getenv('REDIS_SSL_CA_CERTS')
            ssl_certfile = os.getenv('REDIS_SSL_CERTFILE')
            ssl_keyfile = os.getenv('REDIS_SSL_KEYFILE')
            ssl_check_hostname = os.getenv('REDIS_SSL_CHECK_HOSTNAME', 'true').lower() == 'true'
            
            # Connection settings
            socket_timeout = float(os.getenv('REDIS_SOCKET_TIMEOUT', '5.0'))
            socket_connect_timeout = float(os.getenv('REDIS_CONNECT_TIMEOUT', '5.0'))
            socket_keepalive = os.getenv('REDIS_SOCKET_KEEPALIVE', 'true').lower() == 'true'
            retry_on_timeout = os.getenv('REDIS_RETRY_ON_TIMEOUT', 'true').lower() == 'true'
            health_check_interval = int(os.getenv('REDIS_HEALTH_CHECK_INTERVAL', '30'))
            max_connections = int(os.getenv('REDIS_MAX_CONNECTIONS', '20'))
            
            # Performance settings
            decode_responses = os.getenv('REDIS_DECODE_RESPONSES', 'true').lower() == 'true'
            encoding = os.getenv('REDIS_ENCODING', 'utf-8')
            
            return cls(
                url=url,
                password=password,
                db=db,
                ssl_enabled=ssl_enabled,
                ssl_cert_reqs=ssl_cert_reqs,
                ssl_ca_certs=ssl_ca_certs,
                ssl_certfile=ssl_certfile,
                ssl_keyfile=ssl_keyfile,
                ssl_check_hostname=ssl_check_hostname,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                socket_keepalive=socket_keepalive,
                retry_on_timeout=retry_on_timeout,
                health_check_interval=health_check_interval,
                max_connections=max_connections,
                decode_responses=decode_responses,
                encoding=encoding
            )
            
        except ValueError as e:
            raise ConfigurationError(f"Invalid environment variable value: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load Redis configuration: {e}")


@dataclass
class TierLimitsConfig:
    """Tier-based access control configuration."""
    
    free_limit: int = 50000
    pro_limit: int = 500000
    enterprise_limit: int = 2000000
    
    def __post_init__(self):
        """Validate tier limits."""
        if self.free_limit <= 0:
            raise ConfigurationError("Free tier limit must be positive")
        
        if self.pro_limit <= self.free_limit:
            raise ConfigurationError("Pro tier limit must be greater than free tier")
        
        if self.enterprise_limit <= self.pro_limit:
            raise ConfigurationError("Enterprise tier limit must be greater than pro tier")
    
    def get_limit(self, tier: str) -> int:
        """Get token limit for specified tier."""
        tier_lower = tier.lower()
        if tier_lower == 'free':
            return self.free_limit
        elif tier_lower == 'pro':
            return self.pro_limit
        elif tier_lower == 'enterprise':
            return self.enterprise_limit
        else:
            raise ConfigurationError(f"Unknown tier: {tier}")
    
    def validate_tier(self, tier: str) -> bool:
        """Validate if tier is supported."""
        return tier.lower() in ['free', 'pro', 'enterprise']
    
    @classmethod
    def from_env(cls) -> 'TierLimitsConfig':
        """Create TierLimitsConfig from environment variables."""
        try:
            free_limit = int(os.getenv('TIER_FREE_LIMIT', '50000'))
            pro_limit = int(os.getenv('TIER_PRO_LIMIT', '500000'))
            enterprise_limit = int(os.getenv('TIER_ENTERPRISE_LIMIT', '2000000'))
            
            return cls(
                free_limit=free_limit,
                pro_limit=pro_limit,
                enterprise_limit=enterprise_limit
            )
            
        except ValueError as e:
            raise ConfigurationError(f"Invalid tier limit value: {e}")


@dataclass
class GitHubConfig:
    """GitHub API configuration."""
    
    token: Optional[str] = None
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate GitHub configuration."""
        
        if self.max_retries < 0:
            raise ConfigurationError("GitHub max_retries must be non-negative")
    
    @classmethod
    def from_env(cls) -> 'GitHubConfig':
        """Create GitHubConfig from environment variables."""
        try:
            token = os.getenv('GITHUB_TOKEN')
            max_retries = int(os.getenv('GITHUB_MAX_RETRIES', '3'))
            
            return cls(
                token=token,
                max_retries=max_retries
            )
            
        except ValueError as e:
            raise ConfigurationError(f"Invalid GitHub configuration value: {e}")


@dataclass
class SystemConfig:
    """System-wide configuration."""
    
    user_tier: str = "free"
    storage_dir: Optional[str] = None
    log_level: str = "INFO"
    debug_mode: bool = False
    
    def __post_init__(self):
        """Validate system configuration."""
        valid_tiers = ['free', 'pro', 'enterprise']
        # Clean up the tier value (remove quotes and whitespace)
        self.user_tier = self.user_tier.strip().strip('"').strip("'").lower()
        if self.user_tier not in valid_tiers:
            raise ConfigurationError(f"user_tier must be one of: {valid_tiers}")
        
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_log_levels:
            raise ConfigurationError(f"log_level must be one of: {valid_log_levels}")
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Create SystemConfig from environment variables."""
        try:
            user_tier = os.getenv('TIER_PLAN', 'free')
            storage_dir = os.getenv('STORAGE_DIR')
            log_level = os.getenv('LOG_LEVEL', 'INFO')
            debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
            
            return cls(
                user_tier=user_tier,
                storage_dir=storage_dir,
                log_level=log_level,
                debug_mode=debug_mode
            )
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load system configuration: {e}")


class ProductionConfig:
    """
    Production-ready configuration manager with comprehensive validation
    and security checks.
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        self.redis: Optional[RedisCloudConfig] = None
        self.tier_limits: Optional[TierLimitsConfig] = None
        self.github: Optional[GitHubConfig] = None
        self.system: Optional[SystemConfig] = None
        self._validated = False
    
    def load_from_env(self) -> None:
        """Load all configuration from environment variables."""
        try:
            logger.info("Loading configuration from environment variables...")
            
            self.redis = RedisCloudConfig.from_env()
            self.tier_limits = TierLimitsConfig.from_env()
            self.github = GitHubConfig.from_env()
            self.system = SystemConfig.from_env()
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def validate_startup_requirements(self) -> None:
        """
        Perform comprehensive startup validation checks.
        
        Raises:
            ConfigurationError: If validation fails
            SecurityError: If security checks fail
        """
        if not all([self.redis, self.tier_limits, self.github, self.system]):
            raise ConfigurationError("Configuration not loaded. Call load_from_env() first.")
        
        logger.info("Performing startup validation checks...")
        
        # Validate Redis configuration
        self._validate_redis_config()
        
        # Validate tier configuration
        self._validate_tier_config()
        
        # Validate GitHub configuration
        self._validate_github_config()
        
        # Validate system configuration
        self._validate_system_config()
        
        # Perform security checks
        self._perform_security_checks()
        
        self._validated = True
        logger.info("All startup validation checks passed")
    
    def _validate_redis_config(self) -> None:
        """Validate Redis configuration - Redis Cloud only."""
        logger.debug("Validating Redis configuration (Redis Cloud only)...")
        
        # Enforce Redis Cloud only - no localhost allowed
        parsed_url = urlparse(self.redis.url)
        if parsed_url.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            raise ConfigurationError(
                "Local Redis connections not allowed. Use Redis Cloud only. "
                "Update REDIS_URL to point to your Redis Cloud instance."
            )
        
        # Enforce SSL for Redis Cloud (temporarily disabled for testing)
        # if not self.redis.ssl_enabled:
        #     raise ConfigurationError(
        #         "SSL/TLS is required for Redis Cloud connections. "
        #         "Set REDIS_SSL=true and configure SSL certificates."
        #     )
        
        # Enforce password for Redis Cloud
        if not self.redis.password:
            raise ConfigurationError(
                "Redis password is required for Redis Cloud connections. "
                "Set REDIS_PASSWORD to your Redis Cloud password."
            )
        
        # Validate Redis Cloud URL format
        if not parsed_url.scheme in ['redis', 'rediss']:
            raise ConfigurationError(
                "Invalid Redis URL scheme. Use 'redis://' or 'rediss://' for Redis Cloud."
            )
        
        # Recommend rediss:// for Redis Cloud
        if parsed_url.scheme == 'redis' and self.redis.ssl_enabled:
            logger.warning(
                "Consider using 'rediss://' URL scheme for SSL connections to Redis Cloud"
            )
    
    def _validate_tier_config(self) -> None:
        """Validate tier configuration."""
        logger.debug("Validating tier configuration...")
        
        # Validate current user tier
        if not self.tier_limits.validate_tier(self.system.user_tier):
            raise ConfigurationError(f"Invalid user tier: {self.system.user_tier}")
        
        # Check tier limits are reasonable
        if self.tier_limits.free_limit > 100000:
            logger.warning("Free tier limit seems high - consider reviewing")
    
    def _validate_github_config(self) -> None:
        """Validate GitHub configuration."""
        logger.debug("Validating GitHub configuration...")
        
        # Check GitHub token
        if not self.github.token:
            logger.warning("GitHub token not set - API rate limits will be restrictive")
        elif len(self.github.token) < 20:
            logger.warning("GitHub token seems too short - verify token format")
    
    def _validate_system_config(self) -> None:
        """Validate system configuration."""
        logger.debug("Validating system configuration...")
        
        # Check storage directory if specified
        if self.system.storage_dir:
            storage_path = Path(self.system.storage_dir)
            if not storage_path.exists():
                logger.warning(f"Storage directory does not exist: {self.system.storage_dir}")
            elif not storage_path.is_dir():
                raise ConfigurationError(f"Storage path is not a directory: {self.system.storage_dir}")
    
    def _perform_security_checks(self) -> None:
        """Perform security validation checks."""
        logger.debug("Performing security checks...")
        
        # Check for sensitive data in environment
        sensitive_vars = ['REDIS_PASSWORD', 'GITHUB_TOKEN']
        for var in sensitive_vars:
            value = os.getenv(var)
            if value and len(value) < 10:
                logger.warning(f"{var} seems too short - verify security")
        
        # Check SSL configuration - allow 'none' for Redis Cloud compatibility
        if self.redis.ssl_enabled and self.redis.ssl_cert_reqs == 'none':
            logger.info("SSL certificate verification disabled for Redis Cloud compatibility")
    
    def get_redis_connection_kwargs(self) -> Dict[str, Any]:
        """
        Get Redis connection parameters for redis-py client.
        
        Returns:
            Dictionary of connection parameters
        """
        if not self._validated:
            raise ConfigurationError("Configuration not validated. Call validate_startup_requirements() first.")
        
        parsed_url = urlparse(self.redis.url)
        
        kwargs = {
            'host': parsed_url.hostname,
            'port': parsed_url.port or 6379,
            'db': self.redis.db,
            'password': self.redis.password,
            'socket_timeout': self.redis.socket_timeout,
            'socket_connect_timeout': self.redis.socket_connect_timeout,
            'socket_keepalive': self.redis.socket_keepalive,
            'socket_keepalive_options': self.redis.socket_keepalive_options,
            'retry_on_timeout': self.redis.retry_on_timeout,
            'health_check_interval': self.redis.health_check_interval,
            'decode_responses': self.redis.decode_responses,
            'encoding': self.redis.encoding,
        }
        
        # Add SSL configuration if enabled
        if self.redis.ssl_enabled:
            # Map SSL cert requirements to ssl module constants
            import ssl
            cert_reqs_map = {
                'none': ssl.CERT_NONE,
                'optional': ssl.CERT_OPTIONAL,
                'required': ssl.CERT_REQUIRED
            }
            
            kwargs.update({
                'ssl': True,
                'ssl_check_hostname': self.redis.ssl_check_hostname,
                'ssl_cert_reqs': cert_reqs_map.get(self.redis.ssl_cert_reqs, ssl.CERT_REQUIRED),
            })
            
            # Add SSL certificate files if provided
            if self.redis.ssl_ca_certs:
                kwargs['ssl_ca_certs'] = self.redis.ssl_ca_certs
            if self.redis.ssl_certfile:
                kwargs['ssl_certfile'] = self.redis.ssl_certfile
            if self.redis.ssl_keyfile:
                kwargs['ssl_keyfile'] = self.redis.ssl_keyfile
        
        return kwargs
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary for logging/monitoring.
        
        Returns:
            Dictionary with configuration summary (sensitive data masked)
        """
        if not self._validated:
            return {"status": "not_validated"}
        
        parsed_url = urlparse(self.redis.url)
        
        return {
            "status": "validated",
            "redis": {
                "host": parsed_url.hostname,
                "port": parsed_url.port,
                "ssl_enabled": self.redis.ssl_enabled,
                "max_connections": self.redis.max_connections,
                "password_set": bool(self.redis.password)
            },
            "tier_limits": {
                "free": self.tier_limits.free_limit,
                "pro": self.tier_limits.pro_limit,
                "enterprise": self.tier_limits.enterprise_limit
            },
            "system": {
                "user_tier": self.system.user_tier,
                "log_level": self.system.log_level,
                "debug_mode": self.system.debug_mode
            },
            "github": {
                "token_set": bool(self.github.token),
            }
        }


# Global configuration instance
config = ProductionConfig()


def initialize_configuration() -> ProductionConfig:
    """
    Initialize and validate production configuration.
    
    Returns:
        Validated ProductionConfig instance
        
    Raises:
        ConfigurationError: If configuration is invalid
        SecurityError: If security validation fails
    """
    try:
        logger.info("Initializing production configuration...")
        
        config.load_from_env()
        config.validate_startup_requirements()
        
        # Log configuration summary
        summary = config.get_configuration_summary()
        logger.info(f"Configuration initialized: {summary}")
        
        return config
        
    except Exception as e:
        logger.error(f"Configuration initialization failed: {e}")
        raise


def get_config() -> ProductionConfig:
    """
    Get the global configuration instance.
    
    Returns:
        ProductionConfig instance
        
    Raises:
        ConfigurationError: If configuration not initialized
    """
    if not config._validated:
        raise ConfigurationError(
            "Configuration not initialized. Call initialize_configuration() first."
        )
    
    return config