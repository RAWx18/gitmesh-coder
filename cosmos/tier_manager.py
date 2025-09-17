"""
Tier-based access control system for repository access.

This module implements tiered access control based on user subscription plans,
validating token limits and preventing repository access when limits are exceeded.
"""

import os
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TierLevel(Enum):
    """Supported tier levels."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TierConfig:
    """Configuration for a specific tier."""
    max_tokens: int
    name: str
    
    def __post_init__(self):
        """Validate tier configuration after initialization."""
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if not self.name:
            raise ValueError("Tier name cannot be empty")


class TierValidationError(Exception):
    """Raised when tier validation fails."""
    pass


class TierAccessDeniedError(Exception):
    """Raised when access is denied due to tier limits."""
    pass


class TierManager:
    """
    Manages user tier validation and access control.
    
    Provides production-ready validation of token limits against user tiers,
    secure configuration loading, and access control logic.
    """
    
    # Default tier limits (can be overridden by environment variables)
    DEFAULT_TIER_LIMITS = {
        TierLevel.FREE.value: 50000,
        TierLevel.PRO.value: 500000,
        TierLevel.ENTERPRISE.value: 2000000
    }
    
    def __init__(self, custom_limits: Optional[Dict[str, int]] = None):
        """
        Initialize TierManager with optional custom limits.
        
        Args:
            custom_limits: Optional dictionary of tier limits to override defaults
            
        Raises:
            TierValidationError: If configuration is invalid
        """
        self.logger = logging.getLogger(__name__)
        self._tier_configs = {}
        
        try:
            self._load_tier_configuration(custom_limits)
            self._validate_configuration()
        except Exception as e:
            self.logger.error(f"Failed to initialize TierManager: {e}")
            raise TierValidationError(f"TierManager initialization failed: {e}")
    
    def _load_tier_configuration(self, custom_limits: Optional[Dict[str, int]] = None) -> None:
        """
        Load tier configuration from environment variables and custom limits.
        
        Args:
            custom_limits: Optional custom tier limits
            
        Raises:
            TierValidationError: If configuration loading fails
        """
        try:
            # Use custom limits if provided, otherwise start with defaults
            if custom_limits:
                limits = custom_limits.copy()
            else:
                # Start with default limits
                limits = self.DEFAULT_TIER_LIMITS.copy()
                
                # Override with environment variables if present
                env_limits = {
                    TierLevel.FREE.value: self._get_env_int("TIER_FREE_LIMIT"),
                    TierLevel.PRO.value: self._get_env_int("TIER_PRO_LIMIT"),
                    TierLevel.ENTERPRISE.value: self._get_env_int("TIER_ENTERPRISE_LIMIT")
                }
                
                # Update limits with valid environment values
                for tier, limit in env_limits.items():
                    if limit is not None:
                        limits[tier] = limit
            
            # Create tier configurations
            for tier_name, max_tokens in limits.items():
                self._tier_configs[tier_name] = TierConfig(
                    max_tokens=max_tokens,
                    name=tier_name
                )
                
            self.logger.info(f"Loaded tier configuration: {list(self._tier_configs.keys())}")
            
        except Exception as e:
            raise TierValidationError(f"Failed to load tier configuration: {e}")
    
    def _get_env_int(self, env_var: str) -> Optional[int]:
        """
        Safely get integer value from environment variable.
        
        Args:
            env_var: Environment variable name
            
        Returns:
            Integer value or None if not set or invalid
        """
        try:
            value = os.getenv(env_var)
            if value is not None:
                return int(value)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid value for {env_var}: {value}, error: {e}")
        return None
    
    def _validate_configuration(self) -> None:
        """
        Validate the loaded tier configuration.
        
        Raises:
            TierValidationError: If configuration is invalid
        """
        if not self._tier_configs:
            raise TierValidationError("No tier configurations loaded")
        
        # Validate all required tiers are present
        required_tiers = {tier.value for tier in TierLevel}
        loaded_tiers = set(self._tier_configs.keys())
        
        missing_tiers = required_tiers - loaded_tiers
        if missing_tiers:
            raise TierValidationError(f"Missing tier configurations: {missing_tiers}")
        
        # Validate tier limits are in ascending order
        free_limit = self._tier_configs[TierLevel.FREE.value].max_tokens
        pro_limit = self._tier_configs[TierLevel.PRO.value].max_tokens
        enterprise_limit = self._tier_configs[TierLevel.ENTERPRISE.value].max_tokens
        
        if not (free_limit <= pro_limit <= enterprise_limit):
            raise TierValidationError(
                f"Tier limits must be in ascending order: "
                f"free({free_limit}) <= pro({pro_limit}) <= enterprise({enterprise_limit})"
            )
    
    def validate_access(self, user_tier: str, estimated_tokens: int) -> Tuple[bool, str]:
        """
        Validate if user can access repository based on tier and token count.
        
        Args:
            user_tier: User's tier level (free, pro, enterprise)
            estimated_tokens: Estimated token count for the repository
            
        Returns:
            Tuple of (is_allowed, message)
            
        Raises:
            TierValidationError: If validation parameters are invalid
        """
        try:
            # Validate inputs
            if not user_tier:
                raise TierValidationError("User tier cannot be empty")
            
            if estimated_tokens < 0:
                raise TierValidationError(f"Estimated tokens cannot be negative: {estimated_tokens}")
            
            # Normalize tier name
            user_tier = user_tier.lower().strip()
            
            # Check if tier exists - access _tier_configs safely
            try:
                tier_configs = self._tier_configs
            except Exception as e:
                self.logger.error(f"Error accessing tier configurations: {e}")
                raise TierValidationError(f"Access validation failed: {e}")
            
            if user_tier not in tier_configs:
                available_tiers = list(tier_configs.keys())
                return False, f"Invalid tier '{user_tier}'. Available tiers: {available_tiers}"
            
            # Get tier limit
            tier_config = tier_configs[user_tier]
            tier_limit = tier_config.max_tokens
            
            # Check access
            if estimated_tokens <= tier_limit:
                self.logger.info(
                    f"Access granted for tier '{user_tier}': "
                    f"{estimated_tokens} tokens <= {tier_limit} limit"
                )
                return True, f"Access granted. Repository size: {estimated_tokens} tokens"
            else:
                self.logger.warning(
                    f"Access denied for tier '{user_tier}': "
                    f"{estimated_tokens} tokens > {tier_limit} limit"
                )
                return False, (
                    f"Repository size ({estimated_tokens} tokens) exceeds your "
                    f"{user_tier} tier limit ({tier_limit} tokens). "
                    f"Please upgrade your plan to access larger repositories."
                )
                
        except TierValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during access validation: {e}")
            raise TierValidationError(f"Access validation failed: {e}")
    
    def get_tier_limit(self, tier: str) -> int:
        """
        Get the token limit for a specific tier.
        
        Args:
            tier: Tier name
            
        Returns:
            Token limit for the tier
            
        Raises:
            TierValidationError: If tier is invalid
        """
        tier = tier.lower().strip()
        if tier not in self._tier_configs:
            available_tiers = list(self._tier_configs.keys())
            raise TierValidationError(f"Invalid tier '{tier}'. Available tiers: {available_tiers}")
        
        return self._tier_configs[tier].max_tokens
    
    def get_all_tier_limits(self) -> Dict[str, int]:
        """
        Get all tier limits.
        
        Returns:
            Dictionary mapping tier names to token limits
        """
        return {tier: config.max_tokens for tier, config in self._tier_configs.items()}
    
    def check_tier_access_with_details(self, user_tier: str, estimated_tokens: int) -> Dict[str, any]:
        """
        Check tier access and return detailed information.
        
        Args:
            user_tier: User's tier level
            estimated_tokens: Estimated token count
            
        Returns:
            Dictionary with access details
        """
        is_allowed, message = self.validate_access(user_tier, estimated_tokens)
        
        tier_limit = self.get_tier_limit(user_tier) if user_tier in self._tier_configs else 0
        
        return {
            "allowed": is_allowed,
            "message": message,
            "user_tier": user_tier,
            "estimated_tokens": estimated_tokens,
            "tier_limit": tier_limit,
            "usage_percentage": (estimated_tokens / tier_limit * 100) if tier_limit > 0 else 0,
            "available_tiers": list(self._tier_configs.keys())
        }
    
    def load_tier_config(self) -> Dict[str, int]:
        """
        Load and return current tier configuration.
        
        Returns:
            Dictionary of tier limits
        """
        return self.get_all_tier_limits()


# Global instance for easy access
_tier_manager_instance: Optional[TierManager] = None


def get_tier_manager(custom_limits: Optional[Dict[str, int]] = None) -> TierManager:
    """
    Get or create global TierManager instance.
    
    Args:
        custom_limits: Optional custom tier limits
        
    Returns:
        TierManager instance
    """
    global _tier_manager_instance
    
    if _tier_manager_instance is None:
        _tier_manager_instance = TierManager(custom_limits)
    
    return _tier_manager_instance


def reset_tier_manager() -> None:
    """Reset global TierManager instance (mainly for testing)."""
    global _tier_manager_instance
    _tier_manager_instance = None