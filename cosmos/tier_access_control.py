"""
Tier access control integration for repository operations.

This module integrates tier checking with repository fetching and caching operations,
providing clear error messages and logging for access attempts and violations.
"""

import os
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from .tier_manager import TierManager, TierValidationError, get_tier_manager


@dataclass
class AccessAttempt:
    """Record of a repository access attempt."""
    timestamp: datetime
    user_tier: str
    repository_url: str
    estimated_tokens: int
    allowed: bool
    message: str
    user_id: Optional[str] = None


class TierAccessController:
    """
    Controls repository access based on user tier limits.
    
    Integrates tier validation with repository operations and provides
    comprehensive logging and error handling.
    """
    
    def __init__(self, tier_manager: Optional[TierManager] = None):
        """
        Initialize TierAccessController.
        
        Args:
            tier_manager: Optional TierManager instance. If None, uses global instance.
        """
        self.logger = logging.getLogger(__name__)
        self.tier_manager = tier_manager or get_tier_manager()
        self._access_log = []  # In-memory log for monitoring
    
    def validate_repository_access(
        self, 
        repository_url: str, 
        estimated_tokens: int, 
        user_tier: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate repository access based on user tier and repository size.
        
        Args:
            repository_url: URL of the repository to access
            estimated_tokens: Estimated token count for the repository
            user_tier: User's tier level. If None, reads from environment
            user_id: Optional user identifier for logging
            
        Returns:
            Tuple of (is_allowed, message, access_details)
            
        Raises:
            TierValidationError: If validation fails due to configuration issues
        """
        try:
            # Get user tier from environment if not provided
            if user_tier is None:
                user_tier = self._get_user_tier_from_env()
            
            # Validate repository access
            is_allowed, message = self.tier_manager.validate_access(user_tier, estimated_tokens)
            
            # Get detailed access information
            access_details = self.tier_manager.check_tier_access_with_details(
                user_tier, estimated_tokens
            )
            
            # Create access attempt record
            access_attempt = AccessAttempt(
                timestamp=datetime.now(),
                user_tier=user_tier,
                repository_url=repository_url,
                estimated_tokens=estimated_tokens,
                allowed=is_allowed,
                message=message,
                user_id=user_id
            )
            
            # Log the access attempt
            self._log_access_attempt(access_attempt)
            
            # Add access attempt to in-memory log
            self._access_log.append(access_attempt)
            
            return is_allowed, message, access_details
            
        except TierValidationError:
            raise
        except Exception as e:
            error_msg = f"Repository access validation failed: {e}"
            self.logger.error(error_msg)
            raise TierValidationError(error_msg)
    
    def _get_user_tier_from_env(self) -> str:
        """
        Get user tier from environment variables.
        
        Returns:
            User tier string
            
        Raises:
            TierValidationError: If tier is not configured
        """
        user_tier = os.getenv("TIER_PLAN", "").strip().lower()
        
        if not user_tier:
            # Try alternative environment variable names
            user_tier = os.getenv("USER_TIER", "").strip().lower()
        
        if not user_tier:
            user_tier = os.getenv("SUBSCRIPTION_TIER", "").strip().lower()
        
        if not user_tier:
            raise TierValidationError(
                "User tier not configured. Please set TIER_PLAN, USER_TIER, or "
                "SUBSCRIPTION_TIER environment variable to 'free', 'pro', or 'enterprise'"
            )
        
        return user_tier
    
    def _log_access_attempt(self, access_attempt: AccessAttempt) -> None:
        """
        Log repository access attempt with appropriate level.
        
        Args:
            access_attempt: Access attempt record to log
        """
        log_data = {
            "timestamp": access_attempt.timestamp.isoformat(),
            "user_tier": access_attempt.user_tier,
            "repository_url": access_attempt.repository_url,
            "estimated_tokens": access_attempt.estimated_tokens,
            "allowed": access_attempt.allowed,
            "user_id": access_attempt.user_id
        }
        
        if access_attempt.allowed:
            self.logger.info(
                f"Repository access GRANTED - "
                f"User: {access_attempt.user_tier} tier, "
                f"Repository: {access_attempt.repository_url}, "
                f"Tokens: {access_attempt.estimated_tokens}, "
                f"User ID: {access_attempt.user_id or 'N/A'}"
            )
        else:
            self.logger.warning(
                f"Repository access DENIED - "
                f"User: {access_attempt.user_tier} tier, "
                f"Repository: {access_attempt.repository_url}, "
                f"Tokens: {access_attempt.estimated_tokens}, "
                f"Reason: {access_attempt.message}, "
                f"User ID: {access_attempt.user_id or 'N/A'}"
            )
    
    def check_repository_access_before_fetch(
        self, 
        repository_url: str, 
        estimated_tokens: int,
        user_tier: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Check repository access before fetching and raise exception if denied.
        
        Args:
            repository_url: URL of the repository to access
            estimated_tokens: Estimated token count for the repository
            user_tier: User's tier level. If None, reads from environment
            user_id: Optional user identifier for logging
            
        Raises:
            TierAccessDeniedError: If access is denied
            TierValidationError: If validation fails
        """
        is_allowed, message, _ = self.validate_repository_access(
            repository_url, estimated_tokens, user_tier, user_id
        )
        
        if not is_allowed:
            raise TierAccessDeniedError(message)
    
    def get_access_summary(self, user_tier: Optional[str] = None) -> Dict[str, Any]:
        """
        Get access summary for a user tier.
        
        Args:
            user_tier: User's tier level. If None, reads from environment
            
        Returns:
            Dictionary with access summary information
        """
        if user_tier is None:
            user_tier = self._get_user_tier_from_env()
        
        tier_limit = self.tier_manager.get_tier_limit(user_tier)
        all_limits = self.tier_manager.get_all_tier_limits()
        
        return {
            "current_tier": user_tier,
            "current_limit": tier_limit,
            "all_tier_limits": all_limits,
            "recent_access_attempts": len([
                attempt for attempt in self._access_log[-10:]  # Last 10 attempts
                if attempt.user_tier == user_tier
            ])
        }
    
    def get_recent_access_attempts(self, limit: int = 50) -> list[AccessAttempt]:
        """
        Get recent access attempts for monitoring.
        
        Args:
            limit: Maximum number of attempts to return
            
        Returns:
            List of recent access attempts
        """
        return self._access_log[-limit:] if self._access_log else []
    
    def clear_access_log(self) -> None:
        """Clear the in-memory access log."""
        self._access_log.clear()
        self.logger.info("Access log cleared")


class TierAccessDeniedError(Exception):
    """Raised when repository access is denied due to tier limits."""
    pass


# Global instance for easy access
_tier_access_controller_instance: Optional[TierAccessController] = None


def get_tier_access_controller(tier_manager: Optional[TierManager] = None) -> TierAccessController:
    """
    Get or create global TierAccessController instance.
    
    Args:
        tier_manager: Optional TierManager instance
        
    Returns:
        TierAccessController instance
    """
    global _tier_access_controller_instance
    
    if _tier_access_controller_instance is None:
        _tier_access_controller_instance = TierAccessController(tier_manager)
    
    return _tier_access_controller_instance


def reset_tier_access_controller() -> None:
    """Reset global TierAccessController instance (mainly for testing)."""
    global _tier_access_controller_instance
    _tier_access_controller_instance = None


def validate_repository_access_simple(
    repository_url: str, 
    estimated_tokens: int,
    user_tier: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Simple function to validate repository access.
    
    Args:
        repository_url: URL of the repository to access
        estimated_tokens: Estimated token count for the repository
        user_tier: User's tier level. If None, reads from environment
        
    Returns:
        Tuple of (is_allowed, message)
    """
    controller = get_tier_access_controller()
    is_allowed, message, _ = controller.validate_repository_access(
        repository_url, estimated_tokens, user_tier
    )
    return is_allowed, message


def check_repository_access_or_raise(
    repository_url: str, 
    estimated_tokens: int,
    user_tier: Optional[str] = None
) -> None:
    """
    Check repository access and raise exception if denied.
    
    Args:
        repository_url: URL of the repository to access
        estimated_tokens: Estimated token count for the repository
        user_tier: User's tier level. If None, reads from environment
        
    Raises:
        TierAccessDeniedError: If access is denied
        TierValidationError: If validation fails
    """
    controller = get_tier_access_controller()
    controller.check_repository_access_before_fetch(
        repository_url, estimated_tokens, user_tier
    )