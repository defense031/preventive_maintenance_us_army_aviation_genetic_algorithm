"""
Policy Factory for Dynamic Policy Loading

Enables plug-in architecture where policies can be swapped via configuration.
Supports loading policies by name and passing configuration parameters.
"""

from typing import Dict, Any, Type, Optional
from policy.base_policy import BasePolicy
from policy.baseline_policy import BaselinePolicy
from policy.decision_tree_policy import DecisionTreePolicy


# Registry of available policies
POLICY_REGISTRY: Dict[str, Type[BasePolicy]] = {
    'baseline': BaselinePolicy,
    'baseline_policy': BaselinePolicy,
    'fly_freshest': BaselinePolicy,
    'decision_tree': DecisionTreePolicy,
    'dt': DecisionTreePolicy,
    'tree': DecisionTreePolicy,
    # Additional policies will be registered here as they're implemented:
    # 'random': RandomPolicy,
    # 'greedy': GreedyPolicy,
    # etc.
}


def register_policy(name: str, policy_class: Type[BasePolicy]) -> None:
    """
    Register a new policy class in the factory.

    This allows custom policies to be added dynamically at runtime.

    Args:
        name: String identifier for the policy
        policy_class: Policy class (must inherit from BasePolicy)

    Raises:
        TypeError: If policy_class does not inherit from BasePolicy
        ValueError: If name already registered

    Example:
        >>> from policy.policy_factory import register_policy
        >>> register_policy('my_policy', MyCustomPolicy)
    """
    if not issubclass(policy_class, BasePolicy):
        raise TypeError(
            f"Policy class must inherit from BasePolicy. "
            f"Got: {policy_class.__name__}"
        )

    if name in POLICY_REGISTRY:
        raise ValueError(
            f"Policy '{name}' already registered. "
            f"Use a different name or unregister first."
        )

    POLICY_REGISTRY[name] = policy_class


def unregister_policy(name: str) -> None:
    """
    Remove a policy from the registry.

    Args:
        name: String identifier of policy to remove

    Raises:
        KeyError: If policy not found in registry
    """
    if name not in POLICY_REGISTRY:
        raise KeyError(f"Policy '{name}' not found in registry")

    del POLICY_REGISTRY[name]


def list_policies() -> list[str]:
    """
    Get list of all registered policy names.

    Returns:
        List of policy name strings
    """
    return sorted(POLICY_REGISTRY.keys())


def create_policy(
    policy_name: str,
    policy_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> BasePolicy:
    """
    Factory function to create policy instances by name.

    This is the main entry point for loading policies. It looks up the
    policy class in the registry and instantiates it with the provided
    configuration.

    Args:
        policy_name: Name of policy to create (case-insensitive)
        policy_config: Dictionary of policy-specific config parameters
        verbose: Enable verbose logging for policy

    Returns:
        Instantiated policy object

    Raises:
        ValueError: If policy_name not found in registry

    Examples:
        >>> # Create baseline policy
        >>> policy = create_policy('baseline', verbose=True)

        >>> # Create decision tree policy with paths
        >>> config = {
        ...     'chromosome_path': 'chromosomes/baseline_dt.json',
        ...     'feature_config_path': 'config/features/simple_dt.yaml',
        ...     'rul_threshold': 100
        ... }
        >>> policy = create_policy('decision_tree', policy_config=config)

        >>> # Load policy from string (useful for CLI args)
        >>> policy_name = 'baseline'
        >>> policy = create_policy(policy_name)
    """
    if policy_config is None:
        policy_config = {}

    # Normalize policy name (case-insensitive lookup)
    policy_name_lower = policy_name.lower()

    if policy_name_lower not in POLICY_REGISTRY:
        available = ', '.join(list_policies())
        raise ValueError(
            f"Unknown policy: '{policy_name}'. "
            f"Available policies: {available}"
        )

    # Get policy class from registry
    policy_class = POLICY_REGISTRY[policy_name_lower]

    # Instantiate with config
    try:
        policy = policy_class(verbose=verbose, **policy_config)
    except TypeError as e:
        raise TypeError(
            f"Failed to instantiate {policy_class.__name__}: {e}. "
            f"Check that policy_config parameters are valid."
        )

    return policy


def create_policy_from_config(config: Dict[str, Any]) -> BasePolicy:
    """
    Create policy from a configuration dictionary.

    This is a convenience wrapper for create_policy() that extracts
    the policy name and config from a single dictionary. Useful for
    loading policies from YAML/JSON config files.

    Args:
        config: Configuration dictionary with keys:
            - 'name' or 'policy_name': Policy identifier
            - 'verbose': (optional) Enable verbose logging
            - 'config' or 'policy_config': (optional) Policy-specific params
            - Any other top-level keys are passed as policy config

    Returns:
        Instantiated policy object

    Raises:
        ValueError: If config missing required fields or invalid policy name

    Examples:
        >>> # From YAML config
        >>> config = {
        ...     'name': 'baseline',
        ...     'verbose': True
        ... }
        >>> policy = create_policy_from_config(config)

        >>> # With nested config
        >>> config = {
        ...     'name': 'decision_tree',
        ...     'verbose': False,
        ...     'config': {
        ...         'tree_depth': 5,
        ...         'chromosome_path': 'results/best.json'
        ...     }
        ... }
        >>> policy = create_policy_from_config(config)

        >>> # With flat config
        >>> config = {
        ...     'name': 'decision_tree',
        ...     'tree_depth': 5,
        ...     'chromosome_path': 'results/best.json'
        ... }
        >>> policy = create_policy_from_config(config)
    """
    # Extract policy name
    if 'name' in config:
        policy_name = config['name']
    elif 'policy_name' in config:
        policy_name = config['policy_name']
    else:
        raise ValueError(
            "Config must contain 'name' or 'policy_name' field. "
            f"Got keys: {list(config.keys())}"
        )

    # Extract verbose flag
    verbose = config.get('verbose', False)

    # Extract policy-specific config
    if 'config' in config:
        # Nested config structure
        policy_config = config['config']
    elif 'policy_config' in config:
        # Alternative nested structure
        policy_config = config['policy_config']
    else:
        # Flat config structure (use all keys except reserved ones)
        reserved_keys = {'name', 'policy_name', 'verbose'}
        policy_config = {
            k: v for k, v in config.items()
            if k not in reserved_keys
        }

    return create_policy(policy_name, policy_config, verbose)


def get_policy_info(policy_name: str) -> Dict[str, Any]:
    """
    Get metadata about a registered policy.

    Args:
        policy_name: Name of policy to query

    Returns:
        Dictionary with policy metadata:
            - 'name': Policy name
            - 'class_name': Python class name
            - 'module': Module path
            - 'docstring': Class docstring

    Raises:
        ValueError: If policy not found

    Example:
        >>> info = get_policy_info('baseline')
        >>> print(info['docstring'])
    """
    policy_name_lower = policy_name.lower()

    if policy_name_lower not in POLICY_REGISTRY:
        available = ', '.join(list_policies())
        raise ValueError(
            f"Unknown policy: '{policy_name}'. "
            f"Available policies: {available}"
        )

    policy_class = POLICY_REGISTRY[policy_name_lower]

    return {
        'name': policy_name_lower,
        'class_name': policy_class.__name__,
        'module': policy_class.__module__,
        'docstring': policy_class.__doc__ or "No documentation available"
    }
