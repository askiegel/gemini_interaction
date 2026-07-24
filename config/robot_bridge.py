"""Compatibility helpers for Robot Bridge configuration."""

from config.config_manager import ConfigurationManager


def get_robot_bridge_url():
    return ConfigurationManager().robot_bridge_url


ROBOT_BRIDGE_URL = get_robot_bridge_url()
