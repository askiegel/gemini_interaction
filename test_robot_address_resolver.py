#!/usr/bin/env python3

import socket
import unittest

from network.robot_address_resolver import (
    RobotAddressResolutionError,
    is_expected_robot_bridge,
    resolve_robot_address,
)


def address_record(address):
    return (
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        (address, 0),
    )


class RobotAddressResolverTests(unittest.TestCase):
    def test_accepts_configured_ipv4(self):
        result = resolve_robot_address("192.0.2.25")

        self.assertEqual(result.address, "192.0.2.25")
        self.assertEqual(result.source, "configured_ipv4")

    def test_resolves_hostname_to_ipv4(self):
        def resolver(*_args, **_kwargs):
            return [address_record("192.0.2.50")]

        result = resolve_robot_address(
            "minipupperv2.local",
            "192.0.2.99",
            getaddrinfo=resolver,
        )

        self.assertEqual(
            result.configured_host,
            "minipupperv2.local",
        )
        self.assertEqual(result.address, "192.0.2.50")
        self.assertEqual(result.source, "hostname_ipv4")

    def test_hostname_wins_over_stale_fallback(self):
        def resolver(*_args, **_kwargs):
            return [address_record("172.20.10.4")]

        result = resolve_robot_address(
            "minipupperv2.local",
            "192.168.68.124",
            getaddrinfo=resolver,
        )

        self.assertEqual(result.address, "172.20.10.4")
        self.assertEqual(result.source, "hostname_ipv4")

    def test_uses_fallback_when_resolution_fails(self):
        def resolver(*_args, **_kwargs):
            raise socket.gaierror("not found")

        result = resolve_robot_address(
            "minipupperv2.local",
            "192.168.68.124",
            getaddrinfo=resolver,
        )

        self.assertEqual(result.address, "192.168.68.124")
        self.assertEqual(result.source, "configured_fallback")

    def test_fails_without_resolution_or_fallback(self):
        def resolver(*_args, **_kwargs):
            return []

        with self.assertRaises(RobotAddressResolutionError):
            resolve_robot_address(
                "minipupperv2.local",
                None,
                getaddrinfo=resolver,
            )

    def test_accepts_expected_robot_identity(self):
        self.assertTrue(
            is_expected_robot_bridge(
                {
                    "ok": True,
                    "service": "mini_pupper_robot_bridge",
                    "robot": "mini_pupper_2",
                }
            )
        )

    def test_rejects_wrong_service(self):
        self.assertFalse(
            is_expected_robot_bridge(
                {
                    "ok": True,
                    "service": "some_other_service",
                    "robot": "mini_pupper_2",
                }
            )
        )

    def test_rejects_wrong_robot(self):
        self.assertFalse(
            is_expected_robot_bridge(
                {
                    "ok": True,
                    "service": "mini_pupper_robot_bridge",
                    "robot": "unknown_robot",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
