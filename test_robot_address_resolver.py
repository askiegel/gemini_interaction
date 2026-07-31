#!/usr/bin/env python3

import json
import socket
import unittest
from types import SimpleNamespace

from network.robot_address_resolver import (
    RobotAddressResolutionError,
    discover_neighbor_ipv4_candidates,
    is_expected_robot_bridge,
    probe_expected_robot_bridge,
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


def failed_resolver(*_args, **_kwargs):
    raise socket.gaierror("not found")


class Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return json.dumps(self.payload).encode("utf-8")


class RobotAddressResolverTests(unittest.TestCase):
    def test_accepts_configured_ipv4(self):
        result = resolve_robot_address("192.0.2.25")

        self.assertEqual(result.address, "192.0.2.25")
        self.assertEqual(result.source, "configured_ipv4")

    def test_resolves_hostname_to_ipv4(self):
        result = resolve_robot_address(
            "minipupperv2.local",
            "192.0.2.99",
            getaddrinfo=lambda *_args, **_kwargs: [
                address_record("192.0.2.50")
            ],
        )

        self.assertEqual(result.address, "192.0.2.50")
        self.assertEqual(result.source, "hostname_ipv4")

    def test_hostname_wins_over_stale_fallback(self):
        result = resolve_robot_address(
            "minipupperv2.local",
            "192.168.68.124",
            getaddrinfo=lambda *_args, **_kwargs: [
                address_record("172.20.10.4")
            ],
        )

        self.assertEqual(result.address, "172.20.10.4")
        self.assertEqual(result.source, "hostname_ipv4")

    def test_verified_neighbor_wins_when_hostname_fails(self):
        probed = []

        def probe(address):
            probed.append(address)
            return address == "172.20.10.4"

        result = resolve_robot_address(
            "minipupperv2.local",
            "192.168.68.124",
            getaddrinfo=failed_resolver,
            neighbor_candidates=lambda: [
                "172.20.10.1",
                "172.20.10.4",
            ],
            bridge_probe=probe,
        )

        self.assertEqual(result.address, "172.20.10.4")
        self.assertEqual(result.source, "verified_neighbor")
        self.assertEqual(
            probed,
            ["172.20.10.1", "172.20.10.4"],
        )

    def test_fallback_follows_unverified_neighbors(self):
        result = resolve_robot_address(
            "minipupperv2.local",
            "192.168.68.124",
            getaddrinfo=failed_resolver,
            neighbor_candidates=lambda: ["192.0.2.50"],
            bridge_probe=lambda _address: False,
        )

        self.assertEqual(result.address, "192.168.68.124")
        self.assertEqual(result.source, "configured_fallback")

    def test_fails_without_any_usable_address(self):
        with self.assertRaises(RobotAddressResolutionError):
            resolve_robot_address(
                "minipupperv2.local",
                None,
                getaddrinfo=failed_resolver,
                neighbor_candidates=lambda: [],
                bridge_probe=lambda _address: False,
            )

    def test_neighbor_parser_filters_unusable_entries(self):
        result = SimpleNamespace(
            returncode=0,
            stdout=(
                "192.168.68.124 dev eth3 lladdr aa:bb REACHABLE\n"
                "192.168.68.1 dev eth3 lladdr cc:dd STALE\n"
                "192.168.68.55 dev eth3 INCOMPLETE\n"
                "169.254.73.152 dev eth3 lladdr 00:11 PERMANENT\n"
                "224.0.0.251 dev eth3 lladdr ee:ff PERMANENT\n"
            ),
        )

        candidates = discover_neighbor_ipv4_candidates(
            run_command=lambda *_args, **_kwargs: result,
        )

        self.assertEqual(
            candidates,
            ["192.168.68.124", "192.168.68.1"],
        )

    def test_probe_accepts_expected_identity(self):
        payload = {
            "ok": True,
            "service": "mini_pupper_robot_bridge",
            "robot": "mini_pupper_2",
        }

        self.assertTrue(
            probe_expected_robot_bridge(
                "192.0.2.50",
                urlopen=lambda *_args, **_kwargs: Response(payload),
            )
        )

    def test_probe_rejects_wrong_identity(self):
        payload = {
            "ok": True,
            "service": "unrelated_service",
            "robot": "mini_pupper_2",
        }

        self.assertFalse(
            probe_expected_robot_bridge(
                "192.0.2.50",
                urlopen=lambda *_args, **_kwargs: Response(payload),
            )
        )

    def test_identity_helper(self):
        self.assertTrue(
            is_expected_robot_bridge(
                {
                    "ok": True,
                    "service": "mini_pupper_robot_bridge",
                    "robot": "mini_pupper_2",
                }
            )
        )

        self.assertFalse(
            is_expected_robot_bridge(
                {
                    "ok": True,
                    "service": "wrong",
                    "robot": "mini_pupper_2",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
