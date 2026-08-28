#!/usr/bin/env python3

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent

SERVER = (
    ROOT / "voice_relay" / "server.py"
).read_text(encoding="utf-8")

HTML = (
    ROOT / "voice_relay" / "index.html"
).read_text(encoding="utf-8")

JS = (
    ROOT / "voice_relay" / "operator_console.js"
).read_text(encoding="utf-8")


class Tony2MappingDashboardTests(unittest.TestCase):

    def test_server_defines_tony2_mapping_owner(self):
        for marker in (
            "def get_tony2_mapping_runtime():",
            "def tony2_mapping_control_status(self):",
            "def tony2_mapping_control_action(self, action):",
            "def tony2_live_mapping_map_status(self):",
        ):
            self.assertIn(marker, SERVER)

    def test_mapping_get_routes_use_tony2(self):
        self.assertIn(
            "self.tony2_mapping_control_status()",
            SERVER,
        )

        self.assertIn(
            "self.tony2_live_mapping_map_status()",
            SERVER,
        )

    def test_start_stop_reset_are_fixed_routes(self):
        for route in (
            "/dashboard/mapping-start",
            "/dashboard/mapping-stop",
            "/dashboard/mapping-reset",
        ):
            self.assertIn(
                f'"{route}"',
                SERVER,
            )

        self.assertIn(
            "self.tony2_mapping_control_action(",
            SERVER,
        )

    def test_reset_button_exists_once(self):
        self.assertEqual(
            HTML.count(
                'id="resetLiveMappingButton"'
            ),
            1,
        )

        self.assertIn(
            "Reset Live Map",
            HTML,
        )

    def test_reset_endpoint_is_wired_in_browser(self):
        self.assertIn(
            'const RESET_ENDPOINT =',
            JS,
        )

        self.assertIn(
            '"/dashboard/mapping-reset"',
            JS,
        )

        self.assertIn(
            "function resetMapping()",
            JS,
        )

        self.assertIn(
            'reset.addEventListener("click", resetMapping)',
            JS,
        )

    def test_reset_discards_only_unsaved_live_map(self):
        self.assertIn(
            "Reset the current unsaved live map?",
            JS,
        )

        self.assertIn(
            "Saved maps will not be deleted.",
            JS,
        )

    def test_reset_is_blocked_during_navigation(self):
        self.assertIn(
            "latestNavigationActive",
            JS,
        )

        self.assertIn(
            "mapping.navigation_active === true",
            JS,
        )

        self.assertIn(
            "mapping.navigation_status_available !== true",
            JS,
        )

        self.assertIn(
            "Stop guarded navigation before ",
            JS,
        )

        self.assertIn(
            "resetting the live map.",
            JS,
        )

    def test_candidate_save_is_disabled_during_migration(self):
        self.assertIn(
            "latestMapSaveEnabled",
            JS,
        )

        self.assertIn(
            "mapping.map_save_enabled === true",
            JS,
        )

        self.assertIn(
            "Candidate saving remains disabled while ",
            JS,
        )

        self.assertIn(
            "Tony2 owns live mapping.",
            JS,
        )

    def test_mapping_action_requires_mayday_stationary_first(self):
        start = SERVER.index(
            "    def tony2_mapping_control_action("
        )

        end = SERVER.index(
            "    def tony2_live_mapping_map_status(",
            start,
        )

        action = SERVER[start:end]

        stationary = action.index(
            "self.ensure_mayday_stationary()"
        )

        runtime = action.index(
            "get_tony2_mapping_runtime()"
        )

        self.assertLess(
            stationary,
            runtime,
        )

    def test_navigation_status_checked_before_mapping_action(self):
        start = SERVER.index(
            "    def tony2_mapping_control_action("
        )

        end = SERVER.index(
            "    def tony2_live_mapping_map_status(",
            start,
        )

        action = SERVER[start:end]

        navigation = action.index(
            "self.mayday_mapping_navigation_active()"
        )

        stationary = action.index(
            "self.ensure_mayday_stationary()"
        )

        self.assertLess(
            navigation,
            stationary,
        )


if __name__ == "__main__":
    unittest.main()
