"""Exercise object missions with simulated vision and heading updates."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np

import motion_api
from browser_gui import BrowserState


class GotoObjectTests(unittest.TestCase):
    def setUp(self):
        timeout = patch.object(motion_api, "GOTO_TIMEOUT_SECONDS", 60.0)
        timeout.start()
        self.addCleanup(timeout.stop)
        clock = patch.object(motion_api.time, "monotonic", return_value=10.0)
        self.now = clock.start()
        self.addCleanup(clock.stop)
        mission = patch.object(motion_api, "_object_mission", None)
        mission.start()
        self.addCleanup(mission.stop)
        motion_api.move(0, 0, 0, duration=0, new_command=True)
        self.addCleanup(motion_api.move, 0, 0, 0, duration=0, new_command=True)
        self.output = StringIO()
        capture = redirect_stdout(self.output)
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def goto(self, bbox=None, heading=0, **kwargs):
        kwargs.setdefault("robot_position", (3.0, 0.0))
        kwargs.setdefault("object_position", (0.0, 0.0))
        return motion_api.goto_object(
            "chair",
            "red",
            bbox,
            current_yaw_deg=heading,
            frame_size=(100, 100),
            **kwargs,
        )

    def test_rotation_sign_and_proportional_normalized_error(self):
        for bbox, expected in (
            ([80, 10, 100, 30], [0, 0, -0.8]),
            ([60, 10, 80, 30], [0.5, 0, -0.4]),
            ([0, 10, 20, 30], [0, 0, 0.8]),
            ([20, 10, 40, 30], [0.5, 0, 0.4]),
            ([40, 10, 60, 30], [0.5, 0, 0]),
        ):
            with self.subTest(bbox=bbox):
                np.testing.assert_allclose(self.goto(bbox), expected, atol=1e-6)
        self.assertEqual(self.output.getvalue().count("[APPROACH]"), 1)
        self.assertIn("[APPROACH] object=chair color=red", self.output.getvalue())

    def test_alignment_boundary_and_resolution_independence(self):
        np.testing.assert_allclose(self.goto([65, 10, 85, 30]), [0, 0, -0.5])
        np.testing.assert_allclose(self.goto([64, 10, 84, 30]), [0.5, 0, -0.48])
        command = motion_api.goto_object(
            "chair",
            "red",
            [640, 100, 840, 300],
            current_yaw_deg=0,
            robot_position=(3.0, 0.0),
            object_position=(0.0, 0.0),
            frame_size=(1000, 1000),
        )
        np.testing.assert_allclose(command, [0.5, 0, -0.48])

    def test_distance_threshold_completes_and_stops_once(self):
        self.goto([40, 0, 60, 99])
        self.assertEqual(motion_api.get_goto_status(), "RUNNING")
        np.testing.assert_array_equal(
            self.goto([40, 10, 60, 30], robot_position=(1.0, 0)), [0.5, 0, 0]
        )
        np.testing.assert_array_equal(
            self.goto([40, 10, 60, 30], robot_position=(0.9, 0)), [0, 0, 0]
        )
        self.assertEqual(motion_api.get_goto_status(), "SUCCESS")
        np.testing.assert_array_equal(motion_api.move(0, 0, 0), [0, 0, 0])
        np.testing.assert_array_equal(self.goto(), [0, 0, 0])
        self.assertEqual(self.output.getvalue().count("[MISSION] status=SUCCESS"), 1)
        self.assertEqual(self.goto().dtype, np.float32)

    def test_found_log_includes_total_action_time_and_final_distance_once(self):
        self.goto()
        self.now.return_value = 13
        self.goto([40, 10, 60, 30])
        self.now.return_value = 17.5
        self.goto([40, 10, 60, 30], robot_position=(0.3, 0.4))
        self.goto()
        self.assertEqual(
            [
                line
                for line in self.output.getvalue().splitlines()
                if line.startswith("[FOUND]")
            ],
            ["[FOUND] class=chair color=red t=7.50s d=0.500m"],
        )

    def test_failed_mission_does_not_print_found(self):
        self.goto()
        self.now.return_value = 70
        self.goto()
        self.assertNotIn("[FOUND]", self.output.getvalue())

    def test_search_one_full_revolution_across_wrapped_headings(self):
        for heading in (179, -91, -1, 89):
            self.assertGreater(self.goto(heading=heading)[2], 0)
        np.testing.assert_array_equal(self.goto(heading=179), [0, 0, 0])
        self.assertEqual(motion_api.get_goto_status(), "FAIL")
        self.assertIn(
            "reason=object not found after one revolution", self.output.getvalue()
        )
        self.assertEqual(self.output.getvalue().count("[SEARCH]"), 1)
        self.goto(heading=179)
        self.assertEqual(self.output.getvalue().count("[MISSION]"), 1)
        self.assertIsNone(motion_api.get_turn_target_heading())

    def test_pending_vlm_keeps_one_continuous_search_and_waits_for_final_result(self):
        for heading in (179, -91, -1, 89):
            self.assertGreater(self.goto(heading=heading, vision_pending=True)[2], 0)
        np.testing.assert_array_equal(
            self.goto(heading=179, vision_pending=True), [0, 0, 0]
        )
        self.assertEqual(motion_api.get_goto_status(), "RUNNING")
        self.assertEqual(self.output.getvalue().count("[SEARCH]"), 1)
        self.goto(heading=179, vision_pending=False)
        self.assertEqual(motion_api.get_goto_status(), "FAIL")
        self.assertEqual(self.output.getvalue().count("[SEARCH]"), 1)

    def test_final_vlm_result_can_find_target_after_search_turn_completes(self):
        for heading in (0, 90, 180, -90, 0):
            self.goto(heading=heading, vision_pending=True)
        self.assertEqual(motion_api.get_goto_status(), "RUNNING")
        np.testing.assert_array_equal(
            self.goto([40, 10, 60, 30], heading=0), [0.5, 0, 0]
        )
        self.assertEqual(motion_api.get_goto_status(), "RUNNING")
        self.assertEqual(self.output.getvalue().count("[APPROACH]"), 1)

    def test_pending_vlm_holds_approach_without_restarting_search(self):
        self.goto([40, 10, 60, 30])
        for heading in (10, 20, 30):
            np.testing.assert_array_equal(
                self.goto(heading=heading, vision_pending=True), [0, 0, 0]
            )
            self.assertEqual(motion_api._object_mission.phase, "approach")
        self.assertNotIn("[SEARCH]", self.output.getvalue())
        self.assertEqual(motion_api.get_goto_status(), "RUNNING")
        self.goto([40, 10, 60, 30], heading=30)
        self.assertEqual(self.output.getvalue().count("[APPROACH]"), 1)
        self.assertGreater(self.goto(heading=45)[2], 0)
        self.assertEqual(self.output.getvalue().count("[SEARCH]"), 1)
        self.assertEqual(motion_api.get_turn_target_heading(), 405)

    def test_vlm_waits_for_first_result_then_starts_approach(self):
        for timestamp in (10, 11, 12):
            self.now.return_value = timestamp
            np.testing.assert_array_equal(
                self.goto(use_vlm=True, vision_pending=True, vision_received=False),
                [0, 0, 0],
            )
            self.assertIsNone(motion_api._object_mission.phase)
            self.assertIsNone(motion_api.get_turn_target_heading())
        self.assertNotIn("[SEARCH]", self.output.getvalue())
        self.assertNotIn("[APPROACH]", self.output.getvalue())
        np.testing.assert_allclose(
            self.goto([40, 10, 60, 30], use_vlm=True, vision_received=True),
            [0.2, 0, 0],
        )
        self.assertEqual(motion_api._object_mission.phase, "approach")

    def test_vlm_first_not_found_starts_search_and_refresh_keeps_it_running(self):
        self.goto(use_vlm=True, vision_pending=True, vision_received=False)
        self.assertGreater(self.goto(use_vlm=True, vision_received=True)[2], 0)
        self.assertEqual(motion_api._object_mission.phase, "search")
        self.assertGreater(
            self.goto(
                heading=30, use_vlm=True, vision_pending=True, vision_received=False
            )[2],
            0,
        )
        self.assertEqual(self.output.getvalue().count("[SEARCH]"), 1)

    def test_yolo_starts_search_without_vlm_result(self):
        self.assertGreater(self.goto(vision_received=False)[2], 0)
        self.assertEqual(motion_api._object_mission.phase, "search")

    def test_vlm_initial_wait_retains_timeout_and_restarts_for_new_mission(self):
        self.goto(use_vlm=True, vision_received=False, vision_pending=True)
        self.now.return_value = 70
        np.testing.assert_array_equal(
            self.goto(use_vlm=True, vision_received=False, vision_pending=True),
            [0, 0, 0],
        )
        self.assertEqual(motion_api.get_goto_status(), "FAIL")
        self.now.return_value = 80
        self.goto([40, 10, 60, 30], use_vlm=True, new_command=True)
        np.testing.assert_array_equal(
            self.goto(
                use_vlm=True,
                new_command=True,
                vision_received=False,
                vision_pending=True,
            ),
            [0, 0, 0],
        )
        self.assertIsNone(motion_api._object_mission.phase)

    def test_vlm_approach_is_slow_and_continuous_during_refresh(self):
        bbox = [60, 10, 80, 30]
        expected = [0.2, 0, -0.16]
        np.testing.assert_allclose(self.goto(bbox, use_vlm=True), expected)
        for timestamp in (10.3, 11, 12):
            self.now.return_value = timestamp
            np.testing.assert_allclose(
                self.goto(bbox, use_vlm=True, vision_pending=True), expected
            )
            self.assertEqual(motion_api._object_mission.phase, "approach")
        self.assertEqual(self.output.getvalue().count("[APPROACH]"), 1)
        self.assertNotIn("[SEARCH]", self.output.getvalue())
        np.testing.assert_allclose(motion_api.move(0, 0, 0), expected)

    def test_vlm_stops_when_box_unusable_and_recovers_without_phase_reset(self):
        bbox = [40, 10, 60, 30]
        self.goto(bbox, use_vlm=True)
        np.testing.assert_array_equal(
            self.goto(use_vlm=True, vision_pending=True), [0, 0, 0]
        )
        self.assertEqual(motion_api._object_mission.phase, "approach")
        np.testing.assert_allclose(self.goto(bbox, use_vlm=True), [0.2, 0, 0])
        self.assertEqual(self.output.getvalue().count("[APPROACH]"), 1)
        self.assertGreater(self.goto(use_vlm=True)[2], 0)
        self.assertEqual(self.output.getvalue().count("[SEARCH]"), 1)

    def test_vlm_slow_speed_keeps_alignment_and_live_distance_stop(self):
        np.testing.assert_allclose(
            self.goto([80, 10, 100, 30], use_vlm=True), [0, 0, -0.32]
        )
        np.testing.assert_allclose(
            self.goto([40, 10, 60, 30], use_vlm=True), [0.2, 0, 0]
        )
        np.testing.assert_array_equal(
            self.goto(
                [40, 10, 60, 30],
                use_vlm=True,
                vision_pending=True,
                robot_position=(0.9, 0),
            ),
            [0, 0, 0],
        )
        self.assertEqual(motion_api.get_goto_status(), "SUCCESS")

    def test_pending_vlm_preserves_timeout_and_distance_completion(self):
        self.goto(vision_pending=True)
        self.now.return_value = 70
        np.testing.assert_array_equal(self.goto(vision_pending=True), [0, 0, 0])
        self.assertEqual(motion_api.get_goto_status(), "FAIL")
        self.now.return_value = 80
        self.goto(new_command=True, vision_pending=True)
        np.testing.assert_array_equal(
            self.goto(robot_position=(0.5, 0), vision_pending=True), [0, 0, 0]
        )
        self.assertEqual(motion_api.get_goto_status(), "SUCCESS")

    def test_search_stops_immediately_when_target_is_seen(self):
        self.assertGreater(self.goto(heading=30)[2], 0)
        np.testing.assert_array_equal(
            self.goto([40, 10, 60, 30], heading=45), [0.5, 0, 0]
        )
        self.assertIsNone(motion_api.get_turn_target_heading())
        self.assertEqual(motion_api.get_goto_status(), "RUNNING")
        self.assertEqual(self.output.getvalue().count("[SEARCH]"), 1)
        self.assertEqual(self.output.getvalue().count("[APPROACH]"), 1)

    def test_target_loss_restarts_search_from_current_heading(self):
        self.goto([40, 10, 60, 30])
        self.assertGreater(self.goto(heading=45)[2], 0)
        self.assertEqual(motion_api.get_turn_target_heading(), 405)
        self.goto([40, 10, 60, 30], heading=90)
        self.assertGreater(self.goto(heading=100)[2], 0)
        self.assertEqual(motion_api.get_turn_target_heading(), 460)
        self.assertEqual(self.output.getvalue().count("[SEARCH]"), 2)
        self.assertEqual(self.output.getvalue().count("[APPROACH]"), 2)

    def test_configured_timeout_is_captured_at_mission_start(self):
        with patch.object(motion_api, "GOTO_TIMEOUT_SECONDS", 5.0):
            self.goto()
        self.now.return_value = 14.9
        self.assertGreater(self.goto()[2], 0)
        self.now.return_value = 15
        np.testing.assert_array_equal(self.goto(), [0, 0, 0])
        self.assertEqual(motion_api.get_goto_status(), "FAIL")
        self.assertIn("reason=5-second timeout", self.output.getvalue())

    def test_invalid_timeout_configuration_is_rejected(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with (
                self.subTest(timeout=timeout),
                patch.object(motion_api, "GOTO_TIMEOUT_SECONDS", timeout),
            ):
                with self.assertRaises(ValueError):
                    self.goto()
                self.assertIsNone(motion_api.get_goto_status())

    def test_timeout_is_shared_across_phases_and_stops_once(self):
        self.goto()
        self.now.return_value = 40
        self.goto([40, 10, 60, 30], heading=90)
        self.now.return_value = 65
        self.goto(heading=100)
        self.now.return_value = 70
        np.testing.assert_array_equal(self.goto([40, 0, 60, 70]), [0, 0, 0])
        self.assertEqual(motion_api.get_goto_status(), "FAIL")
        self.assertIn(
            "[MISSION] status=FAIL reason=60-second timeout", self.output.getvalue()
        )
        self.assertFalse(motion_api.is_move_active())
        self.goto()
        self.assertEqual(self.output.getvalue().count("[MISSION]"), 1)

    def test_explicit_restart_and_target_change_start_new_missions(self):
        self.assertIsNone(motion_api.get_goto_status())
        self.goto([40, 10, 60, 30], robot_position=(0.5, 0))
        self.now.return_value = 100
        self.assertGreater(self.goto(new_command=True)[2], 0)
        self.assertEqual(motion_api.get_goto_status(), "RUNNING")
        self.now.return_value = 120
        command = motion_api.goto_object(
            "car",
            "blue",
            [40, 10, 60, 30],
            current_yaw_deg=0,
            robot_position=(3.0, 0.0),
            object_position=(0.0, 0.0),
            frame_size=(100, 100),
        )
        np.testing.assert_array_equal(command, [0.5, 0, 0])
        self.assertIn("reason=replaced by a new mission", self.output.getvalue())
        self.now.return_value = 179
        self.assertGreater(
            motion_api.goto_object(
                "car",
                "blue",
                current_yaw_deg=0,
                robot_position=(3.0, 0.0),
                object_position=(0.0, 0.0),
            )[2],
            0,
        )

    def test_invalid_inputs_preserve_current_mission_and_command(self):
        self.goto([40, 10, 60, 30])
        for kwargs in (
            {"bbox": [0, 0, float("nan"), 10]},
            {"bbox": [1, 2, 3]},
            {"bbox": [10, 10, 0, 20]},
            {"bbox": [101, 0, 110, 20]},
            {"frame_size": (0, 100)},
            {"frame_size": (100, float("inf"))},
            {"current_yaw_deg": float("nan")},
            {"yaw_rate_deg_s": float("inf")},
            {"dt": 0},
            {"object_type": ""},
            {"object_color": ""},
            {"robot_position": (float("nan"), 0)},
            {"object_position": (0, 1, 2)},
        ):
            with self.subTest(kwargs=kwargs):
                inputs = {
                    "object_type": "chair",
                    "object_color": "red",
                    "bbox": [40, 10, 60, 30],
                    "current_yaw_deg": 0,
                    "frame_size": (100, 100),
                    "robot_position": (3.0, 0.0),
                    "object_position": (0.0, 0.0),
                }
                inputs.update(kwargs)
                with self.assertRaises(ValueError):
                    motion_api.goto_object(**inputs)
                self.assertEqual(motion_api.get_goto_status(), "RUNNING")
                np.testing.assert_array_equal(motion_api.move(0, 0, 0), [0.5, 0, 0])

    def test_standalone_approach_normalizes_and_stops_on_lost_box(self):
        with patch.object(motion_api, "move", wraps=motion_api.move) as move:
            motion_api.approach(
                "chair",
                "red",
                [400, 100, 600, 300],
                robot_position=(3.0, 0.0),
                object_position=(0.0, 0.0),
                frame_size=(1000, 1000),
                new_command=True,
            )
            move.assert_called_with(0.5, 0, -0.0, new_command=True)
            np.testing.assert_array_equal(
                motion_api.approach(
                    "chair", "red", None, robot_position=(3, 0), object_position=(0, 0)
                ),
                [0, 0, 0],
            )

    def test_mission_logs_reach_gui_console(self):
        state = BrowserState()
        with state.capture_logs():
            self.goto()
            self.goto([40, 10, 60, 30], robot_position=(0.5, 0))
        logs = "".join(entry["text"] for entry in state.logs)
        self.assertIn("[SEARCH]", logs)
        self.assertIn("[APPROACH] object=chair color=red", logs)
        self.assertIn("[MISSION] status=SUCCESS", logs)
        self.assertIn("[FOUND] class=chair color=red t=0.00s d=0.500m", logs)

    def test_distance_uses_both_axes_and_can_finish_after_visual_loss(self):
        self.goto([40, 10, 60, 30], robot_position=(0.6, 0.6))
        self.assertEqual(motion_api.get_goto_status(), "RUNNING")
        np.testing.assert_array_equal(
            self.goto(None, robot_position=(0.4, 0.6)), [0, 0, 0]
        )
        self.assertEqual(motion_api.get_goto_status(), "SUCCESS")

    def test_steering_is_independent_of_object_world_direction(self):
        first = self.goto([80, 10, 100, 90], object_position=(50, 20))
        second = self.goto([80, 10, 100, 90], object_position=(-50, -20))
        np.testing.assert_array_equal(first, second)
        self.assertLess(first[2], 0)


if __name__ == "__main__":
    unittest.main()
