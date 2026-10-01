"""Check COCO map registration and robot scene compatibility."""

import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from runtime_control import RuntimeControl, RuntimeScene
from runtime_control.panel import RuntimeControlPanel

import play


class PlayMapTests(unittest.TestCase):
    def test_yaw_shortcuts_and_quaternion_heading(self):
        config = self.make_config("coco_scene")
        for key in range(1, 10):
            self.assertEqual(
                config["runtime_actions"][f"turn_{key * 10}"]["shortcut"], str(key)
            )
        self.assertAlmostEqual(play.heading_deg([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]), 90)

    def test_browser_command_order_and_latest_override(self):
        # Use the real POST validation without starting a server or browser.
        panel = RuntimeControlPanel.__new__(RuntimeControlPanel)
        panel.lock = RLock()
        panel.actions, panel.pressed_keys = set(), set()
        panel.allowed_actions = {"turn_10", "turn_90", "stop"}
        panel.state = {"linear_x": 1, "linear_y": 1, "yaw": 1}
        events = play.install_browser_motion_queue(panel)
        panel._handle_post("/api/action", {"action": "turn_10"})
        panel._handle_post("/api/key", {"key": "w", "pressed": True})
        self.assertEqual(list(events), [("turn", 10), ("velocity", (1, 0, 0))])
        target, command = play.apply_motion_events(events, None, np.zeros(3), 60)
        self.assertIsNone(target)
        np.testing.assert_array_equal(command, [1, 0, 0])
        panel._handle_post("/api/action", {"action": "turn_90"})
        target, command = play.apply_motion_events(events, None, np.zeros(3), 60)
        self.assertEqual(target, 90)
        self.assertGreater(command[2], 0)
        panel._handle_post("/api/action", {"action": "stop"})
        target, command = play.apply_motion_events(events, target, command, 60)
        self.assertIsNone(target)
        np.testing.assert_array_equal(command, [0, 0, 0])

    def test_keyboard_directions(self):
        self.assertEqual(play.keyboard_velocity({"w", "a", "q"}), (1, 1, 1))
        self.assertEqual(play.keyboard_velocity({"s", "d", "e"}), (-1, -1, -1))
        self.assertEqual(play.keyboard_velocity(set()), (0, 0, 0))

    def test_held_keyboard_submits_only_one_timed_command(self):
        with (
            patch.object(play, "_pressed_keys", {"w"}),
            patch.object(play, "move", wraps=play.move) as command,
            patch.object(
                play.sys,
                "argv",
                [
                    "play.py",
                    "--headless",
                    "--no-policy",
                    "--map",
                    "coco_scene",
                    "--duration",
                    "0.02",
                ],
            ),
            redirect_stdout(StringIO()),
        ):
            play.main()
        movements = [call for call in command.call_args_list if call.args == (1, 0, 0)]
        self.assertGreater(len(movements), 1)
        self.assertEqual(sum(call.kwargs["new_command"] for call in movements), 1)

    def test_invalid_low_rates_are_rejected_before_scene_creation(self):
        for rate in ("0", "-1", "nan", "inf"):
            with (
                self.subTest(rate=rate),
                patch.object(play.sys, "argv", ["play.py", "--low_hz", rate]),
                patch.object(play, "RuntimeScene") as scene,
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                play.main()
            self.assertEqual(error.exception.code, 2)
            scene.assert_not_called()

    def make_config(self, map_name):
        args = SimpleNamespace(map=map_name, gui=False, gui_port=8765)
        return play.build_runtime_config(args, [40.0], [1.0])

    def test_coco_startup_config(self):
        config = self.make_config("coco_scene")
        self.assertEqual(config["runtime_ui"]["default_map"], "coco_scene")
        self.assertEqual(config["runtime_ui"]["maps"]["coco_scene"], "COCO Objects")
        self.assertEqual(config["simulation"]["initial_position"], [0.0, 0.0, 0.42])

    def test_original_map_spawn(self):
        config = self.make_config("rc26_track")
        self.assertEqual(config["runtime_ui"]["default_map"], "rc26_track")
        self.assertEqual(config["simulation"]["initial_position"], [3.7, -9.0, 0.45])

    def test_composed_robot_map(self):
        with RuntimeScene(
            robot_xml=play.DEFAULT_ROBOT_XML,
            map_specs=play.MAP_SPECS,
            runtime_config=self.make_config("coco_scene"),
            robot_body_name="trunk",
            robot_cameras=play.ROBOT_CAMERAS,
            expected_dimensions=(19, 18, 12),
        ) as scene:
            spawn = scene.runtime.maps.activate(scene.model, scene.data, "coco_scene")
            self.assertEqual(spawn["position"], [0.0, 0.0, 0.42])
            geom = scene.model.geom("map_coco_scene_0")
            self.assertGreater(geom.contype[0], 0)
            self.assertTrue(np.isfinite(scene.data.qpos).all())

    def test_gui_initializes_fpv_before_runtime_updates(self):
        make_config = play.make_runtime_config
        update_runtime = RuntimeControl.runtime_control

        def config_without_browser(**kwargs):
            kwargs["gui"] = False
            return make_config(**kwargs)

        with (
            patch.object(play, "VisionModule") as constructor,
            patch.object(
                play, "make_runtime_config", side_effect=config_without_browser
            ),
            patch.object(
                play.sys,
                "argv",
                [
                    "play.py",
                    "--gui",
                    "--map",
                    "coco_scene",
                    "--no-policy",
                    "--duration",
                    "0.02",
                    "--low_hz",
                    "10",
                ],
            ),
            redirect_stdout(StringIO()),
        ):
            vision = constructor.return_value
            vision.stream_fpv.return_value = True

            def update_after_fpv(runtime, model, data):
                self.assertGreater(vision.stream_fpv.call_count, 0)
                return update_runtime(runtime, model, data)

            with patch.object(
                RuntimeControl,
                "runtime_control",
                autospec=True,
                side_effect=update_after_fpv,
            ) as update:
                play.main()
                self.assertGreater(update.call_count, 0)


if __name__ == "__main__":
    unittest.main()
