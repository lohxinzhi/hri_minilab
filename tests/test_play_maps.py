"""Check COCO map registration and robot scene compatibility."""

import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from runtime_control import RuntimeControl, RuntimeScene

import play


class PlayMapTests(unittest.TestCase):
    def setUp(self):
        self.dialogue_patch = patch.object(play, "DialogueManager")
        self.dialogue = self.dialogue_patch.start().return_value
        self.dialogue.poll_actions.return_value = None
        self.dialogue.poll_error.return_value = None
        self.addCleanup(self.dialogue_patch.stop)
        self.addCleanup(play.move, 0, 0, 0, duration=0, new_command=True)

    def test_turn_completes_within_configurable_tolerance(self):
        for tolerance, start, current in [(3.0, 0, 87.1), (5.0, 179, -95.9)]:
            with (
                self.subTest(tolerance=tolerance),
                patch.object(play, "TURN_TOLERANCE_DEG", tolerance),
                redirect_stdout(StringIO()),
            ):
                angle, command = play.update_turn_command(90, start, new_command=True)
                self.assertEqual(angle, 90)
                self.assertGreater(command[2], 0)
                angle, command = play.update_turn_command(angle, current)
                self.assertIsNone(angle)
                np.testing.assert_array_equal(command, [0, 0, 0])
                self.assertIsNone(play.get_turn_target_heading())
                # Completion cancels the controller rather than resuming on drift.
                np.testing.assert_array_equal(play.move(0, 0, 0), [0, 0, 0])

    def test_turn_continues_outside_tolerance(self):
        with redirect_stdout(StringIO()):
            angle, _ = play.update_turn_command(90, 0, new_command=True)
            angle, command = play.update_turn_command(angle, 86.9)
        self.assertEqual(angle, 90)
        self.assertGreater(command[2], 0)
        play.move(0, 0, 0, duration=0, new_command=True)

    def test_heading_plot_saved_on_keyboard_interrupt(self):
        with (
            patch.object(play.sys, "argv", ["play.py", "--headless", "--no-policy"]),
            patch.object(
                RuntimeControl, "runtime_control", side_effect=KeyboardInterrupt
            ),
            patch.object(play, "plot_heading") as save_plot,
            redirect_stdout(StringIO()),
            self.assertRaises(KeyboardInterrupt),
        ):
            play.main()
        save_plot.assert_called_once()
        times, headings, commands = save_plot.call_args.args
        self.assertEqual(len(times), len(headings))
        self.assertEqual(len(times), len(commands))
        self.assertEqual(
            len(times), len(save_plot.call_args.kwargs["angular_velocities"])
        )

    def test_no_keyboard_shortcuts_and_quaternion_heading(self):
        config = self.make_config("coco_scene")
        self.assertFalse(config["runtime_actions"])
        self.assertAlmostEqual(play.heading_deg([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]), 90)

    def test_legacy_browser_panel_is_disabled(self):
        args = SimpleNamespace(map="coco_scene", gui=True, gui_port=8765)
        config = play.build_runtime_config(args, [40.0], [1.0])
        self.assertFalse(config["_runtime_gui"])

    def test_main_polls_dialogue_and_executes_actions_on_main_thread(self):
        with (
            patch.object(play, "DialogueManager") as constructor,
            patch.object(play, "move", wraps=play.move) as move,
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
            dialogue = constructor.return_value
            dialogue.poll_actions.side_effect = lambda: None
            plan = [
                {
                    "action": "move",
                    "velocity": {"vx": 0.4, "vy": 0, "wz": 0},
                    "duration": 2.5,
                }
            ]
            ready = [plan]
            dialogue.poll_actions.side_effect = lambda: ready.pop() if ready else None
            dialogue.poll_error.return_value = None
            play.main()
        constructor.return_value.start.assert_called_once()
        constructor.return_value.close.assert_called_once()
        self.assertTrue(
            any(
                call.args == (0.4, 0, 0) and call.kwargs.get("duration") == 2.5
                for call in move.call_args_list
            )
        )

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

    def test_invalid_vision_confidence_is_rejected_before_scene_creation(self):
        for threshold in ("-0.1", "1.1", "nan", "inf"):
            with (
                self.subTest(threshold=threshold),
                patch.object(
                    play.sys, "argv", ["play.py", "--vision-confidence", threshold]
                ),
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

    def test_gui_uses_two_browser_feeds_without_native_or_cv2_windows(self):
        viewer = RuntimeScene.viewer
        with (
            patch.object(play, "VisionModule") as constructor,
            patch.object(play, "BrowserGUI") as gui_constructor,
            patch.object(
                RuntimeScene, "viewer", autospec=True, side_effect=viewer
            ) as make_viewer,
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
            vision.render_frame.return_value = np.zeros((48, 64, 3), dtype=np.uint8)
            vision.render_fpv.return_value = np.zeros((48, 64, 3), dtype=np.uint8)
            play.main()
            self.assertGreater(vision.render_fpv.call_count, 0)
            self.assertGreater(vision.render_frame.call_count, 0)
            vision.stream_fpv.assert_not_called()
            self.assertTrue(make_viewer.call_args.args[1])
            gui_constructor.return_value.start.assert_called_once()
            gui_constructor.return_value.close.assert_called_once()
            state = gui_constructor.call_args.args[0]
            self.assertEqual(set(state.frames), {"follow", "fpv"})
            self.assertTrue(
                any("warm-up" in log["text"] for log in state.snapshot()["logs"])
            )


if __name__ == "__main__":
    unittest.main()
