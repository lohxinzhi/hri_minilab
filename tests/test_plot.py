"""Check saved absolute-heading plots and normalization."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import motion_api
import plot


class HeadingPlotTests(unittest.TestCase):
    def test_default_filename_has_timestamp_and_does_not_overwrite(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(plot, "DEFAULT_PLOT_DIR", Path(folder)),
        ):
            first = plot.plot_heading([0, 1], [0, 10])
            second = plot.plot_heading([0, 1], [0, 20])
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())
            self.assertRegex(first.name, r"^heading_plot_\d{8}_\d{6}_\d{6}\.png$")

    def test_absolute_heading_and_relative_turn_target(self):
        figures = []
        subplots = plot.plt.subplots

        def capture_figure(**kwargs):
            fig, ax = subplots(**kwargs)
            figures.append(fig)
            return fig, ax

        motion_api.turn(30, 60, new_command=True)
        target = motion_api.get_turn_target_heading()
        self.assertEqual(target, 90)
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(plot.plt, "subplots", side_effect=capture_figure),
        ):
            path = plot.plot_heading(
                [0, 1, 2],
                [60, 75, 90],
                [None, target, target],
                Path(folder) / "heading.png",
            )
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        ax = figures[0].axes[0]
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [60, 75, 90])
        np.testing.assert_allclose(
            ax.lines[1].get_ydata(), [np.nan, 90, 90], equal_nan=True
        )
        self.assertEqual(ax.get_ylim(), (-180, 180))
        self.assertFalse(plot.plt.fignum_exists(figures[0].number))
        motion_api.move(0, 0, 0, duration=0, new_command=True)

    def test_wrap_boundary_breaks_line_and_normalizes_angles(self):
        fig, ax = plot.plt.subplots()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(plot.plt, "subplots", return_value=(fig, ax)),
        ):
            plot.plot_heading(
                [0, 1, 2], [179, 181, 540], output_path=Path(folder) / "heading.png"
            )
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), [179, np.nan, -179, -180], equal_nan=True
        )

    def test_invalid_lengths(self):
        with self.assertRaises(ValueError):
            plot.plot_heading([0, 1], [60])
        with self.assertRaises(ValueError):
            plot.plot_heading([0, 1], [60, 90], [90])


if __name__ == "__main__":
    unittest.main()
