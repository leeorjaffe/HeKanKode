import unittest
import numpy as np
from waveform_analyzer import analyze_waveform, find_pa_mean

class TestWaveformAnalyzer(unittest.TestCase):

    def test_empty_waveform(self):
        """Tests the function with an empty waveform."""
        waveform = []
        result = analyze_waveform(waveform)
        self.assertEqual(result.size, 0)

    def test_simple_waveform_round_default(self):
        """Tests a simple waveform with default rounding."""
        waveform = [[10.2, 0.0], [20.5, 0.2], [10.8, 0.4]]
        result = analyze_waveform(waveform)
        expected = np.zeros(21, dtype=int)
        expected[10] = 1
        expected[11] = 1
        expected[20] = 1
        np.testing.assert_array_equal(result, expected)

    def test_waveform_with_blanking_round_default(self):
        """Tests the blanking functionality with default rounding."""
        waveform = [[10.2, 0.0], [15.5, 0.05], [20.5, 0.1], [25.1, 0.15], [30.9, 0.2]]
        result = analyze_waveform(waveform)
        expected = np.zeros(32, dtype=int)
        expected[10] = 1
        expected[20] = 1
        expected[31] = 1
        np.testing.assert_array_equal(result, expected)

    def test_waveform_with_custom_blanking_time_round_default(self):
        """Tests the blanking functionality with a custom blanking time and default rounding."""
        waveform = [[10.2, 0.0], [15.5, 0.1], [20.5, 0.25], [25.1, 0.3], [30.9, 0.5]]
        result = analyze_waveform(waveform, blanking_time=0.2)
        expected = np.zeros(32, dtype=int)
        expected[10] = 1
        expected[20] = 1
        expected[31] = 1
        np.testing.assert_array_equal(result, expected)

    def test_simple_waveform_floor(self):
        """Tests a simple waveform with floor quantization."""
        waveform = [[10.2, 0.0], [20.5, 0.2], [10.8, 0.4]]
        result = analyze_waveform(waveform, quantize_mode="floor")
        expected = np.zeros(21, dtype=int)
        expected[10] = 2
        expected[20] = 1
        np.testing.assert_array_equal(result, expected)

    def test_invalid_quantize_mode(self):
        """Tests that an invalid quantize mode raises a ValueError."""
        waveform = [[10.2, 0.0]]
        with self.assertRaises(ValueError):
            analyze_waveform(waveform, quantize_mode="invalid")


class TestFindPaMean(unittest.TestCase):

    def test_standard_case(self):
        """Tests a standard case for find_pa_mean."""
        # non-zero counts: [10, 2, 8, 5, 8, 1, 12]. Mode is 8.
        # Bins with count 8 are at indices 3, 6. Median is 4.5.
        bins = np.array([0, 10, 2, 8, 5, 0, 8, 1, 12])
        self.assertAlmostEqual(find_pa_mean(bins), 4.5)

    def test_mode_tie_breaking_highest_wins(self):
        """Tests the mode tie-breaking logic (highest count wins)."""
        # non-zero counts: [5, 2, 5, 2, 8, 8]. Modes are 2, 5, 8. Highest is 8.
        # Bins with count 8 are at indices 5, 6. Median is 5.5.
        bins = np.array([0, 5, 2, 5, 2, 8, 8])
        self.assertAlmostEqual(find_pa_mean(bins), 5.5)

    def test_single_non_zero_bin(self):
        """Tests case with a single non-zero bin."""
        # non-zero count is [5]. Mode is 5.
        # Bin with count 5 is at index 2. Median is 2.0.
        bins = np.array([0, 0, 5, 0, 0])
        self.assertAlmostEqual(find_pa_mean(bins), 2.0)

    def test_no_non_zero_bins(self):
        """Tests case where there are no non-zero bins."""
        bins = np.array([0, 0, 0, 0])
        self.assertIsNone(find_pa_mean(bins))

    def test_example_with_highest_wins_tiebreak(self):
        """Tests an example with the 'highest wins' tie-breaking logic."""
        # non-zero: [5, 2, 8, 5, 8, 1]. Modes are 5, 8. Highest is 8.
        # Bins with count 8 are at indices 3, 6. Median is 4.5.
        bins = np.array([0, 5, 2, 8, 5, 0, 8, 1])
        self.assertAlmostEqual(find_pa_mean(bins), 4.5)


if __name__ == '__main__':
    unittest.main()
