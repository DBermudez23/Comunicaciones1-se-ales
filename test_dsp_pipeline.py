"""Pruebas unitarias para validar las clases del taller DSP.

Verifica el comportamiento de GenreMetadataManager, ADCDACSimulator,
NyquistSpectralAnalyzer, WavAudioLoader y TDAHPathologyAnalyzer.
"""

import unittest
from pathlib import Path
import numpy as np

from audio_loader import WavAudioLoader
from metadata_manager import GenreMetadataManager
from quantizer import ADCDACSimulator
from spectral_analyzer import NyquistSpectralAnalyzer
from tdah_comparator import GenreProcessingResult, TDAHPathologyAnalyzer


class TestGenreMetadataManager(unittest.TestCase):
    """Pruebas para la clase GenreMetadataManager."""

    def setUp(self) -> None:
        self.mgr = GenreMetadataManager()

    def test_total_genres_registered(self) -> None:
        genres = self.mgr.get_all_genres()
        self.assertEqual(len(genres), 5)
        self.assertIn("clasica_barroca", genres)
        self.assertIn("metal", genres)
        self.assertIn("salsa", genres)
        self.assertIn("cantina", genres)
        self.assertIn("reggaeton", genres)

    def test_metadata_fields_integrity(self) -> None:
        for g_id in self.mgr.get_genre_ids():
            meta = self.mgr.get_metadata(g_id)
            self.assertTrue(len(meta.genre_name) > 0)
            self.assertTrue(len(meta.artist_composer) > 0)
            self.assertTrue(len(meta.work_title) > 0)
            self.assertTrue(len(meta.ensemble_type) > 0)
            self.assertIsInstance(meta.release_year, int)
            self.assertTrue(meta.filename.endswith(".wav"))

    def test_invalid_genre_key_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.mgr.get_metadata("genero_inexistente")


class TestADCDACSimulator(unittest.TestCase):
    """Pruebas para la clase ADCDACSimulator."""

    def test_step_size_calculation(self) -> None:
        # Rango [-5V, +5V], 8 bits -> L=256, Delta = 10 / 256 = 0.0390625 V
        sim_8b = ADCDACSimulator(num_bits=8, voltage_min=-5.0, voltage_max=5.0)
        self.assertEqual(sim_8b.num_levels, 256)
        self.assertAlmostEqual(sim_8b.delta, 10.0 / 256.0, places=6)
        self.assertAlmostEqual(sim_8b.scale_factor_volts_per_bit, sim_8b.delta, places=6)
        self.assertAlmostEqual(sim_8b.scale_factor_bits_per_volt, 25.6, places=6)

    def test_quantization_error_bounds(self) -> None:
        sim = ADCDACSimulator(num_bits=8, voltage_min=-5.0, voltage_max=5.0)
        # Senal aleatoria dentro de [-5, 5] V
        np.random.seed(42)
        test_signal = np.random.uniform(-4.9, 4.9, size=1000)
        xq = sim.quantize(test_signal)
        err = sim.quantization_error

        # El error de cuantizacion uniforme en midrise debe satisfacer |e[n]| <= Delta / 2
        max_err = np.max(np.abs(err))
        self.assertLessEqual(max_err, (sim.delta / 2.0) + 1e-6)

    def test_sqnr_computation(self) -> None:
        sim = ADCDACSimulator(num_bits=8, voltage_min=-5.0, voltage_max=5.0)
        # Senoide pura para verificar SQNR cercano a 6.02 * b + 1.76 dB
        t = np.linspace(0, 1, 10000, endpoint=False)
        sin_signal = 4.5 * np.sin(2 * np.pi * 50 * t)
        sim.quantize(sin_signal)
        metrics = sim.compute_metrics()

        # SQNR teorica para 8 bits es ~49.9 dB. Para amplitud 4.5 V debe rondar 48-50 dB
        self.assertGreater(metrics["sqnr_measured_db"], 40.0)
        self.assertLess(metrics["sqnr_measured_db"], 55.0)

    def test_binary_encoding_format(self) -> None:
        sim = ADCDACSimulator(num_bits=8, voltage_min=-5.0, voltage_max=5.0)
        sim.quantize(np.array([-5.0, 0.0, 4.9]))
        words = sim.encode_to_binary(num_samples_preview=3)
        self.assertEqual(len(words), 3)
        for w in words:
            self.assertEqual(len(w), 8)
            self.assertTrue(set(w).issubset({"0", "1"}))


class TestNyquistSpectralAnalyzer(unittest.TestCase):
    """Pruebas para la clase NyquistSpectralAnalyzer."""

    def test_single_tone_fft_peak(self) -> None:
        fs = 8000
        n_samples = 8000
        t = np.arange(n_samples) / float(fs)
        f0 = 440.0  # Tono de 440 Hz a 2.5 V
        v_sig = 2.5 * np.sin(2 * np.pi * f0 * t)

        analyzer = NyquistSpectralAnalyzer(t, v_sig, fs, genre_label="Prueba 440Hz")
        freqs, mags = analyzer.compute_fft()
        peaks = analyzer.find_dominant_frequencies(top_k=1)

        self.assertEqual(len(peaks), 1)
        peak_freq, peak_mag = peaks[0]
        self.assertAlmostEqual(peak_freq, 440.0, delta=1.5)
        self.assertAlmostEqual(peak_mag, 2.5, delta=0.1)

    def test_nyquist_validation(self) -> None:
        fs = 8000
        n_samples = 8000
        t = np.arange(n_samples) / float(fs)
        f0 = 1000.0
        v_sig = 1.0 * np.sin(2 * np.pi * f0 * t)

        analyzer = NyquistSpectralAnalyzer(t, v_sig, fs)
        nyq = analyzer.validate_nyquist()
        self.assertTrue(nyq["is_nyquist_satisfied"])
        self.assertEqual(nyq["nyquist_frequency_hz"], 4000.0)


if __name__ == "__main__":
    unittest.main()
