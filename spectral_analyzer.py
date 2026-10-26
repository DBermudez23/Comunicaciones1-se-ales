"""Modulo de analisis espectral y validacion del Teorema de Nyquist-Shannon.

Implementa la clase NyquistSpectralAnalyzer que aplica la Transformada Rapida
de Fourier (FFT), evalua el ancho de banda efectivo, identifica frecuencias dominantes
y valida matematicamente el criterio de muestreo de Nyquist-Shannon.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq


class NyquistSpectralAnalyzer:
    """Realiza analisis frecuencial y validacion de Nyquist sobre senales muestreadas.

    Calcula la FFT unilateral, extrae el espectro de magnitud en unidades de tension (V),
    localiza las componentes armonicas principales y comprueba la ausencia de aliasing.
    """

    def __init__(
        self,
        time_vector: np.ndarray,
        voltage_signal: np.ndarray,
        sampling_rate: int,
        genre_label: str = "Audio",
    ) -> None:
        """Inicializa el analizador espectral.

        Args:
            time_vector: Vector de tiempo discreto t en segundos.
            voltage_signal: Vector de amplitud en Voltios x(t).
            sampling_rate: Frecuencia de muestreo fs en Hz.
            genre_label: Etiqueta descriptiva del genero o pista musical.
        """
        if len(time_vector) != len(voltage_signal):
            raise ValueError("El vector de tiempo y la senal deben tener la misma longitud.")
        if sampling_rate <= 0:
            raise ValueError("La frecuencia de muestreo debe ser estrictamente positiva.")

        self.time_vector: np.ndarray = np.asarray(time_vector, dtype=np.float64)
        self.voltage_signal: np.ndarray = np.asarray(voltage_signal, dtype=np.float64)
        self.sampling_rate: int = int(sampling_rate)
        self.genre_label: str = genre_label

        self.num_samples: int = len(self.voltage_signal)
        self.nyquist_rate: float = self.sampling_rate / 2.0

        # Arreglos de resultados espectrales
        self.frequencies: np.ndarray = np.empty(0, dtype=np.float64)
        self.magnitude_spectrum: np.ndarray = np.empty(0, dtype=np.float64)
        self.magnitude_dbv: np.ndarray = np.empty(0, dtype=np.float64)
        self.is_analyzed: bool = False

    def compute_fft(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula el espectro de magnitud unilateral mediante la FFT de Scipy.

        Aplica rfft normalizada para senales reales, preservando la escala fisica
        de tension (Voltios pico).

        Returns:
            Tupla (frequencies, magnitude_spectrum):
                - frequencies: Vector de frecuencias positivas [0, fs/2] en Hz.
                - magnitude_spectrum: Magnitud |X(f)| en Voltios.
        """
        n = self.num_samples
        # FFT para senales reales (unilateral)
        fft_complex = rfft(self.voltage_signal)
        self.frequencies = rfftfreq(n, d=1.0 / float(self.sampling_rate))

        # Normalizacion de amplitud unilateral:
        # A(0) = |X[0]| / N  (Componente continua DC)
        # A(k) = 2 * |X[k]| / N para k > 0
        raw_magnitude = np.abs(fft_complex) / float(n)
        magnitude = raw_magnitude.copy()
        if len(magnitude) > 1:
            magnitude[1:] *= 2.0

        self.magnitude_spectrum = magnitude

        # Espectro logaritmico en dBV (referencia 1 Voltio)
        eps = 1e-12
        self.magnitude_dbv = 20.0 * np.log10(np.maximum(magnitude, eps))
        self.is_analyzed = True

        return self.frequencies, self.magnitude_spectrum

    def find_dominant_frequencies(
        self, top_k: int = 5, min_freq_hz: float = 20.0
    ) -> List[Tuple[float, float]]:
        """Identifica los picos espectrales dominantes en la senal.

        Filtra componentes subsónicas menores a min_freq_hz y busca maximos locales.

        Args:
            top_k: Numero de picos espectrales a retornar.
            min_freq_hz: Frecuencia minima a considerar (evita sesgo DC).

        Returns:
            Lista de tuplas (frecuencia_hz, magnitud_volts) ordenadas de mayor a menor amplitud.
        """
        if not self.is_analyzed:
            self.compute_fft()

        mask = self.frequencies >= min_freq_hz
        valid_freqs = self.frequencies[mask]
        valid_mags = self.magnitude_spectrum[mask]

        if len(valid_mags) < 3:
            return []

        # Deteccion de maximos locales (picos)
        is_peak = (valid_mags[1:-1] > valid_mags[:-2]) & (valid_mags[1:-1] > valid_mags[2:])
        peak_indices = np.where(is_peak)[0] + 1

        if len(peak_indices) == 0:
            # Si no hay maximos locales evidentes, ordenar directamente por magnitud
            sorted_idx = np.argsort(valid_mags)[::-1][:top_k]
            return [(float(valid_freqs[i]), float(valid_mags[i])) for i in sorted_idx]

        peak_mags = valid_mags[peak_indices]
        sorted_peak_order = np.argsort(peak_mags)[::-1][:top_k]
        top_indices = peak_indices[sorted_peak_order]

        dominant_peaks = [
            (float(valid_freqs[idx]), float(valid_mags[idx])) for idx in top_indices
        ]
        return dominant_peaks

    def validate_nyquist(self, energy_threshold_ratio: float = 0.99) -> Dict[str, Any]:
        """Valida formalmente el cumplimiento del Teorema de Nyquist-Shannon.

        Calcula el ancho de banda que contiene el porcentaje especificado de la energia
        espectral total y verifica si fs >= 2 * f_max_efectiva.

        Args:
            energy_threshold_ratio: Fraccion de energia espectral para delimitar f_max (ej. 0.99 para 99%).

        Returns:
            Diccionario con diagnostico de Nyquist, frecuencia maxima efectiva y margen.
        """
        if not self.is_analyzed:
            self.compute_fft()

        # Calculo de energia espectral acumulada
        energy = self.magnitude_spectrum**2
        total_energy = float(np.sum(energy))

        if total_energy <= 1e-12:
            f_max_effective = 0.0
        else:
            cumulative_energy = np.cumsum(energy) / total_energy
            idx_threshold = int(np.searchsorted(cumulative_energy, energy_threshold_ratio))
            idx_threshold = min(idx_threshold, len(self.frequencies) - 1)
            f_max_effective = float(self.frequencies[idx_threshold])

        # Frecuencia del pico global (descartando DC < 20 Hz)
        dominant_list = self.find_dominant_frequencies(top_k=1)
        f_dominant = dominant_list[0][0] if dominant_list else 0.0

        nyquist_limit = self.nyquist_rate
        min_required_fs = 2.0 * f_max_effective
        is_nyquist_satisfied = bool(self.sampling_rate >= min_required_fs)
        safety_margin_hz = float(nyquist_limit - f_max_effective)

        return {
            "sampling_rate_hz": self.sampling_rate,
            "nyquist_frequency_hz": nyquist_limit,
            "dominant_peak_hz": round(f_dominant, 2),
            "effective_bandwidth_99_hz": round(f_max_effective, 2),
            "min_required_sampling_rate_hz": round(min_required_fs, 2),
            "is_nyquist_satisfied": is_nyquist_satisfied,
            "safety_margin_hz": round(safety_margin_hz, 2),
            "nyquist_statement": (
                f"fs ({self.sampling_rate} Hz) >= 2 * f_max ({round(min_required_fs, 2)} Hz) "
                f"-> {'CUMPLE NYQUIST' if is_nyquist_satisfied else 'ALERTA DE ALIASING'}"
            ),
        }

    def plot_time_domain(
        self,
        ax: Optional[plt.Axes] = None,
        zoom_window_ms: Optional[float] = 50.0,
        save_path: Optional[str] = None,
    ) -> plt.Axes:
        """Grafica la senal en el dominio del tiempo simulando comportamiento analogico.

        Args:
            ax: Eje de Matplotlib opcional. Si es None, crea una nueva figura.
            zoom_window_ms: Duracion en milisegundos de la ventana de visualizacion ampliada.
            save_path: Ruta de archivo opcional para guardar la figura generada.

        Returns:
            El eje de Matplotlib utilizado.
        """
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            created_fig = True

        if zoom_window_ms is not None and zoom_window_ms > 0:
            max_t = zoom_window_ms / 1000.0
            mask = self.time_vector <= max_t
            t_plot = self.time_vector[mask] * 1000.0  # en ms
            v_plot = self.voltage_signal[mask]
            x_label = "Tiempo [ms]"
            title = f"{self.genre_label} - Senal en el Tiempo x(t) [Ventana {zoom_window_ms} ms]"
        else:
            t_plot = self.time_vector
            v_plot = self.voltage_signal
            x_label = "Tiempo [s]"
            title = f"{self.genre_label} - Senal Completa en el Tiempo x(t)"

        ax.plot(t_plot, v_plot, color="#1f77b4", linewidth=1.0, label="x(t) [V]")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel("Tension [V]", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

        if save_path and created_fig:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()

        return ax

    def plot_frequency_domain(
        self,
        ax: Optional[plt.Axes] = None,
        max_freq_display_hz: float = 12000.0,
        save_path: Optional[str] = None,
    ) -> plt.Axes:
        """Grafica el espectro de magnitud unilateral |X(f)|.

        Args:
            ax: Eje de Matplotlib opcional.
            max_freq_display_hz: Frecuencia maxima para el eje horizontal en Hz.
            save_path: Ruta de guardado opcional.

        Returns:
            El eje de Matplotlib utilizado.
        """
        if not self.is_analyzed:
            self.compute_fft()

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            created_fig = True

        mask = self.frequencies <= max_freq_display_hz
        f_plot = self.frequencies[mask]
        mag_plot = self.magnitude_spectrum[mask]

        ax.plot(f_plot, mag_plot, color="#d62728", linewidth=1.0, label="|X(f)| [V]")

        # Resaltar los 3 picos principales
        peaks = self.find_dominant_frequencies(top_k=3)
        for rank, (fp, mp) in enumerate(peaks, 1):
            if fp <= max_freq_display_hz:
                ax.plot(fp, mp, "ko", markersize=4)
                ax.annotate(
                    f"P{rank}: {fp:.1f} Hz\n({mp:.3f} V)",
                    xy=(fp, mp),
                    xytext=(fp + 150, mp * 1.05),
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
                )

        ax.set_title(f"{self.genre_label} - Espectro de Magnitud FFT |X(f)|", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frecuencia [Hz]", fontsize=10)
        ax.set_ylabel("Magnitud [V]", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

        if save_path and created_fig:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()

        return ax
