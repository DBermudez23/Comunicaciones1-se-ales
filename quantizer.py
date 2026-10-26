"""Modulo de simulacion de conversion ADC-DAC y analisis de cuantizacion uniforme.

Implementa la clase ADCDACSimulator que modela el proceso de digitalizacion (ADC),
calculo del tamano de escalon (Delta), codificacion a palabras binarias, reconstruccion
DAC mediante retenedor de orden cero (ZOH), calculo del error e[n] y determinacion de la SQNR.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np


class ADCDACSimulator:
    """Simulador de convertidor Analogico-Digital (ADC) y Digital-Analogico (DAC).

    Aplica cuantizacion uniforme de b bits sobre senales de entrada en el rango
    [voltage_min, voltage_max], evalua metricas de distorsion y modela la reconstruccion ZOH.
    """

    def __init__(
        self,
        num_bits: int = 8,
        voltage_min: float = -5.0,
        voltage_max: float = 5.0,
    ) -> None:
        """Inicializa los parametros del convertidor ADC/DAC.

        Args:
            num_bits: Resolucion del convertidor en bits (b >= 1).
            voltage_min: Tension minima de entrada full-scale (V).
            voltage_max: Tension maxima de entrada full-scale (V).
        """
        if num_bits < 1:
            raise ValueError("El numero de bits debe ser un entero positivo mayor o igual a 1.")
        if voltage_min >= voltage_max:
            raise ValueError("voltage_min debe ser estrictamente menor que voltage_max.")

        self.num_bits: int = int(num_bits)
        self.voltage_min: float = float(voltage_min)
        self.voltage_max: float = float(voltage_max)

        # Calculo de parametros basicos de cuantizacion
        self.voltage_range: float = self.voltage_max - self.voltage_min
        self.num_levels: int = 2**self.num_bits

        # Tamano de escalon Delta = (Vmax - Vmin) / 2^b
        self.delta: float = self.voltage_range / float(self.num_levels)

        # Factores de escala
        self.scale_factor_volts_per_bit: float = self.delta  # [V/LSB]
        self.scale_factor_bits_per_volt: float = 1.0 / self.delta  # [LSB/V]

        # Estados de la senal procesada
        self.original_signal: np.ndarray = np.empty(0, dtype=np.float64)
        self.quantized_signal: np.ndarray = np.empty(0, dtype=np.float64)
        self.quantization_indices: np.ndarray = np.empty(0, dtype=np.int64)
        self.quantization_error: np.ndarray = np.empty(0, dtype=np.float64)
        self.is_quantized: bool = False

    def quantize(self, signal: np.ndarray) -> np.ndarray:
        """Ejecuta la cuantizacion uniforme midrise sobre la senal muestreada.

        Mapea cada muestra analogica al indice de intervalo correspondiente y
        asigna el valor de reconstruccion central del intervalo.

        Args:
            signal: Arreglo de muestras en Voltios x[n].

        Returns:
            Arreglo con la senal cuantizada x_q[n] en Voltios.
        """
        self.original_signal = np.asarray(signal, dtype=np.float64)

        # Indice de nivel entero: k = floor((x - Vmin) / Delta)
        normalized_steps = (self.original_signal - self.voltage_min) / self.delta
        raw_indices = np.floor(normalized_steps).astype(np.int64)

        # Saturacion (clipping) a los limites fisicos del conversor [0, 2^b - 1]
        self.quantization_indices = np.clip(raw_indices, 0, self.num_levels - 1)

        # Reconstruccion centrada en el intervalo (Midrise):
        # x_q = Vmin + (k + 0.5) * Delta
        self.quantized_signal = self.voltage_min + (self.quantization_indices + 0.5) * self.delta

        # Error de cuantizacion: e[n] = x[n] - x_q[n]
        self.quantization_error = self.original_signal - self.quantized_signal
        self.is_quantized = True

        return self.quantized_signal

    def encode_to_binary(self, num_samples_preview: int = 10) -> List[str]:
        """Convierte los niveles cuantizados a palabras de codigo binarias de b bits.

        Args:
            num_samples_preview: Cantidad de primeras muestras a codificar para visualizacion.

        Returns:
            Lista de cadenas binarias con longitud fija igual a num_bits.
        """
        if not self.is_quantized:
            raise RuntimeError("Debe ejecutar quantize() antes de codificar a binario.")

        preview_indices = self.quantization_indices[:num_samples_preview]
        format_spec = f"0{self.num_bits}b"
        return [format(int(idx), format_spec) for idx in preview_indices]

    def compute_metrics(self) -> Dict[str, float]:
        """Calcula los parametros estadisticos del error y la relacion SQNR.

        Returns:
            Diccionario con potencia de senal, potencia de ruido, SQNR medida (dB),
            SQNR teorica senoidal de referencia (dB), y error maximo absoluto.
        """
        if not self.is_quantized:
            raise RuntimeError("Debe ejecutar quantize() antes de calcular metricas.")

        signal_power = float(np.mean(self.original_signal**2))
        noise_power = float(np.mean(self.quantization_error**2))

        # SQNR experimental
        if noise_power > 1e-15:
            sqnr_db = float(10.0 * np.log10(signal_power / noise_power))
        else:
            sqnr_db = float("inf")

        # SQNR teorica para senoide a plena escala: SQNR = 6.02 * b + 1.76 dB
        sqnr_theoretical_sin_db = 6.02 * float(self.num_bits) + 1.76

        # Varianza teorica del ruido uniforme Delta^2 / 12
        theoretical_noise_var = (self.delta**2) / 12.0

        max_abs_error = float(np.max(np.abs(self.quantization_error)))
        theoretical_max_error = self.delta / 2.0

        return {
            "num_bits": self.num_bits,
            "num_levels": self.num_levels,
            "delta_volts": self.delta,
            "scale_factor_v_per_bit": self.scale_factor_volts_per_bit,
            "scale_factor_bits_per_v": self.scale_factor_bits_per_volt,
            "signal_power_watts": signal_power,
            "noise_power_watts": noise_power,
            "theoretical_noise_power": theoretical_noise_var,
            "sqnr_measured_db": round(sqnr_db, 2),
            "sqnr_theoretical_sin_db": round(sqnr_theoretical_sin_db, 2),
            "max_abs_error_measured": round(max_abs_error, 5),
            "max_abs_error_theoretical": round(theoretical_max_error, 5),
        }

    def plot_adc_dac_comparison(
        self,
        time_vector: np.ndarray,
        num_samples_plot: int = 120,
        genre_label: str = "Audio",
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """Grafica la senal analogica x(t), la reconstruccion DAC en escalon ZOH y el error e[n].

        Args:
            time_vector: Vector de tiempo discreto correspondiente a las muestras.
            num_samples_plot: Cantidad de muestras a visualizar en el acercamiento.
            genre_label: Etiqueta descriptiva del genero.
            save_path: Ruta de guardado opcional.

        Returns:
            Tupla (figura, (ax_top, ax_bottom)).
        """
        if not self.is_quantized:
            raise RuntimeError("Debe ejecutar quantize() antes de graficar.")

        n_pts = min(num_samples_plot, len(self.original_signal))
        t_sub = time_vector[:n_pts] * 1000.0  # convertir a milisegundos
        x_sub = self.original_signal[:n_pts]
        xq_sub = self.quantized_signal[:n_pts]
        err_sub = self.quantization_error[:n_pts]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

        # Subgrafica 1: Reconstruccion DAC vs Original
        ax1.plot(t_sub, x_sub, color="#1f77b4", linewidth=1.5, label="x(t) Original (Analogica simulada)")
        ax1.step(t_sub, xq_sub, where="post", color="#ff7f0e", linewidth=1.2, linestyle="-", label=f"DAC ZOH Reconstruida ({self.num_bits} bits)")
        ax1.scatter(t_sub, xq_sub, color="#ff7f0e", s=12, alpha=0.7, label="Muestras ADC")

        ax1.set_title(
            f"{genre_label} - Digitalizacion ADC y Reconstruccion DAC en Escalon (ZOH) [{self.num_bits} bits, Delta = {self.delta:.4f} V]",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_ylabel("Tension [V]", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", fontsize=9)

        # Subgrafica 2: Error de cuantizacion e[n] = x[n] - x_q[n]
        ax2.plot(t_sub, err_sub, color="#d62728", linewidth=1.0, label="Error e[n] = x[n] - x_q[n]")
        ax2.axhline(self.delta / 2.0, color="gray", linestyle="--", linewidth=0.8, label="+Delta/2 Limite")
        ax2.axhline(-self.delta / 2.0, color="gray", linestyle="--", linewidth=0.8, label="-Delta/2 Limite")
        ax2.set_xlabel("Tiempo [ms]", fontsize=10)
        ax2.set_ylabel("Error [V]", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()

        return fig, (ax1, ax2)
