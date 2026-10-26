"""Modulo de carga, acondicionamiento y normalizacion de senales de audio WAV.

Implementa la clase WavAudioLoader encargada de la lectura de archivos de audio
desde disco, conversion de canales estéreo a mono y escalamiento de amplitud
a un rango estandarizado de voltaje analogico bipolar.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy.io import wavfile


class WavAudioLoader:
    """Carga y acondiciona senales de audio en formato WAV para analisis DSP.

    Realiza el procesamiento previo necesario para simular senales analogicas de
    entrada a un convertidor ADC: conversion de estereo a mono, recorte/ajuste temporal
    y normalizacion lineal a niveles de tension electrica en Voltios.
    """

    def __init__(
        self,
        voltage_min: float = -5.0,
        voltage_max: float = 5.0,
        target_duration_sec: Optional[float] = 4.0,
    ) -> None:
        """Inicializa los parametros de acondicionamiento de senal.

        Args:
            voltage_min: Limite inferior del rango de voltaje de entrada (V).
            voltage_max: Limite superior del rango de voltaje de entrada (V).
            target_duration_sec: Duracion objetivo en segundos (4.0 seg por defecto).
        """
        if voltage_min >= voltage_max:
            raise ValueError("voltage_min debe ser estrictamente menor que voltage_max")

        self.voltage_min: float = float(voltage_min)
        self.voltage_max: float = float(voltage_max)
        self.target_duration_sec: Optional[float] = target_duration_sec

        self.file_path: Optional[Path] = None
        self.sampling_rate: int = 0
        self.num_samples: int = 0
        self.duration_sec: float = 0.0
        self.time_vector: np.ndarray = np.empty(0, dtype=np.float64)
        self.voltage_signal: np.ndarray = np.empty(0, dtype=np.float64)
        self.original_channels: int = 1
        self.is_loaded: bool = False

    def load_file(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, int]:
        """Carga un archivo WAV, convierte a mono y escala al rango de voltaje.

        Args:
            file_path: Ruta al archivo .wav en el sistema de archivos.

        Returns:
            Tupla (time_vector, voltage_signal, sampling_rate):
                - time_vector: Vector de tiempo discreto t en segundos.
                - voltage_signal: Amplitud de la senal acondicionada en Voltios.
                - sampling_rate: Frecuencia de muestreo fs en Hertz (Hz).

        Raises:
            FileNotFoundError: Si el archivo no existe en la ruta especificada.
            ValueError: Si el formato del archivo es invalido o vacio.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"No se encontro el archivo de audio: {path.resolve()}")

        # Lectura mediante scipy.io.wavfile
        fs, raw_data = wavfile.read(str(path))
        if raw_data.size == 0:
            raise ValueError(f"El archivo {path.name} no contiene muestras de audio.")

        self.file_path = path
        self.sampling_rate = int(fs)

        # Determinar numero de canales y convertir a mono si es necesario
        if raw_data.ndim == 1:
            self.original_channels = 1
            audio_data = raw_data.astype(np.float64)
        elif raw_data.ndim == 2:
            self.original_channels = raw_data.shape[1]
            # Promediado de canales estereo/multicanal
            audio_data = np.mean(raw_data, axis=1, dtype=np.float64)
        else:
            raise ValueError(f"Formato multicanal no soportado: dimension={raw_data.ndim}")

        # Normalizacion segun el tipo de datos numerico original
        dtype_name = raw_data.dtype.name
        if dtype_name == "uint8":
            normalized = (audio_data - 128.0) / 128.0
        elif dtype_name == "int16":
            normalized = audio_data / 32768.0
        elif dtype_name == "int32":
            normalized = audio_data / 2147483648.0
        elif dtype_name.startswith("float"):
            normalized = audio_data
        else:
            # Normalizacion generica por valor maximo
            max_val = np.max(np.abs(audio_data))
            normalized = audio_data / max_val if max_val > 0 else audio_data

        # Garantizar que no existan valores fuera de [-1.0, 1.0]
        max_abs = np.max(np.abs(normalized))
        if max_abs > 1.0:
            normalized = normalized / max_abs

        # Ajuste de duracion a 4 segundos (target_duration_sec) si esta configurado
        if self.target_duration_sec is not None and self.target_duration_sec > 0:
            target_samples = int(self.target_duration_sec * self.sampling_rate)
            if len(normalized) >= target_samples:
                normalized = normalized[:target_samples]
            else:
                # Si el audio es menor a 4 segundos, se conserva la longitud real
                pass

        self.num_samples = len(normalized)
        self.duration_sec = self.num_samples / float(self.sampling_rate)

        # Generacion del vector de tiempo t = n / fs
        self.time_vector = np.arange(self.num_samples, dtype=np.float64) / float(self.sampling_rate)

        # Mapeo lineal al rango de voltaje [voltage_min, voltage_max]
        # Para un rango simetrico [-V, +V], normalized [-1, 1] mapea a [-V, +V]
        if abs(self.voltage_min + self.voltage_max) < 1e-9:
            # Caso simetrico (ej. -5V a +5V)
            self.voltage_signal = normalized * self.voltage_max
        else:
            # Caso asimetrico general
            norm_zero_one = (normalized + 1.0) / 2.0
            self.voltage_signal = self.voltage_min + norm_zero_one * (self.voltage_max - self.voltage_min)

        self.is_loaded = True
        return self.time_vector, self.voltage_signal, self.sampling_rate

    def get_summary(self) -> Dict[str, Any]:
        """Retorna un diccionario con las caracteristicas tecnicas del audio cargado.

        Returns:
            Diccionario con fs, duracion, numero de muestras, rango de tension y canales.
        """
        if not self.is_loaded:
            raise RuntimeError("No se ha cargado ningun archivo de audio todavia.")

        v_min_actual = float(np.min(self.voltage_signal))
        v_max_actual = float(np.max(self.voltage_signal))
        v_rms = float(np.sqrt(np.mean(self.voltage_signal**2)))

        return {
            "file_name": self.file_path.name if self.file_path else "desconocido",
            "sampling_rate_hz": self.sampling_rate,
            "num_samples": self.num_samples,
            "duration_seconds": round(self.duration_sec, 4),
            "original_channels": self.original_channels,
            "voltage_range_nominal": (self.voltage_min, self.voltage_max),
            "voltage_min_measured": round(v_min_actual, 4),
            "voltage_max_measured": round(v_max_actual, 4),
            "voltage_rms": round(v_rms, 4),
        }
