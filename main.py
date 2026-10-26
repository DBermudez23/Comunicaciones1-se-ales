"""Script principal de ejecucion de la suite DSP para el taller ADC-DAC y analisis TDAH.

Orquesta la ejecucion secuencial y modular de:
1. Gestion documental e historica de metadatos (GenreMetadataManager).
2. Carga y acondicionamiento electrico de archivos WAV reales (WavAudioLoader).
3. Analisis espectral FFT y verificacion formal de Nyquist-Shannon (NyquistSpectralAnalyzer).
4. Simulacion de conversion ADC-DAC uniforme, calculo de Delta y SQNR (ADCDACSimulator).
5. Analisis comparativo integrador y generacion de dictamen clinico para TDAH (TDAHPathologyAnalyzer).
"""

import os
from pathlib import Path
import sys
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from audio_loader import WavAudioLoader
from metadata_manager import GenreMetadata, GenreMetadataManager
from quantizer import ADCDACSimulator
from spectral_analyzer import NyquistSpectralAnalyzer
from tdah_comparator import GenreProcessingResult, TDAHPathologyAnalyzer


def find_audio_file(audios_dir: Path, expected_filename: str, genre_id: str) -> Optional[Path]:
    """Busca el archivo de audio correspondiente en la carpeta ./audios/.

    Primero busca por coincidencia exacta con el nombre esperado. Si no lo encuentra,
    realiza una busqueda por coincidencia de subcadena en los archivos .wav existentes.

    Args:
        audios_dir: Ruta al directorio ./audios/.
        expected_filename: Nombre de archivo recomendado.
        genre_id: Identificador del genero para busqueda secundaria.

    Returns:
        Ruta Path del archivo si existe, o None si no se encuentra.
    """
    direct_path = audios_dir / expected_filename
    if direct_path.is_file():
        return direct_path

    # Busqueda alternativa por palabra clave en la carpeta
    if audios_dir.is_dir():
        for f in audios_dir.glob("*.wav"):
            fname = f.name.lower()
            if genre_id in fname or expected_filename.lower().replace(".wav", "") in fname:
                return f

    return None


def print_banner(title: str) -> None:
    """Imprime un separador de seccion formal en consola sin emojis."""
    separator = "=" * 82
    print("\n" + separator)
    print(title.upper())
    print(separator)


def run_pipeline(
    audios_directory: str = "./audios",
    output_directory: str = "./output_plots",
    show_plots: bool = True,
) -> None:
    """Ejecuta la tuberia de procesamiento DSP para los 5 generos requeridos.

    Args:
        audios_directory: Carpeta donde residen los 5 archivos WAV provistos por el usuario.
        output_directory: Carpeta donde se guardaran las graficas generadas.
        show_plots: Si es True, invoca plt.show() al finalizar para visualizacion interactiva.
    """
    audios_path = Path(audios_directory).resolve()
    output_path = Path(output_directory).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print_banner("INICIO DEL PROCESAMIENTO DIGITAL DE SENALES - TALLER ADC-DAC Y TDAH")
    print(f"Directorio de entrada de audios: {audios_path}")
    print(f"Directorio de salida de graficas: {output_path}")

    # 1. Gestion de Metadatos Historicos
    print_banner("1. CATALOGO DE METADATOS HISTORICOS Y INVESTIGATIVOS")
    metadata_mgr = GenreMetadataManager()
    metadata_mgr.display_catalog()

    genre_ids = metadata_mgr.get_genre_ids()

    # Verificacion preliminar de archivos en disco
    missing_files: List[str] = []
    audio_files_map: Dict[str, Path] = {}

    for g_id in genre_ids:
        meta = metadata_mgr.get_metadata(g_id)
        found_file = find_audio_file(audios_path, meta.filename, g_id)
        if found_file is not None:
            audio_files_map[g_id] = found_file
        else:
            missing_files.append(f"{meta.filename} (para '{meta.genre_name}')")

    if missing_files:
        print_banner("ADVERTENCIA: ARCHIVOS DE AUDIO PENDIENTES EN ./audios/")
        print("El programa requiere archivos de audio reales de 4 segundos provistos por el usuario.")
        print(f"Asegurese de colocar los siguientes archivos .wav en: {audios_path}\n")
        for mf in missing_files:
            print(f"  [FALTANTE] {mf}")
        print("\nEstructura esperada de nombres:")
        for g_id in genre_ids:
            meta = metadata_mgr.get_metadata(g_id)
            print(f"  - {meta.filename:<22} -> {meta.genre_name}")
        print("\nPor favor agregue los archivos y vuelva a ejecutar el programa.")
        return

    # 2. Inicializacion del Analizador Comparador para TDAH
    tdah_analyzer = TDAHPathologyAnalyzer()

    # 3. Procesamiento secuencial de cada genero
    print_banner("2. EJECUCION DE MEDICIONES DSP, FFT Y CONVERSION ADC-DAC POR GENERO")

    for index, g_id in enumerate(genre_ids, start=1):
        meta = metadata_mgr.get_metadata(g_id)
        audio_file = audio_files_map[g_id]

        print("-" * 82)
        print(f"[{index}/5] PROCESANDO: {meta.genre_name.upper()}")
        print(f"      Archivo: {audio_file.name}")
        print(f"      Obra: {meta.work_title} | Autor: {meta.artist_composer}")
        print("-" * 82)

        # A. Carga de audio y acondicionamiento electrico [-5V, +5V]
        loader = WavAudioLoader(voltage_min=-5.0, voltage_max=5.0, target_duration_sec=4.0)
        time_vec, v_signal, fs = loader.load_file(audio_file)
        summary = loader.get_summary()

        print(f"  Muestreo (fs):         {summary['sampling_rate_hz']} Hz")
        print(f"  Numero de muestras N:  {summary['num_samples']}")
        print(f"  Duracion efectiva:     {summary['duration_seconds']} segundos")
        print(f"  Rango de tension med:  [{summary['voltage_min_measured']} V, {summary['voltage_max_measured']} V]")
        print(f"  Tension RMS:           {summary['voltage_rms']} V")

        # B. Analisis Espectral y Validacion de Nyquist-Shannon
        spec_analyzer = NyquistSpectralAnalyzer(
            time_vector=time_vec,
            voltage_signal=v_signal,
            sampling_rate=fs,
            genre_label=meta.genre_name,
        )
        freqs, mag_spec = spec_analyzer.compute_fft()
        nyq_info = spec_analyzer.validate_nyquist(energy_threshold_ratio=0.99)
        dominant_peaks = spec_analyzer.find_dominant_frequencies(top_k=5)

        print("\n  [ANALISIS ESPECTRAL Y NYQUIST]")
        print(f"  Frecuencia de Nyquist: {nyq_info['nyquist_frequency_hz']} Hz")
        print(f"  Ancho de banda 99%:    {nyq_info['effective_bandwidth_99_hz']} Hz")
        print(f"  Tasa minima requerida: {nyq_info['min_required_sampling_rate_hz']} Hz")
        print(f"  Diagnostico formal:    {nyq_info['nyquist_statement']}")
        print("  Picos espectrales dominantes:")
        for r_idx, (f_peak, v_peak) in enumerate(dominant_peaks[:3], start=1):
            print(f"    Pico {r_idx}: {f_peak:8.2f} Hz | Magnitud: {v_peak:6.4f} V")

        # C. Conversion ADC-DAC: Cuantizacion uniforme a 8 bits y 16 bits
        # Simulador ADC-DAC a 8 bits
        adc_8b = ADCDACSimulator(num_bits=8, voltage_min=-5.0, voltage_max=5.0)
        xq_8b = adc_8b.quantize(v_signal)
        metrics_8b = adc_8b.compute_metrics()
        sample_words_8b = adc_8b.encode_to_binary(num_samples_preview=6)

        # Simulador ADC-DAC a 16 bits para contraste
        adc_16b = ADCDACSimulator(num_bits=16, voltage_min=-5.0, voltage_max=5.0)
        xq_16b = adc_16b.quantize(v_signal)
        metrics_16b = adc_16b.compute_metrics()

        print("\n  [CUANTIZACION ADC UNIFORME Y METRICAS DAC]")
        print(f"  Resolucion 8 bits:  Delta = {metrics_8b['delta_volts']:.5f} V (L = {metrics_8b['num_levels']} niveles)")
        print(f"  Factor de escala:   {metrics_8b['scale_factor_v_per_bit']:.5f} V/LSB | {metrics_8b['scale_factor_bits_per_v']:.2f} LSB/V")
        print(f"  Muestras binarias:  {' '.join(sample_words_8b)}")
        print(f"  SQNR medido (8b):   {metrics_8b['sqnr_measured_db']} dB (Teorico senoidal: {metrics_8b['sqnr_theoretical_sin_db']} dB)")
        print(f"  Error maximo (8b):  {metrics_8b['max_abs_error_measured']:.5f} V (Teorico: {metrics_8b['max_abs_error_theoretical']:.5f} V)")
        print(f"  Resolucion 16 bits: Delta = {metrics_16b['delta_volts']:.8f} V (L = {metrics_16b['num_levels']} niveles)")
        print(f"  SQNR medido (16b):  {metrics_16b['sqnr_measured_db']} dB (Teorico senoidal: {metrics_16b['sqnr_theoretical_sin_db']} dB)")

        # D. Registro de resultados en el analizador integrador
        result_entry = GenreProcessingResult(
            genre_id=g_id,
            metadata=meta,
            time_vector=time_vec,
            voltage_signal=v_signal,
            sampling_rate=fs,
            frequencies=freqs,
            magnitude_spectrum=mag_spec,
            dominant_peaks=dominant_peaks,
            nyquist_results=nyq_info,
            quantization_metrics_8b=metrics_8b,
            quantized_signal_8b=xq_8b,
            quantization_error_8b=adc_8b.quantization_error,
        )
        tdah_analyzer.add_genre_result(result_entry)

    # 4. Generacion de Graficas Individuales
    print_banner("3. GENERACION DE FIGURAS INDIVIDUALES POR GENERO")
    for g_id in genre_ids:
        meta = metadata_mgr.get_metadata(g_id)
        tdah_analyzer.plot_individual_genre(genre_id=g_id, output_dir=str(output_path), zoom_ms=35.0)
        print(f"  Figura guardada: {output_path / f'analisis_individual_{g_id}.png'}")

    # 5. Generacion de Grafica Integradora Comparativa
    print_banner("4. GENERACION DE FIGURA INTEGRADORA MULTIGENERO")
    integrated_fig_path = output_path / "grafica_integradora_tdah.png"
    tdah_analyzer.plot_integrated_comparison(output_path=str(integrated_fig_path), max_freq_hz=6000.0)
    print(f"  Figura integradora guardada en: {integrated_fig_path}")

    # 6. Emision del Informe Tecnico y Clinico para TDAH
    print_banner("5. INFORME ARGUMENTADO DE DSP Y NEUROACUSTICA PARA TDAH")
    clinical_report = tdah_analyzer.generate_clinical_report()
    print(clinical_report)

    # Guardar informe textual en disco
    report_file_path = output_path / "informe_neuroacustica_tdah.txt"
    with open(report_file_path, "w", encoding="utf-8") as f_rep:
        f_rep.write(clinical_report)
    print(f"\nInforme textual guardado en: {report_file_path}")

    print_banner("PROCESAMIENTO FINALIZADO EXITOSAMENTE")

    # Visualizacion interactiva si esta habilitada y existe backend grafico
    if show_plots:
        print("Desplegando suite de graficas interactivas...")
        try:
            plt.show()
        except Exception as exc:
            print(f"Nota: Visualizacion en ventana no disponible ({exc}). Las imagenes fueron guardadas en disco.")


if __name__ == "__main__":
    run_pipeline(
        audios_directory="./audios",
        output_directory="./output_plots",
        show_plots=False,  # En ejecucion de consola por defecto guarda sin bloquear
    )
