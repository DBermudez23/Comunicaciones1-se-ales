"""Modulo de gestion de metadatos historicos y tecnicos de generos musicales.

Este modulo define la informacion contextual e historica de las obras y generos
analizados en el taller de procesamiento digital de senales (DSP) y neuroacustica.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GenreMetadata:
    """Estructura de datos inmutable para almacenar los metadatos de una obra musical.

    Atributos:
        genre_id: Identificador unico en formato clave (ej. 'clasica_barroca').
        genre_name: Nombre estandarizado del genero musical.
        artist_composer: Nombre del musico, compositor o interprete principal.
        work_title: Titulo formal de la pieza musical analizada.
        ensemble_type: Tipo de agrupacion (ej. Orquesta barroca, Banda, Conjunto).
        release_year: Ano de composicion o publicacion original.
        filename: Nombre del archivo de audio .wav esperado en disco.
        historical_context: Resena historica relevante de la pieza y el movimiento.
        musical_characteristics: Rasgos de instrumentacion, tempo y textura sonora.
    """

    genre_id: str
    genre_name: str
    artist_composer: str
    work_title: str
    ensemble_type: str
    release_year: int
    filename: str
    historical_context: str
    musical_characteristics: str


class GenreMetadataManager:
    """Gestiona el catalogo documental e historico de los generos musicales requeridos.

    Proporciona metodos de busqueda, validacion y formateo para los 5 generos
    estudiados en la investigacion de DSP y TDAH.
    """

    def __init__(self) -> None:
        """Inicializa el gestor de metadatos con la base documental predefinida."""
        self._catalog: Dict[str, GenreMetadata] = {}
        self._initialize_catalog()

    def _initialize_catalog(self) -> None:
        """Carga en memoria los registros investigativos de los 5 generos requeridos."""
        genres = [
            GenreMetadata(
                genre_id="clasica_barroca",
                genre_name="Clasica - Barroca",
                artist_composer="Antonio Vivaldi",
                work_title="Las Cuatro Estaciones: Concierto No. 4 en Fa menor, Op. 8, RV 297 'El Invierno' (I. Allegro non molto)",
                ensemble_type="Orquesta de camara barroca (cuerdas frotadas y bajo continuo)",
                release_year=1725,
                filename="clasica_barroca.wav",
                historical_context=(
                    "Obra maestra del barroco tardio veneciano perteneciente a Il cimento dell'armonia e dell'inventione. "
                    "Vivaldi recrea acusticamente el temblor por el frio glacial, el viento tempestuoso y el pataleo "
                    "ritmico para combatir el invierno, mediante pizzicatos y articulaciones contrapuntisticas rigurosas."
                ),
                musical_characteristics=(
                    "Estructura ritmica simetrica de compas binario con ataques repetitivos de semicorcheas en staccato "
                    "(108-112 BPM). Timbre acustico natural dominado por la seccion de violines y chelos, con armónicos "
                    "organicos suaves distribuidos entre 400 Hz y 6 kHz y nula distorsion de origen electronico."
                ),
            ),
            GenreMetadata(
                genre_id="metal",
                genre_name="Metal",
                artist_composer="Metallica (James Hetfield / Lars Ulrich / Kirk Hammett)",
                work_title="Enter Sandman",
                ensemble_type="Banda de Heavy Metal (guitarras electricas distorsionadas, bajo y bateria acustica pesada)",
                release_year=1991,
                filename="metal.wav",
                historical_context=(
                    "Cancion de apertura y corte promocional insignia del album homonimo ('The Black Album'). "
                    "Consolido la proyeccion masiva del metal a escala global, construida sobre un riff principal en "
                    "tono de Mi menor (Mi, Si bemol, Fa, Mi) desarrollado a partir de la triada del tritono."
                ),
                musical_characteristics=(
                    "Tempo constante a 123 BPM con base de bateria potente y un muro de guitarras saturadas por clipping "
                    "analogico-valvular (overdrive/high-gain). Densa concentracion de energia armonica en medios agresivos "
                    "(1 kHz a 5 kHz) y transientes percusivos de gran pegada."
                ),
            ),
            GenreMetadata(
                genre_id="salsa",
                genre_name="Salsa",
                artist_composer="Hector Lavoe (Composicion de Ruben Blades / Arreglos de Willie Colon)",
                work_title="El Cantante",
                ensemble_type="Orquesta de Salsa Clasica (trompetas, trombones, percusion afroantillana y piano)",
                release_year=1978,
                filename="salsa.wav",
                historical_context=(
                    "Himno indiscutible de la salsa caribena producido bajo el sello Fania Records en el album Comedia. "
                    "Compuesta magistralmente por Ruben Blades y arreglada por Willie Colon con una introduccion sinfonica "
                    "que luego estalla en el tumbao afroantillano narrando la dicotomia del artista."
                ),
                musical_characteristics=(
                    "Cadencia sincopada sobre patron de clave 3-2 en compas 4/4 a ~98 BPM. Seccion polirritmica rica "
                    "con timbales, congas y bongo que generan transientes de alta frecuencia, respaldada por metales "
                    "brillantes (trompetas) que alcanzan energias significativas entre 2.5 kHz y 8 kHz."
                ),
            ),
            GenreMetadata(
                genre_id="cantina",
                genre_name="Cantina al Hombro (Musica Popular / Despecho)",
                artist_composer="Julio Jaramillo ('El Ruisenor de America' / Composicion de Benito de Jesus)",
                work_title="Nuestro Juramento",
                ensemble_type="Conjunto de Musica Popular y Cantina (requinto solista, guitarras acusticas y percusion menor)",
                release_year=1957,
                filename="cantina.wav",
                historical_context=(
                    "Emblema absoluto de la musica de cantina, arrabal y despecho en America Latina. Grabado en Guayaquil "
                    "con el virtuoso requinto de Rosalino Quintero, consolidandose como una de las piezas mas universales "
                    "del sentimiento popular hispanoamericano."
                ),
                musical_characteristics=(
                    "Compas binario lento de bolero/cancion a ~82 BPM. Destaca el fraseo inicial del requinto con "
                    "punteado penetrante y timbre metalico agudo (1.5 kHz a 4.5 kHz), bajo acustico de pulsacion "
                    "espaciada y voz solista calida centrada en el rango fundamental vocal (180 Hz a 1.2 kHz)."
                ),
            ),
            GenreMetadata(
                genre_id="reggaeton",
                genre_name="Reggaeton",
                artist_composer="Daddy Yankee (Produccion de Luny Tunes)",
                work_title="Gasolina",
                ensemble_type="Produccion de Musica Urbana Digital (sintetizadores digitales, cajas de ritmo y secuenciadores)",
                release_year=2004,
                filename="reggaeton.wav",
                historical_context=(
                    "Cancion que consagro el reggaeton a escala mundial desde Puerto Rico como fenomeno cultural "
                    "y de masas. Pista central del disco Barrio Fino estructurada para alto impacto auditivo en recintos "
                    "cerrados y sistemas de sonido automotrices."
                ),
                musical_characteristics=(
                    "Patron de dembow riguroso a 96 BPM (bombo a tierra con redoble de caja en tresillo sincopado). "
                    "Contenido extremo de presion acustica en sub-graves (40 Hz a 90 Hz mediante bombos sintetizados 808) "
                    "con procesamiento de compresion dinamica y limitacion agresiva en la etapa de masterizacion."
                ),
            ),
        ]

        for item in genres:
            self._catalog[item.genre_id] = item

    def get_metadata(self, genre_id: str) -> GenreMetadata:
        """Obtiene el registro documental de un genero especifico.

        Args:
            genre_id: Clave identificadora del genero.

        Returns:
            Instancia de GenreMetadata correspondiente.

        Raises:
            KeyError: Si el identificador no existe en el catalogo.
        """
        if genre_id not in self._catalog:
            valid_keys = ", ".join(self._catalog.keys())
            raise KeyError(f"Genero '{genre_id}' no reconocido. Claves validas: {valid_keys}")
        return self._catalog[genre_id]

    def get_all_genres(self) -> Dict[str, GenreMetadata]:
        """Retorna una copia completa del catalogo de generos musicales.

        Returns:
            Diccionario que mapea cada genre_id a su objeto GenreMetadata.
        """
        return dict(self._catalog)

    def get_genre_ids(self) -> List[str]:
        """Retorna la lista de identificadores registrados en el catalogo.

        Returns:
            Lista de cadenas con los genre_ids.
        """
        return list(self._catalog.keys())

    def display_catalog(self) -> None:
        """Imprime en consola un resumen ordenado y tabular del catalogo de generos."""
        separator = "=" * 80
        sub_separator = "-" * 80
        print(separator)
        print("CATALOGO HISTORICO-DOCUMENTAL DE GENEROS MUSICALES")
        print(separator)
        for genre_id, meta in self._catalog.items():
            print(f"Clave ID:          {genre_id}")
            print(f"Genero:            {meta.genre_name}")
            print(f"Compositor/Musico: {meta.artist_composer}")
            print(f"Obra:              {meta.work_title}")
            print(f"Agrupacion:        {meta.ensemble_type}")
            print(f"Ano:               {meta.release_year}")
            print(f"Archivo esperado:  {meta.filename}")
            print(f"Contexto historico: {meta.historical_context}")
            print(f"Caracteristicas:   {meta.musical_characteristics}")
            print(sub_separator)
