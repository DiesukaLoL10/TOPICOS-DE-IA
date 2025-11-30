
"""
Módulo plate_ocr

Define la clase PlateOCR, que se encarga de leer el texto de la placa
a partir de una imagen recortada (BGR) usando EasyOCR.

Tecnologías usadas:
- EasyOCR: librería de OCR basada en deep learning.
- OpenCV: se usa para preprocesar la imagen (escala de grises, resize, binarización).
"""

import easyocr
import cv2


class PlateOCR:
    """
    Clase PlateOCR

    Recibe un recorte de la placa y devuelve el texto leído (solo letras y números),
    ya normalizado en mayúsculas.

    Parámetros del constructor:
    - languages: lista de códigos de idioma para EasyOCR, por ejemplo ['en'] o ['es'].
                 Para matrículas suele bastar con 'en' porque se trata de caracteres
                 alfanuméricos (A-Z, 0-9).
    """

    def __init__(self, languages=None):
        # Si no se especifican idiomas, usamos inglés por defecto (suficiente para placas)
        if languages is None:
            languages = ["en"]

        # Inicializamos el lector de EasyOCR con los idiomas deseados
        self.reader = easyocr.Reader(languages)

    def read_plate(self, plate_image_bgr) -> str | None:
        """
        Lee el texto de la placa a partir de una imagen BGR recortada.

        Parámetros:
        - plate_image_bgr: imagen BGR que contiene únicamente la placa
                           (sale del recorte hecho por PlateDetector).

        Flujo:
        1. Convierte la imagen a escala de grises.
        2. Aumenta el tamaño para facilitar el reconocimiento.
        3. Aplica un suavizado y umbralización para resaltar el contraste.
        4. Ejecuta EasyOCR con un allowlist limitado a A-Z y 0-9.
        5. Une los fragmentos de texto detectados y limpia caracteres no alfanuméricos.

        Retorno:
        - Cadena con la placa en mayúsculas (por ejemplo "ABC123"),
          o None si no se detectó texto.
        """
        if plate_image_bgr is None:
            return None

        # Convertimos a escala de grises (reducción de canales y ruido de color)
        gray = cv2.cvtColor(plate_image_bgr, cv2.COLOR_BGR2GRAY)

        # Aumentamos el tamaño para mejorar la lectura del OCR
        scale = 2.0
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

        # Suavizado + binarización (umbral adaptativo por Otsu)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # OCR con allowlist: restringimos a caracteres típicos de placas
        textos = self.reader.readtext(
            thresh,
            detail=0,  # detail=0 → solo texto, no bounding boxes
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )

        if not textos:
            return None

        # Unimos todos los fragmentos detectados (en caso de que la placa salga separada)
        raw_text = "".join(textos)

        # Normalizamos: quedarnos solo con caracteres alfanuméricos en mayúsculas
        plate_text = "".join(ch for ch in raw_text.upper() if ch.isalnum())

        return plate_text if plate_text else None
