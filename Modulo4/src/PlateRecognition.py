
"""
Módulo plateRecognition

Define la clase PlateDetector, que se encarga de usar un modelo YOLO
(preentrenado en formato .pt) para localizar la placa dentro de cada frame.

Tecnologías usadas:
- Ultralytics YOLO: framework basado en PyTorch para detección de objetos.
- OpenCV: manejo de imágenes y matrices BGR.
- pathlib: para construir rutas de forma portátil.
"""

from ultralytics import YOLO
from pathlib import Path
from typing import Optional, Tuple
import cv2


class PlateDetector:
    """
    Clase PlateDetector

    Su función es recibir imágenes (frames de OpenCV) y devolver:
    - Un recorte de la placa (imagen BGR).
    - Las coordenadas del bounding box (x1, y1, x2, y2).

    Parámetros del constructor:
    - model_path: ruta al archivo .pt del modelo YOLO. Si es None, por defecto
                  busca en <raiz_del_proyecto>/model/best.pt.
    - conf_threshold: umbral de confianza mínimo para considerar válida una detección.
                      Por ejemplo, 0.5 = 50% de confianza mínima.
    """

    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.5):
        # base_dir = carpeta raíz del proyecto (src/..)
        base_dir = Path(__file__).resolve().parent.parent

        # Si no se pasa ruta, usamos model/best.pt por defecto
        if model_path is None:
            model_path = base_dir / "model" / "best.pt"
        else:
            model_path = Path(model_path)

        # Cargamos el modelo YOLO desde el archivo .pt
        self.model = YOLO(str(model_path))

        # Umbral de confianza para filtrar detecciones débiles
        self.conf_threshold = conf_threshold

    def detect_plate_from_image(
        self, image_bgr
    ) -> Tuple[Optional[any], Optional[Tuple[int, int, int, int]]]:
        """
        Detecta la placa a partir de un frame en formato BGR (OpenCV).

        Parámetros:
        - image_bgr: imagen/fotograma en formato BGR (típico de OpenCV),
                     normalmente viene directo de la cámara.

        Retorno:
        - (crop, bbox)
          * crop: imagen BGR recortada donde se encuentra la placa.
          * bbox: tupla (x1, y1, x2, y2) con las coordenadas del bounding box,
                  en el sistema de coordenadas de la propia imagen.
        - (None, None) si no se detecta ninguna placa.

        Detalles:
        - Selecciona la detección con mayor confianza.
        - Amplía ligeramente la caja (márgenes) para asegurar que la placa
          completa quede dentro del recorte.
        """

        # Ejecutamos el modelo en la imagen. conf filtra por confianza mínima.
        results = self.model(image_bgr, conf=self.conf_threshold)

        # Obtenemos las cajas detectadas del primer resultado
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            # No se detectó ninguna placa
            return None, None

        # Tomamos la detección con mayor confianza
        confidences = boxes.conf
        best_idx = confidences.argmax().item()
        best_box = boxes[best_idx]

        # Coordenadas del bounding box en el frame original
        x1, y1, x2, y2 = best_box.xyxy[0].int().tolist()
        h, w, _ = image_bgr.shape

        # Márgenes extra para evitar recortar demasiado ajustado
        margin_x = int((x2 - x1) * 0.05)  # 5% del ancho
        margin_y = int((y2 - y1) * 0.10)  # 10% del alto

        # Aplicamos márgenes y aseguramos que no se salgan de la imagen
        x1 = max(0, x1 - margin_x)
        x2 = min(w - 1, x2 + margin_x)
        y1 = max(0, y1 - margin_y)
        y2 = min(h - 1, y2 + margin_y)

        # Si el resultado no tiene área válida, lo consideramos nulo
        if x2 <= x1 or y2 <= y1:
            return None, None

        # Recortamos la placa de la imagen original
        crop = image_bgr[y1:y2, x1:x2].copy()
        return crop, (x1, y1, x2, y2)
