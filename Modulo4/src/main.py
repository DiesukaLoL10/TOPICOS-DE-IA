
"""
main.py

Punto de entrada del sistema de detección de matrículas en tiempo real usando webcam.

Arquitectura general:
- Captura de video: OpenCV (cv2.VideoCapture).
- Detección de placa: PlateDetector (YOLO/Ultralytics) en plateRecognition.py.
- OCR del texto de la placa: PlateOCR (EasyOCR) en plate_ocr.py.
- Búsqueda en base de datos: SearchVehicle (consultas MySQL) en search_vehicle.py.
- Conexión a MySQL (XAMPP): MySQLDatabase en Db_conector.py.

¿Por qué estas tecnologías?
- OpenCV: estándar de facto para manejo de imágenes y video en tiempo real.
- YOLO (Ultralytics): modelo moderno de detección de objetos rápido y preciso.
- EasyOCR: librería de OCR sencilla de integrar y suficiente para texto corto
  como matrículas.
- MySQL + XAMPP: pila clásica de base de datos, fácil de configurar en entorno
  académico y bien soportada en Python.

Flujo de alto nivel:
1. Se abre la webcam.
2. Cada N frames se redimensiona el frame para reducir carga y se pasa a YOLO.
3. Se obtiene el bounding box de la placa, se reescala al tamaño original y
   se recorta la placa.
4. El recorte se pasa a EasyOCR para obtener el texto de la matrícula.
5. Con la matrícula se consulta la base de datos para obtener los datos
   del propietario.
6. Se dibuja el bounding box, la placa y el nombre del propietario en el frame
   y se muestra en una ventana.
"""

import cv2
from Db_conector import MySQLDatabase
from search_vehicle import SearchVehicle
from PlateRecognition import PlateDetector
from plate_ocr import PlateOCR


def run_on_webcam(camera_index: int = 0) -> None:
    """
    Función principal de ejecución en modo webcam.

    Parámetros:
    - camera_index: índice de la cámara en el sistema (0 = cámara principal).
      Si el equipo tiene varias cámaras, se puede probar con 1, 2, etc.

    Responsabilidades:
    - Crear la conexión a la base de datos.
    - Instanciar las clases de detección de placas, OCR y búsqueda de vehículo.
    - Abrir la cámara y procesar los frames en un bucle.
    - Mostrar en pantalla la imagen anotada con la info de la placa y el propietario.
    """

    # 1. Conexión a la base de datos (MySQL en XAMPP)
    db = MySQLDatabase(
        host="localhost",  # Servidor MySQL (XAMPP)
        port=3306,         # Puerto por defecto de MySQL
        user="root",       # Usuario típico en XAMPP
        password="123",       # Cambiar si configuraste contraseña
        database="placas_db",  # Nombre de la BD del proyecto
    )
    db.connect()

    # SearchVehicle se encarga de las consultas a la BD a partir de la placa
    search_vehicle = SearchVehicle(db)

    # PlateDetector usa YOLO para encontrar la placa en el frame
    # conf_threshold controla el umbral mínimo de confianza (0.5 = 50%)
    detector = PlateDetector(conf_threshold=0.5)

    # PlateOCR hace el reconocimiento de caracteres sobre el recorte de la placa
    # languages=['en'] porque trabajamos con caracteres alfanuméricos, basta inglés.
    ocr = PlateOCR(languages=["en"])

    # 2. Abrimos la webcam con OpenCV
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        db.close()
        return

    print("Cámara iniciada. Presiona 'q' para salir.")

    # Variables de estado para no repetir OCR/consultas innecesarias
    last_plate = None   # Última placa leída (texto)
    last_owner = None   # Último propietario encontrado (diccionario de BD)
    last_bbox = None    # Último bounding box (x1, y1, x2, y2)
    frame_count = 0     # Contador de frames para espaciar las detecciones

    try:
        while True:
            # Leemos un frame de la cámara
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame de la cámara.")
                break

            frame_count += 1

            # Para no saturar CPU, solo corremos YOLO + OCR cada N frames
            RUN_EVERY_N_FRAMES = 10
            run_this_frame = (frame_count % RUN_EVERY_N_FRAMES == 0)

            if run_this_frame:
                # ----- 1) Reducir resolución del frame para la detección -----
                # Esto reduce la carga computacional de YOLO sin afectar demasiado
                # la calidad del bounding box.
                orig_h, orig_w, _ = frame.shape

                target_w = 640  # Ancho deseado para detección (ajustable)
                scale_factor = target_w / float(orig_w)
                target_h = int(orig_h * scale_factor)

                scaled_frame = cv2.resize(
                    frame,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )

                # ----- 2) Detectar la placa en el frame reducido -----
                plate_crop_small, bbox_small = detector.detect_plate_from_image(
                    scaled_frame
                )

                if bbox_small is not None:
                    x1_s, y1_s, x2_s, y2_s = bbox_small

                    # ----- 3) Reescalar bounding box al frame original -----
                    x1 = int(x1_s / scale_factor)
                    x2 = int(x2_s / scale_factor)
                    y1 = int(y1_s / scale_factor)
                    y2 = int(y2_s / scale_factor)

                    # Aseguramos que las coordenadas estén dentro de la imagen
                    x1 = max(0, min(x1, orig_w - 1))
                    x2 = max(0, min(x2, orig_w - 1))
                    y1 = max(0, min(y1, orig_h - 1))
                    y2 = max(0, min(y2, orig_h - 1))

                    if x2 > x1 and y2 > y1:
                        # Guardamos el último bounding box válido
                        last_bbox = (x1, y1, x2, y2)

                        # ----- 4) Recortar la placa del frame original -----
                        # Usamos el frame original para que el OCR tenga mejor calidad.
                        plate_crop = frame[y1:y2, x1:x2].copy()

                        # ----- 5) Ejecutar OCR sobre el recorte de la placa -----
                        plate_text = ocr.read_plate(plate_crop)

                        if plate_text:
                            # Solo si la placa cambió, volvemos a consultar la BD.
                            # Esto evita repetir consultas y OCR innecesariamente.
                            if plate_text != last_plate:
                                last_plate = plate_text
                                print(f"\nPlaca detectada: {plate_text}")

                                # ----- 6) Búsqueda del propietario en la base de datos -----
                                owner_data = search_vehicle.find_owner_by_plate(
                                    plate_text
                                )
                                last_owner = owner_data

                                if owner_data:
                                    print("Información del propietario:")
                                    print(f"  Placa:      {owner_data['placa']}")
                                    print(f"  Marca:      {owner_data['marca']}")
                                    print(f"  Modelo:     {owner_data['modelo']}")
                                    print(f"  Año:        {owner_data['anio']}")
                                    print(f"  Propietario:{owner_data['nombre']}")
                                    print(f"  Teléfono:   {owner_data['telefono']}")
                                    print(f"  Email:      {owner_data['email']}")
                                else:
                                    print(
                                        "La placa no se encontró en la base de datos."
                                    )
                        else:
                            # Si el OCR falla, no se actualiza last_plate/last_owner.
                            pass
                    else:
                        # Bounding box inválido tras reescalar (descartamos)
                        pass
                else:
                    # No hubo detección en este frame reducido
                    pass

            # ----- 7) Dibujar la información en el frame original -----
            if last_bbox is not None:
                x1, y1, x2, y2 = last_bbox

                # Dibujamos la caja de la última placa detectada
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Mostramos el texto de la placa arriba del bounding box
                if last_plate:
                    cv2.putText(
                        frame,
                        last_plate,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                # Mostramos el nombre del propietario o un mensaje si no está en BD
                owner_text = "No encontrada en BD"
                if last_owner:
                    owner_text = last_owner["nombre"]

                y_text = y2 + 25
                if y_text >= frame.shape[0]:
                    y_text = y2 - 10

                cv2.putText(
                    frame,
                    owner_text,
                    (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            # ----- 8) Mostrar el frame en una ventana -----
            cv2.imshow("Sistema de detección de matrículas (Webcam)", frame)

            # Salir del bucle cuando se presione la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Liberamos recursos siempre, incluso si hay errores
        cap.release()
        cv2.destroyAllWindows()
        db.close()
        print("Cámara y conexión a BD cerradas.")


def main() -> None:
    """
    Función main.

    Llama a run_on_webcam con la cámara por defecto (índice 0).
    Si el equipo tiene varias cámaras, se puede cambiar el índice.
    """
    run_on_webcam(camera_index=0)


if __name__ == "__main__":
    main()
