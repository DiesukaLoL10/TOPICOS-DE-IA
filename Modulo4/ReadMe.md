<h1>Sistema de Detección de Matrículas con Visión Artificial</h1>

<p>
Este repositorio contiene un sistema modular para la detección y reconocimiento de matrículas vehiculares 
usando un modelo de visión artificial (YOLO), OCR y una base de datos MySQL.
</p>

<h2>Integrantes del equipo</h2>

<ul>
  <li><strong>Diego Alejandro Gomez Gastelum</strong> </li>
  <li><strong>Marco Josue Meza Manjarrez</strong> </li>
 
</ul>

<hr />

<h2>Estructura del repositorio</h2>

<p>La organización del proyecto está pensada para separar claramente la lógica de visión, OCR, base de datos y orquestación.</p>

<pre><code>.
├─ model/
│   └─ best.pt              # Modelo YOLO preentrenado para detección de matrículas
├─ src/
│   ├─ Db_conector.py       # Clase MySQLDatabase: conexión y operaciones con MySQL
│   ├─ search_vehicle.py    # Clase SearchVehicle: consultas de vehículo y propietario por placa
│   ├─ plateRecognition.py  # Clase PlateDetector: detección de placas mediante YOLO
│   ├─ plate_ocr.py         # Clase PlateOCR: lectura del texto de la placa usando EasyOCR
│   └─ main.py              # Orquestador principal: integra cámara, modelo, OCR y base de datos
├─ database/
│   ├─ schema.sql           # Definición del esquema de la base de datos (vehiculos, propietarios, etc.)
│   └─ datos_prueba.sql     # Datos de ejemplo para pruebas (opcional)
└─ README.md                # Descripción general del proyecto
</code></pre>

<p>
El archivo <code>main.py</code> coordina el flujo completo: captura de video, detección de placa, OCR 
y consulta a la base de datos. Cada responsabilidad está encapsulada en un módulo distinto para 
favorecer la claridad y el mantenimiento del código.
</p>

<hr />

<h2>Dependencias principales</h2>

<p>El proyecto utiliza las siguientes librerías y herramientas:</p>

<ul>
  <li><strong>Python 3.x</strong> – Lenguaje principal del proyecto.</li>
  <li><strong>Ultralytics YOLO</strong> – Para la detección de matrículas a partir de imágenes o video.</li>
  <li><strong>EasyOCR</strong> – Para el reconocimiento óptico de caracteres en la región de la placa.</li>
  <li><strong>OpenCV (opencv-python)</strong> – Para la captura de video y manejo de imágenes (frames de la cámara).</li>
  <li><strong>mysql-connector-python</strong> – Para la conexión y consultas a la base de datos MySQL.</li>
  <li><strong>MySQL (XAMPP)</strong> – Servidor de base de datos donde se almacenan vehículos y propietarios.</li>
</ul>

