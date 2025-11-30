
"""
Módulo search_vehicle

Define la clase SearchVehicle, encargada de consultar la base de datos
para obtener la información de un vehículo y su propietario a partir
del texto de la placa detectada por el sistema de visión + OCR.
"""

from Db_conector import MySQLDatabase


class SearchVehicle:
    """
    Clase SearchVehicle

    Esta clase encapsula las consultas relacionadas con vehículos y propietarios.

    Se apoya en MySQLDatabase para ejecutar las consultas SQL.
    """

    def __init__(self, db: MySQLDatabase):
        """
        Constructor.

        Parámetros:
        - db: instancia de MySQLDatabase ya configurada (host, usuario, etc.).
              SearchVehicle no crea la conexión, solo la reutiliza.
        """
        self.db = db

    def find_owner_by_plate(self, plate: str):
        """
        Busca en la base de datos la información del vehículo y propietario
        a partir de la placa.

        Flujo:
        1. Normaliza la placa (quita espacios y la pasa a mayúsculas) para evitar
           problemas por formato o minúsculas.
        2. Ejecuta un SELECT con JOIN entre vehiculos y propietarios.
        3. Devuelve un diccionario con la información, o None si no existe.

        Parámetros:
        - plate: cadena de texto con la placa detectada por el OCR, por ejemplo "ABC123".

        Retorno:
        - Diccionario con datos de vehículo y propietario (placa, marca, modelo, año,
          nombre, teléfono, email).
        - None si no se encontró ningún registro.
        """
        if not plate:
            return None

        # Normalizamos: sin espacios y en mayúsculas
        normalized_plate = plate.strip().upper().replace(" ", "")

        # Consulta SQL: ajusta los nombres de columnas/tablas a tu esquema real
        query = """
            SELECT 
                v.placa,
                v.marca,
                v.modelo,
                v.anio,
                p.id_propietario,
                p.nombre,
                p.telefono,
                p.email
            FROM vehiculos v
            JOIN propietarios p ON v.id_propietario = p.id_propietario
            -- REPLACE y UPPER para comparar independientemente de espacios o mayúsculas
            WHERE REPLACE(UPPER(v.placa), ' ', '') = %s
        """

        results = self.db.execute_query(query, (normalized_plate,))
        return results[0] if results else None
