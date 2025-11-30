
"""
Módulo Db_conector

Encapsula la lógica de conexión a la base de datos MySQL que corre en XAMPP.
Se utiliza la librería mysql-connector-python para conectarse al servidor,
ejecutar consultas (SELECT) y operaciones de escritura (INSERT/UPDATE/DELETE).

Esta clase se usa en todo el proyecto como "puente" entre Python y la BD.
"""

import mysql.connector
from mysql.connector import Error


class MySQLDatabase:
    """
    Clase MySQLDatabase

    Responsable de abrir y cerrar la conexión con MySQL, así como ejecutar
    consultas de lectura y escritura.

    Parámetros del constructor:
    - host: dirección del servidor MySQL, normalmente "localhost" en XAMPP.
    - port: puerto del servidor MySQL, por defecto 3306.
    - user: usuario de la base de datos (por ejemplo "root" en XAMPP).
    - password: contraseña del usuario (vacía si no configuraste una).
    - database: nombre de la base de datos donde están las tablas de placas.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "123",
        database: str = "placas_db",
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

        # Estos atributos se llenan cuando se llama a connect()
        self.conn = None   # Objeto de conexión a MySQL
        self.cursor = None # Cursor para ejecutar consultas

    def connect(self) -> None:
        """
        Abre la conexión con el servidor MySQL.

        Si la conexión es exitosa, inicializa self.conn y self.cursor.
        En caso de error, muestra el mensaje en consola.
        """
        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            # dictionary=True hace que el resultado de fetchall() sea una lista de diccionarios
            # en lugar de tuplas. Es más cómodo porque podemos acceder por nombre de columna.
            self.cursor = self.conn.cursor(dictionary=True)
            print("Conectado a MySQL correctamente.")
        except Error as e:
            print(f"Error al conectar a MySQL: {e}")
            self.conn = None
            self.cursor = None

    def close(self) -> None:
        """
        Cierra el cursor y la conexión con la base de datos.

        Es importante llamarlo al final del programa para liberar recursos.
        """
        if self.cursor is not None:
            self.cursor.close()
        if self.conn is not None:
            self.conn.close()

        self.cursor = None
        self.conn = None
        print("Conexión a la base de datos cerrada.")

    def execute_query(self, query: str, params: tuple | None = None):
        """
        Ejecuta una consulta de lectura (SELECT) y devuelve todos los resultados.

        Parámetros:
        - query: cadena SQL con la consulta SELECT.
                  Se recomienda usar placeholders (%s) para parámetros.
        - params: tupla con los valores a sustituir en los placeholders.

        Retorno:
        - Lista de diccionarios con los resultados, o lista vacía si hubo error.
        """
        if self.conn is None or self.cursor is None:
            raise RuntimeError(
                "No hay conexión a la base de datos. Llama primero a connect()."
            )

        try:
            self.cursor.execute(query, params or ())
            results = self.cursor.fetchall()
            return results
        except Error as e:
            print(f"Error al ejecutar la consulta: {e}")
            return []

    def execute_non_query(self, query: str, params: tuple | None = None) -> int:
        """
        Ejecuta una operación de escritura (INSERT, UPDATE o DELETE) y hace commit.

        Parámetros:
        - query: cadena SQL con la sentencia (INSERT/UPDATE/DELETE).
        - params: tupla con los valores para los placeholders.

        Retorno:
        - Número de filas afectadas, o 0 si hubo algún error.
        """
        if self.conn is None or self.cursor is None:
            raise RuntimeError(
                "No hay conexión a la base de datos. Llama primero a connect()."
            )

        try:
            self.cursor.execute(query, params or ())
            self.conn.commit()
            return self.cursor.rowcount
        except Error as e:
            print(f"Error al ejecutar la operación: {e}")
            # Si hay error, revertimos la transacción
            self.conn.rollback()
            return 0
