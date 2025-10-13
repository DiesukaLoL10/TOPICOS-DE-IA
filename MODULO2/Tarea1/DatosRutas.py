import pandas as pd
import numpy as np

class DatosRutas:
    def __init__(self, ruta_distancias, ruta_combustible, ruta_ubicaciones):
        self.ruta_distancias = ruta_distancias
        self.ruta_combustible = ruta_combustible
        self.ruta_ubicaciones = ruta_ubicaciones
        self.matriz_distancias = None
        self.matriz_combustible = None
        self.ubicaciones = None
        self.mapa_nombres = {}

    def cargar(self):
        """Carga las matrices y crea el mapa de nombres."""
        self.matriz_distancias = pd.read_csv(self.ruta_distancias, index_col=0).astype(float).values
        self.matriz_combustible = pd.read_csv(self.ruta_combustible, index_col=0).astype(float).values
        self.ubicaciones = pd.read_csv(self.ruta_ubicaciones)

        for i, fila in self.ubicaciones.iterrows():
            self.mapa_nombres[i + 1] = fila["Nombre"]

    def inicializar_rutas(self, num_vehiculos, centros_idx):
        """Distribuye las tiendas entre los vehículos."""
        num_nodos = self.matriz_distancias.shape[0]
        tiendas_idx = list(range(max(centros_idx) + 1, num_nodos + 1))
        np.random.shuffle(tiendas_idx)

        rutas = np.array_split(tiendas_idx, num_vehiculos)
        for i in range(num_vehiculos):
            rutas[i] = [centros_idx[i]] + list(rutas[i]) + [centros_idx[i]]
        return rutas