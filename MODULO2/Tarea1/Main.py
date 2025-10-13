from RecocidoSimulado import RecocidoSimulado  
from DatosRutas import DatosRutas

class SimuladorRutas:
    def __init__(self, ruta_distancias, ruta_combustible, ruta_ubicaciones, num_vehiculos=10):
        self.datos = DatosRutas(ruta_distancias, ruta_combustible, ruta_ubicaciones)
        self.num_vehiculos = num_vehiculos
        self.centros = list(range(1, num_vehiculos + 1))
        self.rutas_iniciales = None
        self.recocido = None
        self.resultado_rutas = None
        self.costo_final = None

    def ejecutar(self):
        """Carga datos, inicializa rutas y ejecuta la optimización."""
        self.datos.cargar()

        self.rutas_iniciales = self.datos.inicializar_rutas(self.num_vehiculos, self.centros)

        self.recocido = RecocidoSimulado(
            self.datos.matriz_distancias,
            self.datos.matriz_combustible,
            self.rutas_iniciales
        )

        self.resultado_rutas, self.costo_final = self.recocido.recocidoSimulado()
        self.mostrar_resultados()

    def mostrar_resultados(self):
        """Imprime las rutas optimizadas."""
        print("\n=== RESULTADOS FINALES ===\n")
        print(f"Costo total óptimo: {self.costo_final:,.2f}\n")

        for i, ruta in enumerate(self.resultado_rutas, 1):
            nombres = [self.datos.mapa_nombres[idx] for idx in ruta]
            print(f"Ruta {i:02d} -> {' -> '.join(nombres)}\n")


if __name__ == "__main__":
    simulador = SimuladorRutas(
        ruta_distancias="matriz_distancias.csv",
        ruta_combustible="matriz_costos_combustible.csv",
        ruta_ubicaciones="datos_distribucion_tiendas.csv",
        num_vehiculos=10
    )

    simulador.ejecutar()