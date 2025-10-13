import numpy as np
import math

class RecocidoSimulado:
    def __init__(self, distancias, combustible, rutas_iniciales,
                 T_inicial=100, T_min=0.01, alpha=0.995, iteraciones=200):
        self.distancias = distancias
        self.combustible = combustible
        self.rutas = rutas_iniciales
        self.T = T_inicial
        self.T_min = T_min
        self.alpha = alpha
        self.iteraciones = iteraciones

    def calcular_costo(self, rutas, alpha=1.0, beta=1.0, epsilon=0.1):
        """
        Calcula el costo total de un conjunto de rutas considerando:
        - Peso de la distancia (alpha)
        - Peso del combustible (beta)
        - Penalización adicional por tramos largos/costosos (epsilon)
    
        Este enfoque permite que swaps que reduzcan tramos muy costosos
        tengan un impacto más visible en el costo total.
        """
        costo_total = 0

        for ruta in rutas:
            origenes = np.array(ruta[:-1]) - 1
            destinos = np.array(ruta[1:]) - 1

            for o, d in zip(origenes, destinos):
                distancia = self.distancias[o, d]
                combustible = self.combustible[o, d]

                costo_tramo = alpha * distancia + beta * combustible

                costo_tramo += epsilon * (distancia + combustible)

                costo_total += costo_tramo

        return costo_total

    def generar_vecino(self, rutas):
        """Genera un vecino intercambiando una tienda entre dos rutas (misma lógica, distinto enfoque)."""
        nuevas_rutas = [r[:] for r in rutas]

        indices = np.arange(len(nuevas_rutas))
        np.random.shuffle(indices)
        r1, r2 = indices[:2]

        ruta1, ruta2 = nuevas_rutas[r1], nuevas_rutas[r2]

        if len(ruta1) > 2 and len(ruta2) > 2:
            i = np.random.randint(1, len(ruta1) - 1)
            j = np.random.randint(1, len(ruta2) - 1)

            temp = ruta1[i]
            ruta1[i] = ruta2[j]
            ruta2[j] = temp

            nuevas_rutas[r1], nuevas_rutas[r2] = ruta1, ruta2

        return nuevas_rutas

    def recocidoSimulado(self):
        """Ejecuta el proceso de recocido."""
        actual = self.rutas
        mejor = self.rutas
        costo_actual = self.calcular_costo(actual)
        costo_mejor = costo_actual
        sin_mejora = 0

        while self.T > self.T_min:
            for _ in range(self.iteraciones):
                vecino = self.generar_vecino(actual)
                costo_vecino = self.calcular_costo(vecino)
                delta = costo_vecino - costo_actual

                if delta < 0 or np.random.rand() < math.exp(-delta / self.T):
                    actual = vecino
                    costo_actual = costo_vecino
                    if costo_actual < costo_mejor:
                        mejor = actual
                        costo_mejor = costo_actual
                        sin_mejora = 0
                    else:
                        sin_mejora += 1

            if sin_mejora > 500:
                self.T *= 0.9
                sin_mejora = 0
            else:
                self.T *= self.alpha

            print(f"Temperatura: {self.T:.4f} | Mejor Costo: {costo_mejor:.2f}")

        return mejor, costo_mejor