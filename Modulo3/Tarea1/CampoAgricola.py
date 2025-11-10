# ==========================================================
# OPTIMIZACIÓN DE COLOCACIÓN DE SENSORES DE HUMEDAD CON PSO
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps


class CampoAgricola:
    def __init__(self, tamaño=100, semilla=20):
        """
        Simula un campo agrícola con variables ambientales:
        - topografía (elevación)
        - humedad del suelo
        - calidad del suelo
        """
        np.random.seed(semilla)
        self.tamaño = tamaño
        self.topografia = np.random.normal(loc=25, scale=5, size=(tamaño, tamaño))
        self.humedad = np.random.uniform(low=0.2, high=0.9, size=(tamaño, tamaño))
        self.suelo = np.random.uniform(low=0.5, high=1.0, size=(tamaño, tamaño))
    
    def obtener_valores(self, x, y):
        """Devuelve humedad y calidad del suelo en una posición (x, y)."""
        x = int(np.clip(x, 0, self.tamaño - 1))
        y = int(np.clip(y, 0, self.tamaño - 1))
        return self.humedad[x][y], self.suelo[x][y]

# ----------------------------------------------------------
# Función objetivo (fitness)
# ----------------------------------------------------------

def evaluar_configuracion(posiciones, campo, n_sensores):
    """
    Calcula el valor de aptitud de cada configuración de sensores.
    Premia:
        - Cobertura de zonas con variabilidad de humedad
        - Suelo de buena calidad
    Penaliza:
        - Sensores muy cercanos entre sí
    """
    fitness = []
    for p in posiciones:
        coords = p.reshape(n_sensores, 2)

        distancias = []
        for i in range(n_sensores):
            for j in range(i+1, n_sensores):
                d = np.linalg.norm(coords[i] - coords[j])
                distancias.append(d)
        penalizacion_cercania = np.sum(np.exp(-np.array(distancias)/10))  # penaliza cercanos

        humedad_total = 0
        suelo_total = 0
        for (x, y) in coords:
            h, s = campo.obtener_valores(x, y)
            humedad_total += h
            suelo_total += s

        var_humedad = np.var([campo.obtener_valores(x, y)[0] for x, y in coords])

        score = (humedad_total + var_humedad + suelo_total) - 0.4 * penalizacion_cercania
        fitness.append(-score)  
    return np.array(fitness)

# ----------------------------------------------------------
# Función principal
# ----------------------------------------------------------

def optimizar_sensores():
    campo = CampoAgricola(tamaño=100)

    N_SENSORES = 10
    DIMENSIONES = N_SENSORES * 2
    LIM_INF = np.zeros(DIMENSIONES)
    LIM_SUP = np.ones(DIMENSIONES) * campo.tamaño
    BOUNDS = (LIM_INF, LIM_SUP)

    OPCIONES = {
        'c1': 2.0,
        'c2': 2.0,
        'w': 0.7
    }

    def fitness_wrapper(pos):
        return evaluar_configuracion(pos, campo, N_SENSORES)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=50,
        dimensions=DIMENSIONES,
        options=OPCIONES,
        bounds=BOUNDS
    )

    mejor_costo, mejor_pos = optimizer.optimize(fitness_wrapper, iters=150)

    print("\n========== RESULTADOS ==========")
    print(f"Mejor costo encontrado: {mejor_costo:.4f}")
    print("Coordenadas óptimas de sensores (x, y):")
    print(mejor_pos.reshape(N_SENSORES, 2))

    coords_opt = mejor_pos.reshape(N_SENSORES, 2)


    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axs[0].imshow(campo.humedad, cmap='Blues', origin='lower')
    axs[0].scatter(coords_opt[:, 0], coords_opt[:, 1], color='red', marker='x', s=80)
    axs[0].set_title('Distribución de Humedad')
    axs[0].set_xlabel('Coordenada X (m)')
    axs[0].set_ylabel('Coordenada Y (m)')
    plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04, label='Nivel de Humedad')

    im2 = axs[1].imshow(campo.suelo, cmap='YlOrBr', origin='lower')
    axs[1].scatter(coords_opt[:, 0], coords_opt[:, 1], color='blue', marker='x', s=80)
    axs[1].set_title('Calidad del Suelo')
    axs[1].set_xlabel('Coordenada X (m)')
    plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04, label='Índice de Calidad del Suelo')

    im3 = axs[2].imshow(campo.topografia, cmap='terrain', origin='lower')
    axs[2].scatter(coords_opt[:, 0], coords_opt[:, 1], color='red', marker='x', s=80)
    axs[2].set_title('Topografía del Terreno')
    axs[2].set_xlabel('Coordenada X (m)')
    plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04, label='Elevación (m)')

    plt.suptitle('Distribución Óptima de Sensores de Humedad — Campo Agrícola', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    optimizar_sensores()
