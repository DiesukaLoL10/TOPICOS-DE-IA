from __future__ import annotations
import random
import math
from typing import List, Tuple

class Ciudad:
    """Representa una ciudad con coordenadas (x, y)."""
    def __init__(self, x: float, y: float, name: str = None):
        self.x = float(x)
        self.y = float(y)
        self.name = name if name else f"({x},{y})"

    def distancia(self, other: "Ciudad") -> float:
        """Distancia Euclidiana entre esta ciudad y otra."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.hypot(dx, dy)

    def __repr__(self):
        return f"{self.name}"

class Aptitud:
    """Calcula distancia de ruta y aptitud."""
    def __init__(self, route: List[Ciudad]):
        self.route = route
        self._distancia = None
        self._fitness = None

    def distancia(self) -> float:
        """Devuelve la distancia total del ciclo (vuelta al inicio)."""
        if self._distancia is None:
            total = 0.0
            for i in range(len(self.route)):
                start = self.route[i]
                end = self.route[(i + 1) % len(self.route)]
                total += start.distancia(end)
            self._distancia = total
        return self._distancia

    def fitness(self) -> float:
        """Aptitud: inversa de la distancia. Maneja distancia 0 por seguridad."""
        if self._fitness is None:
            d = self.distancia()
            self._fitness = 1.0 / d if d > 0 else float("inf")
        return self._fitness

def crear_ruta(ciudades: List[Ciudad]) -> List[Ciudad]:
    """Crear una ruta aleatoria (permutación de ciudades)."""
    return random.sample(ciudades, len(ciudades))

def poblacion_inicial(tamano_poblacion: int, ciudades: List[Ciudad]) -> List[List[Ciudad]]:
    """Genera una población inicial de rutas."""
    return [crear_ruta(ciudades) for _ in range(tamano_poblacion)]

def rank_rutas(poblacion: List[List[Ciudad]]) -> List[Tuple[int, float]]:
    """Devuelve lista de pares (index, fitness) ordenada por fitness descendente."""
    fitness_results = [(i, Aptitud(route).fitness()) for i, route in enumerate(poblacion)]
    return sorted(fitness_results, key=lambda x: x[1], reverse=True)

def seleccion(pop_ranked: List[Tuple[int, float]], tam_elite: int) -> List[int]:
    """Selecciona rutas para el apareamiento usando elitismo y ruleta."""
    resultados = []
    for i in range(tam_elite):
        resultados.append(pop_ranked[i][0])

    fitness_sum = sum([f for _, f in pop_ranked])
    cumulative = []
    acc = 0.0
    for _, f in pop_ranked:
        acc += f
        cumulative.append(acc / fitness_sum)

    for _ in range(len(pop_ranked) - tam_elite):
        r = random.random()
        for idx, cum_prob in enumerate(cumulative):
            if r <= cum_prob:
                resultados.append(pop_ranked[idx][0])
                break
    return resultados

def pool_de_apareamiento(poblacion: List[List[Ciudad]], resultados: List[int]) -> List[List[Ciudad]]:
    """Construye el pool de apareamiento usando los índices seleccionados."""
    return [poblacion[i] for i in resultados]

def crossover_ordenado(parent1: List[Ciudad], parent2: List[Ciudad]) -> List[Ciudad]:
    """
    Realiza crossover ordenado entre dos padres para producir un hijo.
    """
    child = [None] * len(parent1)
    start = random.randint(0, len(parent1) - 1)
    end = random.randint(start, len(parent1) - 1)

    for i in range(start, end + 1):
        child[i] = parent1[i]

    p2_idx = 0
    for i in range(len(child)):
        if child[i] is None:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
            p2_idx += 1
    return child

def cruzar_poblacion(matingpool: List[List[Ciudad]], elite_size: int) -> List[List[Ciudad]]:
    """Genera la próxima generación mediante crossover."""
    children = []
    length = len(matingpool) - elite_size
    for i in range(elite_size):
        children.append(matingpool[i])

    pool = random.sample(matingpool, len(matingpool))
    for i in range(length):
        child = crossover_ordenado(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

def mutacion(individual: List[Ciudad], mutation_rate: float) -> List[Ciudad]:
    """Swap mutation: intercambia dos genes con probabilidad mutation_rate por posición."""
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = random.randint(0, len(individual) - 1)
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual

def mutar_poblacion(poblacion: List[List[Ciudad]], tasa_mutacion: float) -> List[List[Ciudad]]:
    """Aplica mutación a toda la población."""
    return [mutacion(ind[:], tasa_mutacion) for ind in poblacion]  # copiar antes de mutar

def siguiente_generacion(poblacion_actual: List[List[Ciudad]], tam_elite: int, tasa_mutacion: float) -> List[List[Ciudad]]:
    """Genera la siguiente generación completa."""
    pop_ranked = rank_rutas(poblacion_actual)
    selection_results = seleccion(pop_ranked, tam_elite)
    matingpool = pool_de_apareamiento(poblacion_actual, selection_results)
    children = cruzar_poblacion(matingpool, tam_elite)
    next_gen = mutar_poblacion(children, tasa_mutacion)
    return next_gen

def algoritmo_genetico(ciudades: List[Ciudad],
                      tam_poblacion: int = 100,
                      tam_elite: int = 20,
                      tasa_mutacion: float = 0.01,
                      generaciones: int = 500,
                      verbose: bool = True) -> Tuple[List[Ciudad], float]:
    """Ejecuta el AG y devuelve la mejor ruta encontrada y su distancia."""
    pop = poblacion_inicial(tam_poblacion, ciudades)
    if verbose:
        initial_distance = 1 / rank_rutas(pop)[0][1]
        print(f"Distancia inicial: {initial_distance:.4f}")

    for i in range(generaciones):
        pop = siguiente_generacion(pop, tam_elite, tasa_mutacion)
        if verbose and (i + 1) % max(1, generaciones // 30) == 0:
            best_dist = 1 / rank_rutas(pop)[0][1]
            print(f"Generación {i+1} / {generaciones} - Mejor distancia: {best_dist:.4f}")

    best_index = rank_rutas(pop)[0][0]
    best_route = pop[best_index]
    best_distance = 1 / rank_rutas(pop)[0][1]
    if verbose:
        print(f"Distancia final: {best_distance:.4f}")
    return best_route, best_distance


if __name__ == "__main__":
    import random
    random.seed(46)
    ciudades_example = [Ciudad(random.uniform(0,100), random.uniform(0,100), f"C{i}") for i in range(15)]
    best_route, best_dist = algoritmo_genetico(ciudades_example, tam_poblacion=100, tam_elite=20,
                                              tasa_mutacion=0.02, generaciones=200, verbose=True)
    print("Mejor distancia encontrada:", best_dist)
