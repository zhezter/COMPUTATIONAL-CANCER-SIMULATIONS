from manim import *
import numpy as np

"""
ESTE PROGRAMA DISEÑA UNA ANIMACIÓN ESTOCÁSTICA PARA SIMULAR LAS DINÁMICAS DE COMPETENCIA DE CÉLULAS SANAS Y CANCEROSAS DE  UN AC
PARA EL PROYECTO DE AULA DE CÁLCULO DIFERENCIAL.
"""

# El programa está diseñado con librerías de Python: ManimCE (Community Edition) y NumPy.
# Por lo que; se necesitan sus instalaciones y dependencias.

""" LIBRERÍA |    INSTALACIÓN      | DEPENDENCIAS """

""" MANIMCE  | pip install manimce | https://docs.manim.community/en/stable/installation.html """
""" NUMPY    | pip install numpy   | *pip las descarga en automático* """


# CONTEXTOS
# ----------------------------------------------------------------
# AC = Autómata Celular
# Cell/s = célula/s
# EMPTY = Vacío (Estado 0)
# HEALTHY = Sano (Estado 1)
# CANCEROUS = Canceroso (Estado 2)
# NECROTIC = Necrótico (Estado 3)
# mobject (manim) = Mathematical Object
# Population/s = Poblacion/es
# Grid = Grilla - Cuadrícula

"""
CONFIGURACIONES INICIALS
"""

# Simplemente dar un valor númerico a los diferentes estados que puede tomar una celda, el llamado "Alfabeto"
EMPTY = 0
HEALTHY = 1
CANCEROUS = 2
NECROTIC = 3

# Configuración del AC | Cambiar los valores para obtener distintos resultados

grid_size = 20
initial_cancer_cells = [
    (10, 10)
]  # Tupla que contiene la cantidad inicial de células canserosas
initial_healthy_cells = [
    (i, j) for i in range(8, 13) for j in range(8, 13) if (i, j) != (10, 10)
]  # ' ' ' ' sanas

""" FUNCIONES """


def initialize_grid():  # Comenzar la cuadrícula con una distibucion de celulas
    grid = np.zeros((grid_size, grid_size), dtype=int)
    for i, j in initial_cancer_cells:
        grid[i, j] = CANCEROUS
    for i, j in initial_healthy_cells:
        grid[i, j] = HEALTHY
    return grid


P_mcs = 0.05  # Probabilidad de muerte de células sanas por tratamiento (10%)
P_mcc = 0.10  # Probabilidad de muerte de células cancerosas por tratamiento (20%)


def update_grid(grid: np.ndarray) -> np.ndarray:
    """
    Actualiza la cuadrícula según las reglas de transición locales del AC y determina el estado siguiente del AC basado en el estado actual.

    Parámetros:
    `grid` (np.ndarray): El estado actual del AC.

    Devuelve:
    np.ndarray: Un nuevo array 2D de numpy que representa el siguiente estado del AC.
    """
    new_grid = grid.copy()  # Se guarda una copia del estado del AC en el instante

    for i in range(grid_size):  # Recorrer la grilla en i
        for j in range(grid_size):  # Recorrer la grilla en j
            if grid[i, j] == CANCEROUS:
                # Regla de necrosis
                if np.random.rand() < P_mcc:
                    new_grid[i, j] = EMPTY
                else:
                    # Vecindad de Moore (ni: neighboordhood i, nj: neighboordhood j)
                    for ni, nj in [
                        (i - 1, j),
                        (i + 1, j),
                        (i, j - 1),
                        (i, j + 1),
                        (i - 1, j - 1),
                        (i - 1, j + 1),
                        (i + 1, j - 1),
                        (i + 1, j + 1),
                    ]:
                        if (
                            0 <= ni < grid_size
                            and 0 <= nj < grid_size
                            and grid[ni, nj] == HEALTHY
                            and np.random.rand() < 0.25
                        ):
                            # Regla de propagación del cáncer
                            new_grid[ni, nj] = CANCEROUS
                    # Regla de necrosis secundaria
                    if np.random.rand() < 0.15:
                        new_grid[i, j] = NECROTIC
            elif grid[i, j] == HEALTHY and np.random.rand() < P_mcs:
                new_grid[i, j] = EMPTY
            elif grid[i, j] == EMPTY and np.random.rand() < 0.05:
                new_grid[i, j] = HEALTHY

    return new_grid  # Devuelve el nuevo estado del AC


def count_cells(grid):
    """
    Esta Función toma el AC y devuelve un Diccionario de pares clave-valor
    """

    unique, counts = np.unique(grid, return_counts=True)
    return dict(zip(unique, counts))


# Inicializar conteo de células
def initialize_counts(grid):
    counts = count_cells(grid)
    cell_counts = {
        EMPTY: [counts.get(EMPTY, 0)],
        HEALTHY: [counts.get(HEALTHY, 0)],
        CANCEROUS: [counts.get(CANCEROUS, 0)],
        NECROTIC: [counts.get(NECROTIC, 0)],
    }
    return cell_counts


def get_grid_mobject(grid):
    colors = {
        EMPTY: "#d3d3d3",
        HEALTHY: "#adf30f",
        CANCEROUS: "#ff4500",
        NECROTIC: "#4b0082",
    }
    cell_size = 0.3
    grid_mobject = VGroup()  # Inicializamos un grupo de Mobjects vectorizados

    for i in range(grid_size):
        for j in range(grid_size):
            cell_color = colors[grid[i, j]]
            cell = (
                Square(side_length=cell_size)
                .set_fill(color=cell_color, opacity=1)
                .set_stroke(color=WHITE, width=0.5)
            )
            cell.move_to(
                np.array([i - grid_size / 2, j - grid_size / 2, 0]) * cell_size
            )
            grid_mobject.add(cell)
    return grid_mobject


def get_population_plot_mobject(cell_counts, plot):
    colors = {
        EMPTY: "#d3d3d3",
        HEALTHY: "#adf30f",
        CANCEROUS: "#ff4500",
        NECROTIC: "#4b0082",
    }

    time_points = list(range(len(cell_counts[EMPTY])))
    plots = []
    for cell_type in [EMPTY, HEALTHY, CANCEROUS, NECROTIC]:
        population_points = cell_counts[cell_type]
        plot_points = [
            (time, population)
            for time, population in zip(time_points, population_points)
        ]
        plots.append(
            plot.plot_line_graph(
                x_values=[p[0] for p in plot_points],
                y_values=[p[1] for p in plot_points],
                add_vertex_dots=False,
                line_color=colors[cell_type],
            )
        )
    return VGroup(*plots)


class CellGridWithPlot(Scene):
    def construct(self):
        # Inicializar la cuadrícula y el conteo de células
        grid = initialize_grid()
        cell_counts = initialize_counts(grid)

        # Inicializar la animación
        grid_mobject = get_grid_mobject(grid).to_edge(LEFT)
        self.add(grid_mobject)

        # Creamos los ejes donde graficaremos las diferentes poblaciones por paso de tiempo
        plot = (
            Axes(
                x_range=[0, 50, 2],
                y_range=[0, 400, 50],
                axis_config={"include_numbers": True},
                x_axis_config={"include_tip": False, "label_direction": DOWN},
                y_axis_config={"include_tip": False, "label_direction": LEFT},
            )
            .scale(0.5)
            .to_edge(RIGHT)
        )
        plot.add(plot.get_axis_labels(x_label="Tiempo", y_label="Colonia"))
        plot_mobject = get_population_plot_mobject(cell_counts, plot)
        self.add(plot, plot_mobject)

        for step in range(50):
            grid = update_grid(grid)
            counts = count_cells(grid)
            for key in cell_counts:
                cell_counts[key].append(counts.get(key, 0))
            new_grid_mobject = get_grid_mobject(grid).to_edge(LEFT)
            new_plot_mobject = get_population_plot_mobject(cell_counts, plot)
            self.play(
                Transform(grid_mobject, new_grid_mobject),
                Transform(plot_mobject, new_plot_mobject),
                run_time=0.5,
            )
            grid_mobject = new_grid_mobject
            plot_mobject = new_plot_mobject


# Renderizado de la Animación

if __name__ == "__main__":
    config.pixel_height = 1080
    config.pixel_width = 1920
    config.frame_rate = 30
    config.output_file = "cell_automaton_with_plot.mp4"
    scene = CellGridWithPlot()
    scene.render()

# Encender una terminal con Control + ñ y a continuación usar el comando:
# manim .\with_treatment_cellular_automaton.py CellGridWithPlot -pqh --disable_caching

# O si desea solo ver el final de la simulación en una imágen:
# manim .\with_treatment_cellular_automaton.py CellGridWithPlotNoTreatment -ps
