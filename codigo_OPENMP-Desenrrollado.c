/***    ___ _  _  ___ _   _   _ ___  ___ 
 *     |_ _| \| |/ __| | | | | |   \| __|
 *      | || .` | (__| |_| |_| | |) | _| 
 *     |___|_|\_|\___|____\___/|___/|___|   */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif


/***    ___  ___ ___ ___ _  _ ___ 
 *     |   \| __| __|_ _| \| | __|
 *     | |) | _|| _| | || .` | _| 
 *     |___/|___|_| |___|_|\_|___|          */
// Dimensiones y parámetros
#define WIDTH 1000
#define HEIGHT 1000
#define ITERATION 500
#define SPEED 500
#define ALIVE '#'
#define DEAD '.'

// Rejilla: 0 = muerto, 1 = vivo (usar uint8_t mejora locality/casts)
static uint8_t grid[HEIGHT][WIDTH];
int generation = 0;

/**
 * @brief Inicializa la rejilla con valores aleatorios (vivo/muerto).
 *
 * Esta función rellena la rejilla con valores 0 (muerto) o 1 (vivo).
 * Se usa para inicializar el tablero al comienzo del Juego de la Vida.
 */
void initGrid() {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            grid[i][j] = rand() & 1; // faster equivalence a rand()%2
        }
    }
}

/**
 * @brief Imprime el estado actual de la rejilla en la consola.
 *
 * Limpia la pantalla e imprime la rejilla usando `ALIVE` para células
 * vivas y `DEAD` para células muertas. Esta función es para uso
 * interactivo; en benchmarks active `-DBENCHMARK` para deshabilitarla.
 */
void printGrid() {
    system("clear");
    //system("cls"); //PARA COMPILAR EN LINUX
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%c", grid[i][j] ? ALIVE : DEAD);
        }
        printf("\n");
    }
}

/**
 * @brief Cuenta los vecinos vivos de una celda dada.
 *
 * Devuelve el número de vecinos vivos en la vecindad Moore (8 vecinos).
 * Se evita usar el operador módulo (%) por ser costoso; en su lugar
 * se manejan explícitamente los índices frontera (wrap-around).
 *
 * @param i Fila de la celda.
 * @param j Columna de la celda.
 * @return Número de vecinos vivos.
 */
static inline int countAliveNeighbours(int i, int j) {
    int count = 0;

    int im1 = (i == 0) ? HEIGHT - 1 : i - 1;
    int ip1 = (i == HEIGHT - 1) ? 0 : i + 1;
    int jm1 = (j == 0) ? WIDTH - 1 : j - 1;
    int jp1 = (j == WIDTH - 1) ? 0 : j + 1;

    count += grid[im1][jm1];
    count += grid[im1][j];
    count += grid[im1][jp1];
    count += grid[i][jm1];
    count += grid[i][jp1];
    count += grid[ip1][jm1];
    count += grid[ip1][j];
    count += grid[ip1][jp1];

    return count;
}

/**
 * @brief Actualiza la rejilla según las reglas del Juego de la Vida.
 *
 * Crea una rejilla temporal `newGrid`, calcula el siguiente estado
 * para cada celda y copia el resultado de vuelta. Contiene una
 * directiva OpenMP opcional para paralelizar por filas.
 */
void updateGrid() {
    uint8_t newGrid[HEIGHT][WIDTH];

#if defined(_OPENMP)
#pragma omp parallel for default(none) shared(grid, newGrid) schedule(static)
#endif
    for (int i = 0; i < HEIGHT; i++) {

        int j;

        // Desenrollado de bucle: procesamos 2 celdas por iteración
        for (j = 0; j < WIDTH - 1; j += 2) {
            int count1 = countAliveNeighbours(i, j);
            newGrid[i][j] =
                (grid[i][j] == 1)
                ? ((count1 == 2 || count1 == 3) ? 1 : 0)
                : (count1 == 3);

            int count2 = countAliveNeighbours(i, j + 1);
            newGrid[i][j + 1] =
                (grid[i][j + 1] == 1)
                ? ((count2 == 2 || count2 == 3) ? 1 : 0)
                : (count2 == 3);
        }

        // Por si WIDTH es impar
        for (; j < WIDTH; j++) {
            int count = countAliveNeighbours(i, j);
            newGrid[i][j] =
                (grid[i][j] == 1)
                ? ((count == 2 || count == 3) ? 1 : 0)
                : (count == 3);
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for default(none) shared(grid, newGrid) schedule(static)
#endif
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            grid[i][j] = newGrid[i][j];
        }
    }
}

/**
 * @brief Punto de entrada principal del Juego de la Vida.
 *
 * Inicializa la rejilla y ejecuta `ITERATION` iteraciones. En modo
 * interactivo imprime la rejilla y espera `SPEED` milisegundos entre
 * iteraciones; en modo benchmark (compilar con `-DBENCHMARK`) se omite
 * la visualización y la espera para medir el rendimiento real.
 */
int main() {
    srand(42);
    initGrid();

#ifdef BENCHMARK
    #ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        printf("Numero de hilos: %d\n", omp_get_num_threads());
    }

    double t1 = omp_get_wtime();
    #else
    double t1 = (double)clock() / CLOCKS_PER_SEC;
    #endif

    for (int it = 0; it < ITERATION; ++it) {
        updateGrid();
    }

    #ifdef _OPENMP
    double t2 = omp_get_wtime();
    #else
    double t2 = (double)clock() / CLOCKS_PER_SEC;
    #endif

    printf("Tiempo de ejecucion: %f segundos\n", t2 - t1);

#else
    for (int i = 0; i < ITERATION; i++) {
        printGrid();
        updateGrid();
        usleep(SPEED * 1000);
    }
#endif

    return 0;
}
