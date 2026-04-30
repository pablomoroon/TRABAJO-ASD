#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif


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


void updateGridMPI() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > HEIGHT) {
        if (rank == 0) fprintf(stderr, "Error: demasiados procesos (%d) para HEIGHT=%d\n", size, HEIGHT);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int base = HEIGHT / size;
    int rem = HEIGHT % size;
    int filas = base + (rank < rem ? 1 : 0);

    uint8_t localGrid[filas + 2][WIDTH];
    uint8_t newLocalGrid[filas + 2][WIDTH];

    MPI_Status status;

    // Aqui si rank == 0, cada proceso recibe su parte de la rejilla global
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            int rows_p = base + (p < rem ? 1 : 0); // filas para proceso p teniendo en cuenta el resto
            int offset = p * base + (p < rem ? p : rem); // desplazamiento en la rejilla global para proceso p
            MPI_Send(&grid[offset][0], rows_p * WIDTH, MPI_UNSIGNED_CHAR, p, 0, MPI_COMM_WORLD);
        }
        for (int i = 0; i < filas; i++)
            for (int j = 0; j < WIDTH; j++)
                localGrid[i + 1][j] = grid[i][j];
    } else {
        MPI_Recv(&localGrid[1][0], filas * WIDTH, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
    }

    // Aqui se hace el intercambio de filas frontera entre procesos 
    int up = (rank == 0) ? size - 1 : rank - 1; // vecino de arriba 
    int down = (rank == size - 1) ? 0 : rank + 1; // vecino de abajo 

    MPI_Send(&localGrid[1][0], WIDTH, MPI_UNSIGNED_CHAR, up, 1, MPI_COMM_WORLD); // enviar fila superior a vecino de arriba
    MPI_Recv(&localGrid[filas + 1][0], WIDTH, MPI_UNSIGNED_CHAR, down, 1, MPI_COMM_WORLD, &status); // recibir fila inferior de vecino de abajo

    MPI_Send(&localGrid[filas][0], WIDTH, MPI_UNSIGNED_CHAR, down, 2, MPI_COMM_WORLD); // enviar fila inferior a vecino de abajo
    MPI_Recv(&localGrid[0][0], WIDTH, MPI_UNSIGNED_CHAR, up, 2, MPI_COMM_WORLD, &status); // recibir fila superior de vecino de arriba

    // COMPUTO LOCAL Aqui se calcula el siguiente estado para cada celda del bloque local
#if defined(_OPENMP)
#pragma omp parallel for default(none) shared(localGrid, newLocalGrid, filas) schedule(static)
#endif
    for (int i = 1; i <= filas; i++) {

        int j;

        //DESENROLLADO para procesar 2 celdas por iteracion
        for (j = 0; j < WIDTH - 1; j += 2) {

            int im1 = i - 1;
            int ip1 = i + 1;

            int jm1 = (j == 0) ? WIDTH - 1 : j - 1;
            int jp1 = (j == WIDTH - 1) ? 0 : j + 1;

            int count1 =
                localGrid[im1][jm1] + localGrid[im1][j] + localGrid[im1][jp1] +
                localGrid[i][jm1]   +                     localGrid[i][jp1] +
                localGrid[ip1][jm1] + localGrid[ip1][j] + localGrid[ip1][jp1];

            newLocalGrid[i][j] =
                (localGrid[i][j] == 1)
                ? ((count1 == 2 || count1 == 3) ? 1 : 0)
                : (count1 == 3);

            // Segunda celda (desenrollado)
            int jm1_2 = (j + 1 == 0) ? WIDTH - 1 : j;
            int jp1_2 = (j + 1 == WIDTH - 1) ? 0 : j + 2;

            int count2 =
                localGrid[im1][jm1_2] + localGrid[im1][j + 1] + localGrid[im1][jp1_2] +
                localGrid[i][jm1_2]   +                        localGrid[i][jp1_2] +
                localGrid[ip1][jm1_2] + localGrid[ip1][j + 1] + localGrid[ip1][jp1_2];

            newLocalGrid[i][j + 1] =
                (localGrid[i][j + 1] == 1)
                ? ((count2 == 2 || count2 == 3) ? 1 : 0)
                : (count2 == 3);
        }

        // resto
        for (; j < WIDTH; j++) {

            int im1 = i - 1;
            int ip1 = i + 1;

            int jm1 = (j == 0) ? WIDTH - 1 : j - 1;
            int jp1 = (j == WIDTH - 1) ? 0 : j + 1;

            int count =
                localGrid[im1][jm1] + localGrid[im1][j] + localGrid[im1][jp1] +
                localGrid[i][jm1]   +                     localGrid[i][jp1] +
                localGrid[ip1][jm1] + localGrid[ip1][j] + localGrid[ip1][jp1];

            newLocalGrid[i][j] =
                (localGrid[i][j] == 1)
                ? ((count == 2 || count == 3) ? 1 : 0)
                : (count == 3);
        }
    }

    // Recolección de resultados: el proceso 0 recopila los bloques locales
    if (rank == 0) {
        // copiar bloque local al inicio
        for (int i = 0; i < filas; i++)
            for (int j = 0; j < WIDTH; j++)
                grid[i][j] = newLocalGrid[i + 1][j];

        for (int p = 1; p < size; p++) {
            int rows_p = base + (p < rem ? 1 : 0);
            int offset = p * base + (p < rem ? p : rem);
            MPI_Recv(&grid[offset][0], rows_p * WIDTH, MPI_UNSIGNED_CHAR, p, 3, MPI_COMM_WORLD, &status);
        }
    } else {
        MPI_Send(&newLocalGrid[1][0], filas * WIDTH, MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD);
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
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

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
        updateGridMPI();
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
        updateGridMPI();
        usleep(SPEED * 1000);
    }
#endif

    MPI_Finalize();
    return 0;
}
