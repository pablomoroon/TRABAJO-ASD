/***    ___ _  _  ___ _   _   _ ___  ___ 
 *     |_ _| \| |/ __| | | | | |   \| __|
 *      | || .` | (__| |_| |_| | |) | _| 
 *     |___|_|\_|\___|____\___/|___/|___|   */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <mpi.h>

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
static uint8_t global_grid[HEIGHT][WIDTH];
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
            global_grid[i][j] = rand() & 1; // faster equivalence a rand()%2
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
            printf("%c", global_grid[i][j] ? ALIVE : DEAD);
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
int countAliveNeighbours(const uint8_t local[][WIDTH],
        const uint8_t halo_top[WIDTH],
        const uint8_t halo_bottom[WIDTH],
        int local_rows,
        int i, int j){

    int jm1 = (j == 0) ? WIDTH - 1 : j - 1; // columna anterior (wrap-around)
    int jp1 = (j == WIDTH - 1) ? 0 : j + 1; // columna siguiente (wrap-around)

    /* Fila superior: si i==0 usamos halo_top, si no la fila i-1 local */
    const uint8_t *row_above = (i == 0)             ? halo_top    : local[i - 1];
    /* Fila inferior: si es la última usamos halo_bottom, si no i+1 local */
    const uint8_t *row_below = (i == local_rows - 1) ? halo_bottom : local[i + 1];
    const uint8_t *row_cur   = local[i];

    return row_above[jm1] + row_above[j]   + row_above[jp1]
         + row_cur[jm1]                    + row_cur[jp1]
         + row_below[jm1] + row_below[j]   + row_below[jp1];//row above[jm1] es el vecino de arriba a la izquierda, row above[j] el vecino de arriba, row above[jp1] el vecino de arriba a la derecha, row_cur[jm1] el vecino de la izquierda, row_cur[jp1] el vecino de la derecha, row_below[jm1] el vecino de abajo a la izquierda, row_below[j] el vecino de abajo, row_below[jp1] el vecino de abajo a la derecha
}

/**
 * @brief Actualiza la rejilla según las reglas del Juego de la Vida.
 *
 * Crea una rejilla temporal `newGrid`, calcula el siguiente estado
 * para cada celda y copia el resultado de vuelta. Contiene una
 * directiva OpenMP opcional para paralelizar por filas.
 */
void updateRows(
        uint8_t local[][WIDTH],
        uint8_t new_local[][WIDTH],
        const uint8_t halo_top[WIDTH],
        const uint8_t halo_bottom[WIDTH],
        int local_rows,
        int row_start,
        int row_end)
{
    /* OpenMP paraleliza el bucle de filas si está disponible */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = row_start; i < row_end; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int n = countAliveNeighbours(local, halo_top, halo_bottom,
                                    local_rows, i, j);
            new_local[i][j] = (local[i][j])
                              ? (n == 2 || n == 3)
                              : (n == 3);
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
int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (HEIGHT % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: HEIGHT (%d) debe ser divisible por np (%d)\n",
                    HEIGHT, size);
        MPI_Finalize();
        return 1;
    }

    int local_rows = HEIGHT / size;

    /* Reserva dinámica de los buffers locales */
    uint8_t (*local)[WIDTH]     = malloc(local_rows * WIDTH * sizeof(uint8_t)); //local es un puntero a un array de anchura (width) WIDTH, con local_rows filas. Es decir, local[i][j] es el elemento de la fila i y columna j del bloque local de este proceso.
    uint8_t (*new_local)[WIDTH] = malloc(local_rows * WIDTH * sizeof(uint8_t)); //new_local es otro bloque local para almacenar la siguiente generación, con la misma estructura que local.
    uint8_t *halo_top           = malloc(WIDTH * sizeof(uint8_t)); //halo_top es un array de WIDTH elementos que almacena la fila fantasma superior, que corresponde a la última fila del proceso anterior en el anillo.
    uint8_t *halo_bottom        = malloc(WIDTH * sizeof(uint8_t)); //halo_bottom es otro array de WIDTH elementos que almacena la fila fantasma inferior, que corresponde a la primera fila del proceso siguiente en el anillo.

     /* ── Solo rank 0 inicializa la rejilla completa ──────────────── */
    if (rank == 0) {
        srand(42);
        initGrid();
    }


    MPI_Scatter(
        global_grid, local_rows * WIDTH, MPI_UINT8_T,
        local,       local_rows * WIDTH, MPI_UINT8_T,
        0, MPI_COMM_WORLD
    );

    /* ── Información de configuración (rank 0) ───────────────────── */
    if (rank == 0) {
        printf("Configuracion:\n");
        printf("  Tablero: %dx%d, Iteraciones: %d\n", WIDTH, HEIGHT, ITERATION);
        printf("  Procesos MPI: %d, Filas por proceso: %d\n", size, local_rows);
        #ifdef _OPENMP
        printf("  Hilos OpenMP por proceso: %d\n", omp_get_max_threads());
        #endif
    }

    int prev_rank = (rank - 1 + size) % size;
    int next_rank = (rank + 1)        % size;

    double t_start = MPI_Wtime();

    for (int it = 0; it < ITERATION; it++) {
        MPI_Request reqs[4];

        /* Recibir halo_top desde prev_rank (él nos manda su última fila) */
        MPI_Irecv(halo_top,    WIDTH, MPI_UINT8_T,
                  prev_rank, 1, MPI_COMM_WORLD, &reqs[0]);

        /* Recibir halo_bottom desde next_rank (él nos manda su primera fila) */
        MPI_Irecv(halo_bottom, WIDTH, MPI_UINT8_T,
                  next_rank, 0, MPI_COMM_WORLD, &reqs[1]);

        /* Enviar nuestra primera fila a prev_rank (para su halo_bottom) */
        MPI_Isend(local[0],             WIDTH, MPI_UINT8_T,
                  prev_rank, 0, MPI_COMM_WORLD, &reqs[2]);

        /* Enviar nuestra última fila a next_rank (para su halo_top) */
        MPI_Isend(local[local_rows - 1], WIDTH, MPI_UINT8_T,
                  next_rank, 1, MPI_COMM_WORLD, &reqs[3]);

        if (local_rows > 2) {
            updateRows(local, new_local,
                       halo_top, halo_bottom,   /* no se usan aquí, pero se pasan */
                       local_rows,
                       1, local_rows - 1);      /* filas interiores */
        }

        /* ── Esperar a que lleguen los halos ─────────────────────── */
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        /* ── Calcular filas frontera (ahora sí tenemos los halos) ── */
        updateRows(local, new_local,
                   halo_top, halo_bottom,
                   local_rows,
                   0, 1);                       /* primera fila */

        if (local_rows > 1)
            updateRows(local, new_local,
                       halo_top, halo_bottom,
                       local_rows,
                       local_rows - 1, local_rows); /* última fila */
        
    uint8_t (*tmp)[WIDTH] = local;
        local     = new_local;
        new_local = tmp;

    } /* fin del bucle de iteraciones */

    double t_end = MPI_Wtime();

    MPI_Gather(
        local,       local_rows * WIDTH, MPI_UINT8_T,
        global_grid, local_rows * WIDTH, MPI_UINT8_T,
        0, MPI_COMM_WORLD
    );

    double local_time = t_end - t_start;
    double max_time;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nTiempo de ejecucion (proceso mas lento): %.6f segundos\n", max_time);
        printf("Throughput: %.2f Mceldas/s\n",
               (double)WIDTH * HEIGHT * ITERATION / max_time / 1e6);

        #ifndef BENCHMARK
        /* Opcional: imprimir las últimas 10 filas del tablero final */
        printf("\nUltimas 5 filas del tablero final:\n");
        for (int i = HEIGHT - 5; i < HEIGHT; i++) {
            for (int j = 0; j < (WIDTH < 80 ? WIDTH : 80); j++)
                putchar(global_grid[i][j] ? '#' : '.');
            printf(WIDTH > 80 ? "...\n" : "\n");
        }
        #endif
    }

    /* ── Liberación de memoria ───────────────────────────────────── */
    free(local);
    free(new_local);
    free(halo_top);
    free(halo_bottom);

    MPI_Finalize();
    return 0;
}
