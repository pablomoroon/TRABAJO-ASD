#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <mpi.h>
#include <omp.h>

#define WIDTH 1000
#define HEIGHT 1000
#define ITERATION 500
#define SPEED 500
#define ALIVE '#'
#define DEAD '.'

#define IDX(i,j) ((i) * WIDTH + (j))

static uint8_t *grid;
static uint8_t *newGrid;

int generation = 0;
int rank, size, local_rows;

void initGrid() {
    srand(42 + rank);

    for (int i = 1; i <= local_rows; i++) {
        for (int j = 0; j < WIDTH; j++) {
            grid[IDX(i, j)] = rand() & 1;
        }
    }

    for (int j = 0; j < WIDTH; j++) {
        grid[IDX(0, j)] = 0;
        grid[IDX(local_rows + 1, j)] = 0;
    }
}

void printGrid() {
    system("clear");
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%c", grid[IDX(i, j)] ? ALIVE : DEAD);
        }
        printf("\n");
    }
}

int countAliveNeighbours(int i, int j) {
    int count = 0;

    int im1 = i - 1;
    int ip1 = i + 1;
    int jm1 = (j == 0) ? WIDTH - 1 : j - 1;
    int jp1 = (j == WIDTH - 1) ? 0 : j + 1;

    count += grid[IDX(im1, jm1)];
    count += grid[IDX(im1, j)];
    count += grid[IDX(im1, jp1)];
    count += grid[IDX(i, jm1)];
    count += grid[IDX(i, jp1)];
    count += grid[IDX(ip1, jm1)];
    count += grid[IDX(ip1, j)];
    count += grid[IDX(ip1, jp1)];

    return count;
}

void updateGrid() {
    int up = (rank == 0) ? size - 1 : rank - 1;
    int down = (rank == size - 1) ? 0 : rank + 1;

    MPI_Sendrecv(
        &grid[IDX(1, 0)], WIDTH, MPI_UINT8_T, up, 0,
        &grid[IDX(local_rows + 1, 0)], WIDTH, MPI_UINT8_T, down, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    MPI_Sendrecv(
        &grid[IDX(local_rows, 0)], WIDTH, MPI_UINT8_T, down, 1,
        &grid[IDX(0, 0)], WIDTH, MPI_UINT8_T, up, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    #pragma omp parallel for schedule(static)
    for (int i = 1; i <= local_rows; i++) {
        #pragma omp simd
        for (int j = 0; j < WIDTH; j++) {
            int count = countAliveNeighbours(i, j);
            newGrid[IDX(i, j)] =
                grid[IDX(i, j)] ? ((count == 2 || count == 3) ? 1 : 0)
                                : (count == 3);
        }
    }

    uint8_t *tmp = grid;
    grid = newGrid;
    newGrid = tmp;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (HEIGHT % size != 0) {
        if (rank == 0) {
            printf("HEIGHT debe ser divisible por el numero de procesos\n");
        }
        MPI_Finalize();
        return 1;
    }

    local_rows = HEIGHT / size;

    grid = (uint8_t *)malloc((local_rows + 2) * WIDTH * sizeof(uint8_t));
    newGrid = (uint8_t *)malloc((local_rows + 2) * WIDTH * sizeof(uint8_t));

    if (grid == NULL || newGrid == NULL) {
        printf("Error reservando memoria en el proceso %d\n", rank);
        MPI_Finalize();
        return 1;
    }

    initGrid();

#ifdef BENCHMARK
    #pragma omp parallel
    {
        #pragma omp single
        printf("Proceso %d de %d - hilos OpenMP: %d\n", rank, size, omp_get_num_threads());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    for (int it = 0; it < ITERATION; ++it) {
        updateGrid();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    double local_time = t2 - t1;
    double total_time = 0.0;

    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Tiempo total (maximo entre procesos): %f segundos\n", total_time);
    }

#else
    for (int i = 0; i < ITERATION; i++) {
        printGrid();
        updateGrid();
        usleep(SPEED * 1000);
    }
#endif

    free(grid);
    free(newGrid);

    MPI_Finalize();
    return 0;
}