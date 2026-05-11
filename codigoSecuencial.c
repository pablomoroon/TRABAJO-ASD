#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>



#define WIDTH 1000
#define HEIGHT 1000
#define ITERATION 1000
#define SPEED 500
#define ALIVE '#'
#define DEAD '.'

int grid[HEIGHT][WIDTH];
int generation = 0;

/**
 * @brief Initializes the grid with random alive/dead values.
 *
 * This function populates the grid with random values, either 0 (dead)
 * or 1 (alive). This is used to initialize the grid at the start of
 * the Game of Life.
 */
void initGrid() {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            grid[i][j] = rand() % 2;
        }
    }
}

/**
 * @brief Prints the current state of the grid to the console.
 *
 * This function clears the console and prints the current state
 * of the grid to the console. The grid is represented as a series
 * of '#' characters for alive cells and ' ' characters for dead
 * cells. The function is intended to be used in the main loop of
 * the program to print the state of the grid at each iteration.
 */
void printGrid() {
    system("clear");
    //system("cls"); //PARA COMPILAR EN Windows
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%c", grid[i][j] ? ALIVE : DEAD);
        }
        printf("\n");
    }
}

/**
 * @brief Counts the number of alive neighbours for a given cell.
 *
 * This function takes two arguments, i and j, which represent the
 * coordinates of the cell in the grid. It then iterates over the
 * eight cells in the Moore neighbourhood of the cell, counting the
 * number of cells that are alive. Note that the cell itself is
 * excluded from the count.
 *
 * The function uses the modulo operator to wrap around the edges of
 * the grid, so the function works correctly even when the cell is
 * located at the edge of the grid.
 *
 * @param i The row of the cell in the grid.
 * @param j The column of the cell in the grid.
 * @return The number of alive neighbours of the cell.
 */
int countAliveNeighbours(int i, int j) {
    int count = 0;
    for (int k = -1; k <= 1; k++) {
        for (int l = -1; l <= 1; l++) {
            int x = (i + k + HEIGHT) % HEIGHT;
            int y = (j + l + WIDTH) % WIDTH;
            count += grid[x][y];
        }
    }
    count -= grid[i][j]; 
    return count;
}

/**
 * @brief Updates the grid according to the rules of the Game of Life.
 *
 * This function creates a new grid with the same dimensions as the
 * current grid, then iterates over each cell in the current grid.
 * For each cell, it counts the number of alive neighbours, and then
 * applies the rules of the Game of Life to determine whether the cell
 * should be alive or dead in the new grid. The new grid is then
 * copied back into the current grid.
 */
void updateGrid() {
    int newGrid[HEIGHT][WIDTH];
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int count = countAliveNeighbours(i, j);
            newGrid[i][j] = (grid[i][j] == 1) ? (count == 2 || count == 3) : (count == 3); // Si la celda está viva, permanece viva si tiene 2 o 3 vecinos vivos; de lo contrario, muere. Si la celda está muerta, se vuelve viva si tiene exactamente 3 vecinos vivos; de lo contrario, permanece muerta.
        }
    }
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            grid[i][j] = newGrid[i][j];
        }
    }
}

/**
 * @brief The main entry point for the Game of Life.
 *
 * This function initializes the grid with random values, then iterates
 * through the grid for a specified number of iterations, printing the
 * grid and updating it each time. The usleep() call is used to slow down
 * the iteration.
 */

int main() {
    srand(42);
    initGrid();

#ifdef BENCHMARK
    double t1 = omp_get_wtime();

    for (int i = 0; i < ITERATION; i++) {
        updateGrid();
    }

    double t2 = omp_get_wtime();

    printf("Tiempo OpenMP: %f segundos\n", t2 - t1);
#else
    for (int i = 0; i < ITERATION; i++) {
        printGrid();
        updateGrid();
        usleep(SPEED * 1000);
    }
#endif

    return 0;
}