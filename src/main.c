#include <stdlib.h>
#include "summa_opts.h"
#include "utils.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

double get_elapsed_time(struct timeval start_t, struct timeval end_t)
{
    return (end_t.tv_sec - start_t.tv_sec) + ((end_t.tv_usec - start_t.tv_usec) / 1e6);
}

void scatter_matrix_blocks(float *matrix, float *local_matrix, int full_rows, int full_cols,
                           int block_rows, int block_cols, int num_process, MPI_Comm *grid_comm, int rank_process)
{
    MPI_Datatype block_type;

    // 1. Create a subarray type for matrix block distribution
    {
        MPI_Datatype temp_type;
        int sizes[2] = {full_rows, full_cols};      // Full matrix dimensions
        int subsizes[2] = {block_rows, block_cols}; // Block dimensions
        int starts[2] = {0, 0};                     // Always starts from (0,0), displacement handles the offsets

        MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &temp_type);
        MPI_Type_create_resized(temp_type, 0, sizeof(float), &block_type);
        MPI_Type_commit(&block_type);
        MPI_Type_free(&temp_type);
    }

    // 2. Prepare sendcounts and displacements
    int *sendcounts = (int *)malloc(num_process * sizeof(int));
    int *displacements = (int *)malloc(num_process * sizeof(int));

    for (int i = 0; i < num_process; i++)
    {
        int proc_coords[2];
        MPI_Cart_coords(*grid_comm, i, 2, proc_coords);

        sendcounts[i] = 1; // Each process gets one block

        // Compute displacement for 2D row-major storage
        int row_idx = proc_coords[0] * block_rows;
        int col_idx = proc_coords[1] * block_cols;
        displacements[i] = row_idx * full_cols + col_idx;
    }

    double scatter_start, scatter_end, scatter_time;
    scatter_start = MPI_Wtime();

    // 3. Perform the scatter operation
    MPI_Scatterv(matrix, sendcounts, displacements, block_type,
                 local_matrix, block_rows * block_cols, MPI_FLOAT,
                 0, *grid_comm);

    scatter_end = MPI_Wtime();
    scatter_time = scatter_end - scatter_start;

    double global_scatter_time;
    MPI_Reduce(&scatter_time, &global_scatter_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank_process == 0)
    {
        printf("Total Scatter Time: %f seconds\n", global_scatter_time);
    }
    // Synchronization after scattering
    // Cleanup
    free(sendcounts);
    free(displacements);
    MPI_Type_free(&block_type);
}

void gather_matrix_blocks(float *local_matrix, float *matrix, int full_rows, int full_cols,
                          int block_rows, int block_cols, int num_process, MPI_Comm *grid_comm,int rank_process)
{
    MPI_Datatype block_type;

    // 1. Create the subarray type for structured gathering
    {
        MPI_Datatype temp_type;
        int sizes[2] = {full_rows, full_cols};      // Full matrix dimensions
        int subsizes[2] = {block_rows, block_cols}; // Block dimensions
        int starts[2] = {0, 0};                     // Always starts from (0,0), displacement handles offsets

        MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &temp_type);
        MPI_Type_create_resized(temp_type, 0, sizeof(float), &block_type);
        MPI_Type_commit(&block_type);
        MPI_Type_free(&temp_type);
    }

    // 2. Prepare recvcounts and displacements (only used on rank 0)
    int *recvcounts = NULL;
    int *displacements = NULL;

    if (matrix != NULL) // Only allocate on root (rank 0)
    {
        recvcounts = (int *)malloc(num_process * sizeof(int));
        displacements = (int *)malloc(num_process * sizeof(int));

        for (int i = 0; i < num_process; i++)
        {
            int proc_coords[2];
            MPI_Cart_coords(*grid_comm, i, 2, proc_coords);

            recvcounts[i] = 1; // Each process sends one block

            // Compute displacement for 2D row-major storage
            int row_idx = proc_coords[0] * block_rows;
            int col_idx = proc_coords[1] * block_cols;
            displacements[i] = row_idx * full_cols + col_idx;
        }
    }

    double gather_start, gather_end, gather_time;
    gather_start = MPI_Wtime();

    // // 3. Perform the gather operation
    MPI_Gatherv(local_matrix, block_rows * block_cols, MPI_FLOAT, // Send raw block
                matrix, recvcounts, displacements, block_type,    // Receive structured blocks
                0, *grid_comm);

    // MPI_Allgatherv(local_matrix, block_rows * block_cols, MPI_FLOAT,
    //     matrix, recvcounts, displacements, block_type,
    //     *grid_comm);

    gather_end = MPI_Wtime();
    gather_time = gather_end - gather_start;

    double global_gather_time;
    MPI_Reduce(&gather_time, &global_gather_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank_process == 0)
    {
        printf("Total Gather Time: %f seconds\n", global_gather_time);
    }

    // Cleanup (only on rank 0)
    if (matrix != NULL)
    {
        free(recvcounts);
        free(displacements);
    }

    MPI_Type_free(&block_type);
}

void distribute_matrix_blocks(float *A, float *B, float *C, int m, int n, int k,
                              float *A_local, float *B_local, float *C_local, int num_process, MPI_Comm *grid_comm,
                              int block_m, int block_n, int block_k,int rank_process)
{
    // Scatter matrix A
    if(rank_process == 0){
        printf("Scattering Matrix A\n");
    }
    scatter_matrix_blocks(A, A_local, m, k, block_m, block_k, num_process, grid_comm,rank_process);
    MPI_Barrier(*grid_comm);
    if(rank_process == 0){
        printf("Scattering Matrix B\n");
    }
    scatter_matrix_blocks(B, B_local, k, n, block_k, block_n, num_process, grid_comm,rank_process);
    MPI_Barrier(*grid_comm);
    // **Zero Initialize C_local Instead of Scattering C**
    for (int i = 0; i < block_m * block_n; i++)
    {
        C_local[i] = 0.0f;
    }
}

void print_matrix(float *matrix, int rows, int cols, int rank)
{
    printf("Process %d:\n", rank);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%0.6f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
    fflush(stdout);
}

void summa_stationary_c(int m, int n, int k, int num_process, int rank_process,
                        MPI_Comm *grid_comm, MPI_Comm *row_comm, MPI_Comm *col_comm, int *coords)
{
    int dims[2], periods[2] = {0, 0};

    // Assuming square process grid (P = p x p)
    int p = (int)sqrt(num_process);
    dims[0] = dims[1] = p;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, grid_comm);
    MPI_Cart_coords(*grid_comm, rank_process, 2, coords);

    // Create row and column communicators
    MPI_Comm_split(*grid_comm, coords[0], coords[1], row_comm);
    MPI_Comm_split(*grid_comm, coords[1], coords[0], col_comm);

    // Calculate block sizes
    int block_m = (m + p - 1) / p; // Ceiling division
    int block_k = (k + p - 1) / p;
    int block_n = (n + p - 1) / p;

    printf("Process %d (coords %d,%d) -> block_m=%d, block_k=%d, block_n=%d\n",
           rank_process, coords[0], coords[1], block_m, block_k, block_n);

    // Allocate and initialize matrices
    float *A = NULL, *B = NULL, *C = NULL;

    if (rank_process == 0)
    {
        A = generate_matrix_A(m, k, rank_process);
        B = generate_matrix_B(k, n, rank_process);
        C = (float *)calloc(m * n, sizeof(float));

        // printf("<<<<<<Full Matrix A:>>>>>>\n");
        // print_matrix(A, m, k, rank_process);
        // printf("<<<<<<<Full Matrix B:>>>>>>\n");
        // print_matrix(B, k, n, rank_process);
    }
    else
    {
        A = (float *)malloc(m * k * sizeof(float));
        B = (float *)malloc(k * n * sizeof(float));
    }

    printf("Done with getting the matrices \n");

    // Allocate local matrices
    float *A_local = (float *)malloc(block_m * block_k * sizeof(float));
    float *B_local = (float *)malloc(block_k * block_n * sizeof(float));
    float *C_local = (float *)calloc(block_m * block_n, sizeof(float));

    // Distribute matrices - each process extracts its own block

    distribute_matrix_blocks(A, B, C, m, n, k, A_local, B_local, C_local, num_process, grid_comm, block_m, block_n, block_k,rank_process);
    MPI_Barrier(*grid_comm);
    if(rank_process == 0){
    printf("Matrix Distribution is done \n");
    }
    // printf("<<<<<<Full Matrix A:>>>>>>\n");
    // print_matrix(A_local, block_m, block_k, rank_process);

    // **Main Computation using SUMMA Algorithm**

    double comm_start, comm_end, comm_time = 0.0;

    for (int step = 0; step < p; step++)
    {
        float *A_broadcast = (float *)malloc(block_m * block_k * sizeof(float));
        float *B_broadcast = (float *)malloc(block_k * block_n * sizeof(float));

        comm_start = MPI_Wtime();

        // Broadcast A_local along the row from process (i, step)
        if (coords[1] == step)
        {
            memcpy(A_broadcast, A_local, block_m * block_k * sizeof(float));
        }
        MPI_Bcast(A_broadcast, block_m * block_k, MPI_FLOAT, step, *row_comm);

        // Broadcast B_local along the column from process (step, j)
        if (coords[0] == step)
        {
            memcpy(B_broadcast, B_local, block_k * block_n * sizeof(float));
        }
        MPI_Bcast(B_broadcast, block_k * block_n, MPI_FLOAT, step, *col_comm);

        comm_end = MPI_Wtime();
        comm_time += (comm_end - comm_start);

        matmul(A_broadcast, B_broadcast, C_local, block_m, block_n, block_k);
    }

    MPI_Barrier(*grid_comm);
    double global_comm_time;
    MPI_Reduce(&comm_time, &global_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank_process == 0)
    {
        printf("Total Communication Time: %f seconds\n", global_comm_time);
    }

    gather_matrix_blocks(C_local, C, m, n, block_m, block_n, num_process, grid_comm,rank_process);

    // Print results
    if (rank_process == 0)
    {
        printf("Parallel SUMMA (Stationary C) Matrix Multiplication is complete\n");
        printf("Verifing the results after the Parallel Execution\n");
        verify_result(C, A, B, m, n, k);
    }

    free(C);
    free(A_local);
    free(B_local);
    free(C_local);
    free(A);
    free(B);
}

void summa_stationary_a(int m, int n, int k, int num_process, int rank_process,
                        MPI_Comm *grid_comm, MPI_Comm *row_comm, MPI_Comm *col_comm, int *coords)
{

    /*In SUMMA Stationary A matrix, the Matrix A is kept stationary in each of the process
    Only B and C are broadcasted accordingly inorder to calculate the matrix multiplication*/
    int dims[2], periods[2] = {0, 0};
    // Defining the dimensions as 2 since this is 2D partition.

    // Assuming square process grid (P = p x p)
    int p = (int)sqrt(num_process);
    dims[0] = dims[1] = p;
    /*This process sets up a 2D Cartesian topology in MPI,
    enabling efficient communication for grid-based parallel computations.
    The setup involves creating a Cartesian communicator, assigning process coordinates, and
    splitting communicators for row-wise and column-wise communication.*/
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, grid_comm);

    /*MPI_Cart_create() establishes a 2D grid topology from MPI_COMM_WORLD.

    Parameters:
    MPI_COMM_WORLD: The initial communicator containing all processes.
    2: The number of dimensions (creating a 2D grid).
    dims[]: Array defining the number of processes in each dimension (e.g., {dim_x, dim_y}).
    periods[]: Array defining whether dimensions are periodic (e.g., {0, 0} means no wrap-around).
    0: Flag indicating whether to reorder ranks for better topology mapping (0 = no reordering).
    grid_comm: The new Cartesian communicator.

    */

    MPI_Cart_coords(*grid_comm, rank_process, 2, coords);
    /*MPI_Cart_coords() maps a process rank to its (x, y) coordinates in the Cartesian grid.
    Parameters:
    *grid_comm: The Cartesian communicator.
    rank_process: The rank of the process.
    2: The number of dimensions.
    coords[]: Output array storing the process's coordinates.*/

    // Create row and column communicators
    MPI_Comm_split(*grid_comm, coords[0], coords[1], row_comm);
    MPI_Comm_split(*grid_comm, coords[1], coords[0], col_comm);

    // Calculate block sizes
    // Then the total matrix size is divided amount the processor this is done with ceiling division.
    int block_m = (m + p - 1) / p; // Ceiling division
    int block_k = (k + p - 1) / p;
    int block_n = (n + p - 1) / p;

    printf("Process %d (coords %d,%d) -> block_m=%d, block_k=%d, block_n=%d\n",
           rank_process, coords[0], coords[1], block_m, block_k, block_n);

    // Allocate and initialize matrices
    float *A = NULL, *B = NULL, *C = NULL;
    // Getting the matrix values in the process zero.

    if (rank_process == 0)
    {

        A = generate_matrix_A(m, k, rank_process);
        B = generate_matrix_B(k, n, rank_process);
        C = (float *)calloc(m * n, sizeof(float));

        // printf("<<<<<<Full Matrix A:>>>>>>\n");
        // print_matrix(A, m, k, rank_process);
        // printf("<<<<<<<Full Matrix B:>>>>>>\n");
        // print_matrix(B, k, n, rank_process);
    }
    // else
    // {
    //     A = (float *)malloc(m * k * sizeof(float));
    //     B = (float *)malloc(k * n * sizeof(float));
    // }
    if (rank_process == 0)
    {
        printf("Done with getting the matrices \n");
    }
    // Broadcast full matrices to all processes
    // MPI_Bcast(A, m * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // MPI_Bcast(B, k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Allocate local matrices
    float *A_local = (float *)malloc(block_m * block_k * sizeof(float));
    float *B_local = (float *)malloc(block_k * block_n * sizeof(float));
    float *C_local = (float *)calloc(block_m * block_n, sizeof(float));

    // Distribute matrices - each process extracts its own block
    /* After getting the values for the matrices, divide the matrix A,B and C across the process
     */
    distribute_matrix_blocks(A, B, C, m, n, k, A_local, B_local, C_local, num_process, grid_comm, block_m, block_n, block_k,rank_process);
    MPI_Barrier(*grid_comm);
    if (rank_process == 0)
    {
        printf("Done with distributing the process");
    }
    // Allocate buffers for broadcasts
    // float *A_slab = (float *)malloc(block_m * block_k * sizeof(float));
    //float *B_broadcast = (float *)malloc(block_k * block_n * sizeof(float));

    int chunk_size = block_m * block_k;
    int slab_size = block_m * k; // A_slab should store all row blocks

    // Allocate A_slab if not already
    float *A_slab = (float *)malloc(slab_size * sizeof(float));
    if (!A_slab)
    {
        fprintf(stderr, "Process %d: Error allocating A_slab\n", rank_process);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
    }

    // Gather all A_local blocks from row processes
    MPI_Allgather(A_local, chunk_size, MPI_FLOAT, A_slab, chunk_size, MPI_FLOAT, *row_comm);

    //   printf("Printing A_localSlab of process\n");
    //   print_matrix(A_slab, block_m, k,rank_process);
    //   printf("\n");

    float *B_temp = (float *)calloc(block_k * block_n, sizeof(float));

    double comm_start, comm_end, comm_time = 0.0;

    for (int j = 0; j < p; j++)
    {

        comm_start = MPI_Wtime();
        int col_rank;
        MPI_Comm_rank(*col_comm, &col_rank);

        int root = -1;
        if (coords[0] == j)
        {

            memcpy(B_temp, B_local, block_k * block_n * sizeof(float));
        }

        MPI_Bcast(B_temp, block_k * block_n, MPI_FLOAT, j, *col_comm);

        comm_end = MPI_Wtime();
        comm_time += (comm_end - comm_start);


        matmul(A_slab + j * block_m * block_k, B_temp, C_local, block_m, block_n, block_k);
    }

    MPI_Barrier(*grid_comm);
    double global_comm_time;
    MPI_Reduce(&comm_time, &global_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank_process == 0)
    {
        printf("Total Communication Time: %f seconds\n", global_comm_time);
    }

    // Gather final result
    gather_matrix_blocks(C_local, C, m, n, block_m, block_n, num_process, grid_comm,rank_process);

    MPI_Barrier(*grid_comm);

    // Print results
    if (rank_process == 0)
    {
        printf("Parallel Process SUMMA Matrix Multiplication is done\n");
        // print_matrix(C, m, n, rank_process);

        //float *C_ref = (float *)calloc(m * n, sizeof(float));
        printf("Verifing the results after the Parallel Execution\n");
        verify_result(C, A, B, m, n, k);
    }

    free(C);
    free(A_local);
    free(B_local);
    free(C_local);
    free(A);
    free(B);
    free(B_temp);
}

void summa_stationary_b(int m, int n, int k, int num_process, int rank_process, MPI_Comm *grid_comm, MPI_Comm *row_comm, MPI_Comm *col_comm, int *coords)
{

    int dims[2], periods[2] = {0, 0};

    int p = (int)sqrt(num_process);
    dims[0] = dims[1] = p;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, grid_comm);

    MPI_Cart_coords(*grid_comm, rank_process, 2, coords);

    MPI_Comm_split(*grid_comm, coords[0], coords[1], row_comm);
    MPI_Comm_split(*grid_comm, coords[1], coords[0], col_comm);

    // Calculate block sizes
    // Then the total matrix size is divided amount the processor this is done with ceiling division.
    int block_m = (m + p - 1) / p; // Ceiling division
    int block_k = (k + p - 1) / p;
    int block_n = (n + p - 1) / p;

    printf("Process %d (coords %d,%d) -> block_m=%d, block_k=%d, block_n=%d\n",
           rank_process, coords[0], coords[1], block_m, block_k, block_n);

    // Allocate and initialize matrices
    float *A = NULL, *B = NULL, *C = NULL;
    // Getting the matrix values in the process zero.

    if (rank_process == 0)
    {

        A = generate_matrix_A(m, k, rank_process);
        B = generate_matrix_B(k, n, rank_process);
        C = (float *)calloc(m * n, sizeof(float));

        // printf("<<<<<<Full Matrix A:>>>>>>\n");
        // print_matrix(A, m, k, rank_process);
        // printf("<<<<<<<Full Matrix B:>>>>>>\n");
        // print_matrix(B, k, n, rank_process);
    }
    // else
    // {
    //     A = (float *)malloc(m * k * sizeof(float));
    //     B = (float *)malloc(k * n * sizeof(float));
    // }

    if (rank_process == 0)
    {
        printf("Done with getting the matrices \n");
    }

    // Broadcast full matrices to all processes
    // MPI_Bcast(A, m * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // MPI_Bcast(B, k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Allocate local matrices
    float *A_local = (float *)malloc(block_m * block_k * sizeof(float));
    float *B_local = (float *)malloc(block_k * block_n * sizeof(float));
    float *C_local = (float *)calloc(block_m * block_n, sizeof(float));

    // Distribute matrices - each process extracts its own block
    /* After getting the values for the matrices, divide the matrix A,B and C across the process
     */
    distribute_matrix_blocks(A, B, C, m, n, k, A_local, B_local, C_local, num_process, grid_comm, block_m, block_n, block_k,rank_process);
    MPI_Barrier(*grid_comm);
    if (rank_process == 0)
    {
        printf("Done with distributing the process");
    }
    // Allocate B_slab for column-wise collection
    int slab_size = k * block_n; // B_slab should store all column blocks
    float *B_slab = (float *)malloc(slab_size * sizeof(float));
    if (!B_slab)
    {
        fprintf(stderr, "Process %d: Error allocating B_slab\n", rank_process);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
    }

    // Gather all B_local blocks from column processes
    MPI_Allgather(B_local, block_k * block_n, MPI_FLOAT, B_slab, block_k * block_n, MPI_FLOAT, *col_comm);

    // Temporary buffer for broadcasting A
    float *A_temp = (float *)calloc(block_m * block_k, sizeof(float));
    double comm_start, comm_end, comm_time = 0.0;
    for (int i = 0; i < p; i++)
    {
        comm_start = MPI_Wtime();
        int row_rank;
        MPI_Comm_rank(*row_comm, &row_rank);

        if (coords[1] == i)
        {
            memcpy(A_temp, A_local, block_m * block_k * sizeof(float));
        }

        // Broadcast A_temp across row communicators
        MPI_Bcast(A_temp, block_m * block_k, MPI_FLOAT, i, *row_comm);

        comm_end = MPI_Wtime();
        comm_time += (comm_end - comm_start);

        // Perform local matrix multiplication using the correct B_slab portion
        matmul(A_temp, B_slab + i * block_k * block_n, C_local, block_m, block_n, block_k);
    }

    MPI_Barrier(*grid_comm);
    double global_comm_time;
    MPI_Reduce(&comm_time, &global_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank_process == 0)
    {
        printf("Total Communication Time: %f seconds\n", global_comm_time);
    }

    gather_matrix_blocks(C_local, C, m, n, block_m, block_n, num_process, grid_comm,rank_process);

    MPI_Barrier(*grid_comm);

    // Print results
    if (rank_process == 0)
    {
        printf("Parallel Process SUMMA Matrix Multiplication is done\n");
        // print_matrix(C, m, n, rank_process);

        //float *C_ref = (float *)calloc(m * n, sizeof(float));
        // for (int i = 0; i < m; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         float sum = 0.0f;
        //         for (int p = 0; p < k; p++)
        //         {
        //             sum += A[i * k + p] * B[p * n + j];
        //         }
        //         C_ref[i * n + j] = sum;
        //     }
        // }

        printf("Verifing the results after the Parallel Execution\n");
        verify_result(C, A, B, m, n, k);
    }

    free(C);
    free(A_local);
    free(B_local);
    free(C_local);
    free(A);
    free(B);
    free(A_temp); // TODO: Implement SUMMA algorithm with stationary B
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    /*Initialise the MPI process and get the rank and number of process that is present in the MPI communication world*/

    int rank_process, num_process;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_process);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);

    int coords[2] = {0, 0}; // Ensure coords are initialized
    MPI_Comm grid_comm, row_comm, col_comm;

    /*SUMMA works by creating a 2D grid. And the matrix is divided by number of process grid*/

    SummaOpts opts;

    if (rank_process == 0)
    {
        opts = parse_args(argc, argv);
    }
    // Getting the aruguments from the user

    // Broadcasting struct fields separately for portability
    MPI_Bcast(&opts.m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&opts.n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&opts.k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&opts.block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&opts.stationary, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&opts.verbose, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Initializing the grid size, grid size is square root of num of processes. And the number of process should be a perfect square for us to get the grid size
    int grid_size = (int)sqrt(num_process);

    if (grid_size * grid_size != num_process)
    {
        if (rank_process == 0)
        {
            printf("Error: Number of MPI processes (%d) must be a perfect square.\n", num_process);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank_process == 0)
    {
        printf("Using a %d x %d process grid.\n", grid_size, grid_size);
    }

    if (opts.m % grid_size != 0 || opts.n % grid_size != 0 || opts.k % grid_size != 0)
    {
        if (rank_process == 0)
        {
            printf("Error: Matrix dimensions must be divisible by grid size (%d)\n", grid_size);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank_process == 0)
    {
        printf("\nMatrix Dimensions:\n");
        printf("A: %d x %d\n", opts.m, opts.k);
        printf("B: %d x %d\n", opts.k, opts.n);
        printf("C: %d x %d\n", opts.m, opts.n);
        printf("Grid size: %d x %d\n", grid_size, grid_size);
        printf("Block size: %d\n", opts.block_size);
        printf("Algorithm: Stationary %c\n", opts.stationary);
        printf("Verbose: %s\n", opts.verbose ? "true" : "false");
    }

    struct timeval end_t; // Creating a object to get the end and the start time when the messages are begin send and recieved.
    struct timeval start_t;
    double time_elapsed;

    if (rank_process == 0)
    {
        gettimeofday(&start_t, NULL);
    }

    if (opts.stationary == 'a')
    {
        summa_stationary_a(opts.m, opts.n, opts.k, num_process, rank_process,
                           &grid_comm, &row_comm, &col_comm, coords);
    }
    else if (opts.stationary == 'c')
    {
        summa_stationary_c(opts.m, opts.n, opts.k, num_process, rank_process, &grid_comm, &row_comm, &col_comm, coords);
    }
    else if (opts.stationary == 'b')
    {
        summa_stationary_b(opts.m, opts.n, opts.k, num_process, rank_process, &grid_comm, &row_comm, &col_comm, coords);
    }
    else
    {
        if (rank_process == 0)
        {
            printf("Error: Unknown stationary option '%c'. Use 'A' or 'B'.\n", opts.stationary);
        }
        MPI_Finalize();
        return 1;
    }

    // printf("Process %d: coords=(%d,%d)\n", rank_process, coords[0], coords[1]);

    // Stop the timer and calculate elapsed time
    if (rank_process == 0)
    {
        gettimeofday(&end_t, NULL);
        time_elapsed = get_elapsed_time(start_t, end_t);
        printf("Execution time for summa_stationary: '%c' %.6f seconds\n", opts.stationary, time_elapsed);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);

    MPI_Finalize();
    return 0;
}