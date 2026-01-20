#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Print out the rank and size
    std::cout << "Hello from process " << rank << " of " << size << std::endl;

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}