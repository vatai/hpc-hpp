// Compile and run:
// mpicc mympi.c -o mympi && mpiexec -n 4 ./mympi
// Calculates $\sum_{k=1}^n 1/(k*k) ~ pi^2/6

#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Rank: %d / %d\n", rank, size);

  const size_t n = 1000;
  const size_t chunk_size = n / size;
  size_t offset = rank * chunk_size;

  double local_sum = 0.0;
  for (size_t i = 0; i < chunk_size; i++) {
    double k = (double)(offset + i + 1);
    local_sum += 1 / (k * k);
  }
  printf("%lf\n", local_sum);

  if (rank == 0) {
    double buffer;
    MPI_Status status;
    for (size_t i = 1; i < size; i++) {
      MPI_Recv(&buffer, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
      local_sum += buffer;
    }
    printf("Final result: %lf\n", local_sum);
    printf("PI: %lf\n", 6 * local_sum / 3.14);
  } else {
    MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return 0;
}
