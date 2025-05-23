/*  Copyright 2014 IST Austria

Contributed by: Jan Reininghaus

This file is part of DIPHA.

DIPHA is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DIPHA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with DIPHA.  If not, see <http://www.gnu.org/licenses/>. */

#include <dipha/includes.h>
#include <chrono>

void print_help_and_exit()
{
  std::cerr << "Usage: " << "dipha " << "[options] input_filename output_filename" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << std::endl;
  std::cerr << "--help    --  prints this screen" << std::endl;
  std::cerr << "--dual    --  use dualization" << std::endl;
  std::cerr << "--upper_dim N   --  maximal dimension to compute" << std::endl;
  std::cerr << "--benchmark --  prints timing info" << std::endl;
  /* Added for Neuron Fragment Application */
  std::cerr << "--pixel_threshold   --  thresholding value above which we ignore the image" << std::endl;
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

void parse_command_line(int argc, char** argv, bool& benchmark, bool& dualize, int64_t& upper_dim, int64_t& pixel_threshold,
                        std::string& input_filename, std::string& output_filename, std::string& edge_filename, int64_t& nx, int64_t& ny)
{

  std::cout << "argc: " << argc << std::endl;
  if (argc < 6)
    print_help_and_exit();

  input_filename = argv[argc - 5];
  output_filename = argv[argc - 4];
  edge_filename = argv[argc - 3];
  nx = std::atoi(argv[argc - 2]);
  ny = std::atoi(argv[argc - 1]);

  //std::cout << "Check: " << nx << std::endl;

  for (int idx = 1; idx < argc - 5; idx++)
  {
    const std::string option = argv[idx];
    if (option == "--benchmark")
    {
      benchmark = true;
    }
    else if (option == "--help")
    {
      print_help_and_exit();
    }
    else if (option == "--dual")
    {
      dualize = true;
    }
    else if (option == "--upper_dim")
    {
      idx++;
      if (idx >= argc - 2)
        print_help_and_exit();
      std::string parameter = std::string(argv[idx]);
      size_t pos_last_digit;
      upper_dim = std::stoll(parameter, &pos_last_digit);
      if (pos_last_digit != parameter.size())
        print_help_and_exit();
    }
    else if (option == "--pixel_threshold")
    {
      idx++;
      if (idx >= argc - 2)
        print_help_and_exit();
      std::string parameter = std::string(argv[idx]);
      size_t pos_last_digit;
      pixel_threshold = std::stoll(parameter, &pos_last_digit);
      if (pos_last_digit != parameter.size())
        print_help_and_exit();
    }
    else print_help_and_exit();
  }
}

template< typename Complex >
void compute(const std::string& input_filename, bool dualize, int64_t upper_dim, int64_t pixel_threshold, const std::string& output_filename, const std::string& edge_filename, int64_t nx, int64_t ny)
{
  Complex complex;

  auto start = std::chrono::steady_clock::now();

  DIPHA_MACROS_BENCHMARK(complex.load_binary(input_filename, upper_dim); );

  auto end = std::chrono::steady_clock::now();
  std::cout << "Reading elapsed time in seconds: "
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << " sec" <<std::endl;

  if (dipha::globals::benchmark)
    dipha::mpi_utils::cout_if_root() << std::endl << "Number of cells in input: " << std::endl << complex.get_num_cells() << std::endl;
  dipha::data_structures::distributed_vector< int64_t > filtration_to_cell_map;
  dipha::data_structures::write_once_column_array reduced_columns;

  start = std::chrono::steady_clock::now();
  dipha::algorithms::compute_reduced_columns(complex, dualize, upper_dim, pixel_threshold, filtration_to_cell_map, reduced_columns);
  end = std::chrono::steady_clock::now();
  std::cout << "total persistence elapsed time in seconds: "
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << " sec" <<std::endl;

  std::cout << "size: " << reduced_columns.get_size() << std::endl;

  start = std::chrono::steady_clock::now();
  DIPHA_MACROS_BENCHMARK(dipha::outputs::save_persistence_diagram(output_filename, edge_filename, nx, ny, complex, filtration_to_cell_map, reduced_columns, 
                                                                  dualize, upper_dim); );
  end = std::chrono::steady_clock::now();
  std::cout << "total output elapsed time in seconds: "
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << " sec" <<std::endl;
}

int main(int argc, char** argv)
{

  auto start = std::chrono::steady_clock::now();
  // mandatory MPI initilization call
  MPI_Init(&argc, &argv);

  // take time at beggining of execution
  double time_at_start = MPI_Wtime();

  std::string input_filename; // name of file that contains the weighted cell complex
  std::string output_filename; // name of file that will contain the persistence diagram
  std::string edge_filename; //
  int64_t nx;
  int64_t ny;
  bool benchmark = false; // print timings / info
  bool dualize = false; // primal / dual computation toggle
  int64_t upper_dim = std::numeric_limits< int64_t >::max();
  int64_t pixel_threshold = 0;
  parse_command_line(argc, argv, benchmark, dualize, upper_dim, pixel_threshold, input_filename, output_filename, edge_filename, nx, ny);

  //std::cout << "Check 2: " << nx << ' ' << ny << ' ' << nz << std::endl;

  if (benchmark)
  {
    dipha::globals::benchmark = true;

    dipha::mpi_utils::cout_if_root() << std::endl << "Input filename: " << std::endl << input_filename << std::endl;

    dipha::mpi_utils::cout_if_root() << std::endl << "upper_dim: " << upper_dim << std::endl;

    dipha::mpi_utils::cout_if_root() << std::endl << "Number of processes used: " << std::endl << dipha::mpi_utils::get_num_processes() 
                                     << std::endl;

    dipha::mpi_utils::cout_if_root() << std::endl << "Detailed information for rank 0:" << std::endl;

    dipha::mpi_utils::cout_if_root() << std::setw(11) << "time" << std::setw(13) << "prior mem" << std::setw(13) << "peak mem" 
                                     << std::setw(13) << "bytes recv" << std::endl;
  }

  switch (dipha::file_types::get_file_type(input_filename))
  {
  case dipha::file_types::IMAGE_DATA:
    compute< dipha::inputs::weighted_cubical_complex >(input_filename, dualize, upper_dim, pixel_threshold, output_filename, edge_filename, nx, ny);
    break;
  default:
    dipha::mpi_utils::error_printer_if_root() << "Unknown complex type in DIPHA file" << input_filename << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (benchmark)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    dipha::mpi_utils::cout_if_root() << std::setiosflags(std::ios::fixed) << std::setiosflags(std::ios::showpoint) << std::setprecision(1);
    dipha::mpi_utils::cout_if_root() << std::endl << "Overall running time in seconds: " << std::endl;
    dipha::mpi_utils::cout_if_root() << std::setprecision(1) << MPI_Wtime() - time_at_start << std::endl;

    dipha::mpi_utils::cout_if_root() << std::endl << "Reduction kernel running time in seconds: " << std::endl;
    dipha::mpi_utils::cout_if_root() << std::setprecision(1) << dipha::globals::reduction_kernel_running_time << std::endl;

    int64_t peak_mem = getPeakRSS() >> 20;
    std::vector< int64_t > peak_mem_per_rank(dipha::mpi_utils::get_num_processes());
    MPI_Gather(&peak_mem, 1, MPI_LONG_LONG, peak_mem_per_rank.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    dipha::mpi_utils::cout_if_root() << std::endl << "Overall peak mem in GB of all ranks: " << std::endl;
    dipha::mpi_utils::cout_if_root() << (double)*std::max_element(peak_mem_per_rank.begin(), peak_mem_per_rank.end()) / 1024.0 
                                     << std::endl;

    dipha::mpi_utils::cout_if_root() << std::endl << "Individual peak mem in GB of per rank: " << std::endl;
    for (int64_t peak_mem : peak_mem_per_rank)
    {
      dipha::mpi_utils::cout_if_root() << (double)peak_mem / 1024.0 << std::endl;
    }

    std::vector< int64_t > bytes_received_per_rank(dipha::mpi_utils::get_num_processes());
    MPI_Gather(&dipha::globals::bytes_received, 1, MPI_LONG_LONG, bytes_received_per_rank.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    dipha::mpi_utils::cout_if_root() << std::endl << "Maximal communication traffic (without sorting) in GB between any pair of nodes:" 
                                     << std::endl;
    const int64_t max_num_MB_traffic = *std::max_element(bytes_received_per_rank.begin(), bytes_received_per_rank.end()) >> 20;
    dipha::mpi_utils::cout_if_root() << std::setprecision(1) << (double)(max_num_MB_traffic) / 1024.0 << std::endl;

    dipha::mpi_utils::cout_if_root() << std::endl << "Total communication traffic (without sorting) in GB between all pairs of nodes:" 
                                     << std::endl;
    const int64_t total_num_MB_traffic = std::accumulate(bytes_received_per_rank.begin(), bytes_received_per_rank.end(), 0LL) >> 20;
    dipha::mpi_utils::cout_if_root() << std::setprecision(1) << (double)(total_num_MB_traffic) / 1024.0 << std::endl;
  }

  MPI_Finalize();
  auto end = std::chrono::steady_clock::now();

  std::cout << "Elapsed time in seconds: "
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << " sec" <<std::endl;
}
