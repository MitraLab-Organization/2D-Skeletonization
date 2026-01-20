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

#pragma once

#include <dipha/includes.h>

namespace dipha
{
  namespace algorithms
  {
    template<typename Complex>
    void compute_reduced_columns(const inputs::abstract_weighted_cell_complex<Complex>& complex,
                                 bool dualize,
                                 int64_t upper_dim,
                                 int64_t pixel_threshold,
                                 data_structures::distributed_vector<int64_t>& filtration_to_cell_map,
                                 data_structures::write_once_column_array& reduced_columns)
    {
      std::cout << "computing reduced columns" << std::endl;
      DIPHA_MACROS_BENCHMARK(get_filtration_to_cell_map(complex, dualize, pixel_threshold, filtration_to_cell_map); );
      std::cout << "filtration to cell map" << std::endl;
      data_structures::distributed_vector<int64_t> cell_to_filtration_map;
      DIPHA_MACROS_BENCHMARK(get_cell_to_filtration_map(complex.get_num_cells(), filtration_to_cell_map, cell_to_filtration_map); );
      std::cout << "cell to filtration map" << std::endl;

      reduced_columns.init(filtration_to_cell_map.get_size());
      const int64_t max_dim = std::min(upper_dim, complex.get_max_dim());
      for (int64_t idx = 0; idx < max_dim; idx++)
      {
        std::cout << "dim: " << idx << std::endl;
        int64_t cur_dim = dualize ? idx : max_dim - idx;
        data_structures::flat_column_stack unreduced_columns;
        DIPHA_MACROS_BENCHMARK(generate_unreduced_columns(complex, filtration_to_cell_map, cell_to_filtration_map, 
                                                          cur_dim, dualize, unreduced_columns); );
        std::cout << "generate_unreduced_columns" << std::endl;
        DIPHA_MACROS_BENCHMARK(reduction_kernel(complex.get_num_cells(), unreduced_columns, reduced_columns); );
        std::cout << "reduction_kernel" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}
