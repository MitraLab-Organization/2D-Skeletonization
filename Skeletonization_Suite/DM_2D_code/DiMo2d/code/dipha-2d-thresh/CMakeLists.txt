#  Copyright 2014 IST Austria
#
#  Contributed by: Ulrich Bauer, Jan Reininghaus
#
#  This file is part of DIPHA.
#
#  DIPHA is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  DIPHA is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with DIPHA.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.5)

project("Dipha")

set(CMAKE_BUILD_TYPE "Release")

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    add_definitions(-std=c++11 -w)
endif()

FIND_PACKAGE(MPI REQUIRED)

include_directories (include externals ${MPI_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS}")

FILE(GLOB_RECURSE all_dipha_includes "include/dipha/*.h")
FILE(GLOB dipha_general_includes "include/dipha/*.h")
FILE(GLOB dipha_data_structures_includes "include/dipha/data_structures/*.h")
FILE(GLOB dipha_input_includes "include/dipha/inputs/*.h")
FILE(GLOB dipha_output_includes "include/dipha/outputs/*.h")
FILE(GLOB dipha_algorithms_includes "include/dipha/algorithms/*.h")

add_executable (dipha src/dipha.cpp ${all_dipha_includes})
target_link_libraries(dipha ${MPI_LIBRARIES})

source_group(Header\ Files FILES ${dipha_general_includes})
source_group(Header\ Files\\inputs FILES ${dipha_input_includes})
source_group(Header\ Files\\outputs FILES ${dipha_output_includes})
source_group(Header\ Files\\data_structures FILES ${dipha_data_structures_includes})
source_group(Header\ Files\\algorithms FILES ${dipha_algorithms_includes})

enable_testing()

string(RANDOM LENGTH 1 ALPHABET 0123456789 NUM_PROCESSES)
math(EXPR NUM_PROCESSES "${NUM_PROCESSES} + 1") 
#set(NUM_PROCESSES 1)
message( STATUS "NUM_PROCESSES = ${NUM_PROCESSES}") 

FILE(GLOB functional_test_temp_data "test_data/*.DIPHA_TEST")
if(NOT( functional_test_temp_data STREQUAL "" ) )
	execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${functional_test_temp_data}) 
endif()

SET(DEFAULT_UPPER_DIM 9223372036854775807)

