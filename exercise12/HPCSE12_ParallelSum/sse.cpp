// Skeleton code for HPCSE Exam, 18.12.2012
// Profs. P. Koumoutsakos and M. Troyer
// Question 5c)

#include <vector>
#include <numeric>
#include <iostream>
#include <x86intrin.h>
#include "aligned_allocator.hpp"


int main( int argc, char** argv )
{
    // vector size
    const int N = 1600000;

    // initialize 16 byte aligned vectors
    std::vector< float, hpc12::aligned_allocator<float,16> > x(N,-1.2), y(N,3.4), z(N);
    
    
    // DO THE SUM z = x + y
    ...
    
    
    // print result checksum
    std::cout << std::accumulate(z.begin(), z.end(), 0.) << std::endl;
}

