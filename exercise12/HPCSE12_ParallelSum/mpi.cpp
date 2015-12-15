// Skeleton code for HPCSE Exam, 18.12.2012
// Profs. P. Koumoutsakos and M. Troyer
// Question 5d)

#include <vector>
#include <numeric>
#include <cassert>
#include <mpi.h>
#include <iostream>

int main( int argc, char** argv )
{
    // vector size
    const int N = 1600000;
    int num_processes, rank;
    
    // SET UP COMMUNICATION.
    // DETERMINE num_processes AND OUR rank
    ...
    
    
    // initialize local parts of the vectors and do the sum z = x + y
    assert( N % num_processes == 0 );
    int nlocal = N / num_processes;
    std::vector<float> x(nlocal,-1.2), y(nlocal,3.4), z(nlocal);
    for( int i = 0; i < nlocal; i++ )
        z[i] = x[i] + y[i];
    
    if( rank == 0 )
    {
        std::vector<float> fullz(N);
        
        
        // COLLECT ALL PARTS INTO fullz
        ...
        
        
        // print result checksum
        std::cout << std::accumulate( fullz.begin(), fullz.end(), 0. ) << std::endl;
    }
    else
    {
        // SEND LOCAL PART z TO ROOT PROCESS
        ...
    }
    
    // CLEAN UP
    ...
}
