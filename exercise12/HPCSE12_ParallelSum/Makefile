CXX=g++
MPICXX=mpic++
CXXFLAGS=-Wall -std=c++11 -lpthread

all: omp threads sse mpi

omp: omp.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o omp omp.cpp

threads: threads.cpp
	$(CXX) $(CXXFLAGS) -o threads threads.cpp

sse: sse.cpp
	$(CXX) $(CXXFLAGS) -msse -o sse sse.cpp

mpi: mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o mpi mpi.cpp

clean:
	rm -f omp threads mpi

