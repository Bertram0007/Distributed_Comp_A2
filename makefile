MPI:MPI.c
	mpicc -o MPI MPI.c -lm -lpthread
clean:
	rm MPI