#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <math.h>
#include <mpi.h>

struct thread_data
{
    float** matrix;
    float** resultMatrix;
    float* blockingResult;
    float** blockIndex1;
    float** blockIndex2;
    int iteration;
    int n;
    int m;
    int b;
    int startI;
    int startJ;
    int number;
    int threadCounter;
    int blocks;
    int startNumA;
    int startIterateA;
};

void test(float **matrix, float *testResult, int n, int m){
    int testCounter = 0;
    for(int row=0; row<n; row++){
        for(int iterateRow = row; iterateRow < n; iterateRow++){
            float temp = 0;
            for(int column=0; column<m; column++){
                temp += matrix[row][column] * matrix[iterateRow][column];
            }
            testResult[testCounter++] = temp;
        }
    }
}

float** generateMatrix(int n, int m)
{
    float* matrix0 = malloc(n * m * sizeof(float));
    float** matrix = malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++)
    {
        matrix[i] = &matrix0[i * m];
    }
    return matrix;
}

void *baseline(void *data){
    struct thread_data *p = (struct thread_data *)data;
    int i, j;
    int numberTemp = p->number;
    //start from startI for outer for loop
    for (i = p->startI; i < p->n; ++i) {
        //start from startJ for inner for loop
        for(j = i == p->startI? p->startJ : 0; j < p->n; j++){
            float sum = 0;
            numberTemp--;
            for(int column=0; column<p->m; column++){
                //loop unrolling with factor of 4,3,2, and 1
                if(column + 4 < p->m && numberTemp >= 4){
                    sum += p->blockIndex1[i][column] * p->blockIndex2[j][column];
                    sum += p->blockIndex1[i][column+1] * p->blockIndex2[j][column+1];
                    sum += p->blockIndex1[i][column+2] * p->blockIndex2[j][column+2];
                    sum += p->blockIndex1[i][column+3] * p->blockIndex2[j][column+3];
                    sum += p->blockIndex1[i][column+4] * p->blockIndex2[j][column+4];
                    column+=4;
                }else if(column + 3 < p->m && numberTemp >= 3){
                    sum += p->blockIndex1[i][column] * p->blockIndex2[j][column];
                    sum += p->blockIndex1[i][column+1] * p->blockIndex2[j][column+1];
                    sum += p->blockIndex1[i][column+2] * p->blockIndex2[j][column+2];
                    sum += p->blockIndex1[i][column+3] * p->blockIndex2[j][column+3];
                    column+=3;
                }else if(column + 2 < p->m && numberTemp >= 2){
                    sum += p->blockIndex1[i][column] * p->blockIndex2[j][column];
                    sum += p->blockIndex1[i][column+1] * p->blockIndex2[j][column+1];
                    sum += p->blockIndex1[i][column+2] * p->blockIndex2[j][column+2];
                    column+=2;
                }else if(column + 1 < p->m && numberTemp >= 1){
                    sum += p->blockIndex1[i][column] * p->blockIndex2[j][column];
                    sum += p->blockIndex1[i][column+1] * p->blockIndex2[j][column+1];
                    column+=1;
                }else{
                    sum += p->blockIndex1[i][column] * p->blockIndex2[j][column];
                }
            }
            p->resultMatrix[i][j + p->iteration * p->b] = sum;
            //all the items has been calculated
            if(numberTemp == 0){
                return NULL;
            }
        }
    }
    return NULL;
}


int main(int argc, char** argv){
    struct timeval
            start,
            end;
    int N, M, processId, P, B, T;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);


    if(argc == 4){
        N = atoi(argv[1]);
        M = atoi(argv[2]);
        T = atoi(argv[3]);
        if(processId == 0){
            printf("N = %d, M = %d, T = %d\n\n", N, M, T);
        }
    }
    else{
        printf("Usage: %s N M\n\n"
               " N: matrix row length\n"
               " M: matrix column length\n"
               " T: T threads for each process\n\n",argv[0]);
        return 1;
    }

    struct thread_data thread_data_array[T];
    pthread_t tid[T];

    if (P % 2 != 1){
        if (processId == 0){
            printf("the process number must be odd!");
        }
        goto end;
    }else if (N % P != 0){
        if (processId == 0){
            printf("the remainder of N/P must be 0.");
        }
        goto end;
    }

    B = N/P;
    int iteration = 0;
    MPI_Request requests;
    MPI_Status status;

    //1. create the original matrix and assign random values
    float** matrix = generateMatrix(N,M);
    if(processId == 0){
        for (int i = 0; i < N; i++){
            for (int j = 0; j < M; j++){
                matrix[i][j] = (float)rand()/RAND_MAX;
            }
        }
    }
    //create the matrix for blockIndex1
    float** blockIndex1 = generateMatrix(B, M);
    //create the matrix for blockIndex2
    float** blockIndex2 = generateMatrix(B,M);


    if(processId == 0) {
        srand(time(0));
        //used to calculate the running time
        gettimeofday(&start, 0);
        //Compute the block matrix on the diagonal
        for(int i=0; i<B; i++){
            for(int j=0; j<M; j++){
                blockIndex1[i][j] = matrix[i][j];
            }
        }
        if(P > 1){
            //send the corresponding part of the original matrix to the blockIndex1
            for(int i=1; i<P; i++){
                MPI_Isend(matrix[i*B], M * B, MPI_FLOAT, i, 1, MPI_COMM_WORLD,  &requests);
            }
        }
    }
    if(processId > 0){
        //receive the corresponding part of the original matrix and store the data to the blockIndex1
        MPI_Recv(blockIndex1[0], M * B, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
    }

    //in the diagonal, all the items in blockIndex1 = blockIndex2
    for(int i=0; i<B; i++){
        for(int j=0; j<M; j++){
            blockIndex2[i][j] = blockIndex1[i][j];
        }
    }
    //create the result matrix for each block multiplication
    float** resultMatrix = generateMatrix(B, B * (P + 1) / 2);
    //Build the final result matrix
    float** TotalResult = generateMatrix(N, B * (P + 1) / 2);
    if(P > 0) {
        //in the diagonal, calculate the block matrix when iteration == 0
        if (iteration == 0) {
            for (int i = 0; i < B; i++) {
                for (int j = i; j < B; j++) {
                    float sum = 0;
                    for (int k = 0; k < M; k++) {
                        sum += blockIndex1[i][k] * blockIndex2[j][k];
                    }
                    resultMatrix[i][j] = sum;
                }
            }
            iteration += 1;
            if(P == 1){
                //calculate the time when P = 1
                gettimeofday(&end, 0);
                long seconds = end.tv_sec - start.tv_sec;
                long microseconds = end.tv_usec - start.tv_usec;
                double period = seconds + microseconds/1000000.0;
                printf("time period for single process:  %8f \n", period);
                goto end;
            }
        }
        if(iteration > 0){
            while(iteration < (P+1)/2){
                    //send the block(index2) to the previous process, and receive the block(index2) from the next process
                    MPI_Sendrecv_replace(blockIndex2[0], M*B, MPI_FLOAT, processId == 0 ? P - 1 : processId - 1, 1, (processId + 1)%P, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    float total = B*B;   //total number of multiplications
                    if(total < T){
                        T = total;
                    }
                    float each = total/(float )T;   //number of multiplications for each thread
                    int numberCeil = ceil(each);  //the ceiling number of multiplications for each thread
                    int numberFloor = floor(each);   //the floor number of multiplications for each thread
                    int number = 0;     //will be assigned with numberCeil/numberFloor
                    int startI = 0, startJ = 0;    //start index for the outer for loop and inner for loop
                    int counter = 0;
                    for(int m=0; m<T; m++){
                        if((total-counter)/(T-m) > each){
                            number = numberCeil;
                        }else{
                            number = numberFloor;
                        }
                        counter += number;
                        thread_data_array[m].matrix = matrix;
                        thread_data_array[m].resultMatrix = resultMatrix;
                        thread_data_array[m].n = B;
                        thread_data_array[m].m = M;
                        thread_data_array[m].b = B;
                        thread_data_array[m].number = number;
                        thread_data_array[m].startI = startI;
                        thread_data_array[m].startJ = startJ;
                        thread_data_array[m].iteration = iteration;
                        thread_data_array[m].blockIndex1 = blockIndex1;
                        thread_data_array[m].blockIndex2 = blockIndex2;
                        pthread_create(&tid[m], NULL, baseline, &thread_data_array[m]);

                        //calculate the start index for inner for loop and outer for loop
                        startJ += number;
                        while(startJ >= B && startI <= B-1){
                            startI++;
                            startJ = startJ - B;
                            if(startI >= B || (startI == B-1 && startJ > B-1)){
                                break;
                            }
                        }
                    }
                    for (int m = 0; m < T; m++){
                        pthread_join(tid[m], NULL);
                    }
                    iteration++;
            }
            if(P > 1){
                MPI_Isend(resultMatrix[0], B*B*(P+1)/2, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &requests);
            }
            if(processId == 0){
                for(int i=0; i<P; i++){
                    MPI_Recv(TotalResult[i*B], B*B*(P+1)/2, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
                }
            }
        }
    }

    if(processId == 0){
        float * finalResult = (float*)malloc(N*(N+1)/2*sizeof(float));
        for(int i=0; i<N; i++){
            for(int j=i % B; j<B*(P+1)/2; j++){
                int previousItems = 0;
                //if it's the top half of a triangle, keep the position the same
                if(i <= (int )floor((float )N/(float )2)){
                    previousItems = (2*N - i + 1)*i/2 + j - i % B ;
                }else{
                    //if it's the bottom left of the triangle, keep the position the same
                    if(j - i%B< N - i){
                        previousItems = (2*N - i + 1)*i/2 + j - i%B;
                    }else{
                        //otherwise, convert the index and move to the up right of the triangle
                        int blockIndexI = (i/B);
                        int blockIndexJ = j/B - (int )ceil((float )N-(float )i)/(float )B;
                        int row = i%B;
                        int column = j%B;
                        int newI = blockIndexJ*B + column;
                        int newJ = blockIndexI*B + row;
                        if(newI == 0){
                            previousItems = newJ;
                        }else if(newI == 1){
                            int merge = B == 1 ? newI : newI % B;
                            previousItems = N + newJ - merge;
                        }else{
                            previousItems = (2*N - newI + 1)*newI/2 + newJ - newI ;
                        }
                    }
                }
                finalResult[previousItems] = TotalResult[i][j];
            }
        }
        gettimeofday(&end, 0);
        long seconds = end.tv_sec - start.tv_sec;
        long microseconds = end.tv_usec - start.tv_usec;
        double period = seconds + microseconds/1000000.0;
        printf("time period for mpi:  %8f \n", period);

        float *testResult = (float*)malloc(N*(N+1)/2*sizeof(float));
        test(matrix, testResult, N, M);

        for(int i=0; i<N*(N+1)/2; i++){
            if(testResult[i] != finalResult[i]){
                printf("\nresult %d false", i);
            }
        }
    }
    free(matrix);
    free(resultMatrix);
    free(TotalResult);
    free(blockIndex1);
    free(blockIndex2);

    end:
    MPI_Finalize();
    return 0;
}
