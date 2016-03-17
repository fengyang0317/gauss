#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include "mpi.h"


#define NSIZE       128
#define VERIFY      1

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}
#define SWAPINT(a,b)       {register int tmp; tmp = a; a = b; b = tmp;}
#define ABS(a)          (((a) > 0) ? (a) : -(a))

double **matrix,*B,*V,*C;
int *swap;

/* Allocate the needed arrays */

void allocate_memory(int size)
{
	double *tmp;
	int i;

	matrix = (double**)malloc(size*sizeof(double*));
	assert(matrix != NULL);
    tmp = (double*)malloc(size*(size+1)*sizeof(double));
	assert(tmp != NULL);

	for(i = 0; i < size; i++){
		matrix[i] = tmp;
        tmp = tmp + size + 1;
    }

	B = (double*)malloc(size * sizeof(double));
	assert(B != NULL);
	V = (double*)malloc(size * sizeof(double));
	assert(V != NULL);
	C = (double*)malloc(size * sizeof(double));
	assert(C != NULL);
	swap = (int*)malloc(size * sizeof(int));
	assert(swap != NULL);
}

/* Initialize the matirx with some values that we know yield a
 * solution that is easy to verify. A correct solution should yield
 * -0.5, and 0.5 for the first and last C values consecutively, and 0
 * for the rest, though it should work regardless */

void initMatrix(int nsize)
{
	int i,j;
	for(i = 0 ; i < nsize ; i++){
		for(j = 0; j < nsize ; j++) {
			matrix[i][j] = ((j < i )? 2*(j+1) : 2*(i+1));
		}
        matrix[i][j] = (double)i;
		swap[i] = i;
	}
}

int numtasks, rank;

/* Get the pivot row. If the value in the current pivot position is 0,
 * try to swap with a non-zero row. If that is not possible bail
 * out. Otherwise, make sure the pivot value is 1.0, and return. */

void getPivot(int nsize, int currow)
{
    int i, irow = -1, lrank = (numtasks - currow % numtasks + rank) % numtasks;

    for (i = currow + lrank; i < nsize; i+=numtasks) {
        if (matrix[i][currow] > 0) {
            irow = i;
            break;
        }
    }

    int *recvbuf;
    if (rank == currow % numtasks) {
        recvbuf = malloc(numtasks * sizeof(int));
    }

    MPI_Gather(&irow, 1, MPI_INT, recvbuf, 1, MPI_INT, currow % numtasks, MPI_COMM_WORLD);

    if (rank == currow % numtasks) {
        i = currow % numtasks;
        while (i < numtasks && recvbuf[i] == -1)
            i++;

        if (i == numtasks){
            printf("The matrix is singular\n");
            exit(-1);
        }
    }

    MPI_Bcast(&i, 1, MPI_INT, currow % numtasks, MPI_COMM_WORLD);

    if (i != currow % numtasks){
        if (rank == i) {
            MPI_Status stat;
            int rc = MPI_Sendrecv_replace(matrix[irow], nsize+1, MPI_DOUBLE, currow % numtasks, 0, currow % numtasks, 0, MPI_COMM_WORLD, &stat);
        }
        if (rank == currow % numtasks) {
            MPI_Status stat;
            int rc = MPI_Sendrecv_replace(matrix[currow], nsize+1, MPI_DOUBLE, i, 0, i, 0, MPI_COMM_WORLD, &stat);
        }
    }

    if (rank == currow % numtasks)
	{
		double pivotVal;
		pivotVal = matrix[currow][currow];

		if (pivotVal != 1.0){
			matrix[currow][currow] = 1.0;
            for(i = currow + 1; i <= nsize; i++){
				matrix[currow][i] /= pivotVal;
			}
        }
	}

    MPI_Bcast(matrix[currow], nsize+1, MPI_DOUBLE, currow % numtasks, MPI_COMM_WORLD);
}


/* For all the rows, get the pivot and eliminate all rows and columns
 * for that particular pivot row. */

void computeGauss(int nsize)
{
	int i,j,k;
	double pivotVal;

	for(i = 0; i < nsize; i++){
		getPivot(nsize,i);

		pivotVal = matrix[i][i];

        int lrank = (numtasks - (i+1) % numtasks + rank) % numtasks;
        for (j = i + 1 + lrank ; j < nsize; j+=numtasks){
			pivotVal = matrix[j][i];
			matrix[j][i] = 0.0;
            for (k = i + 1 ; k <= nsize; k++){
				matrix[j][k] -= pivotVal * matrix[i][k];
			}
		}
	}
}


/* Solve the equation. That is for a given A*B = C type of equation,
 * find the values corresponding to the B vector, when B, is all 1's */

void solveGauss(int nsize)
{
	int i,j;

	V[nsize-1] = B[nsize-1];
	for (i = nsize - 2; i >= 0; i --){
		V[i] = B[i];
		for (j = nsize - 1; j > i ; j--){
			V[i] -= matrix[i][j] * V[j];
		}
	}

	for(i = 0; i < nsize; i++){
		C[i] = V[i];//V[swap[i]];
	}
}

extern char * optarg;
int main(int argc,char *argv[])
{
	int i;
	struct timeval start;
	struct timeval finish;
	long compTime;
	double Time;
	int nsize = NSIZE;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	while((i = getopt(argc,argv,"s:")) != -1){
		switch(i){
			case 's':
				{
					int s;
					s = atoi(optarg);
					if (s > 0){
						nsize = s;
					} else {
						fprintf(stderr,"Entered size is negative, hence using the default (%d)\n",(int)NSIZE);
					}
				}
				break;
			default:
				assert(0);
				break;
		}
	}

	allocate_memory(nsize);

	initMatrix(nsize);
	gettimeofday(&start, 0);
	computeGauss(nsize);
	if (rank == 0) {
	gettimeofday(&finish, 0);
        for (i = 0; i < nsize; i++)
            B[i] = matrix[i][nsize];
#if VERIFY
	solveGauss(nsize);
#endif

	compTime = (finish.tv_sec - start.tv_sec) * 1000000;
	compTime = compTime + (finish.tv_usec - start.tv_usec);
	Time = (double)compTime;

	printf("Application time: %f Secs\n",(double)Time/1000000.0);

#if VERIFY
	for(i = 0; i < nsize; i++)
		printf("%6.5f %5.5f\n",B[i],C[i]);
#endif
    }
    MPI_Finalize();
	return 0;
}
