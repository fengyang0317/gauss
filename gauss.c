/* 
 * Original author:  UNKNOWN
 *
 * Modified:         Kai Shen (January 2010)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <semaphore.h>
#include <sched.h>

/* #define DEBUG */

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}

/* Solve the equation:
 *   matrix * X = R
 */

double **matrix, *X, *R;

/* Pre-set solution. */

double *X__;
int nsize = 0;
int num_threads = 1;
sem_t th;

/* Initialize the matirx. */

int initMatrix(const char *fname)
{
	FILE *file;
	int l1, l2, l3;
	double d;
	int nsize;
	int i, j;
	double *tmp;
	char buffer[1024];

	if ((file = fopen(fname, "r")) == NULL) {
		fprintf(stderr, "The matrix file open error\n");
		exit(-1);
	}

	/* Parse the first line to get the matrix size. */
	fgets(buffer, 1024, file);
	sscanf(buffer, "%d %d %d", &l1, &l2, &l3);
	nsize = l1;
#ifdef DEBUG
	fprintf(stdout, "matrix size is %d\n", nsize);
#endif

	/* Initialize the space and set all elements to zero. */
	matrix = (double**)malloc(nsize*sizeof(double*));
	assert(matrix != NULL);
	tmp = (double*)malloc(nsize*nsize*sizeof(double));
	assert(tmp != NULL);    
	for (i = 0; i < nsize; i++) {
		matrix[i] = tmp;
		tmp = tmp + nsize;
	}
	for (i = 0; i < nsize; i++) {
		for (j = 0; j < nsize; j++) {
			matrix[i][j] = 0.0;
		}
	}

	/* Parse the rest of the input file to fill the matrix. */
	for (;;) {
		fgets(buffer, 1024, file);
		sscanf(buffer, "%d %d %lf", &l1, &l2, &d);
		if (l1 == 0) break;

		matrix[l1-1][l2-1] = d;
#ifdef DEBUG
		fprintf(stdout, "row %d column %d of matrix is %e\n", l1-1, l2-1, matrix[l1-1][l2-1]);
#endif
	}

	fclose(file);
	return nsize;
}

/* Initialize the right-hand-side following the pre-set solution. */

void initRHS(int nsize)
{
	int i, j;

	X__ = (double*)malloc(nsize * sizeof(double));
	assert(X__ != NULL);
	for (i = 0; i < nsize; i++) {
		X__[i] = i+1;
	}

	R = (double*)malloc(nsize * sizeof(double));
	assert(R != NULL);
	for (i = 0; i < nsize; i++) {
		R[i] = 0.0;
		for (j = 0; j < nsize; j++) {
			R[i] += matrix[i][j] * X__[j];
		}
	}
}

/* Initialize the results. */

void initResult(int nsize)
{
	int i;

	X = (double*)malloc(nsize * sizeof(double));
	assert(X != NULL);
	for (i = 0; i < nsize; i++) {
		X[i] = 0.0;
	}
}

/* Get the pivot - the element on column with largest absolute value. */

void getPivot(int nsize, int currow)
{
	int i, pivotrow;

	pivotrow = currow;
	for (i = currow+1; i < nsize; i++) {
		if (fabs(matrix[i][currow]) > fabs(matrix[pivotrow][currow])) {
			pivotrow = i;
		}
	}

	if (fabs(matrix[pivotrow][currow]) == 0.0) {
		fprintf(stderr, "The matrix is singular\n");
		exit(-1);
	}

	if (pivotrow != currow) {
#ifdef DEBUG
		fprintf(stdout, "pivot row at step %5d is %5d\n", currow, pivotrow);
#endif
		//for (i = currow; i < nsize; i++) {
		//	SWAP(matrix[pivotrow][i],matrix[currow][i]);
		//}
		double *t = matrix[currow];
		matrix[currow] = matrix[pivotrow];
		matrix[pivotrow] = t;
		SWAP(R[pivotrow],R[currow]);
	}
}

pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int arrived = 0;

void print_core() {
	int i;
	cpu_set_t get;
	if (pthread_getaffinity_np(pthread_self(), sizeof(get), &get) < 0) {
		printf("get error\n");
		exit(1);
	}
	for (i = 0; i < 8; i++) {
		if (CPU_ISSET(i, &get))
			printf("Thread %d is running in processor %d\n", (int)pthread_self(), i);
	}
}

void *fact_row(void* _id) {
	print_core();
	int id = (int)_id;
	int cols;
	int i = 0, j, k;
	while (1) {
		pthread_mutex_lock (&mut);	//lock
		arrived++;
		if (arrived == num_threads) {
			arrived = 0;
			sem_post(&th);
		}
		pthread_cond_wait (&cond, &mut);
		pthread_mutex_unlock(&mut);
		cols = nsize - i - 1;
		if (!cols)
			break;
		double pivotval;
		for (j = i + 1 + id * cols / num_threads; j < i + 1 + (id + 1) * cols / num_threads; j++) {
			pivotval = matrix[j][i];
			matrix[j][i] = 0.0;
			for (k = i + 1; k < nsize; k++) {
				matrix[j][k] -= pivotval * matrix[i][k];
			}
			R[j] -= pivotval * R[i];
		}
		i++;
	}
	pthread_exit(NULL);
}

/* For all the rows, get the pivot and eliminate all rows and columns
 * for that particular pivot row. */

void computeGauss(int nsize)
{
	int i, j;
	double pivotval;
	void *status;
	sem_init(&th, 0, 0);
	pthread_t *threads = (pthread_t*) malloc(num_threads * sizeof(pthread_t));

	cpu_set_t mask;
	for (i = 0; i < num_threads; i++) {
		pthread_create(threads + i, NULL, fact_row, (void*)i);
		CPU_ZERO(&mask);
		CPU_SET(i, &mask);
		if (pthread_setaffinity_np(threads[i], sizeof(mask), &mask) < 0) {
			printf("pin error\n");
			exit(1);
		}
	}

	CPU_ZERO(&mask);
	CPU_SET(i, &mask);
	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
		printf("pin error\n");
		exit(1);
	}
	print_core();

	int num = sysconf(_SC_NPROCESSORS_CONF);
	for (i = 0; i < nsize; i++) {
		sem_wait(&th);
		getPivot(nsize,i);

		/* Scale the main row. */
		pivotval = matrix[i][i];
		if (pivotval != 1.0) {
			matrix[i][i] = 1.0;
			for (j = i + 1; j < nsize; j++) {
				matrix[i][j] /= pivotval;
			}
			R[i] /= pivotval;
		}
		pthread_cond_broadcast(&cond);
	}
	for (j = 0; j < num_threads; j++)
		pthread_join(threads[j], &status);
}

/* Solve the equation. */

void solveGauss(int nsize)
{
	int i, j;

	X[nsize-1] = R[nsize-1];
	for (i = nsize - 2; i >= 0; i --) {
		X[i] = R[i];
		for (j = nsize - 1; j > i; j--) {
			X[i] -= matrix[i][j] * X[j];
		}
	}

#ifdef DEBUG
	fprintf(stdout, "X = [");
	for (i = 0; i < nsize; i++) {
		fprintf(stdout, "%.6f ", X[i]);
	}
	fprintf(stdout, "];\n");
#endif
}

int main(int argc, char *argv[])
{
	int i;
	struct timeval start, finish;
	double error;
	char rc;

	while ((rc = getopt (argc, argv, "hp:")) != -1)
		switch (rc) 
		{
			case 'p':
				num_threads = atoi (optarg);
				break;
			case 'h':
				printf("Usage: ./gauss [-p <num threads>] matrix file\n");
				return 0;
		}

	if (argc <= optind) {
		printf("Usage: ./gauss [-p <num threads>] matrix file\n");
		return 0;
	}

	printf ("%d tasks\n", num_threads);

	nsize = initMatrix(argv[optind]);

	initRHS(nsize);
	initResult(nsize);

	gettimeofday(&start, 0);
	computeGauss(nsize);
	gettimeofday(&finish, 0);

	solveGauss(nsize);

	fprintf(stdout, "Time:  %f seconds\n", (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001);

	error = 0.0;
	for (i = 0; i < nsize; i++) {
		double error__ = (X__[i]==0.0) ? 1.0 : fabs((X[i]-X__[i])/X__[i]);
		if (error < error__) {
			error = error__;
		}
	}
	fprintf(stdout, "Error: %e\n", error);

	return 0;
}
