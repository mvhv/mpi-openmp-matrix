#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

static const int MASTER = 0;
static const int FROM_MASTER = 1;
static const int FROM_WORKER = 2;

static MPI_Datatype MPI_MAT_ENTRY;

typedef struct MatEntry {
    int i;
    int j;
    double value;
} MatEntry;

int compRowMajor(const void * a, const void * b) {
    MatEntry matA = *((MatEntry*)a);
    MatEntry matB = *((MatEntry*)b);

    // sort by row
    if (matA.i > matB.i) return 1;
    if (matA.i < matB.i) return -1;
    // if on same row, secondary sort by column
    if (matA.j > matB.j) return 1;
    if (matA.j < matB.j) return -1;
    return 0;
}

int compColMajor(const void * a, const void * b) {
    MatEntry matA = *((MatEntry*)a);
    MatEntry matB = *((MatEntry*)b);

    // sort by column
    if (matA.j > matB.j) return 1;
    if (matA.j < matB.j) return -1;
    // if in same column, secondary sort by row
    if (matA.i > matB.i) return 1;
    if (matA.i < matB.i) return -1;
    return 0;
}

int countLines(FILE *fp) {
    int count = 0;
    char ch;

    // count newlines
    while (!feof(fp)) {
        ch = fgetc(fp);
        if (ch == '\n') count++;
    }
    // reset file pointer for actual use
    fseek(fp, 0, SEEK_SET);
    return count;
}

MatEntry *loadMatrixFromFile(char *path, int *count) {
    FILE *fp;
    int length, i, j;
    double value;
    MatEntry *matrix;

    // open file and check success
    fp = fopen(path, "r");
    if (fp == NULL) {
        perror(path);
        exit(EXIT_FAILURE);
    }

    // count lines and rewind
    length = countLines(fp);

    // allocate memory for matrix
    matrix = malloc(length * sizeof(MatEntry));

    // read file and record entries
    for (int n = 0; n < length; n++) {
        fscanf(fp, "%d %d %lf", &i, &j, &value);
        matrix[n].i = i;
        matrix[n].j = j;
        matrix[n].value = value;
    }

    *count = length;
    return matrix;
}

void updateEntry(MatEntry *result, int *size, int i, int j, double value) {
    // check for existing entry
    for (int n = 0; n < *size; n++) {
        if (result[n].i == i && result[n].j == j) {
            result[n].value += value;
            return;
        }
    }

    // else insert new struct
    result[*size] = (MatEntry) {.i = i, .j = j, .value = value};
    (*size)++;
    return;
}

int main(int argc, char **argv) {
    MatEntry *matA, *matB, *reduced, *partial, *result;
    char *pathA, *pathB;
    int sizeA, sizeB, rowsA, rowsB, colsA, colsB;
    int commRank, commSize, commWorkers, rc, provided;
    int entriesPart, entriesExtra, entriesOffset, entries;
    int reducedSize, partialSize, reducedMax;
    double value;
    MPI_Status status;

    // init MPI in funneled thread mode to ensure one thread per node
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    // don't count the head node as a worker
    commWorkers = commSize - 1;

    // define MPI type for MatEntry structs
    int count = 3;
    int blocklengths[] = {1, 1, 1};
    MPI_Aint offsets[] = {offsetof(MatEntry, i),
                          offsetof(MatEntry, j),
                          offsetof(MatEntry, value)};
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_DOUBLE};
    MPI_Datatype tempType;
    MPI_Type_create_struct(count, blocklengths, offsets, types, &tempType);
    MPI_Aint lb, extent;
    MPI_Type_get_extent(tempType, &lb, &extent);
    MPI_Type_create_resized(tempType, lb, extent, &MPI_MAT_ENTRY);
    MPI_Type_commit(&MPI_MAT_ENTRY);


    if (commRank == MASTER) { // MASTER NODE --------------------------------------
        // parse args
        if (argc != 7) {
            printf("Expected 6 arguments, recieved %d\n", argc);
            MPI_Abort(MPI_COMM_WORLD, rc);
            exit(EXIT_FAILURE);
        } else {
            pathA = argv[1];
            rowsA = atoi(argv[2]);
            colsA = atoi(argv[3]);
            pathB = argv[4];
            rowsA = atoi(argv[5]);
            colsB = atoi(argv[6]);
        }

        // load and sort first matrix in row major order
        matA = loadMatrixFromFile(pathA, &sizeA);
        qsort(matA, sizeA, sizeof(*matA), compRowMajor);
        // load and sort second matrix in column major order
        matB = loadMatrixFromFile(pathB, &sizeB);
        qsort(matB, sizeB, sizeof(*matB), compColMajor);

        // partition tasks by entries from the first matrix
        entriesPart = sizeA / commWorkers;
        entriesExtra = sizeA % commWorkers;
        entriesOffset = 0;

        // send data to workers
        for (int node = 1; node < commSize; node++) {
            // send matA partition size
            entries = entriesPart;
            if (node <= entriesExtra) entries++;
            MPI_Send(&entries, 1, MPI_INT, node, FROM_MASTER, MPI_COMM_WORLD);
            // send matA partition
            MPI_Send(matA+entriesOffset, entries, MPI_MAT_ENTRY, node, FROM_MASTER, MPI_COMM_WORLD);
            // send matB size
            MPI_Send(&sizeB, 1, MPI_INT, node, FROM_MASTER, MPI_COMM_WORLD);
            // send matB
            MPI_Send(matB, sizeB, MPI_MAT_ENTRY, node, FROM_MASTER, MPI_COMM_WORLD);
            entriesOffset += entries;
        }

        // prepare to store data from workers for later reduction
        int taskResultSizes[commWorkers];
        int taskResultSize;
        MatEntry *taskResults[commWorkers];
        MatEntry *taskResult;
        reducedMax = 0;

        // recieve data from workers
        for (int node = 1; node < commSize; node++) {
            // recieve number of result entries
            MPI_Recv(&taskResultSize, 1, MPI_INT, node, FROM_WORKER, MPI_COMM_WORLD, &status);
            // recieve result entries
            taskResult = malloc(taskResultSize * sizeof(*taskResult));
            MPI_Recv(taskResult, taskResultSize, MPI_MAT_ENTRY, node, FROM_WORKER, MPI_COMM_WORLD, &status);
            taskResultSizes[node - 1] = taskResultSize;
            taskResults[node - 1] = taskResult;
            reducedMax += taskResultSize;
        }

        // reduce data from workers
        reducedSize = 0;
        MatEntry reduced[reducedMax];
        for (int task = 0; task < commWorkers; task++) {
            for (int entry = 0; entry < taskResultSizes[task]; entry++) {
                result = &taskResults[task][entry];
                updateEntry(reduced, &reducedSize, result->i, result->j, result->value);
            }
        }

        // sort and print result
        qsort(reduced, reducedSize, sizeof(*reduced), compRowMajor);
        for (int n = 0; n < reducedSize; n++) {
            printf("%d %d %lf\n", reduced[n].i, reduced[n].j, reduced[n].value);
        }
    } else { // WORKER NODES --------------------------------------------------
        // recieve matA allocation size
        MPI_Recv(&sizeA, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        // recieve matA allocation
        matA = malloc(sizeA * sizeof(*matA));
        MPI_Recv(matA, sizeA, MPI_MAT_ENTRY, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        // recieve matB size
        MPI_Recv(&sizeB, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        // recieve matB
        matB = malloc(sizeB * sizeof(*matB));
        MPI_Recv(matB, sizeB, MPI_MAT_ENTRY, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);

        // target for worker reduction
        MatEntry reduced[sizeA * sizeB];
        reducedSize = 0;

        // paralellise updates
        #pragma omp parallel 
        {
            // malloc memory for thread
            MatEntry partial[sizeA * sizeB];
            int partialSize = 0;

            // compare all pairs of mat entries
            #pragma omp for
            {
                for (int a = 0; a < sizeA; a++) {
                    for (int b = 0; b < sizeB; b++) {
                        if (matA[a].j == matB[b].i) {
                            updateEntry(partial, &partialSize, matA[a].i, matB[b].j, matA[a].value * matB[b].value);
                        }
                    }
                }
            }

            // concatenate private arrays
            #pragma omp critical
            {
                for (int n = 0; n < partialSize; n++) {
                    updateEntry(reduced, &reducedSize, partial[n].i, partial[n].j, partial[n].value);
                }
            }
        }

        // send result to master
        MPI_Send(&reducedSize, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(reduced, reducedSize, MPI_MAT_ENTRY, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

// A * B = C
// Cij = Ai1 * B1j + Ai2 * B2j + ... + Ain * Bnj