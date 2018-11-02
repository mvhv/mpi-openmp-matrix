#include <stdlib.h>
#include <stdio.h>

#define MASTER 0

typedef struct MatEntry {
    int i;
    int j;
    double value;
} MatEntry;

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
    FILE *fp;
    MatEntry *matA, *matB, *result;
    char *pathA, *pathB;
    int sizeA, sizeB, rowsA, rowsB, colsA, colsB, size;
    double value;

    // parse args
    if (argc != 7) {
        return EXIT_FAILURE;
    } else {
        pathA = argv[1];
        rowsA = atoi(argv[2]);
        colsA = atoi(argv[3]);
        pathB = argv[4];
        rowsA = atoi(argv[5]);
        colsB = atoi(argv[6]);
    }

    // read matrices from file
    matA = loadMatrixFromFile(pathA, &sizeA);
    matB = loadMatrixFromFile(pathB, &sizeB);

    // sparse result can't be bigger than sizeA * sizeB worst case
    result = malloc(sizeA * sizeB * sizeof(MatEntry));
    size = 0;

    // compare all pairs of mat market entries and sum to result if valid
    for (int a = 0; a < sizeA; a++) {
        for (int b = 0; b < sizeB; b++) {
            if (matA[a].j == matB[b].i) {
                updateEntry(result, &size, matA[a].i, matB[b].j, matA[a].value * matB[b].value);
            }
        }
    }

    // print result
    for (int n = 0; n < size; n++) {
        printf("%d %d %lf\n", result[n].i, result[n].j, result[n].value);
    }

    return EXIT_SUCCESS;
}