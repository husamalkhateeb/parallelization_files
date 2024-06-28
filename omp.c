#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define ROWS 10000
#define COLS 300

int main() {
    double cpu_time;
    double begin, end;
    int rows = ROWS;  // Number of data points
    int cols = COLS;  // Number of features

    // Generate larger random data
    double data[ROWS][COLS];
    srand(42); // Seed for reproducibility

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            data[i][j] = rand() % 100; // You can replace this with your data generation logic
        }
    }

    int k = 1;  // Number of principal components

    double principal_components[1][COLS] = {0.0};

    // Apply PCA
    // pca(data1, rows, cols, k, principal_components);

    // Step 1: Standardize the data
    double mean[COLS] = {0.0};
    double stddev[COLS] = {0.0};
    begin = omp_get_wtime();
    #pragma omp parallel for
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            mean[j] += data[i][j];
        }
        mean[j] /= rows;

        for (int i = 0; i < rows; i++) {
            stddev[j] += pow(data[i][j] - mean[j], 2);
        }
        stddev[j] = sqrt(stddev[j] / rows);
    }

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = (data[i][j] - mean[j]) / stddev[j];
        }
    }

    // Step 2: Compute the covariance matrix
    double covariance_matrix[COLS][COLS] = {{0.0}};

    #pragma omp parallel for reduction(+:covariance_matrix[:cols][:cols])
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            covariance_matrix[j][j] += data[i][j] * data[i][j];

            for (int l = j + 1; l < cols; l++) {
                covariance_matrix[j][l] += data[i][j] * data[i][l];
                covariance_matrix[l][j] += data[i][j] * data[i][l];
            }
        }
    }

    #pragma omp parallel for
    for (int j = 0; j < cols; j++) {
        for (int l = 0; l < cols; l++) {
            covariance_matrix[j][l] /= rows;
        }
    }

    // For simplicity, we assume k = 1
    int top_component = 0;

    #pragma omp parallel for
    for (int j = 0; j < cols; j++) {
        principal_components[0][j] = covariance_matrix[top_component][j];
    }
    end = omp_get_wtime();
    cpu_time = end-begin;

    // Display the principal components
    printf("Principal Components:\n");
    for (int j = 0; j < cols; j++) {
        printf("%.2lf ", principal_components[0][j]);
    }
    printf("\n");
    printf("cpu time = %.4f\n", cpu_time);

    return 0;
}