#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Функция инициализации матрицы A и вектора b.
// Матрица A заполняется так, что A(i, j) = 2.0, если i == j, иначе 1.0.
// Вектор b: для i = 0 и i = 1, b[i] = N + 1, для остальных b[i] = N.
void initializeData(int N, vector<double> &A, vector<double> &b) {
    A.resize(N * N);
    b.resize(N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0 : 1.0;
        }
    }
    for (int i = 0; i < N; i++) {
        b[i] = (i == 0 || i == 1) ? (N + 1) : N;
    }
}

int main(int argc, char *argv[]) {
    int N = 4000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // Выделение памяти для матрицы A, векторов b, x и r.
    // x - начальное приближение (инициализируем нулями)
    // r - для хранения результата умножения A*x
    vector<double> A, b, x(N, 0.0), r(N, 0.0);
    initializeData(N, A, b);

    // Параметры итерационного метода
    double tau = 1e-4;  // шаг
    double eps = 1e-7;  // требуемая относительная точность
    int maxIter = 10000;// максимальное число итераций

    // Вычисление нормы вектора b
    double normB = 0.0;
    for (int i = 0; i < N; i++) {
        normB += b[i] * b[i];
    }
    normB = sqrt(normB);

    // Замер времени начала вычислений
    auto start = chrono::high_resolution_clock::now();

    int iter;
    for (iter = 0; iter < maxIter; iter++) {
// 1) Вычисление произведения матрицы A на вектор x: r = A*x
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += A[i * N + j] * x[j];
            }
            r[i] = sum;
        }

        // 2) Вычисление суммы квадратов невязки: res_sq = ||b - r||^2
        double res_sq = 0.0;
#pragma omp parallel for reduction(+ : res_sq)
        for (int i = 0; i < N; i++) {
            double diff = b[i] - r[i];
            res_sq += diff * diff;
        }
        double normRes = sqrt(res_sq);
        double relRes = normRes / normB;

        // Вывод информации каждые 100 итераций (вывод может выполняться одним потоком)
        if (iter % 100 == 0) {
            cout << "Iter " << iter << ", relative residual = " << relRes << endl;
        }
        if (relRes < eps) {
            break;
        }

// 3) Обновление вектора x: x = x + tau*(b - r)
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            x[i] += tau * (b[i] - r[i]);
        }
    }

    int maxThreads = omp_get_max_threads();
    std::cout << "Maximum available threads: " << maxThreads << std::endl;

    // Замер времени окончания вычислений
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "\nConverged after " << iter << " iterations." << endl;
    cout << "Total elapsed time: " << elapsed.count() << " seconds." << endl;

    char *ompThreads = getenv("OMP_NUM_THREADS");
    if (ompThreads != nullptr) {
        cout << "OMP_NUM_THREADS: " << ompThreads << endl;
    } else {
        cout << "OMP_NUM_THREADS is not set." << endl;
    }
    
    cout << "\nComputed solution x (first 10 entries):" << endl;
    for (int i = 0; i < min(N, 10); i++) {
        cout << "x[" << i << "] = " << x[i] << endl;
    }
    cout << "\nExpected solution x (vector of ones):" << endl;
    for (int i = 0; i < min(N, 10); i++) {
        cout << "1 ";
    }
    cout << endl;

    return 0;
}
