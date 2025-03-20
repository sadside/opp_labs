#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Функция инициализации матрицы A (размера N x N) и вектора b (размера N).
// Матрица A: A(i,j) = 2.0, если i == j, иначе 1.0.
// Вектор b: для i = 0 и i = 1 значение b[i] = N + 1, для остальных b[i] = N.
void initializeData(int N, vector<double>& A, vector<double>& b) {
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

int main(int argc, char* argv[]) {
    int N = 4000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // Выделение памяти под матрицу A и векторы b, x, r.
    vector<double> A, b, x(N, 0.0), r(N, 0.0);
    initializeData(N, A, b);

    // Параметры итерационного метода
    double tau = 1e-4;      // шаг
    double eps = 1e-7;      // требуемая относительная точность
    int maxIter = 10000;    // максимальное число итераций

    // Вычисление нормы вектора b
    double normB = 0.0;
    for (int i = 0; i < N; i++) {
        normB += b[i] * b[i];
    }
    normB = sqrt(normB);

    auto start = chrono::high_resolution_clock::now();

    int iter = 0;
    bool done = false;

    // Объявляем переменную для редукции вне цикла while, чтобы она была shared для всех потоков.
    double localResSq;

// Единая параллельная область, охватывающая весь итерационный алгоритм.
#pragma omp parallel shared(iter, done, x, r, localResSq)
    {
        while (true) {
// Синхронизация перед началом итерации.
#pragma omp barrier

// 1) Вычисление r = A * x
#pragma omp for
            for (int i = 0; i < N; i++) {
                double sum = 0.0;
                for (int j = 0; j < N; j++) {
                    sum += A[i * N + j] * x[j];
                }
                r[i] = sum;
            }

#pragma omp barrier

// Инициализируем переменную для редукции в одном потоке
#pragma omp single
            {
                localResSq = 0.0;
            }
#pragma omp barrier

// 2) Вычисление суммы квадратов невязки ||b - r||^2 с использованием редукции
#pragma omp for reduction(+:localResSq)
            for (int i = 0; i < N; i++) {
                double diff = b[i] - r[i];
                localResSq += diff * diff;
            }

#pragma omp barrier

            double relRes = 0.0;
// 3) Вычисление относительной невязки и проверка условия останова
#pragma omp single
            {
                double normRes = sqrt(localResSq);
                relRes = normRes / normB;
                if (iter % 100 == 0) {
                    cout << "Iter " << iter << ", relative residual = " << relRes << endl;
                }
                if (relRes < eps || iter >= maxIter) {
                    done = true;
                }
                iter++; // Увеличиваем счетчик итераций
            }

#pragma omp barrier

            // Если условие останова выполнено, выходим из цикла
            if (done) {
                break;
            }

// 4) Обновление вектора x: x = x + tau*(b - r)
#pragma omp for
            for (int i = 0; i < N; i++) {
                x[i] += tau * (b[i] - r[i]);
            }
#pragma omp barrier
        } // конец цикла while
    } // конец параллельной области

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "\nConverged after " << iter << " iterations." << endl;
    cout << "Total elapsed time: " << elapsed.count() << " seconds." << endl;

    // Вывод первых 10 значений решения
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
