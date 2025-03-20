#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>

// Функция инициализации локальной части матрицы A и вектора b.
// Каждый процесс получает строки с индексами от startRow до startRow + localRows - 1.
//  матрица заполняется так:
//    A(i, j) = 2.0, если j == global row index, и A(i, j) = 1.0, иначе.
// Вектор b рассчитывается так: для первых двух строк b = N + 1, для остальных b = N.
void initializeData(int N, std::vector<double>& A_local, std::vector<double>& b,
                   int rank, int size, int startRow, int localRows) {
   // Заполняем локальную часть матрицы A
   for (int i = 0; i < localRows; i++) {
       int globalRow = startRow + i;
       for (int j = 0; j < N; j++) {
           // Диагональный элемент = 2.0, все остальные = 1.0
           A_local[i * N + j] = (j == globalRow) ? 2.0 : 1.0;
       }
   }

   // Инициализация вектора b производится только на процессе 0,
   // затем рассылается всем процессам.
   if (rank == 0) {
       for (int i = 0; i < N; i++) {
           if (i == 0 || i == 1)
               b[i] = N + 1; // для первых двух строк
           else
               b[i] = N;     // для остальных строк
       }
   }

   // Шарим другим процессам
   MPI_Bcast(b.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Функция для умножения локальной части матрицы A_local (размера localRows x N)
// на глобальный вектор x (размера N). Результат сохраняется в r_local (размера localRows).
void matVecMul(const std::vector<double>& A_local, const std::vector<double>& x,
              std::vector<double>& r_local, int N, int localRows) {
   for (int i = 0; i < localRows; i++) {
       double sum = 0.0;
       for (int j = 0; j < N; j++) {
           sum += A_local[i * N + j] * x[j];
       }
       r_local[i] = sum;
   }
}

int main(int argc, char** argv) {
   MPI_Init(&argc, &argv);

   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   // Размерность системы. Если передан параметр командной строки, используем его, иначе N = 8.
   int N = 1000;
   if (argc > 1) {
       N = std::atoi(argv[1]);
   }

   // Определяем, сколько строк достанется каждому процессу.
   int rowsPerProc = N / size;
   int remainder   = N % size;
   int startRow    = rank * rowsPerProc + std::min(rank, remainder);
   int localRows   = rowsPerProc + (rank < remainder ? 1 : 0);

   // Выделяем память под локальную часть матрицы, а также под глобальные векторы b и x.
   std::vector<double> A_local(localRows * N, 0.0);
   std::vector<double> b(N, 0.0);
   std::vector<double> x(N, 0.0);       // Начальное приближение: x = 0
   std::vector<double> r_local(localRows, 0.0); // Локальный результат умножения A_local * x

   // Инициализация данных
   initializeData(N, A_local, b, rank, size, startRow, localRows);

   // Вычисляем норму вектора b для относительного критерия остановки.
   double localNormB_sq = 0.0;
   for (int i = 0; i < localRows; i++) {
       double val = b[startRow + i];
       localNormB_sq += val * val;
   }
   double globalNormB_sq = 0.0;
   MPI_Allreduce(&localNormB_sq, &globalNormB_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   double normB = std::sqrt(globalNormB_sq);
   // Если normB == 0, то задача тривиальна (b == 0).

   // Параметры итерационного метода
   double tau    = 1e-4;  // уменьшенный шаг для стабильности
   double eps    = 1e-5;  // относительная точность
   int maxIter   = 10000;


   // Подготовка для сбора локальных результатов в глобальный вектор r
   std::vector<double> r(N, 0.0);
   std::vector<int> recvCounts(size), displs(size);
   for (int p = 0; p < size; p++) {
       int rp = N / size;
       int rm = N % size;
       int st = p * rp + std::min(p, rm);
       int lr = rp + (p < rm ? 1 : 0);
       recvCounts[p] = lr;
       displs[p] = st;
   }

   double startTime = MPI_Wtime();

   int iter;
   for (iter = 0; iter < maxIter; iter++) {
       // 1) Вычисляем локальную часть произведения: r_local = A_local * x
       matVecMul(A_local, x, r_local, N, localRows);

       // 2) Собираем результаты с помощью MPI_Allgatherv в глобальный вектор r.
       MPI_Allgatherv(r_local.data(), localRows, MPI_DOUBLE,
                      r.data(), recvCounts.data(), displs.data(),
                      MPI_DOUBLE, MPI_COMM_WORLD);

       // 3) Вычисляем локальную сумму квадратов невязки: diff = b - r
       double localRes_sq = 0.0;
       for (int i = 0; i < localRows; i++) {
           double diff = b[startRow + i] - r[startRow + i];
           localRes_sq += diff * diff;
       }
       double globalRes_sq = 0.0;
       MPI_Allreduce(&localRes_sq, &globalRes_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
       double normRes = std::sqrt(globalRes_sq);

       // 4) Вычисляем относительную невязку: relRes = ||r||/||b||
       double relRes = normRes / normB;

       // 5) Вывод локальной невязки из каждого процесса каждые 100 итераций.
       if (iter % 100 == 0) {
           double localResidual = std::sqrt(localRes_sq);
           std::cout << "Iter " << iter
                     << ", Process " << rank
                     << ", local residual = " << localResidual << std::endl;
           // Дополнительно на процессе 0 выводим глобальную относительную невязку
           if (rank == 0) {
               std::cout << "Iter " << iter
                         << ", Global relative residual = " << relRes << std::endl;
           }
       }

       // 6) Проверяем критерий остановки: если относительная невязка < eps, завершаем итерации.
       if (relRes < eps) {
           break;
       }

       // 7) Обновляем вектор x: x = x + tau*(b - r)
       for (int i = 0; i < N; i++) {
           x[i] += tau * (b[i] - r[i]);
       }
       // Синхронизируем обновлённый x между всеми процессами.
       MPI_Bcast(x.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   }

   double endTime = MPI_Wtime();


   if (rank == 0) {
       std::cout << "\nConverged after " << iter << " iterations." << std::endl;

       std::cout << "Computed solution x:" << std::endl;
       for (int i = 0; i < N; i++) {
           std::cout << "x[" << i << "] = " << x[i] << std::endl;
       }

       std::cout << "\nExpected solution x (vector of ones):" << std::endl;
       for (int i = 0; i < N; i++) {
           std::cout << "1 ";
       }

       std::cout << "Total elapsed time: " << (endTime - startTime) << " seconds." << std::endl;
   }

   MPI_Finalize();
   return 0;
}
