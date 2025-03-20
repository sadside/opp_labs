#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>

// Функция инициализации локальной части матрицы A и вектора b.
// Каждый процесс получает строки с индексами от startRow до startRow+localRows-1.
// В данной демонстрации матрица заполняется так:
//   A(i,j) = 2.0, если j == global row index, и A(i,j) = 1.0, иначе.
// Вектор b задаётся так: для первых двух строк b = N+1, для остальных b = N.
void initializeData(int N, std::vector<double>& A_local, std::vector<double>& b,
                   int rank, int size, int startRow, int localRows) {
   for (int i = 0; i < localRows; i++) {
       int globalRow = startRow + i;
       for (int j = 0; j < N; j++) {
           A_local[i * N + j] = (j == globalRow) ? 2.0 : 1.0;
       }
   }
   if (rank == 0) {
       for (int i = 0; i < N; i++) {
           if (i == 0 || i == 1)
               b[i] = N + 1; // первые две строки
           else
               b[i] = N;     // остальные строки
       }
   }
   MPI_Bcast(b.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// умножение локальной части матрицы (A_local) на вектор v.
// Результат записывается в r_local.
void matVecMul(const std::vector<double>& A_local, const std::vector<double>& v,
              std::vector<double>& r_local, int N, int localRows) {
   for (int i = 0; i < localRows; i++) {
       double sum = 0.0;
       for (int j = 0; j < N; j++) {
           sum += A_local[i * N + j] * v[j];
       }
       r_local[i] = sum;
   }
}

int main(int argc, char** argv) {
   MPI_Init(&argc, &argv);

   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   // Задаём размерность системы: N x N.
   int N = 10000;
   if (argc > 1) {
       N = std::atoi(argv[1]);
   }

   // Определяем, сколько строк получит каждый процесс.
   int rowsPerProc = N / size;
   int remainder   = N % size;
   int startRow    = rank * rowsPerProc + std::min(rank, remainder);
   int localRows   = rowsPerProc + (rank < remainder ? 1 : 0);

   // Выделяем память: локальная часть матрицы, глобальные векторы b, x,
   // а также вспомогательные векторы для хранения результатов умножения.
   std::vector<double> A_local(localRows * N, 0.0);
   std::vector<double> b(N, 0.0);
   std::vector<double> x(N, 0.0);       // начальное приближение (нулевой вектор)
   std::vector<double> r_local(localRows, 0.0); // для A*x и для A*r (локально)
   std::vector<double> r(N, 0.0);       // глобальная невязка r = b - A*x
   std::vector<double> Ar_local(localRows, 0.0); // локальная часть A*r
   std::vector<double> Ar(N, 0.0);      // глобальный A*r

   // Подготавливаем массивы для MPI_Allgatherv (для сбора векторов r и Ar).
   std::vector<int> recvCounts(size), displs(size);
   for (int p = 0; p < size; p++) {
       int rp = N / size;
       int rm = N % size;
       int st = p * rp + std::min(p, rm);
       int lr = rp + (p < rm ? 1 : 0);
       recvCounts[p] = lr;
       displs[p] = st;
   }

   // Инициализируем данные: заполняем A_local и b.
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

   // Параметры метода
   double eps    = 1e-6;   // точность (относительная невязка)
   int maxIter   = 10000;  // макс. число итераций

   // Таймер запускается перед итерационным процессом.
   double startTime = MPI_Wtime();

   int iter;
   for (iter = 0; iter < maxIter; iter++) {
       // 1. Вычисляем A*x (локально) и собираем глобальный результат.
       matVecMul(A_local, x, r_local, N, localRows);
       MPI_Allgatherv(r_local.data(), localRows, MPI_DOUBLE,
                      r.data(), recvCounts.data(), displs.data(),
                      MPI_DOUBLE, MPI_COMM_WORLD);

       // 2. Вычисляем невязку: r = b - A*x.
       for (int i = 0; i < N; i++) {
           r[i] = b[i] - r[i];
       }

       // 3. Вычисляем норму невязки ||r||.
       double localRes_sq = 0.0;
       // Каждый процесс считает сумму квадратов для своих элементов (с учётом своего диапазона)
       for (int i = 0; i < localRows; i++) {
           double diff = r[startRow + i];
           localRes_sq += diff * diff;
       }
       double globalRes_sq = 0.0;
       MPI_Allreduce(&localRes_sq, &globalRes_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
       double normRes = std::sqrt(globalRes_sq);
       double relRes = normRes / normB;

       // Выводим локальную невязку каждые 100 итераций.
       if (iter % 100 == 0) {
           double localResidual = std::sqrt(localRes_sq);
           std::cout << "Iter " << iter
                     << ", Process " << rank
                     << ", local residual = " << localResidual << std::endl;
           if (rank == 0) {
               std::cout << "Iter " << iter
                         << ", Global relative residual = " << relRes << std::endl;
           }
       }

       // Проверка сходимости.
       if (relRes < eps) {
           break;
       }

       // 4. Вычисляем A*r: сначала вычисляем локально, затем собираем глобально.
       matVecMul(A_local, r, Ar_local, N, localRows);
       MPI_Allgatherv(Ar_local.data(), localRows, MPI_DOUBLE,
                      Ar.data(), recvCounts.data(), displs.data(),
                      MPI_DOUBLE, MPI_COMM_WORLD);

       // 5. Вычисляем скалярное произведение (r, Ar) и норму ||Ar||^2.
       double local_r_Ar = 0.0, local_Ar_Ar = 0.0;
       for (int i = 0; i < localRows; i++) {
           double r_val = r[startRow + i];
           double Ar_val = Ar[startRow + i];
           local_r_Ar += r_val * Ar_val;
           local_Ar_Ar += Ar_val * Ar_val;
       }
       double global_r_Ar = 0.0, global_Ar_Ar = 0.0;
       MPI_Allreduce(&local_r_Ar, &global_r_Ar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
       MPI_Allreduce(&local_Ar_Ar, &global_Ar_Ar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

       // Если global_Ar_Ar слишком мал, можно прекратить (избежим деления на ноль).
       if (global_Ar_Ar < 1e-15) {
           if (rank == 0)
               std::cout << "global_Ar_Ar is nearly zero; stopping iteration." << std::endl;
           break;
       }

       // 6. Вычисляем шаг alpha.
       double alpha = global_r_Ar / global_Ar_Ar;

       // 7. Обновляем решение: x = x + alpha * r.
       for (int i = 0; i < N; i++) {
           x[i] += alpha * r[i];
       }

       // Синхронизируем обновлённое решение x между процессами.
       MPI_Bcast(x.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   }

   // Останавливаем таймер.
   double endTime = MPI_Wtime();

   if (rank == 0) {
       std::cout << "\nConverged after " << iter << " iterations." << std::endl;
       std::cout << "Computed solution x:" << std::endl;
       for (int i = 0; i < N; i++) {
           std::cout << "x[" << i << "] = " << x[i] << std::endl;
       }

       // Для сравнения: ожидаемое решение – вектор единиц.
       std::cout << "\nExpected solution x (vector of ones):" << std::endl;
       for (int i = 0; i < N; i++) {
           std::cout << "1 ";
       }
       std::cout << std::endl;

       std::cout << "Total elapsed time: " << (endTime - startTime)
                 << " seconds." << std::endl;
   }

   MPI_Finalize();
   return 0;
}
