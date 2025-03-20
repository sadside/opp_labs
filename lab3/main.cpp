#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Утилита: индексация в матрице, хранящейся построчно (row-major).
inline int idx(int row, int col, int cols) {
    return row * cols + col;
}

// Функция умножения локальных блоков A_block (размера loc_n1 x n2)
// и B_block (размера n2 x loc_n3), результат C_block (loc_n1 x loc_n3).
void localMatrixMultiply(const double* A_block, const double* B_block,
                         double* C_block, int loc_n1, int n2, int loc_n3) {
    // C_block = A_block * B_block
    for (int i = 0; i < loc_n1; i++) {
        for (int j = 0; j < loc_n3; j++) {
            double sum = 0.0;
            for (int k = 0; k < n2; k++) {
                sum += A_block[idx(i, k, n2)] * B_block[idx(k, j, loc_n3)];
            }
            C_block[idx(i, j, loc_n3)] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Параметры задачи. Предположим, что n1, n2, n3 делятся на p1, p2.
    // Для примера зададим по умолчанию n1=n2=n3=8, p1=2, p2=2.
    // Можно переопределить из аргументов командной строки или по-своему.
    int n1 = 8, n2 = 8, n3 = 8;
    int p1 = 2, p2 = 2; // p1 * p2 = size (ожидаем, что это совпадает)

    if (argc > 5) {
        n1 = std::atoi(argv[1]);
        n2 = std::atoi(argv[2]);
        n3 = std::atoi(argv[3]);
        p1 = std::atoi(argv[4]);
        p2 = std::atoi(argv[5]);
    }

    // Проверка, что число процессов совпадает с p1*p2
    if (p1 * p2 != size) {
        if (rank == 0) {
            std::cerr << "Error: p1*p2 != MPI size" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Определяем размеры локальных блоков
    // Предполагаем, что n1 % p1 == 0 и n3 % p2 == 0.
    int loc_n1 = n1 / p1; // Число строк A (и C) на каждом процессе по оси x
    int loc_n3 = n3 / p2; // Число столбцов B (и C) на каждом процессе по оси y

    // Создадим подкоммуникаторы: rowComm (для каждой строки) и colComm (для каждого столбца).
    // Это упростит broadcast.
    // Сначала создадим 2D Cart коммуникатор, а из него выделим sub-комм.
    int dims[2]    = {p1, p2};
    int periods[2] = {0, 0}; // без периодичности
    MPI_Comm cartComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/0, &cartComm);

    // Получаем координаты (i, j) в 2D решётке
    int coords[2];
    MPI_Cart_coords(cartComm, rank, 2, coords);
    int i = coords[0]; // координата по "строкам"
    int j = coords[1]; // координата по "столбцам"

    // Создадим sub-комм для строк (rowComm) и для столбцов (colComm).
    MPI_Comm rowComm, colComm;
    // rowComm: фиксируем i, меняется j
    // colComm: фиксируем j, меняется i
    MPI_Comm_split(cartComm, i, j, &rowComm); // все процессы в строке i
    MPI_Comm_split(cartComm, j, i, &colComm); // все процессы в столбце j

    // Определяем "головные" процессы для каждой строки и столбца.
    // Например, (i,0) - головной процесс по строке, (0,j) - по столбцу.
    int rowRoot = 0; // j=0
    int colRoot = 0; // i=0

    // Создадим (только на rank=0) исходные матрицы A(n1 x n2) и B(n2 x n3).
    // И матрицу C(n1 x n3) для итогового результата.
    std::vector<double> A, B, C;
    if (rank == 0) {
        A.resize(n1 * n2);
        B.resize(n2 * n3);
        C.resize(n1 * n3, 0.0);

        // Инициализируем A, B произвольными значениями (для теста)
        for (int r = 0; r < n1; r++) {
            for (int c = 0; c < n2; c++) {
                A[idx(r, c, n2)] = (r + 1) * 1.0; // упрощённо
            }
        }
        for (int r = 0; r < n2; r++) {
            for (int c = 0; c < n3; c++) {
                B[idx(r, c, n3)] = (c + 1) * 1.0; // упрощённо
            }
        }
    }

    // Локальные блоки, которые получит каждый процесс
    // A_block имеет размер loc_n1 x n2
    // B_block имеет размер n2 x loc_n3
    // C_block имеет размер loc_n1 x loc_n3 (результат локального умножения)
    std::vector<double> A_block(loc_n1 * n2, 0.0);
    std::vector<double> B_block(n2 * loc_n3, 0.0);
    std::vector<double> C_block(loc_n1 * loc_n3, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    //--------------------------------------------------------------------------
    // 1. Scatter A по строкам (ось x): процесс (i,0) получает i-ю горизонтальную полосу.
    //    У нас p1 полос, каждая содержит loc_n1 строк (всего n1 строк).
    //    После этого каждая полоса будет в (i,0).
    //--------------------------------------------------------------------------
    if (j == 0) {
        // Только в столбце j=0 участвуют в scatter для A
        if (rank == 0) {
            // Раздаём полосы A процессам (i,0)
            for (int dest_i = 0; dest_i < p1; dest_i++) {
                int destRank;
                int coordsSend[2] = {dest_i, 0};
                MPI_Cart_rank(cartComm, coordsSend, &destRank);

                if (destRank == 0) {
                    // Копируем свою (0-я) полосу A
                    for (int row = 0; row < loc_n1; row++) {
                        int globalRow = row;
                        // globalRow = row + dest_i*loc_n1, но здесь dest_i=0
                        for (int col = 0; col < n2; col++) {
                            A_block[idx(row, col, n2)] = A[idx(globalRow, col, n2)];
                        }
                    }
                } else {
                    // Отправляем полосу для процесса (dest_i, 0)
                    std::vector<double> temp(loc_n1 * n2);
                    // Вычислим, какие строки ему принадлежат
                    int rowStart = dest_i * loc_n1;
                    for (int row = 0; row < loc_n1; row++) {
                        for (int col = 0; col < n2; col++) {
                            temp[idx(row, col, n2)] =
                                    A[idx(rowStart + row, col, n2)];
                        }
                    }
                    MPI_Send(temp.data(), loc_n1 * n2, MPI_DOUBLE,
                             destRank, 101, cartComm);
                }
            }
        } else {
            // (i,0), i>0: получаем свою полосу A
            if (i != 0) {
                MPI_Status st;
                MPI_Recv(A_block.data(), loc_n1 * n2, MPI_DOUBLE,
                         0, 101, cartComm, &st);
            }
        }
    }

    //--------------------------------------------------------------------------
    // 2. Scatter B по столбцам (ось y): процесс (0,j) получает j-ю вертикальную полосу.
    //--------------------------------------------------------------------------
    if (i == 0) {
        // Только в строке i=0 участвуют в scatter для B
        if (rank == 0) {
            // Раздаём полосы B процессам (0,j)
            for (int dest_j = 0; dest_j < p2; dest_j++) {
                int destRank;
                int coordsSend[2] = {0, dest_j};
                MPI_Cart_rank(cartComm, coordsSend, &destRank);

                if (destRank == 0) {
                    // Копируем свою (0-я) полосу B
                    for (int col = 0; col < loc_n3; col++) {
                        int globalCol = col;
                        for (int row = 0; row < n2; row++) {
                            B_block[idx(row, col, loc_n3)] =
                                    B[idx(row, globalCol, n3)];
                        }
                    }
                } else {
                    // Отправляем полосу для процесса (0, dest_j)
                    std::vector<double> temp(n2 * loc_n3);
                    int colStart = dest_j * loc_n3;
                    for (int col = 0; col < loc_n3; col++) {
                        for (int row = 0; row < n2; row++) {
                            temp[idx(row, col, loc_n3)] =
                                    B[idx(row, colStart + col, n3)];
                        }
                    }
                    MPI_Send(temp.data(), n2 * loc_n3, MPI_DOUBLE,
                             destRank, 202, cartComm);
                }
            }
        } else {
            // (0,j), j>0: получаем свою полосу B
            if (j != 0) {
                MPI_Status st;
                MPI_Recv(B_block.data(), n2 * loc_n3, MPI_DOUBLE,
                         0, 202, cartComm, &st);
            }
        }
    }

    //--------------------------------------------------------------------------
    // 3. Broadcast A_block по оси y (строка i).
    //    Процесс (i,0) рассылает A_block всем (i,j) в той же строке i.
    //--------------------------------------------------------------------------
    MPI_Bcast(A_block.data(), loc_n1 * n2, MPI_DOUBLE, rowRoot, rowComm);

    //--------------------------------------------------------------------------
    // 4. Broadcast B_block по оси x (столбец j).
    //    Процесс (0,j) рассылает B_block всем (i,j) в том же столбце j.
    //--------------------------------------------------------------------------
    MPI_Bcast(B_block.data(), n2 * loc_n3, MPI_DOUBLE, colRoot, colComm);

    //--------------------------------------------------------------------------
    // 5. Каждый процесс (i,j) вычисляет свою подматрицу C_block
    //--------------------------------------------------------------------------
    localMatrixMultiply(A_block.data(), B_block.data(),
                        C_block.data(), loc_n1, n2, loc_n3);

    //--------------------------------------------------------------------------
    // 6. Сбор результирующей матрицы C (n1 x n3) в процесс 0.
    //    Каждый процесс (i,j) шлёт C_block в (0,0), который раскладывает их на место.
    //--------------------------------------------------------------------------
    if (rank == 0) {
        // Скопируем свой C_block (i=0,j=0) на место
        for (int row = 0; row < loc_n1; row++) {
            for (int col = 0; col < loc_n3; col++) {
                C[idx(row, col, n3)] = C_block[idx(row, col, loc_n3)];
            }
        }
        // Получаем блоки от остальных процессов
        for (int src_i = 0; src_i < p1; src_i++) {
            for (int src_j = 0; src_j < p2; src_j++) {
                // (0,0) пропускаем - это уже скопировано
                if (src_i == 0 && src_j == 0) continue;

                int srcRank;
                int coordsRecv[2] = {src_i, src_j};
                MPI_Cart_rank(cartComm, coordsRecv, &srcRank);

                // Вычислим, куда в глобальной матрице C вставлять блок (src_i, src_j)
                int rowStart = src_i * loc_n1;
                int colStart = src_j * loc_n3;

                std::vector<double> temp(loc_n1 * loc_n3);
                MPI_Status st;
                MPI_Recv(temp.data(), loc_n1 * loc_n3, MPI_DOUBLE,
                         srcRank, 303, cartComm, &st);

                // Копируем temp в нужное место C
                for (int row = 0; row < loc_n1; row++) {
                    for (int col = 0; col < loc_n3; col++) {
                        C[idx(rowStart + row, colStart + col, n3)] =
                                temp[idx(row, col, loc_n3)];
                    }
                }
            }
        }
    } else {
        // Процесс (i,j) отправляет свой C_block в (0,0)
        MPI_Send(C_block.data(), loc_n1 * loc_n3, MPI_DOUBLE, 0, 303, cartComm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();
    double localElapsed = endTime - startTime;

    // Собираем время выполнения: хотим узнать максимум (или среднее) по всем процессам
    double globalElapsed;
    MPI_Reduce(&localElapsed, &globalElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Проверим результат (для теста) на rank=0
    if (rank == 0) {
        std::cout << "\nElapsed time (max across ranks): "
                  << globalElapsed << " seconds.\n";

        std::cout << "\nResult C (n1 x n3) = " << n1 << " x " << n3 << "\n";
        // Для отладки можно вывести небольшую матрицу
        if (n1 <= 8 && n3 <= 8) {
            for (int r = 0; r < n1; r++) {
                for (int c = 0; c < n3; c++) {
                    std::cout << C[idx(r, c, n3)] << " ";
                }
                std::cout << "\n";
            }
        } else {
            std::cout << "(matrix too large to print)\n";
        }
    }

    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&cartComm);

    MPI_Finalize();
    return 0;
}
