#pragma once

void matmul_ref(float* a, float* b, float* c, int M, int N, int K, int lda, int ldb, int ldc, float* ops = nullptr);