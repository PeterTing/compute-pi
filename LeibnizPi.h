#include <stdio.h>

double leibniz_pi_baseline(size_t N);
double leibniz_pi_openmp(size_t N, int threads);
double leibniz_pi_avx(size_t N);
double leibniz_pi_avx_unroll(size_t N);

