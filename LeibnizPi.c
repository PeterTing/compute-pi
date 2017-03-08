#include <stdio.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include "LeibnizPi.h"

#define SAMPLE_SIZE 10


double leibniz_pi_baseline(size_t N)
{
    double pi = 0.0;
    for(size_t i = 0; i < N; i++) {
        int sign = i % 2 == 0 ? 1 : -1;             //if the power of sign is odd, sign will be -1, else will be 1
        pi +=  (sign / (2.0 * (double) i + 1.0));
    }

    return pi * 4.0;

}

double leibniz_pi_openmp(size_t N, int threads)
{
    double pi = 0.0;

    #pragma omp parallel for num_threads(threads) reduction(+:pi)
    for(size_t i = 0; i < N; i++) {
        int sign = i % 2 == 0 ? 1 : -1;
        pi += (sign / (2.0 * (double) i + 1.0));
    }

    return pi * 4.0;
}

/* Use avx to improve the performance of leibniz formula */

double leibniz_pi_avx(size_t N)
{
    double pi = 0.0;
    register __m256d ymm0, ymm1, ymm2, ymm3, ymm4;
    ymm0 = _mm256_set_pd(1.0, -1.0, 1.0, -1.0); // sign
    ymm1 = _mm256_set1_pd(2.0);                 // constant of k
    ymm2 = _mm256_set1_pd(1.0);                 // the constant of the denominator 2K+1
    ymm4 = _mm256_setzero_pd();                 // sum of pi

    for (int i = 0; i < N - 4; i+= 4) {
        ymm3 = _mm256_set_pd(i, i+1, i+2, i+3);
        ymm3 = _mm256_mul_pd(ymm3, ymm1);       // x = 2.0*i
        ymm3 = _mm256_add_pd(ymm3, ymm2);       // x = 2.0*i + 1.0
        ymm4 += _mm256_div_pd(ymm0, ymm3);       // pi += sign/(2.0*i+1.0)
    }

    double tmp[4] __attribute__((aligned(32))); // to allocate the array tmp on a 32-byte boundary
    _mm256_storeu_pd(tmp, ymm4);                // move packed float64 values to 256-bit aligned memory location
    pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    return pi * 4.0;

}

double leibniz_pi_avx_unroll(size_t N)
{

    double pi = 0.0;
    register __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10;
    ymm0 = _mm256_set_pd(1.0, -1.0, 1.0, -1.0); // sign
    ymm1 = _mm256_set1_pd(2.0);                 // constant of k
    ymm2 = _mm256_set1_pd(1.0);                 // the constant of the denominator 2K+1
    ymm7 = _mm256_setzero_pd();                 // sum of pi
    ymm8 = _mm256_setzero_pd();
    ymm9 = _mm256_setzero_pd();
    ymm10 = _mm256_setzero_pd();

    for (int i = 0; i < N - 16; i+= 16) {
        ymm3 = _mm256_set_pd(i, i+1, i+2, i+3);
        ymm3 = _mm256_mul_pd(ymm3, ymm1);       // x = 2.0*i
        ymm3 = _mm256_add_pd(ymm3, ymm2);       // x = 2.0*i + 1.0
        ymm7 += _mm256_div_pd(ymm0, ymm3);       // pi += sign/(2.0*i+1.0)

        ymm4 = _mm256_set_pd(i+4, i+5, i+6, i+7);
        ymm4 = _mm256_mul_pd(ymm4, ymm1);       // x = 2.0*i
        ymm4 = _mm256_add_pd(ymm4, ymm2);       // x = 2.0*i + 1.0
        ymm8 += _mm256_div_pd(ymm0, ymm4);       // pi += sign/(2.0*i+1.0)

        ymm5 = _mm256_set_pd(i+8, i+9, i+10, i+11);
        ymm5 = _mm256_mul_pd(ymm5, ymm1);       // x = 2.0*i
        ymm5 = _mm256_add_pd(ymm5, ymm2);       // x = 2.0*i + 1.0
        ymm9 += _mm256_div_pd(ymm0, ymm5);       // pi += sign/(2.0*i+1.0)

        ymm6 = _mm256_set_pd(i+12, i+13, i+14, i+15);
        ymm6 = _mm256_mul_pd(ymm6, ymm1);       // x = 2.0*i
        ymm6 = _mm256_add_pd(ymm6, ymm2);       // x = 2.0*i + 1.0
        ymm10 += _mm256_div_pd(ymm0, ymm6);       // pi += sign/(2.0*i+1.0)

    }

    double tmp[4] __attribute__((aligned(32))); // to allocate the array tmp on a 32-byte boundary
    _mm256_storeu_pd(tmp, ymm7);                // move packed float64 values to 256-bit aligned memory location
    pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, ymm8);                // move packed float64 values to 256-bit aligned memory location
    pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, ymm9);                // move packed float64 values to 256-bit aligned memory location
    pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, ymm10);                // move packed float64 values to 256-bit aligned memory location
    pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    return pi * 4.0;
}
