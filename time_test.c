#include <stdio.h>
#ifndef LEIBNIZ
#include "computepi.h"
#else
#include "LeibnizPi.h"
#endif

int main(int argc, char const *argv[])
{
    __attribute__((unused)) int N = 400000000;
    double pi = 0.0;

#if defined(BASELINE)
#if defined(LEIBNIZ)
    pi = leibniz_pi_baseline(N);
#elif
    pi = compute_pi_baseline(N);
#endif
#endif

#if defined(OPENMP_2)
#if defined(LEIBNIZ)
    pi = leibniz_pi_openmp(N, 2);
#elif
    pi = compute_pi_openmp(N, 2);
#endif
#endif

#if defined(OPENMP_4)
#if defined(LEIBNIZ)
    pi = leibniz_pi_openmp(N, 4);
#elif
    pi = compute_pi_openmp(N, 4);
#endif
#endif

#if defined(AVX)
#if defined(LEIBNIZ)
    pi = leibniz_pi_avx(N);
#elif
    pi = compute_pi_avx(N);
#endif
#endif

#if defined(AVXUNROLL)

#if defined(LEIBNIZ)
    pi = leibniz_pi_avx_unroll(N);
#elif
    pi = compute_pi_avx_unroll(N);
#endif
#endif

    printf("N = %d , pi = %lf\n", N, pi);

    return 0;
}
