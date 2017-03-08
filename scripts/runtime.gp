reset
set ylabel 'time(sec)'
set style data lines
set title 'Leibniz formula'
set term png enhanced font 'Verdana,10'
set output 'runtime.png'
set xtics rotate by 45 right
set datafile separator ","

plot [:][:0.05]'result_clock_gettime.csv' using 1:2  title 'Baseline', \
'' using 1:3  title 'OpenMP ( 2 threads )', \
'' using 1:4  title 'OpenMP ( 4 threads )', \
'' using 1:5  title 'AVX', \
'' using 1:6  title 'AVX + Unroll looping', \

