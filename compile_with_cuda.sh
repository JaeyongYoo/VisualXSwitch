cp userlevel/Makefile.for.i386.example  userlevel/Makefile

echo "compiling cuda_dxt_processings.cu..."
nvcc -L/usr/local/cuda/lib -L/usr/lib  -lcuda -lcudart -I. -I/usr/local/cuda/include -gencode=arch=compute_20,code=\"sm_20,compute_20\" -c elements/local/VxS/cuda_dxt_processings.cu -o userlevel/cuda_dxt_processings.o
echo "compiling click..."
make

