echo "compiling dxtc.cu..."
nvcc -L/usr/local/cuda/lib -L/usr/lib  -lcuda -lcudart -I. -I/usr/local/cuda/include -gencode=arch=compute_20,code=\"sm_20,compute_20\" -c elements/local/dxtc.cu -o userlevel/dxtc.o
echo ""
echo "compiling click..."
make

