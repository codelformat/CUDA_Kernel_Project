# 远端 GPU 节点上加载：CUDA 12.6 + Nsight Compute 2024.3.2 + Nsight Systems 2024.5.1
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$CUDA_HOME/nsight-compute-2024.3.2:$CUDA_HOME/nsight-systems-2024.5.1/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
