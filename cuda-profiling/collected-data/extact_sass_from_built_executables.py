# this script simply invokes the cuobjdump and nvdisasm tools to extract SASS from built CUDA executables


# for CUDA codes, the workflow is as follows:
# 1) extract all embedded cubins as standalone files (this creates .cubin files)
# cuobjdump ./bin/cuda/atan2 -xelf all 

# 2) disassemble the cubins to get SASS
# nvdisasm -g ./<something>.sm_86.cubin > <something>.sm_86.sass

# for OpenMP codes, the workflow is as follows:
# 1) extract the sm_86 image (cubin)
# /usr/lib/llvm-21/bin/clang-offload-packager ./bin/omp/atan2 --image=triple=nvptx64-nvidia-cuda,arch=sm_86,file=atan2_sm86.cubin 

# 2) disassemble the cubins to get SASS
# nvdisasm -c -g -fun atan2_sm86.cubin > atan2_sm86.sass





