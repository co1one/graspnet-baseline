Hi, the following procedures work for me. However, I'm not an expert in CUDA and C++ so it would be great if someone professional could refine the solution.

Modify knn/src/cuda/vision.h
First, go to knn/src/cuda/vision.h and comment this line:

#include <THC/THC.h>
Append the following codes after this line:

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
Also in this file we need to change several APIs:

// float *dist_dev = (float*)THCudaMalloc(state, ref_nb * query_nb * sizeof(float));
// Change this to:
float *dist_dev = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(ref_nb * query_nb * sizeof(float));
// THCudaFree(state, dist_dev);
// Change this to:
c10::cuda::CUDACachingAllocator::raw_delete(dist_dev);
About the last change, I am not sure what's the substitute for THError so I temporarily use return 0 here. I believe there should be a proper solution for this.

cudaError_t err = cudaGetLastError();
if (err != cudaSuccess)
{
    printf("error in knn: %s\n", cudaGetErrorString(err));
    return 0;
    // THError("aborting");
}
Modify knn/src/knn.h
Go to knn/src/knn.h and comment these lines:

#include <THC/THC.h>
extern THCState *state;
Install the Package
Finally we can install the package knn/, and the demo.py works fine for me.

Reference
Missing headers in ATen/cuda/DeviceUtils.cuh pytorch/pytorch#72807 (comment)