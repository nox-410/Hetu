#include "gpu_runtime.h"

static const size_t DIM_BLOCK=1024;
#define DIM_GRID(x) ( ((size_t)x + DIM_BLOCK - 1) / DIM_BLOCK )

__global__ void add_l2_regularization(const float* param, float* grad,
    float l2reg, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    grad[ind] = grad[ind] + l2reg * param[ind];
}

int AddL2Regularization(const DLArrayHandle param, DLArrayHandle grad,
    float l2reg, DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < param->ndim; ++i) {
        size *= param->shape[i];
    }
    cudaStream_t stream = stream_handle ? *(cudaStream_t*)stream_handle->handle : cudaStreamDefault;
    const float* param_data = (const float*)param->data;
    float* grad_data = (float*)grad->data;
    add_l2_regularization<<<DIM_GRID(size), DIM_BLOCK, 0, stream>>>(
        param_data, grad_data, l2reg, size);
    return 0;
}

__global__ void sgd_update(float* param, const float* grad, float lr,
    size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    param[ind] = param[ind] - lr * grad[ind];
}

int SGDOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad, float lr,
    DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < param->ndim; ++i) {
        size *= param->shape[i];
    }
    cudaStream_t stream = stream_handle ? *(cudaStream_t*)stream_handle->handle : cudaStreamDefault;
    float* param_data = (float*)param->data;
    const float* grad_data = (const float*)grad->data;
    sgd_update<<<DIM_GRID(size), DIM_BLOCK, 0, stream>>>(param_data, grad_data, lr, size);
    return 0;
}

__global__ void nesterov_momentum_update(float* param, const float* grad,
    float* velocity, float lr,
    float momentum, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float temp = lr * grad[ind];
    velocity[ind] = momentum * (velocity[ind] - temp);
    param[ind] = param[ind] + velocity[ind] - temp;
}

__global__ void nesterov_momentum_process(const float* param, float* grad,
    float* velocity, float lr,
    float momentum, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float temp = lr * grad[ind];
    velocity[ind] = momentum * (velocity[ind] - temp);
    grad[ind] = velocity[ind] - temp;
}

__global__ void momentum_update(float* param, const float* grad,
    float* velocity, float lr, float momentum,
    size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    velocity[ind] = momentum * velocity[ind] - lr * grad[ind];
    param[ind] = param[ind] + velocity[ind];
}

__global__ void momentum_process(const float* param, float* grad,
    float* velocity, float lr, float momentum,
    size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    velocity[ind] = momentum * velocity[ind] - lr * grad[ind];
    grad[ind] = velocity[ind];
}

int MomentumOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
    DLArrayHandle velocity, float lr, float momentum,
    bool nesterov, bool only_process_grad,
    DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < param->ndim; ++i) {
        size *= param->shape[i];
    }
    cudaStream_t stream = stream_handle ? *(cudaStream_t*)stream_handle->handle : cudaStreamDefault;
    float* param_data = (float*)param->data;
    float* grad_data = (float*)grad->data;
    float* velocity_data = (float*)velocity->data;
    if (nesterov && only_process_grad) {
        nesterov_momentum_process<<<DIM_GRID(size), DIM_BLOCK, 0, stream>>>(
            param_data, grad_data, velocity_data, lr, momentum, size);
    } else if (nesterov && !only_process_grad) {
        nesterov_momentum_update<<<DIM_GRID(size), DIM_BLOCK, 0, stream>>>(
            param_data, grad_data, velocity_data, lr, momentum, size);
    } else if (!nesterov && only_process_grad) {
        momentum_process<<<DIM_GRID(size), DIM_BLOCK, 0, stream>>>(
            param_data, grad_data, velocity_data, lr, momentum, size);
    } else {
        momentum_update<<<DIM_GRID(size), DIM_BLOCK, 0, stream>>>(
            param_data, grad_data, velocity_data, lr, momentum, size);
    }
    return 0;
}

__global__ void adagrad_update(float* param, const float* grad, float* acc,
    float lr, float eps, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    acc[ind] = acc[ind] + grad[ind] * grad[ind];
    param[ind] = param[ind] - lr * grad[ind] / (sqrtf(acc[ind]) + eps);
}

__global__ void adagrad_process(const float* param, float* grad, float* acc,
    float lr, float eps, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    acc[ind] = acc[ind] + grad[ind] * grad[ind];
    grad[ind] = -lr * grad[ind] / (sqrtf(acc[ind]) + eps);
}

int AdaGradOptimizerUpdate(DLArrayHandle param, DLArrayHandle grad,
    DLArrayHandle acc, float lr, float eps, bool only_process_grad,
    DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < param->ndim; ++i) {
        size *= param->shape[i];
    }
    cudaStream_t stream = stream_handle ? *(cudaStream_t*)stream_handle->handle : cudaStreamDefault;
    float* param_data = (float*)param->data;
    float* grad_data = (float*)grad->data;
    float* acc_data = (float*)acc->data;
    if (only_process_grad)
        adagrad_process<<<DIM_GRID(size), DIM_BLOCK, 0, stream>>>(
            param_data, grad_data, acc_data, lr, eps, size);
    else
        adagrad_update<<<DIM_GRID(size), DIM_BLOCK, 0, stream>>>(
            param_data, grad_data, acc_data, lr, eps, size);
    return 0;
}

__global__ void adam_update(float* param, const float* grad, float* m, float* v,
    float lr, float beta1, float beta2, float beta1t,
    float beta2t, float eps, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    m[ind] = beta1 * m[ind] + (1 - beta1) * grad[ind];
    v[ind] = beta2 * v[ind] + (1 - beta2) * grad[ind] * grad[ind];
    float m_local = m[ind] / (1 - beta1t);
    float v_local = v[ind] / (1 - beta2t);
    param[ind] = param[ind] - lr * m_local / (sqrtf(v_local) + eps);
}

__global__ void adam_process(const float* param, float* grad, float* m, float* v,
    float lr, float beta1, float beta2, float beta1t,
    float beta2t, float eps, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    m[ind] = beta1 * m[ind] + (1 - beta1) * grad[ind];
    v[ind] = beta2 * v[ind] + (1 - beta2) * grad[ind] * grad[ind];
    float m_local = m[ind] / (1 - beta1t);
    float v_local = v[ind] / (1 - beta2t);
    grad[ind] = -lr * m_local / (sqrtf(v_local) + eps);
}

int AdamOptimizerUpdate(DLArrayHandle param, DLArrayHandle grad,
    DLArrayHandle expavg, DLArrayHandle expavgsq, float lr,
    float beta1, float beta2, float beta1t, float beta2t,
    float eps, bool only_process_grad, DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < param->ndim; ++i) {
        size *= param->shape[i];
    }
    cudaStream_t stream = stream_handle ? *(cudaStream_t*)stream_handle->handle : cudaStreamDefault;
    float* param_data = (float*)param->data;
    float* grad_data = (float*)grad->data;
    float* m_data = (float*)expavg->data;
    float* v_data = (float*)expavgsq->data;
    if (only_process_grad)
        adam_process<<<DIM_GRID(size), DIM_BLOCK, 0, stream>>>(
            param_data, grad_data, m_data, v_data, lr, beta1, beta2, beta1t,
            beta2t, eps, size);
    else
        adam_update<<<DIM_GRID(size), DIM_BLOCK, 0, stream>>>(
            param_data, grad_data, m_data, v_data, lr, beta1, beta2, beta1t,
            beta2t, eps, size);
    return 0;
}
