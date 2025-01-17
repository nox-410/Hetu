#include "gpu_runtime.h"

__global__ void minus_mean_n_square_kernel(const float *in_arr,
                                           const float *mean, float *out_arr,
                                           int last_dim, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float temp = in_arr[ind] - mean[ind / last_dim];
    out_arr[ind] = temp * temp;
}

__global__ void rescale_kernel(const float *in_arr, const float *mean_arr,
                               const float *var_arr, const float *ln_scale,
                               const float *ln_bias, float *out_arr,
                               int last_dim, float eps, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t mo_ind = ind / last_dim;
    size_t ln_ind = ind % last_dim;
    out_arr[ind] = (in_arr[ind] - mean_arr[mo_ind])
                       / sqrtf(var_arr[mo_ind] + eps) * ln_scale[ln_ind]
                   + ln_bias[ln_ind];
}

int DLGpuLayerNormalization(const DLArrayHandle in_arr,
                            const DLArrayHandle ln_scale,
                            const DLArrayHandle ln_bias, DLArrayHandle mean_arr,
                            DLArrayHandle var_arr, DLArrayHandle out_arr,
                            float eps, DLStreamHandle stream_handle) {
    int dev_id = (in_arr->ctx).device_id;
    cudaSetDevice(dev_id);
    cudnn_init(dev_id, stream_handle);

    float one = 1.0f;
    float zero = 0.0f;

    cudnnReduceTensorDescriptor_t rtd;
    CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&rtd));
    CUDNN_CALL(cudnnSetReduceTensorDescriptor(
        rtd, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

    cudnnTensorDescriptor_t adesc;
    cudnnTensorDescriptor_t cdesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&adesc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&cdesc));

    int ndim = in_arr->ndim;
    int last_dim = in_arr->shape[ndim - 1];
    if (ndim < 4)
        ndim = 4;
    size_t cpu_mem = ndim * sizeof(int);
    int *dimA = (int *)malloc(cpu_mem);
    int *strideA = (int *)malloc(cpu_mem);
    int *dimC = (int *)malloc(cpu_mem);
    int *strideC = (int *)malloc(cpu_mem);

    int temp_strideA = 1;
    int temp_strideC = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        dimA[i] = i < in_arr->ndim ? (int)in_arr->shape[i] : 1;
        dimC[i] = i < in_arr->ndim - 1 ? (int)in_arr->shape[i] : 1;
        strideA[i] = temp_strideA;
        strideC[i] = temp_strideC;
        temp_strideA *= dimA[i];
        temp_strideC *= dimC[i];
    }
    size_t size = temp_strideA * sizeof(float);

    CUDNN_CALL(cudnnSetTensorNdDescriptor(adesc, CUDNN_DATA_FLOAT, ndim, dimA,
                                          strideA));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(cdesc, CUDNN_DATA_FLOAT, ndim, dimC,
                                          strideC));

    CUDNN_CALL(cudnnReduceTensor(cudnn_map[dev_id], rtd, NULL, 0,
                                 (void *)out_arr->data, size, &one, adesc,
                                 (const void *)in_arr->data, &zero, cdesc,
                                 (void *)mean_arr->data));

    dim3 blocks;
    dim3 threads;
    if (temp_strideA <= 1024) {
        threads.x = temp_strideA;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (temp_strideA + 1023) / 1024;
    }

    if (stream_handle)
        minus_mean_n_square_kernel<<<blocks, threads, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            (const float *)in_arr->data, (const float *)mean_arr->data,
            (float *)out_arr->data, last_dim, temp_strideA);
    else
        minus_mean_n_square_kernel<<<blocks, threads>>>(
            (const float *)in_arr->data, (const float *)mean_arr->data,
            (float *)out_arr->data, last_dim, temp_strideA);

    CUDNN_CALL(cudnnReduceTensor(cudnn_map[dev_id], rtd, NULL, 0,
                                 (void *)out_arr->data, size, &one, adesc,
                                 (const void *)out_arr->data, &zero, cdesc,
                                 (void *)var_arr->data));

    if (temp_strideA <= 1024) {
        threads.x = temp_strideA;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (temp_strideA + 1023) / 1024;
    }
    if (stream_handle)
        rescale_kernel<<<blocks, threads, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            (const float *)in_arr->data, (const float *)mean_arr->data,
            (const float *)var_arr->data, (const float *)ln_scale->data,
            (const float *)ln_bias->data, (float *)out_arr->data, last_dim, eps,
            temp_strideA);
    else
        rescale_kernel<<<blocks, threads>>>(
            (const float *)in_arr->data, (const float *)mean_arr->data,
            (const float *)var_arr->data, (const float *)ln_scale->data,
            (const float *)ln_bias->data, (float *)out_arr->data, last_dim, eps,
            temp_strideA);

    CUDNN_CALL(cudnnDestroyTensorDescriptor(adesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(cdesc));
    CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(rtd));
    free(dimA);
    free(dimC);
    free(strideA);
    free(strideC);
    return 0;
}

__global__ void process_kernel1(const float *grads, const float *in_arr,
                                const float *mean_arr, const float *var_arr,
                                const float *ln_scale, float *ws1, float *ws2,
                                float *out_arr, float eps, int last_dim,
                                size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int mo_ind = ind / last_dim;
    float std = sqrtf(var_arr[mo_ind] + eps);
    float x_centered = in_arr[ind] - mean_arr[mo_ind];
    float x_norm = x_centered / std;
    float gscale = grads[ind] * x_norm;
    ws1[ind] = gscale;
    int ln_ind = ind % last_dim;
    if (ln_ind == 0) {
        ws2[mo_ind] = std;
    }
    float dx_norm = grads[ind] * ln_scale[ln_ind];
    float dvar_temp = dx_norm * x_centered;
    out_arr[ind] = dvar_temp;
}

__global__ void process_kernel2(const float *grads, const float *in_arr,
                                const float *mean_arr, const float *var_arr,
                                const float *ln_scale, float *ws1, float *ws2,
                                float *ws3, float eps, int last_dim,
                                size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int ln_ind = ind % last_dim;
    float dx_norm = grads[ind] * ln_scale[ln_ind];
    int mo_ind = ind / last_dim;
    float dx_mu_1 = dx_norm / ws2[mo_ind];
    float dvar = ws3[mo_ind] * -0.5 / ws2[mo_ind] / (var_arr[mo_ind] + eps);
    float x_centered = in_arr[ind] - mean_arr[mo_ind];
    float dx_mu_2 = dvar * 2 * x_centered / last_dim;
    float dx1 = dx_mu_1 + dx_mu_2;
    ws1[ind] = dx1;
}

__global__ void process_kernel3(const float *ws1, const float *ws3,
                                float *out_arr, int last_dim, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int mo_ind = ind / last_dim;
    out_arr[ind] = ws1[ind] - ws3[mo_ind] / last_dim;
}

int DLGpuLayerNormalizationGradient(
    const DLArrayHandle out_grads, const DLArrayHandle in_arr,
    const DLArrayHandle ln_scale, DLArrayHandle grad_arr,
    DLArrayHandle grad_scale, DLArrayHandle grad_bias,
    const DLArrayHandle mean_arr, const DLArrayHandle var_arr, float eps,
    DLStreamHandle stream_handle) {
    int dev_id = (out_grads->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    float one = 1.0f;
    float zero = 0.0f;

    cudnnReduceTensorDescriptor_t rtd;
    CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&rtd));
    CUDNN_CALL(cudnnSetReduceTensorDescriptor(
        rtd, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

    cudnnTensorDescriptor_t adesc;
    cudnnTensorDescriptor_t cdesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&adesc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&cdesc));
    int ori_ndim = out_grads->ndim;
    int last_dim = out_grads->shape[ori_ndim - 1];
    int ndim = max(ori_ndim, 4);
    size_t cpu_mem = ndim * sizeof(int);
    int *dimA = (int *)malloc(cpu_mem);
    int *strideA = (int *)malloc(cpu_mem);
    int *dimC = (int *)malloc(cpu_mem);
    int *strideC = (int *)malloc(cpu_mem);

    int temp_strideA = 1;
    int temp_strideC = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        dimA[i] = i < ori_ndim ? (int)out_grads->shape[i] : 1;
        dimC[i] = i == ori_ndim - 1 ? (int)out_grads->shape[i] : 1;
        strideA[i] = temp_strideA;
        strideC[i] = temp_strideC;
        temp_strideA *= dimA[i];
        temp_strideC *= dimC[i];
    }
    size_t size = temp_strideA * sizeof(float);

    CUDNN_CALL(cudnnSetTensorNdDescriptor(adesc, CUDNN_DATA_FLOAT, ndim, dimA,
                                          strideA));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(cdesc, CUDNN_DATA_FLOAT, ndim, dimC,
                                          strideC));
    CUDNN_CALL(cudnnReduceTensor(cudnn_map[dev_id], rtd, NULL, 0,
                                 (void *)grad_arr->data, size, &one, adesc,
                                 (const void *)out_grads->data, &zero, cdesc,
                                 (void *)grad_bias->data));

    size_t size1 = temp_strideA * sizeof(float);
    size_t size2 = size1 / last_dim;

    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    float *ws1 = (float *)find_chunk(size1, dev_id);
    float *ws2 = (float *)find_chunk(size2, dev_id);
    float *ws3 = (float *)find_chunk(size2, dev_id);

    dim3 blocks;
    dim3 threads;
    if (temp_strideA <= 1024) {
        threads.x = temp_strideA;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (temp_strideA + 1023) / 1024;
    }

    if (stream_handle)
        process_kernel1<<<blocks, threads, 0,
                          *(cudaStream_t *)stream_handle->handle>>>(
            (const float *)out_grads->data, (const float *)in_arr->data,
            (const float *)mean_arr->data, (const float *)var_arr->data,
            (const float *)ln_scale->data, ws1, ws2, (float *)grad_arr->data,
            eps, last_dim, temp_strideA);
    else
        process_kernel1<<<blocks, threads>>>(
            (const float *)out_grads->data, (const float *)in_arr->data,
            (const float *)mean_arr->data, (const float *)var_arr->data,
            (const float *)ln_scale->data, ws1, ws2, (float *)grad_arr->data,
            eps, last_dim, temp_strideA);

    CUDNN_CALL(cudnnReduceTensor(cudnn_map[dev_id], rtd, NULL, 0, (void *)ws1,
                                 size, &one, adesc, (const void *)ws1, &zero,
                                 cdesc, (void *)grad_scale->data));

    temp_strideC = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        dimC[i] = i < ori_ndim - 1 ? (int)out_grads->shape[i] : 1;
        strideC[i] = temp_strideC;
        temp_strideC *= dimC[i];
    }
    CUDNN_CALL(cudnnSetTensorNdDescriptor(cdesc, CUDNN_DATA_FLOAT, ndim, dimC,
                                          strideC));
    CUDNN_CALL(cudnnReduceTensor(
        cudnn_map[dev_id], rtd, NULL, 0, (void *)grad_arr->data, size, &one,
        adesc, (const void *)grad_arr->data, &zero, cdesc, (void *)ws3));

    if (stream_handle)
        process_kernel2<<<blocks, threads, 0,
                          *(cudaStream_t *)stream_handle->handle>>>(
            (const float *)out_grads->data, (const float *)in_arr->data,
            (const float *)mean_arr->data, (const float *)var_arr->data,
            (const float *)ln_scale->data, ws1, ws2, ws3, eps, last_dim,
            temp_strideA);
    else
        process_kernel2<<<blocks, threads>>>(
            (const float *)out_grads->data, (const float *)in_arr->data,
            (const float *)mean_arr->data, (const float *)var_arr->data,
            (const float *)ln_scale->data, ws1, ws2, ws3, eps, last_dim,
            temp_strideA);

    CUDNN_CALL(cudnnReduceTensor(cudnn_map[dev_id], rtd, NULL, 0,
                                 (void *)grad_arr->data, size, &one, adesc,
                                 (const void *)ws1, &zero, cdesc, (void *)ws3));

    if (stream_handle)
        process_kernel3<<<blocks, threads, 0,
                          *(cudaStream_t *)stream_handle->handle>>>(
            ws1, ws3, (float *)grad_arr->data, last_dim, temp_strideA);
    else
        process_kernel3<<<blocks, threads>>>(ws1, ws3, (float *)grad_arr->data,
                                             last_dim, temp_strideA);

    del_chunk(ws1, dev_id);
    del_chunk(ws2, dev_id);
    del_chunk(ws3, dev_id);

    CUDNN_CALL(cudnnDestroyTensorDescriptor(adesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(cdesc));
    CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(rtd));
    free(dimA);
    free(dimC);
    free(strideA);
    free(strideC);
    return 0;
}

int DLGpuLayerNormalizationInference(
    const DLArrayHandle in_arr, const DLArrayHandle ln_scale,
    const DLArrayHandle ln_bias, DLArrayHandle mean_arr, DLArrayHandle var_arr,
    DLArrayHandle out_arr, float eps, DLStreamHandle stream_handle) {
    int ndim = in_arr->ndim;
    int last_dim = in_arr->shape[ndim - 1];
    if (ndim < 4)
        ndim = 4;
    size_t cpu_mem = ndim * sizeof(int);
    int *dimA = (int *)malloc(cpu_mem);
    int *strideA = (int *)malloc(cpu_mem);
    int *dimC = (int *)malloc(cpu_mem);
    int *strideC = (int *)malloc(cpu_mem);

    int temp_strideA = 1;
    int temp_strideC = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        dimA[i] = i < in_arr->ndim ? (int)in_arr->shape[i] : 1;
        dimC[i] = i < in_arr->ndim - 1 ? (int)in_arr->shape[i] : 1;
        strideA[i] = temp_strideA;
        strideC[i] = temp_strideC;
        temp_strideA *= dimA[i];
        temp_strideC *= dimC[i];
    }

    dim3 blocks;
    dim3 threads;

    if (temp_strideA <= 1024) {
        threads.x = temp_strideA;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (temp_strideA + 1023) / 1024;
    }
    if (stream_handle)
        rescale_kernel<<<blocks, threads, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            (const float *)in_arr->data, (const float *)mean_arr->data,
            (const float *)var_arr->data, (const float *)ln_scale->data,
            (const float *)ln_bias->data, (float *)out_arr->data, last_dim, eps,
            temp_strideA);
    else
        rescale_kernel<<<blocks, threads>>>(
            (const float *)in_arr->data, (const float *)mean_arr->data,
            (const float *)var_arr->data, (const float *)ln_scale->data,
            (const float *)ln_bias->data, (float *)out_arr->data, last_dim, eps,
            temp_strideA);

    free(dimA);
    free(dimC);
    free(strideA);
    free(strideC);
    return 0;
}