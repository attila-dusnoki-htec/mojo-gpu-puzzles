from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp
from bit import log2_ceil
from utils.numerics import max_finite, min_finite


alias SIZE = 128  # This must be equal to INPUT_SIZE in p18.py
alias layout = Layout.row_major(SIZE)
alias GRID_DIM_X = 1
# Tree-based reduction require the number of threads to be the next power of two >= SIZE for correctness.
alias BLOCK_DIM_X = 1 << log2_ceil(SIZE)


fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
):
    shared_max = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    shared_sum = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    tid = thread_idx.x

    local_val = input[tid]

    # init shared max
    local_max = min_finite[dtype]()
    if tid < input_size:
        local_max = rebind[Scalar[dtype]](local_val)

    shared_max[tid] = rebind[Scalar[dtype]](local_max)
    barrier()

    # calc max at 0
    offset = input_size // 2
    while offset > 0:
        tmp = rebind[Scalar[dtype]](shared_max[tid + offset])
        barrier()

        if tmp > local_max:
            shared_max[tid] = rebind[Scalar[dtype]](tmp)
            local_max = rebind[Scalar[dtype]](tmp)

        offset //= 2
        barrier()

    max_val = shared_max[0]

    # init shared sum
    var exp_val: output.element_type = 0
    if tid < input_size:
        exp_val = exp(local_val - max_val)

    shared_sum[tid] = exp_val
    barrier()

    # calc sum at 0
    offset = input_size // 2
    while offset > 0:
        tmp = rebind[Scalar[dtype]](shared_sum[tid + offset])
        barrier()

        shared_sum[tid] += tmp
        offset //= 2
        barrier()

    sum_val = shared_sum[0]
    if tid < input_size:
        output[tid] = exp_val / sum_val


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    var max_val: input.element_type = min_finite[dtype]()
    for i in range(input_size):
        if max_val < input[i]:
            max_val = input[i]

    var denom: input.element_type = 0
    for i in range(input_size):
        denom += exp(input[i] - max_val)

    for i in range(input_size):
        output[i] = exp(input[i]-max_val) / denom


# ANCHOR_END: softmax_cpu_kernel

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](output.to_layout_tensor())
        var input_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](input.to_layout_tensor())

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[output_tensor.dtype]]](
                        output_tensor.ptr
                    ),
                    input_size,
                    owning=False,
                ),
                0,
            )

            gpu_ctx.enqueue_function[
                softmax_gpu_kernel[layout, input_size, dtype]
            ](
                output_tensor,
                input_tensor,
                grid_dim=GRID_DIM_X,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
