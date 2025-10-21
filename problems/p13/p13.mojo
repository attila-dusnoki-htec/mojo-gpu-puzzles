from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from sys import size_of, argv
from testing import assert_equal

# ANCHOR: conv_1d_simple
alias TPB = 8
alias SIZE = 6
alias CONV = 3
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32
alias in_layout = Layout.row_major(SIZE)
alias out_layout = Layout.row_major(SIZE)
alias conv_layout = Layout.row_major(CONV)


fn conv_1d_simple[
    in_layout: Layout, out_layout: Layout, conv_layout: Layout
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, conv_layout],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    shared_a = LayoutTensor[
        dtype,
        Layout.row_major(SIZE),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    shared_b = LayoutTensor[
        dtype,
        Layout.row_major(CONV),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    if global_i < SIZE:
        shared_a[local_i] = a[global_i]
    if global_i < CONV:
        shared_b[local_i] = b[global_i]

    barrier()

    if global_i < SIZE:
        tmp = Scalar[dtype](0)
        @parameter
        for w in range(CONV):
            idx = local_i + w
            if idx < SIZE:
                tmp += rebind[Scalar[dtype]](shared_a[idx]) * rebind[Scalar[dtype]](shared_b[w])
        output[global_i] = tmp

# ANCHOR_END: conv_1d_simple

# ANCHOR: conv_1d_block_boundary
alias SIZE_2 = 15
alias CONV_2 = 4
alias BLOCKS_PER_GRID_2 = (2, 1)
alias THREADS_PER_BLOCK_2 = (TPB, 1)
alias in_2_layout = Layout.row_major(SIZE_2)
alias out_2_layout = Layout.row_major(SIZE_2)
alias conv_2_layout = Layout.row_major(CONV_2)
alias WINDOW_SIZE = TPB + CONV_2 - 1


fn conv_1d_block_boundary[
    in_layout: Layout, out_layout: Layout, conv_layout: Layout, dtype: DType
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, conv_layout],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    shared_a = LayoutTensor[
        dtype,
        Layout.row_major(WINDOW_SIZE),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    shared_b = LayoutTensor[
        dtype,
        Layout.row_major(CONV_2),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Init TPB sized part
    if global_i < SIZE_2:
        shared_a[local_i] = a[global_i]
    else:
        shared_a[local_i] = 0

    # Init overlapping part (values are from the next block)
    if local_i < CONV_2 - 1:
        global_offset = global_i + TPB
        local_offset = local_i + TPB
        if global_offset < SIZE_2:
            shared_a[local_offset] = a[global_offset]
        else:
            shared_a[local_offset] = 0

    # we use the same kernel in each block
    if local_i < CONV_2:
        shared_b[local_i] = b[local_i]

    barrier()

    if global_i < SIZE_2:
        tmp = Scalar[dtype](0)
        @parameter
        for w in range(CONV_2):
            if global_i + w < SIZE_2: # faster than shared boundary
                tmp += rebind[Scalar[dtype]](shared_a[local_i + w]) * rebind[Scalar[dtype]](shared_b[w])
        output[global_i] = tmp


# ANCHOR_END: conv_1d_block_boundary


def main():
    with DeviceContext() as ctx:
        size = SIZE_2 if argv()[1] == "--block-boundary" else SIZE
        conv = CONV_2 if argv()[1] == "--block-boundary" else CONV
        out = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](conv).enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = i

        with b.map_to_host() as b_host:
            for i in range(conv):
                b_host[i] = i

        if len(argv()) != 2 or argv()[1] not in [
            "--simple",
            "--block-boundary",
        ]:
            raise Error(
                "Expected one command-line argument: '--simple' or"
                " '--block-boundary'"
            )

        if argv()[1] == "--simple":
            var out_tensor = LayoutTensor[mut=False, dtype, out_layout](
                out.unsafe_ptr()
            )
            var a_tensor = LayoutTensor[mut=False, dtype, in_layout](
                a.unsafe_ptr()
            )
            var b_tensor = LayoutTensor[mut=False, dtype, conv_layout](
                b.unsafe_ptr()
            )
            ctx.enqueue_function[
                conv_1d_simple[in_layout, out_layout, conv_layout]
            ](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        else:
            var out_tensor = LayoutTensor[mut=False, dtype, out_2_layout](
                out.unsafe_ptr()
            )
            var a_tensor = LayoutTensor[mut=False, dtype, in_2_layout](
                a.unsafe_ptr()
            )
            var b_tensor = LayoutTensor[mut=False, dtype, conv_2_layout](
                b.unsafe_ptr()
            )
            ctx.enqueue_function[
                conv_1d_block_boundary[
                    in_2_layout, out_2_layout, conv_2_layout, dtype
                ]
            ](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )

        ctx.synchronize()
        expected = ctx.enqueue_create_host_buffer[dtype](size).enqueue_fill(0)

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(size):
                for j in range(conv):
                    if i + j < size:
                        expected[i] += a_host[i + j] * b_host[j]

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(size):
                assert_equal(out_host[i], expected[i])
