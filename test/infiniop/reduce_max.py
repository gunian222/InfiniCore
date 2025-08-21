import torch
import ctypes
from ctypes import c_uint64

def test_reduce_max(
    handle,
    device,
    shape,
    axes,
    keep_dims,
    dtype=torch.float16,
    sync=None
):
    # 生成测试数据
    input_tensor = torch.randn(shape, dtype=dtype)
    expected = torch.max(input_tensor, dim=tuple(axes), keepdim=keep_dims)

    # 初始化InfiniOP张量
    input_op = TestTensor(shape, dtype=dtype, device=device, data=input_tensor)
    output_op = TestTensor(expected.shape, dtype=dtype, device=device, mode="zeros")

    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateReduceMaxDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_op.descriptor,
            input_op.descriptor,
            axes,
            len(axes),
            1 if keep_dims else 0
        )
    )

    # 获取工作空间大小
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetReduceMaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # 执行算子
    check_error(
        LIBINFINIOP.infiniopReduceMax(
            descriptor,
            workspace.data(),
            workspace_size.value,
            output_op.data(),
            input_op.data(),
            None
        )
    )

    # 验证结果
    atol, rtol = get_tolerance(dtype, device)
    assert torch.allclose(output_op.actual_tensor(), expected, atol=atol, rtol=rtol)

    # 销毁资源
    check_error(LIBINFINIOP.infiniopDestroyReduceMaxDescriptor(descriptor))

if __name__ == "__main__":
    args = get_args()
    # 测试用例覆盖不同形状和轴
    test_cases = [
        ((2, 3, 4), (0,), True),
        ((5, 5), (0, 1), False),
        ((2, 2, 2, 2), (1, 3), True),
    ]
    # 支持的设备类型
    for device in get_test_devices(args, include_nvidia=True, include_tianshu=True, include_moxi=True):
        test_operator(device, test_reduce_max, test_cases, [torch.float16, torch.float32])
    print("\033[92mAll tests passed!\033[0m")