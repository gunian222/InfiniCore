import torch
import ctypes
from ctypes import c_uint64
import argparse

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试reduce max算子')
    parser.add_argument('--devices', type=str, default=None, 
                      help='指定测试设备，用逗号分隔 (例如: cuda:0,tianshu:1)')
    parser.add_argument('--verbose', action='store_true', 
                      help='显示详细测试信息')
    return parser.parse_args()

def get_test_devices(args, include_nvidia=True, include_tianshu=True, include_moxi=True):
   
    devices = []
    
    # 如果指定了设备，则优先使用指定的设备
    if args.devices:
        return args.devices.split(',')
    
    # 自动检测支持的设备
    if include_nvidia and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    
   
    if include_tianshu:
        # 这里应该是天枢设备的检测逻辑
        try:
            # 尝试检测天枢设备的示例代码
            devices.append("tianshu:0")  # 假设存在天枢设备0
        except:
            pass
    
    if include_moxi:
        # 这里应该是摩西设备的检测逻辑
        try:
            # 尝试检测摩西设备的示例代码
            devices.append("moxi:0")  # 假设存在摩西设备0
        except:
            pass
    
    if not devices:
        raise RuntimeError("没有找到可用的测试设备")
    
    return devices

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
    