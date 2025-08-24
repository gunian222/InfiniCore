import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
_TEST_CASES_ = [
    # shape, x_stride, y_stride, dim
    ((13, 4), None, None, 1),
    ((32, 512), None, None, 1),
    ((32, 20, 512), None, None, 2),
    ((28, 15, 15), None, None, 1),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


_INPLACE = [
    Inplace.INPLACE_X,
    Inplace.OUT_OF_PLACE,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# PyTorch reference reduce_max
def reduce_max_ref(x, dim):
    return torch.max(x, dim=dim).values


def test(
    handle,
    device,
    shape,
    x_stride=None,
    y_stride=None,
    dim=0,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing ReduceMax on {InfiniDeviceNames[device]} with shape:{shape} "
        f"x_stride:{x_stride} y_stride:{y_stride} dim:{dim} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    x = TestTensor(shape, x_stride, dtype, device)
    ans = reduce_max_ref(x.torch_tensor(), dim)

    # y 的 shape 去掉被 reduce 的 dim
    out_shape = shape[:dim] + shape[dim+1:]
    if inplace == Inplace.INPLACE_X:
        y = x
    else:
        y = TestTensor(out_shape, y_stride, dtype, device)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateReduceMaxDescriptor(
            handle, ctypes.byref(descriptor), y.descriptor, x.descriptor, dim
        )
    )

    # Invalidate descriptor
    x.destroy_desc()
    y.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetReduceMaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_reduce_max():
        check_error(
            LIBINFINIOP.infiniopReduceMax(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                None,
            )
        )

    lib_reduce_max()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation("PyTorch", lambda: reduce_max_ref(x.torch_tensor(), dim), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_reduce_max(), device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyReduceMaxDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
