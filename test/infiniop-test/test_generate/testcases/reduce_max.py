import numpy as np
import gguf
from typing import List
from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides


# =============== 参考实现 ===============
def reduce_max(x: np.ndarray, axes: List[int], keepdims: bool = True):
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    return np.max(x, axis=tuple(axes), keepdims=keepdims)


# =============== 随机数据生成 ===============
def random_tensor(shape, dtype):
    rate = 1e-3
    var = 0.5 * rate
    return rate * np.random.rand(*shape).astype(dtype) - var


# =============== 测试用例类 ===============
class ReduceMaxTestCase(InfiniopTestCase):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        axes: List[int],
        keepdims: bool,
        shape_x: List[int] | None,
        shape_y: List[int] | None,
        stride_x: List[int] | None,
        stride_y: List[int] | None,
    ):
        super().__init__("reduce_max")
        self.x = x
        self.y = y
        self.axes = axes
        self.keepdims = keepdims
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.stride_x = stride_x
        self.stride_y = stride_y

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # shape / stride 信息
        if self.shape_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.shape"), self.shape_x)
        if self.shape_y is not None:
            test_writer.add_array(test_writer.gguf_key("y.shape"), self.shape_y)
        if self.stride_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*self.stride_x))
        test_writer.add_array(
            test_writer.gguf_key("y.strides"),
            gguf_strides(*self.stride_y if self.stride_y is not None else contiguous_gguf_strides(self.shape_y))
        )

        # 输入输出张量
        test_writer.add_tensor(
            test_writer.gguf_key("x"),
            self.x,
            raw_dtype=np_dtype_to_ggml(self.x.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("y"),
            self.y,
            raw_dtype=np_dtype_to_ggml(self.y.dtype),
        )

        # 正确结果 (ground truth)
        ans = reduce_max(self.x.astype(np.float64), self.axes, keepdims=self.keepdims)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans,
            raw_dtype=gguf.GGMLQuantizationType.F64
        )


# =============== 主程序入口 ===============
if __name__ == "__main__":
    test_writer = InfiniopTestWriter("reduce_max.gguf")
    test_cases = []

    # 配置测试样例
    _TEST_CASES_ = [
        # (shape, axes, keepdims, stride_x, stride_y)
        ((3, 3), [1], True, None, None),
        ((3, 3), [1], False, None, None),
        ((32, 512), [1], True, None, None),
        ((32, 512), [0], True, None, None),
        ((32, 20, 512), [1, 2], True, None, None),
        ((32, 20, 512), [1, 2], False, (10240, 512, 1), None),  # 测试非连续输入
    ]
    _TENSOR_DTYPES_ = [np.float16, np.float32]

    for dtype in _TENSOR_DTYPES_:
        for shape, axes, keepdims, stride_x, stride_y in _TEST_CASES_:
            x = random_tensor(shape, dtype)
            y_shape = np.max(x, axis=tuple(axes), keepdims=keepdims).shape
            y = np.empty(y_shape, dtype=dtype)

            test_case = ReduceMaxTestCase(
                x,
                y,
                axes,
                keepdims,
                list(shape),
                list(y_shape),
                stride_x,
                stride_y,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
