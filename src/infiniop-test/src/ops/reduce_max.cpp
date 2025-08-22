#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::reduce_max {
struct Test::Attributes {
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> y;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (tensors.find("x") == tensors.end()
        || tensors.find("y") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->x = tensors["x"];
    test->_attributes->y = tensors["y"];
    test->_attributes->ans = tensors["ans"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {
    infiniopReduceMaxDescriptor_t op_desc;
    auto x = _attributes->x->to(device, device_id);
    auto y = _attributes->y->to(device, device_id);
// 从测试属性中获取归约参数（假设_attributes包含axes、num_axes、keep_dims）
const int* axes = _attributes->axes.data();
int num_axes = _attributes->num_axes;
int keep_dims = _attributes->keep_dims;

CHECK_OR(infiniopCreateReduceMaxDescriptor(handle, &op_desc,
                                           y->desc(),
                                           x->desc(),
                                           axes,  // 补充归约维度
                                           num_axes,  // 补充维度数量
                                           keep_dims),  // 补充是否保留维度
         return TEST_FAILED(OP_CREATION_FAILED,
                            "Failed to create reduce_max descriptor."));

    size_t workspace_size;
    CHECK_OR(infiniopGetReduceMaxWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED,
                                "Failed to get workspace size."));

    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED,
                                "Failed to allocate workspace."));

    CHECK_OR(infiniopReduceMax(op_desc, workspace, workspace_size,
                               y->data(),
                               x->data(),
                               nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED,
                                "Failed during reduce_max execution."));

    try {
        allClose(y, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = benchmark(
        [=]() {
            infiniopReduceMax(op_desc, workspace, workspace_size,
                              y->data(),
                              x->data(),
                              nullptr);
        },
        warm_ups, iterations);
    if (workspace != nullptr) {
    infinirtFree(workspace);
}
    infiniopDestroyReduceMaxDescriptor(op_desc);  // 销毁算子描述符

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"x", "y", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"y"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- x: " << _attributes->x->info() << std::endl;
    oss << "- y: " << _attributes->y->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::reduce_max
