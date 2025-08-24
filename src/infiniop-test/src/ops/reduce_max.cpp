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
    std::vector<int> axes;   // 归约维度
    int num_axes;            // 维度数量
    int keep_dims;           // 是否保留归约维度
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol)
{
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    // 检查必要张量
    if (tensors.find("x") == tensors.end() ||
        tensors.find("y") == tensors.end() ||
        tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test: missing tensors");
    }

    test->_attributes->x = tensors["x"];
    test->_attributes->y = tensors["y"];
    test->_attributes->ans = tensors["ans"];

    // 初始化归约参数
    if (attributes.find("axes") != attributes.end()) {
        const uint8_t* ptr = attributes["axes"].data();
        test->_attributes->axes.assign(ptr, ptr + attributes["axes"].size());
        test->_attributes->num_axes = static_cast<int>(test->_attributes->axes.size());
    } else {
        throw std::runtime_error("Missing axes for reduce_max");
    }

    if (attributes.find("keep_dims") != attributes.end()) {
        test->_attributes->keep_dims = attributes["keep_dims"][0];
    } else {
        test->_attributes->keep_dims = 0; // 默认不保留
    }

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations)
{
    infiniopReduceMaxDescriptor_t op_desc;
    auto x = _attributes->x->to(device, device_id);
    auto y = _attributes->y->to(device, device_id);

    const int* axes = _attributes->axes.data();
    int num_axes = _attributes->num_axes;
    int keep_dims = _attributes->keep_dims;

    CHECK_OR(infiniopCreateReduceMaxDescriptor(handle, &op_desc,
                                               y->desc(), x->desc(),
                                               axes, num_axes, keep_dims),
             return TEST_FAILED(OP_CREATION_FAILED,
                                "Failed to create reduce_max descriptor."));

    size_t workspace_size = 0;
    CHECK_OR(infiniopGetReduceMaxWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED,
                                "Failed to get workspace size."));

    void* workspace = nullptr;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED,
                                "Failed to allocate workspace."));

    CHECK_OR(infiniopReduceMax(op_desc, workspace, workspace_size,
                               y->data(), x->data(), nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED,
                                "Failed during reduce_max execution."));

    try {
        allClose(y, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        infinirtFree(workspace);
        infiniopDestroyReduceMaxDescriptor(op_desc);
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = benchmark(
        [=]() {
            infiniopReduceMax(op_desc, workspace, workspace_size,
                              y->data(), x->data(), nullptr);
        },
        warm_ups, iterations);

    // 释放资源
    if (workspace) infinirtFree(workspace);
    infiniopDestroyReduceMaxDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() { return {"axes", "keep_dims"}; }
std::vector<std::string> Test::tensor_names() { return {"x", "y", "ans"}; }
std::vector<std::string> Test::output_names() { return {"y"}; }

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- x: " << _attributes->x->info() << std::endl;
    oss << "- y: " << _attributes->y->info() << std::endl;
    oss << "- ans: " << _attributes->ans->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    oss << "- axes=[";
    for (size_t i = 0; i < _attributes->axes.size(); i++)
        oss << _attributes->axes[i] << (i + 1 < _attributes->axes.size() ? ", " : "");
    oss << "], keep_dims=" << _attributes->keep_dims << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::reduce_max
