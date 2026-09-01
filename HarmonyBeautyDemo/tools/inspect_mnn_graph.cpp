#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <MNN_generated.h>

namespace {
std::string TensorName(const MNN::Net* net, int index)
{
    const auto* names = net->tensorName();
    if (names == nullptr || index < 0 || index >= static_cast<int>(names->size())) {
        return std::to_string(index);
    }
    return std::to_string(index) + ":" + names->Get(index)->str();
}

std::string TensorList(const MNN::Net* net, const flatbuffers::Vector<int32_t>* indexes)
{
    if (indexes == nullptr) return "[]";
    std::string result = "[";
    for (flatbuffers::uoffset_t index = 0; index < indexes->size(); ++index) {
        if (index > 0) result += ", ";
        result += TensorName(net, indexes->Get(index));
    }
    return result + "]";
}

std::string TensorShape(const MNN::Net* net, int index)
{
    const auto* descriptions = net->extraTensorDescribe();
    if (descriptions == nullptr) return "?";
    for (flatbuffers::uoffset_t item = 0; item < descriptions->size(); ++item) {
        const auto* description = descriptions->Get(item);
        if (description->index() != index || description->blob() == nullptr || description->blob()->dims() == nullptr) {
            continue;
        }
        std::string result = "[";
        const auto* dims = description->blob()->dims();
        for (flatbuffers::uoffset_t dim = 0; dim < dims->size(); ++dim) {
            if (dim > 0) result += "x";
            result += std::to_string(dims->Get(dim));
        }
        return result + "] fmt=" + std::to_string(static_cast<int>(description->blob()->dataFormat()));
    }
    return "?";
}
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: inspect_mnn_graph <model.mnn>\n";
        return 2;
    }
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "Cannot open " << argv[1] << "\n";
        return 1;
    }
    const auto size = input.tellg();
    input.seekg(0);
    std::vector<char> buffer(static_cast<size_t>(size));
    input.read(buffer.data(), size);

    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size());
    if (!MNN::VerifyNetBuffer(verifier)) {
        std::cerr << "Invalid MNN FlatBuffer\n";
        return 1;
    }
    const auto* net = MNN::GetNet(buffer.data());
    const auto* ops = net->oplists();
    if (ops == nullptr) return 0;
    for (flatbuffers::uoffset_t index = 0; index < ops->size(); ++index) {
        const auto* op = ops->Get(index);
        const auto type = op->type();
        std::cout << index << " " << MNN::EnumNameOpType(type)
                  << " name=" << (op->name() ? op->name()->str() : "<anon>")
                  << " in=" << TensorList(net, op->inputIndexes())
                  << " out=" << TensorList(net, op->outputIndexes());
        if (op->inputIndexes() != nullptr) {
            std::cout << " inShape=";
            for (flatbuffers::uoffset_t input = 0; input < op->inputIndexes()->size(); ++input) {
                if (input > 0) std::cout << ",";
                std::cout << TensorShape(net, op->inputIndexes()->Get(input));
            }
        }
        if (op->outputIndexes() != nullptr) {
            std::cout << " outShape=";
            for (flatbuffers::uoffset_t output = 0; output < op->outputIndexes()->size(); ++output) {
                if (output > 0) std::cout << ",";
                std::cout << TensorShape(net, op->outputIndexes()->Get(output));
            }
        }
        if (type == MNN::OpType_Softmax && op->main_as_Axis() != nullptr) {
            std::cout << " axis=" << op->main_as_Axis()->axis();
        }
        if (type == MNN::OpType_Flatten && op->main_as_Flatten() != nullptr) {
            const auto* flatten = op->main_as_Flatten();
            std::cout << " axis=" << flatten->axis() << " endAxis=" << flatten->endAxis();
        }
        if (type == MNN::OpType_Permute && op->main_as_Permute() != nullptr) {
            std::cout << " dims=[";
            const auto* dims = op->main_as_Permute()->dims();
            for (flatbuffers::uoffset_t dim = 0; dims != nullptr && dim < dims->size(); ++dim) {
                if (dim > 0) std::cout << ",";
                std::cout << dims->Get(dim);
            }
            std::cout << "]";
        }
        if (type == MNN::OpType_Const && op->main_as_Blob() != nullptr) {
            const auto* values = op->main_as_Blob()->int32s();
            std::cout << " int32=[";
            for (flatbuffers::uoffset_t value = 0; values != nullptr && value < values->size(); ++value) {
                if (value > 0) std::cout << ",";
                std::cout << values->Get(value);
            }
            std::cout << "]";
        }
        std::cout << "\n";
    }
    return 0;
}
