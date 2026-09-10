// Host-only projection of the compiled ResNet18 graph contract for the
// source-bound manifest gate.  JSONL keeps the comparison independent of
// compiler layout and does not require a JSON library in the CUDA binary.

#include "resnet18_graph_contract.h"

#include <cstdio>
#include <string_view>

namespace {

namespace contract = ringlpn_resnet18;

const char *linear_kind(contract::LinearKind kind) {
    switch (kind) {
        case contract::LinearKind::Conv2D: return "conv2d";
        case contract::LinearKind::Fc: return "fc";
    }
    return "invalid";
}

const char *stock_kind(contract::StockKeyKind kind) {
    switch (kind) {
        case contract::StockKeyKind::MaxPool: return "maxpool";
        case contract::StockKeyKind::ReluExtend: return "relu_extend";
        case contract::StockKeyKind::SignExtend: return "sign_extend";
    }
    return "invalid";
}

const char *stream_kind(contract::StreamKind kind) {
    switch (kind) {
        case contract::StreamKind::Linear: return "linear";
        case contract::StreamKind::Truncation: return "truncation";
        case contract::StreamKind::MaxPool: return "maxpool";
        case contract::StreamKind::ReluExtend: return "relu_extend";
        case contract::StreamKind::GlobalAverage: return "global_average";
        case contract::StreamKind::SignExtend: return "sign_extend";
        case contract::StreamKind::Output: return "output";
    }
    return "invalid";
}

const char *value_kind(contract::ValueSourceKind kind) {
    switch (kind) {
        case contract::ValueSourceKind::Truncation: return "truncation";
        case contract::ValueSourceKind::Stock: return "stock";
        case contract::ValueSourceKind::Residual: return "residual";
    }
    return "invalid";
}

const char *mask_source(contract::MaskSource source) {
    switch (source) {
        case contract::MaskSource::Relu2: return "relu2";
        case contract::MaskSource::Relu4: return "relu4";
        case contract::MaskSource::Relu7: return "relu7";
        case contract::MaskSource::Relu9: return "relu9";
        case contract::MaskSource::Relu12: return "relu12";
        case contract::MaskSource::Relu14: return "relu14";
        case contract::MaskSource::Relu18: return "relu18";
        case contract::MaskSource::Relu20: return "relu20";
        case contract::MaskSource::Relu23: return "relu23";
        case contract::MaskSource::Relu25: return "relu25";
        case contract::MaskSource::Relu29: return "relu29";
        case contract::MaskSource::Relu31: return "relu31";
        case contract::MaskSource::Relu34: return "relu34";
        case contract::MaskSource::Relu36: return "relu36";
        case contract::MaskSource::Relu40: return "relu40";
        case contract::MaskSource::Relu42: return "relu42";
        case contract::MaskSource::SignExtend: return "sign_extend";
    }
    return "invalid";
}

void print_name(std::string_view value) {
    std::printf("%.*s", static_cast<int>(value.size()), value.data());
}

}  // namespace

int main() {
    const bool valid = contract::contract_valid();
    std::printf(
        "{\"section\":\"meta\",\"schema\":\"ringlpn-resnet18-compiled-graph-contract-v1\","
        "\"full_bw\":%d,\"scale\":%d,\"truncated_bw\":%d,"
        "\"linear_count\":%zu,\"truncation_count\":%zu,"
        "\"stock_count\":%zu,\"remask_count\":%zu,"
        "\"residual_count\":%zu,\"stream_count\":%zu,\"status\":\"%s\"}\n",
        contract::kFullBw, contract::kScale, contract::kTruncatedBw,
        contract::kLinearCount, contract::kTruncationCount,
        contract::kStockKeyCount, contract::kRemaskCount,
        contract::kResidualCount, contract::kStreamItemCount,
        valid ? "pass" : "FAIL");
    if (!valid) return 1;

    for (size_t i = 0; i < contract::kLinearSpecs.size(); ++i) {
        const auto &s = contract::kLinearSpecs[i];
        std::printf("{\"section\":\"linear\",\"index\":%zu,\"name\":\"", i);
        print_name(s.name);
        std::printf(
            "\",\"kind\":\"%s\",\"stream_position\":%u,\"source_line\":%u,"
            "\"n\":%d,\"h\":%d,\"w\":%d,\"ci\":%d,\"fh\":%d,"
            "\"fw\":%d,\"co\":%d,\"padding\":%d,\"stride\":%d,"
            "\"rows\":%d,\"inner\":%d,\"cols\":%d,"
            "\"input_words\":%llu,\"weight_words\":%llu,\"output_words\":%llu}\n",
            linear_kind(s.kind), s.stream_position, s.source_line,
            s.n, s.h, s.w, s.ci, s.fh, s.fw, s.co, s.padding, s.stride,
            s.rows, s.inner, s.cols,
            static_cast<unsigned long long>(s.input_words),
            static_cast<unsigned long long>(s.weight_words),
            static_cast<unsigned long long>(s.output_words));
    }

    for (size_t i = 0; i < contract::kTruncationSpecs.size(); ++i) {
        const auto &s = contract::kTruncationSpecs[i];
        std::printf("{\"section\":\"truncation\",\"index\":%zu,\"node\":\"", i);
        print_name(s.node);
        std::printf(
            "\",\"stream_position\":%u,\"linear_index\":%d,\"words\":%llu}\n",
            s.stream_position, s.linear_index,
            static_cast<unsigned long long>(s.words));
    }

    for (size_t i = 0; i < contract::kStockKeySpecs.size(); ++i) {
        const auto &s = contract::kStockKeySpecs[i];
        std::printf("{\"section\":\"stock\",\"index\":%zu,\"node\":\"", i);
        print_name(s.node);
        std::printf(
            "\",\"kind\":\"%s\",\"stream_position\":%u,"
            "\"input_bw\":%d,\"output_bw\":%d,"
            "\"input_words\":%llu,\"output_words\":%llu}\n",
            stock_kind(s.kind), s.stream_position, s.input_bw, s.output_bw,
            static_cast<unsigned long long>(s.input_words),
            static_cast<unsigned long long>(s.output_words));
    }

    for (size_t i = 0; i < contract::kRemaskSpecs.size(); ++i) {
        const auto &s = contract::kRemaskSpecs[i];
        std::printf(
            "{\"section\":\"remask\",\"index\":%zu,\"target_linear_index\":%u,"
            "\"source\":\"%s\",\"words\":%llu}\n",
            i, s.target_linear_index, mask_source(s.source),
            static_cast<unsigned long long>(s.words));
    }

    for (size_t i = 0; i < contract::kResidualSpecs.size(); ++i) {
        const auto &s = contract::kResidualSpecs[i];
        std::printf(
            "{\"section\":\"residual\",\"index\":%zu,"
            "\"main_kind\":\"%s\",\"main_index\":%u,"
            "\"shortcut_kind\":\"%s\",\"shortcut_index\":%u,"
            "\"result_stock_index\":%u,\"words\":%llu}\n",
            i, value_kind(s.main.kind), s.main.index,
            value_kind(s.shortcut.kind), s.shortcut.index,
            s.result_stock_index, static_cast<unsigned long long>(s.words));
    }

    for (size_t i = 0; i < contract::kStockInputSources.size(); ++i) {
        const auto &s = contract::kStockInputSources[i];
        std::printf(
            "{\"section\":\"stock_input\",\"index\":%zu,"
            "\"source_kind\":\"%s\",\"source_index\":%u}\n",
            i, value_kind(s.kind), s.index);
    }

    for (size_t i = 0; i < contract::kStreamItems.size(); ++i) {
        const auto &s = contract::kStreamItems[i];
        std::printf("{\"section\":\"stream\",\"index\":%zu,\"position\":%u,\"node\":\"", i, s.position);
        print_name(s.node);
        std::printf("\",\"kind\":\"%s\"}\n", stream_kind(s.kind));
    }
    return 0;
}
