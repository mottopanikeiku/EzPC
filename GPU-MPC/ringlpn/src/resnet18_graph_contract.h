#pragma once

// Exact source-bound Orca ResNet18 forward graph used by the full known-zero
// composition artifact.  One stream item represents one stock key consumer;
// the live secure-truncation implementation occupies the stock GPUStTRKey
// positions without claiming byte compatibility for those keys.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ringlpn_resnet18 {

constexpr int kFullBw = 32;
constexpr int kScale = 10;
constexpr int kTruncatedBw = kFullBw - kScale;
constexpr size_t kLinearCount = 21;
constexpr size_t kConvCount = 20;
constexpr size_t kTruncationCount = 21;
constexpr size_t kReluCount = 17;
constexpr size_t kStockKeyCount = 19;
constexpr size_t kRemaskCount = 20;
constexpr size_t kResidualCount = 8;
constexpr size_t kStreamItemCount = 62;
constexpr uint64_t kInputWords = 1ULL * 224 * 224 * 3;
constexpr uint64_t kGlobalPoolInputWords = 1ULL * 7 * 7 * 512;
constexpr uint64_t kGlobalPoolOutputWords = 512;
constexpr uint64_t kClassifierOutputWords = 1000;

// Orca's integer global-average implementation multiplies the 7x7 sum by
// floor(2^scale / 49), then consumes the stock position-59 truncation.
constexpr uint64_t kGlobalPoolMultiplier = (uint64_t(1) << kScale) / 49;

// Indexes are zero based inside this contract.  stream_position and
// source_line are one based and match the source-bound execution manifest.
enum class LinearKind : uint32_t { Conv2D = 1, Fc = 2 };

struct LinearSpec {
    std::string_view name;
    LinearKind kind;
    uint32_t stream_position;
    uint32_t source_line;
    int n;
    int h;
    int w;
    int ci;
    int fh;
    int fw;
    int co;
    int padding;
    int stride;
    int rows;
    int inner;
    int cols;
    uint64_t input_words;
    uint64_t weight_words;
    uint64_t output_words;
};

constexpr std::array<LinearSpec, kLinearCount> kLinearSpecs = {{
    {"conv0", LinearKind::Conv2D, 1, 442, 1, 224, 224, 3, 7, 7, 64, 3, 2, 0, 0, 0, 150528, 9408, 802816},
    {"conv3", LinearKind::Conv2D, 5, 445, 1, 56, 56, 64, 3, 3, 64, 1, 1, 0, 0, 0, 200704, 36864, 200704},
    {"conv5", LinearKind::Conv2D, 8, 447, 1, 56, 56, 64, 3, 3, 64, 1, 1, 0, 0, 0, 200704, 36864, 200704},
    {"conv8", LinearKind::Conv2D, 11, 449, 1, 56, 56, 64, 3, 3, 64, 1, 1, 0, 0, 0, 200704, 36864, 200704},
    {"conv10", LinearKind::Conv2D, 14, 451, 1, 56, 56, 64, 3, 3, 64, 1, 1, 0, 0, 0, 200704, 36864, 200704},
    {"conv13", LinearKind::Conv2D, 17, 453, 1, 56, 56, 64, 3, 3, 128, 1, 2, 0, 0, 0, 200704, 73728, 100352},
    {"conv15", LinearKind::Conv2D, 20, 455, 1, 28, 28, 128, 3, 3, 128, 1, 1, 0, 0, 0, 100352, 147456, 100352},
    {"conv16", LinearKind::Conv2D, 22, 456, 1, 56, 56, 64, 1, 1, 128, 0, 2, 0, 0, 0, 200704, 8192, 100352},
    {"conv19", LinearKind::Conv2D, 25, 458, 1, 28, 28, 128, 3, 3, 128, 1, 1, 0, 0, 0, 100352, 147456, 100352},
    {"conv21", LinearKind::Conv2D, 28, 460, 1, 28, 28, 128, 3, 3, 128, 1, 1, 0, 0, 0, 100352, 147456, 100352},
    {"conv24", LinearKind::Conv2D, 31, 462, 1, 28, 28, 128, 3, 3, 256, 1, 2, 0, 0, 0, 100352, 294912, 50176},
    {"conv26", LinearKind::Conv2D, 34, 464, 1, 14, 14, 256, 3, 3, 256, 1, 1, 0, 0, 0, 50176, 589824, 50176},
    {"conv27", LinearKind::Conv2D, 36, 465, 1, 28, 28, 128, 1, 1, 256, 0, 2, 0, 0, 0, 100352, 32768, 50176},
    {"conv30", LinearKind::Conv2D, 39, 467, 1, 14, 14, 256, 3, 3, 256, 1, 1, 0, 0, 0, 50176, 589824, 50176},
    {"conv32", LinearKind::Conv2D, 42, 469, 1, 14, 14, 256, 3, 3, 256, 1, 1, 0, 0, 0, 50176, 589824, 50176},
    {"conv35", LinearKind::Conv2D, 45, 471, 1, 14, 14, 256, 3, 3, 512, 1, 2, 0, 0, 0, 50176, 1179648, 25088},
    {"conv37", LinearKind::Conv2D, 48, 473, 1, 7, 7, 512, 3, 3, 512, 1, 1, 0, 0, 0, 25088, 2359296, 25088},
    {"conv38", LinearKind::Conv2D, 50, 474, 1, 14, 14, 256, 1, 1, 512, 0, 2, 0, 0, 0, 50176, 131072, 25088},
    {"conv41", LinearKind::Conv2D, 53, 476, 1, 7, 7, 512, 3, 3, 512, 1, 1, 0, 0, 0, 25088, 2359296, 25088},
    {"conv43", LinearKind::Conv2D, 56, 478, 1, 7, 7, 512, 3, 3, 512, 1, 1, 0, 0, 0, 25088, 2359296, 25088},
    {"gemm48", LinearKind::Fc, 61, 482, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 512, 1000, 512, 512000, 1000},
}};

struct TruncationSpec {
    std::string_view node;
    uint32_t stream_position;
    // [0, 19] identifies a convolution; -1 identifies globalaveragepool46.
    int linear_index;
    uint64_t words;
};

constexpr std::array<TruncationSpec, kTruncationCount> kTruncationSpecs = {{
    {"conv0", 2, 0, 802816}, {"conv3", 6, 1, 200704},
    {"conv5", 9, 2, 200704}, {"conv8", 12, 3, 200704},
    {"conv10", 15, 4, 200704}, {"conv13", 18, 5, 100352},
    {"conv15", 21, 6, 100352}, {"conv16", 23, 7, 100352},
    {"conv19", 26, 8, 100352}, {"conv21", 29, 9, 100352},
    {"conv24", 32, 10, 50176}, {"conv26", 35, 11, 50176},
    {"conv27", 37, 12, 50176}, {"conv30", 40, 13, 50176},
    {"conv32", 43, 14, 50176}, {"conv35", 46, 15, 25088},
    {"conv37", 49, 16, 25088}, {"conv38", 51, 17, 25088},
    {"conv41", 54, 18, 25088}, {"conv43", 57, 19, 25088},
    {"globalaveragepool46", 59, -1, 512},
}};

enum class StockKeyKind : uint32_t { MaxPool = 1, ReluExtend = 2, SignExtend = 3 };

struct StockKeySpec {
    std::string_view node;
    StockKeyKind kind;
    uint32_t stream_position;
    int input_bw;
    int output_bw;
    uint64_t input_words;
    uint64_t output_words;
};

constexpr std::array<StockKeySpec, kStockKeyCount> kStockKeySpecs = {{
    {"maxpool1", StockKeyKind::MaxPool, 3, 22, 22, 802816, 200704},
    {"relu2", StockKeyKind::ReluExtend, 4, 22, 32, 200704, 200704},
    {"relu4", StockKeyKind::ReluExtend, 7, 22, 32, 200704, 200704},
    {"relu7", StockKeyKind::ReluExtend, 10, 22, 32, 200704, 200704},
    {"relu9", StockKeyKind::ReluExtend, 13, 22, 32, 200704, 200704},
    {"relu12", StockKeyKind::ReluExtend, 16, 22, 32, 200704, 200704},
    {"relu14", StockKeyKind::ReluExtend, 19, 22, 32, 100352, 100352},
    {"relu18", StockKeyKind::ReluExtend, 24, 22, 32, 100352, 100352},
    {"relu20", StockKeyKind::ReluExtend, 27, 22, 32, 100352, 100352},
    {"relu23", StockKeyKind::ReluExtend, 30, 22, 32, 100352, 100352},
    {"relu25", StockKeyKind::ReluExtend, 33, 22, 32, 50176, 50176},
    {"relu29", StockKeyKind::ReluExtend, 38, 22, 32, 50176, 50176},
    {"relu31", StockKeyKind::ReluExtend, 41, 22, 32, 50176, 50176},
    {"relu34", StockKeyKind::ReluExtend, 44, 22, 32, 50176, 50176},
    {"relu36", StockKeyKind::ReluExtend, 47, 22, 32, 25088, 25088},
    {"relu40", StockKeyKind::ReluExtend, 52, 22, 32, 25088, 25088},
    {"relu42", StockKeyKind::ReluExtend, 55, 22, 32, 25088, 25088},
    {"relu45", StockKeyKind::ReluExtend, 58, 22, 32, 25088, 25088},
    {"gemm48-signextend", StockKeyKind::SignExtend, 60, 22, 32, 512, 512},
}};

enum class MaskSource : uint32_t {
    Relu2,
    Relu4,
    Relu7,
    Relu9,
    Relu12,
    Relu14,
    Relu18,
    Relu20,
    Relu23,
    Relu25,
    Relu29,
    Relu31,
    Relu34,
    Relu36,
    Relu40,
    Relu42,
    SignExtend,
};

struct RemaskSpec {
    uint32_t target_linear_index;
    MaskSource source;
    uint64_t words;
};

constexpr std::array<RemaskSpec, kRemaskCount> kRemaskSpecs = {{
    {1, MaskSource::Relu2, 200704},
    {2, MaskSource::Relu4, 200704},
    {3, MaskSource::Relu7, 200704},
    {4, MaskSource::Relu9, 200704},
    {5, MaskSource::Relu12, 200704},
    {6, MaskSource::Relu14, 100352},
    {7, MaskSource::Relu12, 200704},
    {8, MaskSource::Relu18, 100352},
    {9, MaskSource::Relu20, 100352},
    {10, MaskSource::Relu23, 100352},
    {11, MaskSource::Relu25, 50176},
    {12, MaskSource::Relu23, 100352},
    {13, MaskSource::Relu29, 50176},
    {14, MaskSource::Relu31, 50176},
    {15, MaskSource::Relu34, 50176},
    {16, MaskSource::Relu36, 25088},
    {17, MaskSource::Relu34, 50176},
    {18, MaskSource::Relu40, 25088},
    {19, MaskSource::Relu42, 25088},
    {20, MaskSource::SignExtend, 512},
}};

constexpr size_t stock_index(MaskSource source) {
    return source == MaskSource::SignExtend
               ? 18
               : static_cast<size_t>(source) + 1;
}

enum class ValueSourceKind : uint32_t {
    Truncation = 1,
    Stock = 2,
    Residual = 3,
};

struct ValueSource {
    ValueSourceKind kind;
    uint32_t index;
};

struct ResidualSpec {
    ValueSource main;
    ValueSource shortcut;
    uint32_t result_stock_index;
    uint64_t words;
};

constexpr std::array<ResidualSpec, kResidualCount> kResidualSpecs = {{
    {{ValueSourceKind::Truncation, 2}, {ValueSourceKind::Stock, 1}, 3, 200704},
    {{ValueSourceKind::Truncation, 4}, {ValueSourceKind::Stock, 3}, 5, 200704},
    {{ValueSourceKind::Truncation, 6}, {ValueSourceKind::Truncation, 7}, 7, 100352},
    {{ValueSourceKind::Truncation, 9}, {ValueSourceKind::Stock, 7}, 9, 100352},
    {{ValueSourceKind::Truncation, 11}, {ValueSourceKind::Truncation, 12}, 11, 50176},
    {{ValueSourceKind::Truncation, 14}, {ValueSourceKind::Stock, 11}, 13, 50176},
    {{ValueSourceKind::Truncation, 16}, {ValueSourceKind::Truncation, 17}, 15, 25088},
    {{ValueSourceKind::Truncation, 19}, {ValueSourceKind::Stock, 15}, 17, 25088},
}};

constexpr std::array<ValueSource, kStockKeyCount> kStockInputSources = {{
    {ValueSourceKind::Truncation, 0},
    {ValueSourceKind::Stock, 0},
    {ValueSourceKind::Truncation, 1},
    {ValueSourceKind::Residual, 0},
    {ValueSourceKind::Truncation, 3},
    {ValueSourceKind::Residual, 1},
    {ValueSourceKind::Truncation, 5},
    {ValueSourceKind::Residual, 2},
    {ValueSourceKind::Truncation, 8},
    {ValueSourceKind::Residual, 3},
    {ValueSourceKind::Truncation, 10},
    {ValueSourceKind::Residual, 4},
    {ValueSourceKind::Truncation, 13},
    {ValueSourceKind::Residual, 5},
    {ValueSourceKind::Truncation, 15},
    {ValueSourceKind::Residual, 6},
    {ValueSourceKind::Truncation, 18},
    {ValueSourceKind::Residual, 7},
    {ValueSourceKind::Truncation, 20},
}};

enum class StreamKind : uint32_t {
    Linear,
    Truncation,
    MaxPool,
    ReluExtend,
    GlobalAverage,
    SignExtend,
    Output,
};

struct StreamItem {
    uint32_t position;
    std::string_view node;
    StreamKind kind;
};

constexpr std::array<StreamItem, kStreamItemCount> kStreamItems = {{
    {1,"conv0",StreamKind::Linear},{2,"conv0",StreamKind::Truncation},{3,"maxpool1",StreamKind::MaxPool},{4,"relu2",StreamKind::ReluExtend},
    {5,"conv3",StreamKind::Linear},{6,"conv3",StreamKind::Truncation},{7,"relu4",StreamKind::ReluExtend},
    {8,"conv5",StreamKind::Linear},{9,"conv5",StreamKind::Truncation},{10,"relu7",StreamKind::ReluExtend},
    {11,"conv8",StreamKind::Linear},{12,"conv8",StreamKind::Truncation},{13,"relu9",StreamKind::ReluExtend},
    {14,"conv10",StreamKind::Linear},{15,"conv10",StreamKind::Truncation},{16,"relu12",StreamKind::ReluExtend},
    {17,"conv13",StreamKind::Linear},{18,"conv13",StreamKind::Truncation},{19,"relu14",StreamKind::ReluExtend},
    {20,"conv15",StreamKind::Linear},{21,"conv15",StreamKind::Truncation},{22,"conv16",StreamKind::Linear},{23,"conv16",StreamKind::Truncation},{24,"relu18",StreamKind::ReluExtend},
    {25,"conv19",StreamKind::Linear},{26,"conv19",StreamKind::Truncation},{27,"relu20",StreamKind::ReluExtend},
    {28,"conv21",StreamKind::Linear},{29,"conv21",StreamKind::Truncation},{30,"relu23",StreamKind::ReluExtend},
    {31,"conv24",StreamKind::Linear},{32,"conv24",StreamKind::Truncation},{33,"relu25",StreamKind::ReluExtend},
    {34,"conv26",StreamKind::Linear},{35,"conv26",StreamKind::Truncation},{36,"conv27",StreamKind::Linear},{37,"conv27",StreamKind::Truncation},{38,"relu29",StreamKind::ReluExtend},
    {39,"conv30",StreamKind::Linear},{40,"conv30",StreamKind::Truncation},{41,"relu31",StreamKind::ReluExtend},
    {42,"conv32",StreamKind::Linear},{43,"conv32",StreamKind::Truncation},{44,"relu34",StreamKind::ReluExtend},
    {45,"conv35",StreamKind::Linear},{46,"conv35",StreamKind::Truncation},{47,"relu36",StreamKind::ReluExtend},
    {48,"conv37",StreamKind::Linear},{49,"conv37",StreamKind::Truncation},{50,"conv38",StreamKind::Linear},{51,"conv38",StreamKind::Truncation},{52,"relu40",StreamKind::ReluExtend},
    {53,"conv41",StreamKind::Linear},{54,"conv41",StreamKind::Truncation},{55,"relu42",StreamKind::ReluExtend},
    {56,"conv43",StreamKind::Linear},{57,"conv43",StreamKind::Truncation},{58,"relu45",StreamKind::ReluExtend},
    {59,"globalaveragepool46",StreamKind::GlobalAverage},{60,"gemm48",StreamKind::SignExtend},{61,"gemm48",StreamKind::Linear},{62,"output",StreamKind::Output},
}};

inline bool contract_valid() {
    size_t linear = 0;
    size_t truncation = 0;
    size_t maxpool = 0;
    size_t relu = 0;
    size_t global_average = 0;
    size_t signextend = 0;
    size_t output = 0;
    for (size_t i = 0; i < kStreamItems.size(); ++i) {
        if (kStreamItems[i].position != i + 1 || kStreamItems[i].node.empty()) {
            return false;
        }
        switch (kStreamItems[i].kind) {
            case StreamKind::Linear: ++linear; break;
            case StreamKind::Truncation: ++truncation; break;
            case StreamKind::MaxPool: ++maxpool; break;
            case StreamKind::ReluExtend: ++relu; break;
            case StreamKind::GlobalAverage: ++global_average; break;
            case StreamKind::SignExtend: ++signextend; break;
            case StreamKind::Output: ++output; break;
        }
    }
    if (linear != kLinearCount || truncation != 20 || maxpool != 1 ||
        relu != kReluCount || global_average != 1 || signextend != 1 ||
        output != 1) {
        return false;
    }
    for (size_t i = 0; i < kLinearSpecs.size(); ++i) {
        const LinearSpec &spec = kLinearSpecs[i];
        if (spec.name.empty() || spec.stream_position == 0 ||
            spec.input_words == 0 || spec.weight_words == 0 ||
            spec.output_words == 0) {
            return false;
        }
        if (spec.kind == LinearKind::Conv2D) {
            const int oh = (spec.h + 2 * spec.padding - spec.fh) / spec.stride + 1;
            const int ow = (spec.w + 2 * spec.padding - spec.fw) / spec.stride + 1;
            if (spec.n <= 0 || spec.h <= 0 || spec.w <= 0 || spec.ci <= 0 ||
                spec.fh <= 0 || spec.fw <= 0 || spec.co <= 0 ||
                spec.stride <= 0 || oh <= 0 || ow <= 0 ||
                spec.input_words != static_cast<uint64_t>(spec.n) * spec.h * spec.w * spec.ci ||
                spec.weight_words != static_cast<uint64_t>(spec.fh) * spec.fw * spec.ci * spec.co ||
                spec.output_words != static_cast<uint64_t>(spec.n) * oh * ow * spec.co) {
                return false;
            }
        } else if (spec.rows <= 0 || spec.inner <= 0 || spec.cols <= 0 ||
                   spec.input_words != static_cast<uint64_t>(spec.rows) * spec.inner ||
                   spec.weight_words != static_cast<uint64_t>(spec.inner) * spec.cols ||
                   spec.output_words != static_cast<uint64_t>(spec.rows) * spec.cols) {
            return false;
        }
    }
    for (size_t i = 0; i < kRemaskSpecs.size(); ++i) {
        const RemaskSpec &spec = kRemaskSpecs[i];
        const size_t source_index = stock_index(spec.source);
        if (spec.target_linear_index != i + 1 ||
            spec.words != kLinearSpecs[i + 1].input_words ||
            source_index >= kStockKeySpecs.size() ||
            spec.words != kStockKeySpecs[source_index].output_words) {
            return false;
        }
    }
    const auto source_words = [](ValueSource source) -> uint64_t {
        if (source.kind == ValueSourceKind::Truncation &&
            source.index < kTruncationSpecs.size()) {
            return kTruncationSpecs[source.index].words;
        }
        if (source.kind == ValueSourceKind::Stock &&
            source.index < kStockKeySpecs.size()) {
            return kStockKeySpecs[source.index].output_words;
        }
        if (source.kind == ValueSourceKind::Residual &&
            source.index < kResidualSpecs.size()) {
            return kResidualSpecs[source.index].words;
        }
        return 0;
    };
    for (size_t i = 0; i < kResidualSpecs.size(); ++i) {
        const ResidualSpec &spec = kResidualSpecs[i];
        if (spec.words == 0 ||
            source_words(spec.main) != spec.words ||
            source_words(spec.shortcut) != spec.words ||
            spec.result_stock_index >= kStockKeySpecs.size() ||
            kStockKeySpecs[spec.result_stock_index].input_words != spec.words ||
            kStockInputSources[spec.result_stock_index].kind !=
                ValueSourceKind::Residual ||
            kStockInputSources[spec.result_stock_index].index != i) {
            return false;
        }
    }
    for (size_t i = 0; i < kStockInputSources.size(); ++i) {
        const ValueSource source = kStockInputSources[i];
        if (source_words(source) != kStockKeySpecs[i].input_words ||
            (source.kind == ValueSourceKind::Stock && source.index >= i) ||
            (source.kind == ValueSourceKind::Residual &&
             (source.index >= kResidualSpecs.size() ||
              kResidualSpecs[source.index].result_stock_index != i))) {
            return false;
        }
    }
    return kTruncationSpecs.back().words == kGlobalPoolOutputWords &&
           kLinearSpecs.back().output_words == kClassifierOutputWords;
}

}  // namespace ringlpn_resnet18
