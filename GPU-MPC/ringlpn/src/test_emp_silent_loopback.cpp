#include "emp_silent_adapter.h"
#include "two_party_ot.h"

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using ringlpn_2pc::EmpSilentApi;
using ringlpn_2pc::EmpSilentPlan;
using ringlpn_2pc::OtBackend;
using ringlpn_2pc::PartyChannel;
using ringlpn_2pc::U128;
using ringlpn_2pc::Word;

struct Pipe {
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<uint8_t> bytes;
  bool corrupt_first_byte = false;
  bool corrupted = false;
};

struct MemoryEndpoint {
  Pipe *out = nullptr;
  Pipe *in = nullptr;
};

int memory_send(void *opaque, const void *data, size_t n) {
  auto &endpoint = *static_cast<MemoryEndpoint *>(opaque);
  const auto *bytes = static_cast<const uint8_t *>(data);
  {
    std::lock_guard<std::mutex> lock(endpoint.out->mutex);
    for (size_t i = 0; i < n; ++i) {
      uint8_t value = bytes[i];
      if (endpoint.out->corrupt_first_byte && !endpoint.out->corrupted) {
        value ^= 1;
        endpoint.out->corrupted = true;
      }
      endpoint.out->bytes.push_back(value);
    }
  }
  endpoint.out->ready.notify_all();
  return 0;
}

int memory_recv(void *opaque, void *data, size_t n) {
  auto &endpoint = *static_cast<MemoryEndpoint *>(opaque);
  auto *bytes = static_cast<uint8_t *>(data);
  std::unique_lock<std::mutex> lock(endpoint.in->mutex);
  endpoint.in->ready.wait(lock,
                          [&] { return endpoint.in->bytes.size() >= n; });
  for (size_t i = 0; i < n; ++i) {
    bytes[i] = endpoint.in->bytes.front();
    endpoint.in->bytes.pop_front();
  }
  return 0;
}

int memory_flush(void *) { return 0; }

ringlpn_emp_config control_config(int party, uint64_t count,
                                  MemoryEndpoint *endpoint, uint8_t sid_tag) {
  ringlpn_emp_config config{};
  config.abi_version = RINGLPN_EMP_SILENT_ABI_VERSION;
  config.local_party = static_cast<uint32_t>(party);
  config.sender_party = 0;
  config.direction = RINGLPN_EMP_STRAIGHT;
  config.parameter = RINGLPN_EMP_FERRET_B13;
  config.threads = 1;
  config.declared_count = count;
  config.capacity_floor = 13727984;
  for (size_t i = 0; i < sizeof(config.sid); ++i)
    config.sid[i] = static_cast<uint8_t>(sid_tag + i);
  config.io.context = endpoint;
  config.io.send = memory_send;
  config.io.recv = memory_recv;
  config.io.flush = memory_flush;
  return config;
}

void expect_begin_rejected(EmpSilentApi &api, ringlpn_emp_config left_config,
                           ringlpn_emp_config right_config) {
  ringlpn_emp_handle *left = api.create(left_config);
  ringlpn_emp_handle *right = api.create(right_config);
  bool left_failed = false;
  bool right_failed = false;
  std::thread a([&] {
    try { api.begin(left); } catch (...) { left_failed = true; }
  });
  std::thread b([&] {
    try { api.begin(right); } catch (...) { right_failed = true; }
  });
  a.join();
  b.join();
  api.destroy(left);
  api.destroy(right);
  if (!left_failed || !right_failed)
    throw std::runtime_error("malformed public control did not fail closed");
}

void run_control_tests(const std::string &bridge) {
  EmpSilentApi api(bridge);
  Pipe left_to_right;
  Pipe right_to_left;
  MemoryEndpoint left{&left_to_right, &right_to_left};
  MemoryEndpoint right{&right_to_left, &left_to_right};

  // Malformed ABI/config must be rejected without touching the transport.
  auto malformed = control_config(0, 1, &left, 7);
  malformed.abi_version++;
  bool malformed_failed = false;
  try { (void)api.create(malformed); } catch (...) { malformed_failed = true; }
  if (!malformed_failed) throw std::runtime_error("malformed ABI was accepted");

  // Consuming before begin poisons the handle and sends no bytes.
  auto early_config = control_config(0, 1, &left, 9);
  ringlpn_emp_handle *early = api.create(early_config);
  uint8_t zero = 0;
  bool early_failed = false;
  try { api.send(early, 1, &zero, &zero, 1); } catch (...) { early_failed = true; }
  api.destroy(early);
  if (!early_failed || !left_to_right.bytes.empty())
    throw std::runtime_error("consume-before-begin was not fail-closed");

  // SID mismatch is detected before correlation generation.
  expect_begin_rejected(api, control_config(0, 1, &left, 11),
                        control_config(1, 1, &right, 12));

  // Exact-count disagreement is part of the public control transcript.
  expect_begin_rejected(api, control_config(0, 1, &left, 13),
                        control_config(1, 2, &right, 13));

  // A corrupted control byte is rejected by both roles.
  left_to_right.corrupt_first_byte = true;
  expect_begin_rejected(api, control_config(0, 1, &left, 14),
                        control_config(1, 1, &right, 14));
}

struct ChosenOutputs {
  std::vector<uint8_t> bits;
  std::vector<Word> fields;
  std::vector<U128> blocks;
};

struct SuiteResult {
  ChosenOutputs straight;
  ChosenOutputs reversed;
};

void check_outputs(const ChosenOutputs &got) {
  const std::vector<uint8_t> expected_bits{0, 0, 1, 1};
  const std::vector<Word> expected_fields{5, 22, 9};
  const std::vector<U128> expected_blocks{
      (static_cast<U128>(2) << 64) | 1,
      (static_cast<U128>(8) << 64) | 7};
  if (got.bits != expected_bits || got.fields != expected_fields ||
      got.blocks != expected_blocks)
    throw std::runtime_error("chosen-message OT output mismatch");
}

SuiteResult run_suite(OtBackend backend, const std::string &bridge, int port) {
  SuiteResult result;
  std::exception_ptr errors[2];
  std::thread parties[2];
  for (int party = 0; party < 2; ++party) {
    parties[party] = std::thread([&, party] {
      try {
        EmpSilentPlan plan;
        plan.bridge_library = bridge;
        plan.straight_count = 9;
        plan.reversed_count = 9;
        plan.threads = 1;
        for (size_t i = 0; i < plan.public_manifest_digest.size(); ++i)
          plan.public_manifest_digest[i] = static_cast<uint8_t>(i + 1);
        PartyChannel channel(party, "127.0.0.1", port,
                             /*defer_ot_setup=*/true,
                             /*require_loopback_endpoints=*/true, backend,
                             backend == OtBackend::EmpSilent ? &plan : nullptr);
        channel.setup_ots();

        const std::vector<uint8_t> b0{0, 1, 0, 1};
        const std::vector<uint8_t> b1{1, 0, 1, 0};
        const std::vector<uint8_t> bc{0, 1, 1, 0};
        const std::vector<Word> f0{5, 6, 9};
        const std::vector<Word> f1{15, 22, 19};
        const std::vector<uint8_t> fc{0, 1, 0};
        const std::vector<U128> x0{
            (static_cast<U128>(2) << 64) | 1,
            (static_cast<U128>(4) << 64) | 3};
        const std::vector<U128> x1{
            (static_cast<U128>(6) << 64) | 5,
            (static_cast<U128>(8) << 64) | 7};
        const std::vector<uint8_t> xc{0, 1};

        ChosenOutputs mine;
        if (party == 0) {
          channel.ot_send_bits(b0, b1);
          channel.ot_send_field(f0, f1, 62);
          channel.ot_send_128(x0, x1);
          mine.bits = channel.ot_recv_bits(bc);
          mine.fields = channel.ot_recv_field(fc, 62);
          mine.blocks = channel.ot_recv_128(xc);
        } else {
          mine.bits = channel.ot_recv_bits(bc);
          mine.fields = channel.ot_recv_field(fc, 62);
          mine.blocks = channel.ot_recv_128(xc);
          channel.ot_send_bits(b0, b1);
          channel.ot_send_field(f0, f1, 62);
          channel.ot_send_128(x0, x1);
        }
        channel.finish_ots();
        check_outputs(mine);
        if (party == 0) result.reversed = std::move(mine);
        else result.straight = std::move(mine);
      } catch (...) {
        errors[party] = std::current_exception();
      }
    });
    if (party == 0) std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  parties[0].join();
  parties[1].join();
  for (const auto &error : errors)
    if (error) std::rethrow_exception(error);
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 4) {
      std::fprintf(stderr,
                   "Usage: %s BRIDGE [--controls-only|--full] [BASE_PORT]\n",
                   argv[0]);
      return 2;
    }
    const std::string bridge = argv[1];
    const std::string mode = argc >= 3 ? argv[2] : "--controls-only";
    const int port = argc >= 4 ? std::stoi(argv[3]) : 39761;
    run_control_tests(bridge);
    if (mode == "--full") {
      const SuiteResult sci = run_suite(OtBackend::SciIknp, bridge, port);
      const SuiteResult emp = run_suite(OtBackend::EmpSilent, bridge, port + 2);
      if (sci.straight.bits != emp.straight.bits ||
          sci.straight.fields != emp.straight.fields ||
          sci.straight.blocks != emp.straight.blocks ||
          sci.reversed.bits != emp.reversed.bits ||
          sci.reversed.fields != emp.reversed.fields ||
          sci.reversed.blocks != emp.reversed.blocks)
        throw std::runtime_error("SCI/EMP differential mismatch");
    } else if (mode != "--controls-only") {
      throw std::invalid_argument("unknown loopback mode");
    }
    std::printf("emp-silent loopback %s: PASS (unreviewed/unmeasured)\n",
                mode.c_str());
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "emp-silent loopback: FAIL: %s\n", e.what());
    return 1;
  }
}
