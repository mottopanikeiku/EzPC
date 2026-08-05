#pragma once

#include "two_party_dpf_gpu.cuh"
#include "two_party_spfss.h"

namespace ringlpn_spfss {

// Publication-path entry point. Inputs, device frontier, and output keys all
// belong to `party`; the peer's private root/frontier never enters this API.
inline bool generate_party_spfss_keys_gpu_batched(
    int party, const SpfssPublicParams &params,
    const SpfssPartyBatch &batch, ringlpn_2pc::PartyChannel &channel,
    ringlpn_2pc::PartyRandom &rng, AESGlobalContext *gaes,
    GroupedHostKeys &grouped_keys, DpfCounters &counters) {
    counters = DpfCounters{};
    grouped_keys.clear();
    if ((party != 0 && party != 1) || gaes == nullptr ||
        !validate_party_batch(params, batch)) {
        return false;
    }
    const ringlpn_2pc::Counters before = channel.costs;
    std::vector<spfss_host::DPFKey> flat_keys;
    ringlpn_2pdpf::DpfStageCounters stages;
    const bool generated = ringlpn_2pdpf::two_party_dpf_gen_batch_gpu_batched(
        party, params.log_domain, static_cast<Word>(params.modulus),
        batch.offsets, batch.beta_factors, channel, rng, gaes, flat_keys,
        &stages);
    const ringlpn_2pc::Counters after = channel.costs;
    const bool counters_ok =
        record_dpf_generation_counters(before, after, stages, counters);
    return generated && counters_ok &&
           group_party_dpf_keys(batch, flat_keys, grouped_keys, counters);
}

}  // namespace ringlpn_spfss
