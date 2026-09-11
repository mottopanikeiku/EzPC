#!/usr/bin/env python3
"""Host-qualified GPU admission, without GPU access or container execution."""

import copy
import unittest
from unittest.mock import patch

import peer_private_execution as peer


class PublicationGpuIdentityTest(unittest.TestCase):
    def setUp(self):
        self.host_a = "a" * 64
        self.host_b = "b" * 64
        self.parties = (
            {"machine_identity_sha256": self.host_a,
             "gpu": {"requested_cdi_device": "nvidia.com/gpu=2"}},
            {"machine_identity_sha256": self.host_b,
             "gpu": {"requested_cdi_device": "nvidia.com/gpu=3"}},
        )
        self.preparation = {"gpu_identities": {
            "party0": self.device(self.host_a, "2", 1, "00000000:01:00.0"),
            "party1": self.device(self.host_b, "3", 2, "00000000:02:00.0"),
            "checker": self.device(self.host_a, "3", 3, "00000000:02:00.0"),
        }}

    @staticmethod
    def device(host, selector, identity, pci):
        return {
            "machine_identity_sha256": host,
            "requested_cdi_device": "nvidia.com/gpu=" + selector,
            "uuid": "GPU-00000000-0000-0000-0000-%012x" % identity,
            "pci_bus_id": pci,
        }

    def admitted(self, preparation):
        with patch.object(peer, "machine_identity_sha256", return_value=self.host_a):
            try:
                peer.verify_publication_gpu_bindings(preparation, *self.parties)
            except SystemExit as error:
                self.assertEqual(error.code, 2)
                return False
        return True

    def test_equal_remote_ordinal_is_not_a_local_physical_alias(self):
        self.assertTrue(self.admitted(self.preparation))
        for identity in ("uuid", "pci_bus_id"):
            aliased = copy.deepcopy(self.preparation)
            aliased["gpu_identities"]["checker"][identity] = (
                aliased["gpu_identities"]["party0"][identity])
            with self.subTest(identity=identity):
                self.assertFalse(self.admitted(aliased))

    def test_physical_binding_cannot_change_host_or_party_selector(self):
        wrong_host = copy.deepcopy(self.preparation)
        wrong_host["gpu_identities"]["checker"]["machine_identity_sha256"] = self.host_b
        self.assertFalse(self.admitted(wrong_host))
        wrong_selector = copy.deepcopy(self.preparation)
        wrong_selector["gpu_identities"]["party1"]["requested_cdi_device"] = "nvidia.com/gpu=4"
        self.assertFalse(self.admitted(wrong_selector))


if __name__ == "__main__":
    unittest.main()
