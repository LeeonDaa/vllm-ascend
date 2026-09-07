#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import threading
import unittest
from unittest.mock import MagicMock, call

import numpy as np

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerSendingThread,
    KVTransferThread,
    LayerBatchBuilder,
    _build_range_debug_payload,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBlockRange,
    LayerRangeReqMeta,
    LayerTransferTask,
    LayerwiseBlockKey,
    LoadSpec,
    SharedBlockData,
    ReqMeta,
    make_layerwise_block_key,
    parse_layerwise_block_key,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_session_tracker import (
    MooncakeSessionTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

# isort: on


def make_token_database() -> ChunkedTokenDatabase:
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0, 0)], [16], None)
    database.set_group_buffers(
        {0: [1000, 2000, 3000]},
        {0: [10, 20, 30]},
        {0: [100, 200, 300]},
        group_num_layers={0: 2},
        group_layer_cache_entry_offsets={0: [0, 2, 3]},
    )
    return database


class TestMooncakeLayerBatchBuilder(unittest.TestCase):
    def test_range_debug_payload_reports_per_key_bytes_and_offsets(self):
        payload = _build_range_debug_payload(
            "save",
            3,
            [[10, 20], [7]],
            [[30, 40], [50]],
            [30, -1],
        )

        self.assertEqual(payload["event"], "range")
        self.assertEqual(payload["layer_id"], 3)
        self.assertEqual(payload["requested_bytes"], [30, 7])
        self.assertEqual(payload["object_offsets"], [[30, 40], [50]])
        self.assertEqual(payload["results"], [30, -1])

    def test_key_major_ranges_use_full_object_offsets(self):
        request = ReqMeta("r1", block_ids=[2], block_hashes=[])
        request.save_block_keys = ["key"]
        task = LayerTransferTask(
            layer_id=1,
            layer_idx_in_group=1,
            block_ranges=[LayerBlockRange(request, 0, 1)],
            use_key_major_ranges=True,
        )
        builder = LayerBatchBuilder(make_token_database(), page_size_bytes=60, num_layers=2)

        result = builder.build(task)

        self.assertIsInstance(result, LayerRangeReqMeta)
        assert isinstance(result, LayerRangeReqMeta)
        self.assertEqual(result.keys, ["key"])
        self.assertEqual(result.block_ids, [2])
        self.assertEqual(result.all_buffers, [[3600]])
        self.assertEqual(result.all_sizes, [[30]])
        self.assertEqual(result.all_offsets, [[30]])

    def test_range_limits_split_rows_and_large_segments(self):
        batches = KVTransferThread._range_transfer_batches(
            ["k0", "k1"],
            [[100], [200]],
            [[25], [5]],
            [[1000], [2000]],
            max_transfer_blocks=1,
            max_transfer_bytes=10,
        )

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0], (["k0"], [[100, 110, 120]], [[10, 10, 5]], [[1000, 1010, 1020]]))
        self.assertEqual(batches[1], (["k1"], [[200]], [[5]], [[2000]]))


class TestMooncakeLayerSaveSession(unittest.TestCase):
    def test_final_layer_commits_after_all_ranges(self):
        store = MagicMock()
        # Range APIs may return the positive number of bytes moved on success.
        store.batch_copy_put.return_value = [30]
        store.batch_commit.return_value = [0]
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("key", 0)])
        save_finished = [threading.Event(), threading.Event()]
        builder = LayerBatchBuilder(make_token_database(), page_size_bytes=60, num_layers=2)
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=make_token_database(),
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            page_size_bytes=60,
            ready_event=threading.Event(),
            num_layers=2,
            layer_save_finished_events=save_finished,
            sync_save_events=[MagicMock(), MagicMock()],
            group_builders=[builder],
            put_started_keys={"key"},
            session_tracker=tracker,
        )
        request = ReqMeta("r1", block_ids=[2], block_hashes=[], is_last_chunk=True)
        request.save_block_keys = ["key"]

        for layer_id in range(2):
            task = LayerTransferTask(
                layer_id=layer_id,
                layer_idx_in_group=layer_id,
                block_ranges=[LayerBlockRange(request, 0, 1)],
                shared_block_data=builder.build_shared(
                    LayerTransferTask(
                        layer_id=layer_id,
                        block_ranges=[LayerBlockRange(request, 0, 1)],
                        use_key_major_ranges=True,
                    )
                ),
                use_key_major_ranges=True,
            )
            thread.add_stored_request("r1")
            thread.request_queue.put([task])
            thread._handle_request([task])

        self.assertEqual(store.batch_copy_put.call_count, 2)
        store.batch_commit.assert_called_once_with(["key"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("key", 0)])


class TestMooncakeWorkerSessionPreparation(unittest.TestCase):
    @staticmethod
    def _make_worker() -> KVPoolWorker:
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.kv_role = "kv_producer"
        worker.consumer_is_to_put = False
        worker.tp_rank = 0
        worker.put_step = 1
        worker.block_size = 16
        worker.grouped_block_size = [16]
        worker.hash_block_size = 16
        worker.model_name = "model"
        worker.head_or_tp_rank = 0
        worker.backend_name = "mooncake"
        worker.use_block_key_layerwise = True
        worker.layerwise_offload = False
        worker.independent_layers = []
        worker.page_size_bytes = 60
        worker.group_block_len = {0: [10, 20, 30]}
        worker.layerwise_max_transfer_blocks = 0
        worker.use_eagle = False
        worker._put_started_keys = set()
        worker._put_started_keys_lock = threading.Lock()
        worker._mooncake_session_tracker = MooncakeSessionTracker()
        worker.m_store = MagicMock()
        return worker

    def test_put_start_uses_full_current_layout_size_and_skips_hits(self):
        worker = self._make_worker()
        worker.m_store.batch_put_start.return_value = [0]
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            save_start_token=0,
            save_end_token=32,
            block_ids=[1, 2],
            block_hashes=[b"h0", b"h1"],
            can_save=True,
            load_spec=LoadSpec(0, 16, can_load=True),
        )

        worker._prepare_mooncake_put_session(request)

        worker.m_store.batch_put_start.assert_called_once_with(["model@6831@0"], [60])
        self.assertEqual(request.save_key_block_offset, 1)
        self.assertEqual(request.save_block_keys, ["model@6831@0"])

    def test_full_remote_hit_loads_last_block_even_when_vllm_keeps_one_token(self):
        worker = self._make_worker()
        request = ReqMeta(
            "r1",
            block_ids=[1, 2, 3, 4],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],
            load_spec=LoadSpec(0, 63, can_load=True, kvpool_store_skip_tokens=64),
        )

        slots = worker._prepare_mooncake_get_session(request)

        self.assertEqual(len(slots), 4)
        self.assertEqual(request.load_block_keys[-1], "model@6833@0")
        self.assertIsNone(request.load_last_block_key)

    def test_next_chunk_reloads_committed_prefix_without_new_load_spec(self):
        worker = self._make_worker()
        worker._mooncake_session_tracker.register_put_keys(
            "r1",
            [("model@6830@0", 0)],
        )
        worker._mooncake_session_tracker.commit_put_keys(["model@6830@0"])
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            block_ids=[10, 11],
            block_hashes=[b"h0", b"h1"],
            load_spec=None,
            is_last_chunk=False,
        )

        slots = worker._prepare_mooncake_get_session(request)
        worker.layer_load_tasks = [[]]
        worker._process_load_for_layer_batch([request], 0)

        self.assertEqual(slots, [("model@6830@0", 10, 0)])
        self.assertEqual(request.load_block_keys, ["model@6830@0"])
        self.assertEqual(len(worker.layer_load_tasks[0]), 1)
        block_range = worker.layer_load_tasks[0][0].block_ranges[0]
        self.assertEqual((block_range.start_block, block_range.end_block), (0, 1))

    def test_hashless_boundary_key_uses_the_matching_block_slot(self):
        worker = self._make_worker()
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            block_ids=[10, 11],
            block_hashes=[b"h0"],
            load_spec=LoadSpec(0, 32, can_load=True),
        )

        slots = worker._prepare_mooncake_get_session(request)

        self.assertEqual(
            request.load_block_keys,
            ["model@6830@0", "model@r1_lastblock@0"],
        )
        self.assertIsNone(request.load_last_block_key)
        self.assertEqual(slots[-1], ("model@r1_lastblock@0", 11, 1))


class TestMooncakeSessionTracker(unittest.TestCase):
    def test_commit_promotes_shared_put_key_to_every_request_owner(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("shared", 0)])
        tracker.register_put_keys("r2", [("shared", 1)])

        tracker.commit_put_keys(["shared"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [("shared", 0)])
        self.assertEqual(tracker.prepare_load_entries("r2", []), [("shared", 1)])

    def test_complete_key_replaces_partial_key_for_the_same_block(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("partial", 1)])
        tracker.commit_put_keys(["partial"])
        tracker.register_put_keys("r1", [("complete", 1)])
        tracker.commit_put_keys(["complete"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [("complete", 1)])

    def test_shared_get_ends_only_after_the_last_owner_releases_it(self):
        tracker = MooncakeSessionTracker()
        tracker.prepare_load_entries("r1", [("shared", 0)])
        tracker.prepare_load_entries("r2", [("shared", 0)])
        tracker.record_get_result("shared", {"r1", "r2"}, succeeded=True)

        self.assertEqual(tracker.release_terminal({"r1"}), [])
        self.assertEqual(tracker.release_terminal({"r2"}), ["shared"])
        self.assertEqual(tracker.release_terminal({"r2"}), [])

    def test_failed_renewal_retains_desired_keys_for_retry(self):
        tracker = MooncakeSessionTracker()
        tracker.prepare_load_entries("r1", [("shared", 0)])
        tracker.register_put_keys("r1", [("pending", 1)])
        tracker.record_get_result("shared", {"r1"}, succeeded=True)

        tracker.record_get_result("shared", {"r1"}, succeeded=False)
        tracker.commit_put_keys(["pending"])

        self.assertEqual(tracker.release_for_retry({"r1"}), [])
        self.assertEqual(
            tracker.prepare_load_entries("r1", []),
            [("shared", 0), ("pending", 1)],
        )

    def test_failed_get_attempt_preserves_unrelated_shared_owner(self):
        tracker = MooncakeSessionTracker()
        tracker.prepare_load_entries("old-owner", [("shared", 0)])
        tracker.prepare_load_entries(
            "new-owner",
            [("shared", 0), ("new-key", 1)],
        )
        tracker.record_get_result(
            "shared",
            {"old-owner", "new-owner"},
            succeeded=True,
        )

        keys_to_end = tracker.release_failed_get_attempts(
            {
                "shared": {"new-owner"},
                "new-key": {"new-owner"},
            }
        )

        self.assertEqual(keys_to_end, ["new-key"])
        self.assertEqual(tracker.release_terminal({"old-owner"}), ["shared"])
        self.assertEqual(
            tracker.prepare_load_entries("new-owner", []),
            [("shared", 0), ("new-key", 1)],
        )

    def test_terminal_request_loses_pending_put_ownership(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("pending", 0)])

        tracker.release_terminal({"r1"})
        tracker.commit_put_keys(["pending"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [])

    def test_chunk_commit_retry_and_terminal_cleanup(self):
        tracker = MooncakeSessionTracker()
        tracker.register_put_keys("r1", [("k0", 0)])
        tracker.commit_put_keys(["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("k0", 0)])

        tracker.record_get_result("k0", ["r1"], succeeded=True)
        self.assertEqual(tracker.release_for_retry({"r1"}), ["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("k0", 0)])

        tracker.record_get_result("k0", ["r1"], succeeded=True)
        self.assertEqual(tracker.release_terminal({"r1"}), ["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [])


class TestMooncakeLayerwiseGroupKeys(unittest.TestCase):
    """M2a foundation: group-qualified Mooncake layerwise block keys.

    The default group must keep the legacy ``model@hash@rank`` key so existing
    stored objects stay readable; multi-group layouts (DeepSeek-V4-Flash)
    qualify the key with group/cache_role/cache_family.
    """

    def test_legacy_default_key_unchanged(self):
        key = make_layerwise_block_key("model", "abc123", 3)
        self.assertEqual(key, "model@abc123@3")
        self.assertEqual(
            parse_layerwise_block_key(key),
            LayerwiseBlockKey("model", 0, "kv", "default", "abc123", 3),
        )

    def test_group_qualified_key_roundtrip(self):
        key = make_layerwise_block_key(
            "dsv4",
            "feed00",
            1,
            kv_cache_group_id=2,
            cache_role="state",
            cache_family="c4",
        )
        self.assertEqual(key, "dsv4@group:2@cache_role:state@cache_family:c4@feed00@1")
        self.assertEqual(
            parse_layerwise_block_key(key),
            LayerwiseBlockKey("dsv4", 2, "state", "c4", "feed00", 1),
        )

    def test_group_qualified_keys_do_not_collide(self):
        keys = {
            make_layerwise_block_key(
                "model",
                "hash",
                rank,
                kv_cache_group_id=group_id,
                cache_role=cache_role,
                cache_family=cache_family,
            )
            for group_id in (0, 1, 2)
            for cache_role in ("kv", "state")
            for cache_family in ("default", "c4", "c128")
            for rank in (0, 1)
            if not (group_id == 0 and cache_role == "kv" and cache_family == "default")
        }
        legacy_key = make_layerwise_block_key("model", "hash", 0)
        self.assertNotIn(legacy_key, keys)
        # The default (group 0, kv, default) combination intentionally keeps the
        # legacy key format, so it contributes 2 of the 36 rank-key variants.
        self.assertEqual(len(keys), 3 * 2 * 3 * 2 - 2)

    def test_group_qualified_lastblock_key_roundtrip(self):
        key = make_layerwise_block_key(
            "dsv4",
            "req-1_lastblock",
            0,
            kv_cache_group_id=1,
            cache_role="kv",
            cache_family="default",
        )
        parsed = parse_layerwise_block_key(key)
        self.assertEqual(parsed.chunk_hash, "req-1_lastblock")
        self.assertEqual(parsed.kv_cache_group_id, 1)

    def test_group_object_size_bytes(self):
        worker = object.__new__(KVPoolWorker)
        worker.group_block_len = {0: [10, 20], 1: [8]}
        worker.page_size_bytes = 64
        worker.num_kv_cache_groups = 2
        self.assertEqual(worker._mooncake_object_size_bytes(0), 30)
        self.assertEqual(worker._mooncake_object_size_bytes(1), 8)
        self.assertEqual(worker._mooncake_object_size_per_group(), [30, 8])


class TestMooncakeMultiGroupLayerwise(unittest.TestCase):
    """M2a multi-group wiring: per-group keys, sessions and commits."""

    @staticmethod
    def _make_multi_group_worker() -> KVPoolWorker:
        worker = TestMooncakeWorkerSessionPreparation._make_worker()
        worker.num_kv_cache_groups = 2
        worker.grouped_block_size = [16, 16]
        worker.kv_cache_group_families = ["default", "c4"]
        worker.group_block_len = {0: [10, 20, 30], 1: [4, 4]}
        return worker

    def test_request_block_keys_select_by_group(self):
        request = ReqMeta("r1", block_ids=[])
        request.save_block_keys_by_group = [["a0"], ["b0"]]
        request.save_key_block_offset_by_group = [0, 0]
        self.assertEqual(
            LayerBatchBuilder._request_block_keys(request, is_save=True, group_id=1),
            (["b0"], 0, None),
        )
        legacy = ReqMeta("r2", block_ids=[])
        legacy.save_block_keys = ["x"]
        legacy.save_key_block_offset = 2
        self.assertEqual(
            LayerBatchBuilder._request_block_keys(legacy, is_save=True, group_id=0),
            (["x"], 2, None),
        )

    def test_multi_group_put_session_uses_group_qualified_keys(self):
        worker = self._make_multi_group_worker()
        worker.m_store.batch_put_start.return_value = [0]
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            save_start_token=0,
            save_end_token=32,
            block_ids=[1, 2],
            block_hashes=[b"h0"],
            can_save=True,
        )

        worker._prepare_mooncake_multi_group_put_session(request)

        calls = worker.m_store.batch_put_start.call_args_list
        self.assertEqual(len(calls), 2)
        group0_keys = calls[0].args[0]
        group1_keys = calls[1].args[0]
        self.assertTrue(all("@group:" not in key for key in group0_keys))
        self.assertTrue(all("@group:1@cache_role:kv@cache_family:c4@" in key for key in group1_keys))
        self.assertEqual(calls[0].args[1], [60] * len(group0_keys))
        self.assertEqual(calls[1].args[1], [8] * len(group1_keys))

    def test_multi_group_get_session_restores_every_group(self):
        worker = self._make_multi_group_worker()
        group0_key = make_layerwise_block_key("model", "6830", 0)
        group1_key = make_layerwise_block_key(
            "model",
            "6830",
            0,
            kv_cache_group_id=1,
            cache_role="kv",
            cache_family="c4",
        )
        worker._mooncake_session_tracker.register_put_keys("r1", [(group0_key, 0), (group1_key, 0)])
        worker._mooncake_session_tracker.commit_put_keys([group0_key, group1_key])
        request = ReqMeta(
            "r1",
            token_len_chunk=16,
            block_ids_by_group=[[10], [20]],
            block_hashes=[b"h0"],
            load_spec=LoadSpec(0, 16, can_load=True),
        )
        worker.m_store.batch_get_start.return_value = [0, 0]

        slots = worker._prepare_mooncake_multi_group_get_session(request)
        worker._open_mooncake_multi_group_get_sessions(slots)

        self.assertEqual(request.load_block_keys_by_group[0], [group0_key])
        self.assertEqual(request.load_block_keys_by_group[1], [group1_key])
        self.assertEqual(sorted(request.load_keys), sorted([group0_key, group1_key]))

    def test_multi_group_save_commits_each_group_at_its_own_final_layer(self):
        store = MagicMock()
        store.batch_copy_put.return_value = [10]
        store.batch_commit.return_value = [0]
        save_finished = [threading.Event(), threading.Event()]
        sync_events = [MagicMock(), MagicMock()]
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=make_token_database(),
            block_size=[16, 16],
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            page_size_bytes=60,
            ready_event=threading.Event(),
            num_layers=2,
            layer_save_finished_events=save_finished,
            sync_save_events=sync_events,
            group_builders=[MagicMock(), MagicMock()],
            group_final_layer_ids=[0, 1],
        )

        def make_meta(group_id: int, layer_id: int):
            return LayerRangeReqMeta(
                req_ids=["r1"],
                layer_id=layer_id,
                block_ids=[1],
                keys=[f"k{group_id}"],
                all_buffers=[[1000]],
                all_sizes=[[10]],
                all_offsets=[[0]],
            )

        builders = thread.group_builders
        assert builders is not None
        builders[0].build_addrs.side_effect = lambda shared, layer: make_meta(0, layer)
        builders[1].build_addrs.side_effect = lambda shared, layer: make_meta(1, layer)
        thread.add_stored_request("r1")
        thread.add_stored_request("r1")
        thread.add_stored_request("r1")
        thread.add_stored_request("r1")

        for layer_id in range(2):
            tasks = []
            for group_id in range(2):
                shared = SharedBlockData(
                    block_ids_arr=np.asarray([1], dtype=np.int64),
                    block_gvas_arr=None,
                    req_ids=["r1"],
                    is_last_chunks=[True],
                    block_keys=[f"k{group_id}"],
                )
                tasks.append(
                    LayerTransferTask(
                        layer_id=layer_id,
                        block_ranges=[LayerBlockRange(request=None, start_block=0, end_block=1)],
                        shared_block_data=shared,
                        group_id=group_id,
                        layer_idx_in_group=layer_id,
                        use_key_major_ranges=True,
                    )
                )
            # Mirror the run loop: task_done() inside _handle_request must pair
            # with one get() per queued batch.
            thread.request_queue.put(tasks)
            thread.request_queue.get()
            thread._handle_request(tasks)

        store.batch_commit.assert_has_calls(
            [
                call(["k0"]),
                call(["k1"]),
            ]
        )
        self.assertIn("r1", thread.finished_requests)


if __name__ == "__main__":
    unittest.main()
