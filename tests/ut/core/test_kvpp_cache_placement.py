# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.layers.attention import MLAAttention
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

from tests.ut.kvpp_utils import indexer_name, layer_name, make_kvpp_config, make_kvpp_specs
from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.core import kv_cache_placement as placement
from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    AscendSlidingWindowMLASpec,
)


@pytest.mark.parametrize("tp,pcp,expected", [(2, 2, 4), (4, 2, 8)])
def test_kvpp_size_covers_pcp_replicas(tp, pcp, expected):
    config = make_kvpp_config(tp)
    config.parallel_config.prefill_context_parallel_size = pcp
    config.parallel_config.pipeline_parallel_size = 2
    config.parallel_config.data_parallel_size = 2
    assert KVPPConfig.from_vllm_config(config).size == expected


@pytest.mark.parametrize("tp,owners", [(3, [0, 0, 0, 1, 1, 1, 2, 2]), (10, list(range(8)))])
@pytest.mark.parametrize("with_mtp", [False, True])
def test_pp_local_owners_and_bundles(tp, owners, with_mtp):
    specs = make_kvpp_specs()
    if not with_mtp:
        del specs[layer_name(17)]
    config = make_kvpp_config(tp)
    plan = placement.create_kvpp_cache_allocation_plan(config, specs, kvpp_rank=1)
    reverse = placement.create_kvpp_cache_allocation_plan(config, dict(reversed(list(specs.items()))), kvpp_rank=1)
    expected = {layer_name(i): owner for i, owner in zip(range(9, 17), owners)}
    expected[indexer_name(11)] = owners[2]
    assert plan.layer_owner_ranks == expected
    assert list(plan.layer_owner_ranks.items()) == list(reverse.layer_owner_ranks.items())
    assert list(plan.layer_bundles.items()) == list(reverse.layer_bundles.items())
    assert list(plan.layer_bundles) == [layer_name(i) for i in range(9, 18 if with_mtp else 17)]
    assert plan.layer_bundles[layer_name(11)] == (layer_name(11), indexer_name(11))
    assert plan.logical_cache_spec == reverse.logical_cache_spec == specs
    assert plan.tensor_sizes == reverse.tensor_sizes
    assert layer_name(17) not in plan.layer_owner_ranks


@pytest.mark.parametrize(
    "packed,scale_dtype,expected_sizes,expected_layout,total",
    [
        (False, torch.float16, ((32, 16), (8, 4)), (((0, 96), (96, 48)), ((144, 24), (168, 12))), 180),
        (True, torch.float16, ((32,), (8, 4)), (((0, 96),), ((96, 24), (120, 12))), 132),
        (True, torch.float32, ((32,), (8, 8)), (((0, 96),), ((96, 24), (120, 24))), 144),
    ],
)
def test_component_sizes_and_layout(monkeypatch, packed, scale_dtype, expected_sizes, expected_layout, total):
    main, indexer = layer_name(11), indexer_name(11)
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_lora_rank, layer.qk_rope_head_dim = 8, 4
    config = make_kvpp_config()
    specs = {
        main: AscendMLAAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=16 if packed else 12,
            dtype=torch.int8 if packed else torch.bfloat16,
            cache_sparse_sfa_c8=packed,
        ),
        indexer: AscendSFAIndexerCacheSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.int8,
            scale_dim=1,
            scale_dtype=scale_dtype,
            cache_sparse_li_c8=True,
        ),
    }
    monkeypatch.setattr(placement, "get_layers_from_vllm_config", lambda *_args: {main: layer})
    monkeypatch.setattr(placement, "enable_sfa", lambda _: True)
    sizes = placement.build_kvpp_buffer_sizes(config, specs)
    assert sizes == {main: expected_sizes[0], indexer: expected_sizes[1]}
    layout, size = placement.build_kvpp_layer_layout((main, indexer), sizes, 3)
    assert layout == {main: expected_layout[0], indexer: expected_layout[1]}
    assert size == total


def test_unquantized_indexer_and_quantized_mla_sizes(monkeypatch):
    main, indexer = layer_name(11), indexer_name(11)
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_lora_rank, layer.qk_rope_head_dim = 8, 4
    config = make_kvpp_config()
    config.quant_config = Mock()
    config.quant_config.get_kv_quant_split_factor.return_value = (2, 2)
    specs = {
        main: AscendMLAAttentionSpec(block_size=2, num_kv_heads=1, head_size=16, dtype=torch.int8),
        indexer: replace(make_kvpp_specs()[indexer], dtype=torch.bfloat16, scale_dim=0, cache_sparse_li_c8=False),
    }
    monkeypatch.setattr(placement, "get_layers_from_vllm_config", lambda *_args: {main: layer})
    monkeypatch.setattr(placement, "enable_sfa", lambda _: False)
    monkeypatch.setattr(placement, "enable_fa_quant", lambda _: True)
    assert placement.build_kvpp_buffer_sizes(config, specs) == {main: (16, 16), indexer: (16,)}
    config.quant_config.get_kv_quant_split_factor.assert_called_once_with(main, [8, 4])


def test_multigroup_block_sizes_are_supported():
    """Hybrid layouts mix block sizes; each cache is still sized per layout."""
    specs = make_kvpp_specs()
    wide = layer_name(9)
    specs[wide] = replace(specs[wide], block_size=4)

    plan = placement.create_kvpp_cache_allocation_plan(make_kvpp_config(2), specs, kvpp_rank=0)

    assert plan.tensor_sizes[wide] == placement.build_kvpp_buffer_sizes(make_kvpp_config(2), {wide: specs[wide]})[wide]
    assert plan.layer_owner_ranks[wide] == 0


def test_dsv4_hybrid_specs_use_one_page_per_cache():
    """DeepSeek-V4 pages stay whole; the runner splits them per view."""
    swa_layers, indexer, state = layer_name(1), indexer_name(1), "model.layers.1.self_attn.state"
    swa = AscendSlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=192,
        dtype=torch.float8_e4m3fn,
        sliding_window=128,
        model_version="deepseek_v4",
    )
    # The compressor state spec carries no model_version, only packed pages.
    compressor_state = AscendSlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float8_e4m3fn,
        sliding_window=32,
        page_size_padded=64 * 64,
    )
    compressed = AscendMLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        scale_dim=1,
        scale_dtype=torch.float16,
        model_version="deepseek_v4",
    )
    specs = {swa_layers: swa, state: compressor_state, indexer: compressed}
    config = make_kvpp_config(2)
    config.model_config.hf_config.compress_ratios = [1, 4, 128]

    assert placement.build_kvpp_buffer_sizes(config, specs) == {
        swa_layers: (swa.page_size_bytes,),
        state: (compressor_state.page_size_bytes,),
        indexer: (compressed.page_size_bytes,),
    }

    plan = placement.create_kvpp_cache_allocation_plan(config, specs, kvpp_rank=0)
    # Every cache of one layer belongs to the same owner and shares one bundle.
    assert plan.layer_bundles == {swa_layers: (swa_layers, indexer, state)}
    assert plan.layer_owner_ranks == {swa_layers: 0, state: 0, indexer: 0}
    # The runner takes these pages flat instead of a per-component tuple.
    assert placement.is_flat_cache_spec(config, swa)
    assert placement.is_flat_cache_spec(config, compressor_state)
    assert placement.is_flat_cache_spec(config, compressed)
    _, size = placement.build_kvpp_layer_layout((swa_layers, indexer, state), plan.tensor_sizes, num_blocks=8)
    assert size == 8 * (swa.page_size_bytes + compressor_state.page_size_bytes + compressed.page_size_bytes)


def test_models_without_compression_keep_per_component_caches():
    """Non-compressed models keep their previous sizing and tuple layout."""
    spec = AscendSlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=192,
        dtype=torch.float8_e4m3fn,
        sliding_window=128,
    )
    config = make_kvpp_config(2)

    assert not placement.is_flat_cache_spec(config, spec)
    with pytest.raises(ValueError, match="cannot size cache"):
        placement.build_kvpp_buffer_sizes(config, {layer_name(1): spec})


def test_request_owned_tail_caches_are_rejected():
    """The GLM-5-Next tail ring cannot be replaced by another rank's copy."""
    tail = AscendIndexerKPoolTailSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float32,
        sliding_window=4,
        compress_ratio=4,
    )

    with pytest.raises(ValueError, match="tail caches"):
        placement.create_kvpp_cache_allocation_plan(
            make_kvpp_config(2), {"model.layers.1.indexer.tail": tail}, kvpp_rank=0
        )


@pytest.mark.parametrize("tp,rank,cost", [(3, 0, 404), (3, 1, 392), (3, 2, 328), (10, 9, 248)])
def test_physical_cost_per_rank(tp, rank, cost):
    plan = placement.create_kvpp_cache_allocation_plan(make_kvpp_config(tp), make_kvpp_specs(), rank)
    assert plan.get_num_blocks(cost - 1) == 0
    assert plan.get_num_blocks(cost) == 1


@pytest.mark.parametrize("available,blocks", [(0, 0), (391, 0), (392, 1), (1175, 2), (1176, 3)])
def test_budget_floors_complete_blocks(available, blocks):
    plan = placement.create_kvpp_cache_allocation_plan(make_kvpp_config(), make_kvpp_specs(), 1)
    assert plan.get_num_blocks(available) == blocks


@pytest.mark.parametrize("with_mtp,expected", [(True, 3), (False, 0)])
def test_stage_without_target_has_no_scratch_cost(with_mtp, expected):
    specs = make_kvpp_specs()
    specs = {layer_name(17): specs[layer_name(17)]} if with_mtp else {}
    plan = placement.create_kvpp_cache_allocation_plan(make_kvpp_config(), specs, 1)
    assert plan.get_num_blocks(288) == expected


# Six cache groups per layer, mirroring DeepSeek-V4-Flash: 42 cache names over
# 21 physical layers in some groups, 20 over 20 in others.
DSV4_GROUP_KINDS = ("main", "c4", "c128", "idx", "idx_scale", "swa")


def _dsv4_multigroup_specs(n_layers=44):
    def mla(head_size, layers, kind):
        return {
            f"model.layers.{i}.{kind}": AscendMLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=head_size,
                dtype=torch.float8_e4m3fn,
                model_version="deepseek_v4",
            )
            for i in range(layers)
        }

    specs = mla(576, n_layers, "main")
    specs |= mla(144, 21, "c4") | mla(5, 21, "c128")
    specs |= {
        f"model.layers.{i}.{kind}": AscendSFAIndexerCacheSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=32,
            dtype=torch.float8_e4m3fn,
            scale_dim=1,
            scale_dtype=torch.float16,
            cache_sparse_li_c8=True,
            model_version="deepseek_v4",
        )
        for i in range(21)
        for kind in ("idx", "idx_scale")
    }
    specs |= {
        f"model.layers.{i}.swa": AscendSlidingWindowMLASpec(
            block_size=128,
            num_kv_heads=1,
            head_size=36,
            dtype=torch.float8_e4m3fn,
            sliding_window=128,
            model_version="deepseek_v4",
        )
        for i in range(20)
    }
    return specs


def _dsv4_groups(specs):
    by_kind = {kind: {} for kind in DSV4_GROUP_KINDS}
    for name, spec in specs.items():
        by_kind[name.rsplit(".", 1)[-1]][name] = spec
    return [
        KVCacheGroupSpec(
            layer_names=list(by_kind[kind]),
            kv_cache_spec=UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=by_kind[kind]),
        )
        for kind in DSV4_GROUP_KINDS
    ]


def _kvpp_per_block(plan):
    """Bytes one block costs this rank, as allocate_kvpp_cache lays it out."""
    persistent = 0
    scratch = 0
    for name, bundle in plan.layer_bundles.items():
        _, size = placement.build_kvpp_layer_layout(bundle, plan.tensor_sizes, 1)
        owner = plan.layer_owner_ranks.get(name)
        if owner is None or owner == plan.kvpp_rank:
            persistent += size
        if owner is not None:
            scratch = max(scratch, size)
    return persistent + placement.KVPP_SCRATCH_BUFFER_COUNT * scratch


@pytest.mark.parametrize("available_gib", [10, 32])
def test_multigroup_budget_fits_the_per_layer_bundles(monkeypatch, available_gib):
    """The engine must derive back exactly the block count KVPP budgeted.

    The engine divides the advertised budget by its own bytes-per-block, which
    for a shared-tuple layout is far below the sum of every cache's page size.
    Advertising that sum inflated the block count and overflowed NPU memory on
    DeepSeek-V4, whose six groups per layer made the two units diverge.
    """
    specs = _dsv4_multigroup_specs()
    config = make_kvpp_config(tp=4)
    config.model_config.hf_config.compress_ratios = [1, 4, 128]
    plan = placement.create_kvpp_cache_allocation_plan(config, specs, kvpp_rank=0)
    groups = _dsv4_groups(specs)
    monkeypatch.setattr(placement, "get_kv_cache_groups", lambda *_args: groups)

    # Anchor the unit from the engine's own planner, not from the function under
    # test: for a shared-tuple layout it is deliberately below the page-size sum,
    # and that gap is exactly what overflowed device memory.
    from vllm_ascend.patch.platform.patch_kv_cache_utils import _ascend_pool_bytes_per_block

    bytes_per_block = _ascend_pool_bytes_per_block(groups)
    assert bytes_per_block < sum(spec.page_size_bytes for spec in specs.values())

    available = available_gib * 1024**3
    advertised = placement.kvpp_memory_budget(config, plan, available)

    assert advertised // bytes_per_block == plan.get_num_blocks(available)
    assert (advertised // bytes_per_block) * _kvpp_per_block(plan) <= available


def test_single_group_budget_keeps_the_page_sum_unit(monkeypatch):
    """A single group has one bytes-per-block unit, so nothing changes."""
    specs = make_kvpp_specs()
    config = make_kvpp_config()
    plan = placement.create_kvpp_cache_allocation_plan(config, specs, kvpp_rank=1)
    groups = [
        KVCacheGroupSpec(
            layer_names=list(specs),
            kv_cache_spec=UniformTypeKVCacheSpecs(block_size=2, kv_cache_specs=specs),
        )
    ]
    monkeypatch.setattr(placement, "get_kv_cache_groups", lambda *_args: groups)

    page_sum = sum(spec.page_size_bytes for spec in specs.values())
    assert placement.engine_bytes_per_block(config, specs) == page_sum
    assert placement.kvpp_memory_budget(config, plan, 1176) == 3 * page_sum
