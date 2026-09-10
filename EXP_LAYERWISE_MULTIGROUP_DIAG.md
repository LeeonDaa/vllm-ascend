# 实验：多组 Mooncake Layerwise 每层 get 归因诊断（仅实验分支）

> 分支：`exp/mooncake-layerwise-multigroup-diag-0271`（基 `aeb4ca8c6`）。
> **仅实验用，不合并**；诊断默认关闭，关闭时行为与 `range-0271` 完全一致。

## 目的

把 profiler 里每层的 `hixlOpBatchRead`（CPU 侧算子下发）/ `HixlBatchGet`（HIXL 实际行为）
逐项归因到代码：该物理层属于哪些 group、各组命中 key 数、每 key 的 slice/entry 数、
fragment 数、`_range_transfer_batches` 切批数，以及 `submit → issue` 的时序。用于判定
“是下发太晚（锚点），还是单层展开数太多（transport/fragment）”。

## 运行配方

P 节点环境变量：
```bash
export VLLM_ASCEND_KVPOOL_LAYER_DIAG=1     # 新增：打印逐层归因
export VLLM_ASCEND_KVPOOL_RANGE_DEBUG=0    # 保持 0，避免逐层 range 噪音
# 可选：请求期想看到 Mooncake client 侧（client_service/transfer_task/ascend_*）日志
export GLOG_v=2
# export GLOG_logtostderr=1
```

`kv_connector_extra_config`（AscendStoreConnector 段）：
```json
{
  "lookup_rpc_port": "0",
  "backend": "mooncake",
  "use_layerwise": true,
  "layerwise_prefetch_layers": 1,
  "layerwise_anchor": "attention"
}
```

> A/B：`layerwise_anchor` ∈ {`attention`(默认), `immediate`}；`layerwise_prefetch_layers` ∈ {1,2,4}。

## 日志形态

```
(Worker_TPx_EPx ...) INFO ... [KVPOOL_LAYER_DIAG] {"event":"load","ts":<perf>, "layer_id":2,
  "groups":[0,2,4],"keys_per_group":[32,128,2048],"keys_total":2208,"rows_per_group":[...],
  "slices_per_row":[4,1,2],"fragments":4352,"batches":1,"stagger_ms":0.01,"prep_ms":0.3,"req_ids":[...]}
(Worker_TPx_EPx ...) INFO ... [KVPOOL_LAYER_DIAG] {"event":"load_copy","layer_id":2,"copy_ms":19.7,
  "batch_ms":[19.7]}                     # 逐批 batch_copy_get 耗时（每层 1 批时=该层拷贝耗时）
(Worker_TPx_EPx ...) INFO ... [KVPOOL_LAYER_DIAG] {"event":"timing","layer_id":2,"gated":true,
  "waited_for_save":null,"submit_to_issue_ms":X,"issue_to_done_ms":Y}
(Worker_TPx_EPx ...) INFO ... [KVPOOL_LAYER_DIAG] {"event":"save_detail","layer_id":2,"keys_total":K,
  "sync_ms":S,"put_ms":P,"batch_ms":[...]}   # 保存侧：等计算事件 + batch_copy_put 耗时
```

字段含义：`keys_total` = 该层所有组命中 key 数（= HIXL 期望的 get 次数）；`fragments` = Σ(slice)；
`batches` = 切批数（`layerwise_max_transfer_blocks/bytes` 生效时 > 1）；`stagger_ms/prep_ms` = H2D 节流与
Python 组装耗时；`copy_ms`/`batch_ms` = 实际 `batch_copy_get` 耗时；`submit_to_issue_ms` = 提交→准备拷贝
（含等 gate 与 recv 队列排队），`issue_to_done_ms` = stagger+prep+copy。`load.ts` 与 `save.ts` 是同一进程的
`perf_counter`，可判断 **save 的 put 是否与 load 的 get 时间重叠**（同一 Mooncake client/transport 会互堵）。

## “代码计数 ↔ profile 计数”对照表模板

| layer | groups | keys_per_group | keys_total | slices_per_row | fragments | batches | submit_to_issue_ms | issue_to_done_ms | profiler: hixlOpBatchRead | profiler: HixlBatchGet | profiler: dsa_forward(ms) |
| :-- | :-- | :-- | --: | :-- | --: | --: | --: | --: | --: | --: | --: |
| 0 | | | | | | | | | | | |
| 1 | | | | | | | | | | | |
| 2 | | | | | | | | | | | |
| … | | | | | | | | | | | |

判读：
- `HixlBatchGet ≈ fragments`（每 fragment 一次 HIXL get）；`hixlOpBatchRead` = 不同远端 transport 数（CPU 下发）。
- 两族层（属 `{0,2,4}` vs `{1,3,5}`）的 `keys_total/fragments` 差异 → 解释 11/32 交叉。
- `submit_to_issue_ms` 大且落在 `dsa_forward` 中后段 → 锚点太晚；对比 `layerwise_anchor=immediate`。

## A/B 记录模板

| 配置 (anchor, prefetch) | dsa_forward 均值/最大 (ms) | 每层 hixlOpBatchRead/HixlBatchGet | 下发相对 dsa_forward 位置 | 备注 |
| :-- | :-- | :-- | :-- | :-- |
| attention, 1 | | | | 基线 |
| immediate, 1 | | | | 验证“提前下发” |
| attention, 2 | | | | 验证预取窗口 |
| immediate, 2 | | | | |
| attention, 4 | | | | |
