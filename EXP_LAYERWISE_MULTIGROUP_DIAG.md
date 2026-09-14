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
(Worker_TPx_EPx ...) INFO ... [KVPOOL_LAYER_DIAG] {"event":"timing","layer_id":2,"gated":false,
  "waited_for_save":null,"submit_ts":T0,"issue_ts":T1,"done_ts":T2,"submit_to_issue_ms":X,"issue_to_done_ms":Y}
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
- `submit_to_issue_ms` 大且落在 `dsa_forward` 中后段 → 锚点太晚；Mooncake 现在默认
  `layerwise_anchor=immediate`（提交即放行，最大化与上一层计算的重叠），可显式设回 `attention` 做 A/B。
- **重叠度算法**：`submit_ts` = 第 i 层入口提交 i+1 预取的时刻，`issue_ts` = 传输线程真正开始搬的时刻，
  `done_ts` = 该层拷贝结束时刻。相邻两层 `submit_ts` 之差即该层可用计算窗口；把
  `issue_ts - submit_ts`（等待）与 `done_ts - issue_ts`（本体）叠加到该窗口上，即可算出
  "被隐藏/暴露"的比例——`issue_ts` 越接近上一个 `submit_ts`，说明传输等待越小。
- 读会话的关闭（`batch_get_end`）已从计算线程挪到传输线程，因此 `timing` 的等待里不应再出现
  计算线程发起的后端调用；若 `submit_to_issue_ms` 仍大，先看锁等待与 `_load_session_lock`。

## A/B 记录模板

| 配置 (anchor, prefetch) | dsa_forward 均值/最大 (ms) | 每层 hixlOpBatchRead/HixlBatchGet | 下发相对 dsa_forward 位置 | 备注 |
| :-- | :-- | :-- | :-- | :-- |
| attention, 1 | | | | 基线 |
| immediate, 1 | | | | 验证“提前下发” |
| attention, 2 | | | | 验证预取窗口 |
| immediate, 2 | | | | |
| attention, 4 | | | | |

## save 端“重写已存在对象”诊断与修法

- 新增 diag 事件 `[KVPOOL_LAYER_DIAG] {"event":"put_session", ...}`：每组打印
  `save_start_token / save_end_token / start_block / end_block / has_load_spec /
  skipped_pooled_prefix / pool_hit_tokens / store_skip_tokens / keys_total / new_keys /
  newly_started / already_existing / put_start_ms`。
  `already_existing > 0` 即“对已存在对象发起了 put_session_start”；`put_start_ms` 是该 RPC 耗时。
- **修法 A（默认生效）**：scheduler 的 `hits_per_group`（`{group_id: tokens}`，按 group id 取值，
  不假设 group id 连续）透传到 `LoadSpec.kvpool_hits_per_group`，
  `_prepare_mooncake_put_session` 对**每组**用本组自身命中作为 save 起跳。缺省为 None 时行为不变。
- **修法 B（可选兜底）**：`layerwise_put_exists_filter=true` 时，put_session_start 前先用
  `batch_is_exist` 过滤已存在 key。

**读日志的两个坑（2026-09-11 更正）**

1. `store_skip_tokens=0` **不等于**“修法 A 未生效”。LoadSpec 只在 new request 首次调度时构建，
   chunked-prefill 的后续 chunk 走 `_process_running_cached_request`，本来就没有 LoadSpec
   （`metadata.py` 的 `from_request_tracker` 在 `can_load` 为假时把 load_spec 置 None），
   所以运行中 chunk 的 `store_skip_tokens/pool_hit_tokens` 恒为 0。请用
   `has_load_spec` 区分“没有命中检查”与“命中为 0”。
2. `if not requested_keys: continue` 在 diag 打印之前，会出现“整段都命中、键集为空、
   因此看不到该层 put_session 行”的情况。现在这种整段跳过会打印
   `keys_total=0 / skipped_pooled_prefix=true` 的事件。

**“重写已池化前缀”的结论修正**：在 `exp_logs/new_log1.md`（`chatcmpl-94a01242`，
`hits_per_group=[8192]*6`）里，该请求 12 条 `put_session` 的 `start_block` 恰好等于
`8192 / group_block_size`，**没有任何 start_block=0 的行**，即 0–8192 的池化前缀已被跳过；
`already_existing` 落在 8192–16384，是该 chunk 自己的新区间（90% 命中基准下前缀共享，
被并发请求写过）。修法 A 仍然有价值，但只在各组命中不齐（min-over-groups 欠跳）时生效，
收益量级是每 chunk 约 40ms 的 `batch_put_start` RPC——KV payload 本来就没有重复搬运。
- 验收（修正后）：各组命中不齐时 `[min, 本组自身命中)` 段不再 put；master
  `object_already_exists` 只统计“本请求写入区间内的重复”；不以“chunk1 already_existing → 0”为准。

## 每 step 一次提交（与 MemCache 对齐）

早期实现是“每个组在自己最后一层 commit”，一步之内会发出和组数一样多的控制调用
（DSV4 为 6 次），而 TE 是单线程的，这些调用会插在逐层拷贝之间，把下一层的搬运往后推。
现在改为 MemCache `batch_write_finish` 的语义，**put / get 两侧的控制调用同样收敛成每 step 一次**：

| 方向 | 控制调用 | 每 step 次数 | 说明 |
| :-- | :-- | :-- | :-- |
| put | `batch_put_session_start` | 1（原 6） | `_prepare_mooncake_put_session` 只收集候选 key（含各组对象 size），`_start_mooncake_put_session` 统一发一次 |
| put | `batch_put_session_end`（commit） | 1（原 6） | save 线程累积 `_pending_commit_keys`，末层入队 `_LayerCommitTask` 后一次提交 |
| get | `batch_get_session_start` | 1 | 本 step 所有请求/组的 key 合成一次调用 |
| get | `batch_get_session_end` | 1 | 末层把 key 交给 recv 线程，由传输线程一次关闭 |

- 被 revoke 的 key（`batch_copy_put` 失败）不进 `_pending_commit_keys`，也不会出现在这次提交里；
  提交完成后统一清空 revoke 标记，进入下一个 step。
- `put_session` 事件的 `put_start_ms` 现在恒为 0（该 step 只有一次共享调用），真实耗时在新的
  `{"event":"put_start","keys_total":N,"newly_started":M,"batches":B,"ms":X}` 事件里；把各行相加
  不会再重复计数。

可见性因此变为“某 step 的所有组、所有层在 step 末一起可见”，与 MemCache 一致；代价是某个组
在自己末层之后、step 结束之前不可见（可接受：同 step 内没有别的请求会依赖它）。
单组路径同样走这次 step 提交。
