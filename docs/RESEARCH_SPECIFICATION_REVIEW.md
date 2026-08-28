# Research Specification 顶会潜力审核

## 审核结论

**审核对象：** [`RESEARCH_SPECIFICATION.md`](RESEARCH_SPECIFICATION.md) version 1.0
**审核视角：** NeurIPS / ICLR / AAAI 等顶会的严格审稿标准
**审核日期：** 2026-08-26

### 总体判断

当前规格文档体现出较强的工程规范意识、信息边界意识和复现意识，但**如果完全按照当前文档实施，仍不足以稳定支撑顶会主会录用**。它目前更接近一份严谨的实验设计草案，而不是已经完成科学闭环的顶会级研究方案。

在不修改核心问题定义和评测设计的情况下，预期审稿结论更可能是 **Reject / Weak Reject**。粗略估计当前录用概率为 **10%--20%**；完成研究问题级别的补强后，才可能进入 Borderline 或 Weak Accept 区间。

最关键的阻碍不是代码量，而是以下三个科学问题尚未被充分解决：

1. 允许的信息是否真的足以识别不同故障机制？
2. 主动干预带来的恢复收益是否可以进行因果且成本公平的比较？
3. 现有样本量和实验矩阵是否足以支持校准、泛化和优越性结论？

因此，继续增加 uncertainty feature 或 monitor 模块之前，应先重构可识别性和因果评测协议。

## 一、已具备的优点

### 1. 研究问题不局限于二元失败检测

文档将风险估计、故障诊断、干预选择和恢复评估拆开定义（`RESEARCH_SPECIFICATION.md:21-46`）。这比只报告 AUROC 或单一 failure score 更接近实际运行时安全问题。

### 2. 信息边界写得清楚

文档明确禁止使用 `FaultSpec`、故障调度时间、执行动作、MuJoCo 特权状态、终局标签等在线信息（`RESEARCH_SPECIFICATION.md:188-233`）。这是防止标签泄漏和不公平比较的必要条件。

### 3. 对已有工作的创新边界相对诚实

文档明确将 OOD、动作不确定性、action-conditioned prediction、conformal calibration、hidden-feature probe、retry 等列为已有组件，而不是单独声称创新（`RESEARCH_SPECIFICATION.md:73-102`）。这一点有助于避免明显的 novelty overclaim。

### 4. 策略冻结和复现约束较强

参数冻结、参数哈希、checkpoint SHA256、动作合法性、软件版本和输出不可覆盖等要求较完整（`RESEARCH_SPECIFICATION.md:170-186`、`677-698`）。现有 GR00T 审计也记录了 30 个 clean episodes、19 次成功以及评估前后相同的参数哈希（`ROBOCASA_OFFICIAL_CHECKPOINT_AUDIT.md:97-114`）。

需要注意：这些证据能支持 policy integration 和工程 provenance，不能单独证明 monitor 或 recovery 方法有效。

## 二、致命问题（Major / Fatal）

以下问题若不修复，即使最终结果数值较好，也会严重影响顶会录用。

### F1. Latent mode 定义不具备清晰的可识别性

文档将以下对象放入同一个互斥变量 `h_t`：

```text
healthy, actuator_fault, observation_fault, policy/OOD, irrecoverable
```

见 `RESEARCH_SPECIFICATION.md:247-280`。

这些对象并不处于同一语义层级：

- `actuator_fault`、`observation_fault` 是外部干预机制；
- `policy/OOD` 是输入分布或策略不确定性；
- `irrecoverable` 是关于未来可达性的状态属性；
- `healthy` 是剩余状态。

它们也不一定互斥。例如 observation fault 可能导致 policy/OOD，故障后又可能进入 irrecoverable。当前文档没有定义 mode 的时间边界、重叠故障优先级、`irrecoverable` 的 offline 标签规则，或 policy/OOD 与普通困难 clean state 的区分方法。

仅给出

\[
b_t(h)=P(h_t=h\mid x_t)
\]

不能使一个本身未定义且不可观测的分类变量变得可估计。

**必须修改：**

- 将故障机制和任务可恢复性拆成不同变量，例如 `m_t` 与 `r_t`；
- 为每个变量写出明确的 offline labeling protocol；
- 定义多故障和状态转移规则，或明确限制为单故障协议；
- 明确哪些机制在允许的信息边界下理论上不可区分；
- 将不可恢复性作为独立的 hindsight evaluation label，除非能提供正式的 online 判定定义。

### F2. 主动干预收益缺少因果可识别设计

文档提出 recovery critic 和信息增益目标：

\[
V(u\mid x_t)=P(\mathrm{success}\mid x_t,u)-\lambda C(u),
\]

以及

\[
u_t^*=\arg\max_u[V(u\mid x_t)+\eta I(h_t;o_{t+1}\mid x_t,u)].
\]

见 `RESEARCH_SPECIFICATION.md:365-405`。

但当前协议没有规定如何可靠估计 success probability、information gain、intervention cost，以及干预后的状态重置和时间预算。若 critic 使用 episode outcome 训练，再在相近的 fault schedule 上测试，很容易学习到 schedule correlation，而不是恢复能力。

仅比较最终 success rate 也不能证明主动诊断是原因，因为干预改变了后续轨迹。还缺少：

- 同一 decision state 上不同干预的随机化或 counterfactual 设计；
- intention-to-treat 统计；
- 干预次数、时间、计算和人工成本的统一量纲；
- 对 `terminate`、`request_help` 与成功之间效用差异的定义；
- 对“检测到故障后才统计”的 selection bias 处理。

**必须修改：** 将 recovery evaluation 定义为 sequential decision / off-policy evaluation 问题；预先定义干预成本和可用条件；在匹配 decision checkpoint 上随机化或进行严格 paired comparison；同时报告 no-intervention、fixed retry、oracle mechanism 和 learned selector 的 intention-to-treat 结果。

### F3. 样本量不足以支撑同时报告的结论

推荐 split 是 training 100 个 seeds、calibration 50 个 seeds、validation 50 个 seeds、final 30 个 seeds（`RESEARCH_SPECIFICATION.md:450-466`）。

这对工程试验可行，但对顶会级优越性和校准结论偏弱：

- final 只有 30 个 episode；
- 5% clean intervention budget 在 30 个 episode 上只有约 1--2 个事件的分辨率；
- 每个 fault family 还要切分 onset、duration、severity、camera 和 chunk alignment；
- confusion matrix 的每个 cell 会非常稀疏；
- 同时比较多个 baseline、多个 operating point，却没有 primary endpoint 或 power analysis。

现有 clean baseline 的 19/30 与官方 18/30 的差异不能被视为稳定优势。30 个样本的二项比例区间很宽，不能支撑泛化结论。

**必须修改：**

- 预先指定一个 primary endpoint；
- 给出最小可检测效应和 power analysis；
- 增加 independent seeds，而不是只增加同一 episode 的 frame 数；
- 使用分层或 fractional-factorial 设计，避免每个 cell 样本过少；
- 采用 hierarchical bootstrap 或 mixed-effects model；
- 对 calibration、recovery effect 和 transfer 分别安排足够样本。

### F4. Clean false-intervention budget 定义不完整

文档将主要 operating curve 定义为固定 clean false-intervention budget，并举例使用 5% clean episodes（`RESEARCH_SPECIFICATION.md:551-572`）。

但 `re-query_policy`、`switch_camera_subset`、`retry`、`request_help` 和 `terminate` 的风险、时间和计算代价完全不同。把它们都计为一个 non-continue event 会掩盖真实 trade-off。按 episode 计数又会忽略 episode 长度差异。

**必须修改：** 预先定义统一的 cost vector，例如时间、计算、人力和任务风险成本；同时报告 per-episode、per-1,000-steps、weighted intervention cost、clean success degradation 以及 success-cost Pareto frontier。

### F5. “Paired counterfactual”表述过强

文档要求 clean/faulted episode 使用同一 seed，并称 fault 是唯一条件差异（`RESEARCH_SPECIFICATION.md:441-448`）。但 fault 会改变 action、状态、随机数消耗和终止时间；policy 也可能包含随机推理。因此这通常只能称为 **matched-seed comparison**，不自动等于严格 counterfactual。

**必须修改：** 明确区分 common-random-number pairing、same initial state、deterministic replay 和 post-injection divergence；若无法保持外生随机过程一致，应降低措辞并量化随机性影响。

## 三、主要问题（可修复但会显著影响说服力）

### M1. 创新仍主要表现为已有组件的组合

文档列出的主要组件几乎都已有先例（`RESEARCH_SPECIFICATION.md:73-102`）。当前贡献描述是“fault conditioning + prediction + uncertainty + calibration + active recovery”的组合，但没有明确说明该组合产生了什么新的算法原理、可验证命题或理论优势。

审稿人会直接追问：为什么强的 FIPER/Foresight/SAFE 加一个 learned selector 不能达到同样结果？

**建议：** 至少提出一个可检验的核心命题，例如 fault-conditioned residual 在特定观测条件下改善机制区分，或主动 probe 在明确假设下减少 Bayes risk，并围绕该命题设计 ablation 和负结果报告。

### M2. Observation occlusion 可能被低级图像统计捷径解决

当前 RoboCasa adapter 对选定 camera 主要采用 `np.zeros_like` mask（`src/vla_recovery_bench/robocasa_adapter.py:145-169`）。全零图像具有极强的低级可检测特征，image-health baseline 很可能直接识别它；这不能证明 monitor 学会了 observation-vs-actuator mechanism diagnosis。

应加入 partial mask、blur、frozen frame、自然遮挡、曝光/颜色变化、stale image 和多 camera correlated corruption，并对不同故障做 matched observability 控制。

### M3. Actuator dropout 的物理语义过窄

将所有请求动作替换成 zero-like action（`RESEARCH_SPECIFICATION.md:409-420`）可能不等价于真实 actuator fault；对 gripper 和 control mode 尤其如此。至少应比较 all-channel、arm-only、base-only、gripper-only、hold-last、noise 和 intermittent dropout，并说明它们是否属于同一机制。

### M4. Reward timing 未固定

文档允许 reward 作为 online input（`RESEARCH_SPECIFICATION.md:193-210`），但没有统一规定 reward 产生时刻、success reward 是否可用于同一步检测、所有 baseline 是否看到相同 reward。建议 primary track 禁止 reward，另设 reward-enabled ablation。

### M5. `irrecoverable` 没有正式定义

primary fault families 只有 actuator dropout 和 observation occlusion（`RESEARCH_SPECIFICATION.md:407-425`），却要求诊断 irrecoverable state。若没有 reachable-set、hindsight planner 或标注一致性协议，该标签容易退化为终局 success label 的泄漏版本。建议移出 primary online diagnosis，改为独立的 hindsight 分析。

### M6. Intervention protocol 与当前代码接口不一致

规格要求 `requery_policy`、`reissue_current_chunk`、`switch_camera_subset` 和 `diagnostic_probe`（`RESEARCH_SPECIFICATION.md:380-394`），但当前 `RecoveryAction` 只有 `CONTINUE/RETRY/REPLAN/REQUEST_HELP/TERMINATE`（`src/vla_recovery_bench/types.py:19-25`）。

当前 `MonitorContext` 也只传一个 action 和完整 `info`，没有 chunk position、action chunk、camera subset、latency 或 probe state（`src/vla_recovery_bench/types.py:79-87`）。因此当前实现无法执行文档所称的 active diagnosis。

### M7. 当前通用 runner 存在信息边界和 artifact contract 落差

runner 将完整 `transition.info` 传入 monitor（`src/vla_recovery_bench/runner.py:110-120`），同时从 `info["success"]` 判定终局（`src/vla_recovery_bench/runner.py:173-175`）。虽然终局标签用于评估是合理的，但接口没有实现明确的 privileged-info firewall，无法保证任意 monitor 实现不会看到泄漏字段。

此外：

- `EpisodeResult` 缺少规格要求的 `pair_id`、posterior、calibration、latency 和完整 intervention audit（`src/vla_recovery_bench/types.py:100-113`）；
- `JsonlRecorder` 以写模式直接打开文件，未实现拒绝覆盖非空目录（`src/vla_recovery_bench/recording.py:29-37`）；
- 当前 runner 不生成规格要求的 `run_manifest.json`、`monitor_config.json`、`calibration.json` 等完整 artifact set。

### M8. 单 task、单 checkpoint 的外部有效性不足

文档已经正确限制：只有一个 task 和一个 frozen GR00T checkpoint 时，不能宣称 policy-agnostic 或 general RoboCasa（`RESEARCH_SPECIFICATION.md:700-715`）。但在已有跨任务、跨策略 failure detection 工作之后，单 task + 单 policy + 两类 synthetic fault 仍不足以支撑通用方法主张。

至少需要两个 task、两个 frozen policies 或明确将论文定位为 GR00T-specific case study，并加入未见 fault schedule 的 transfer test。

## 四、当前仓库证据能支持什么

### 可以支持的结论

- RoboCasa 观察和结构化 action contract 已有较完整的审计记录；
- GR00T checkpoint 文件和参数冻结状态有 provenance 证据；
- clean baseline runner、adapter validation 和基础 fault orchestration 已有工程基础；
- 当前项目适合继续做 Phase 0 identifiability pilot。

### 当前不能支持的结论

- 不能声称已完成 fault diagnosis；
- 不能声称 active recovery 已优于 fixed retry；
- 不能声称 calibration guarantee 已成立；
- 不能声称 policy-agnostic 或 general RoboCasa；
- 不能把现有 30-episode clean baseline 当作 recovery result。

现有审计文档也明确指出 clean run 是 checkpoint/environment provenance，而不是 detector 或 recovery result（`ROBOCASA_OFFICIAL_CHECKPOINT_AUDIT.md:97-114`）。

## 五、建议的最小研究重构

### 建议收窄后的主张

> 在不访问执行动作和 simulator state 的条件下，外部 runtime monitor 利用 action-conditioned temporal evidence 和有限主动 probe，在固定的加权 clean intervention cost 下，提高冻结 VLA 在两类可复现故障上的 recovery success；该收益在未见 fault timing 和至少一个未见任务上保持。

该主张比“通用 fault diagnosis and recovery framework”更可证伪，也更符合当前环境和 policy 资源。

### 必须新增的实验

1. **Mechanism identifiability study**
   - actuator 与 observation 分开评估；
   - 加入 hard observation corruptions；
   - 报告不可辨识区域和低级统计 baseline；
   - 明确 offline labels 与 upper bound。

2. **Intervention causal study**
   - 在相同 decision checkpoint 比较 continue、fixed retry、probe 和 oracle；
   - 随机化 intervention assignment 或使用严格 paired design；
   - 报告 intention-to-treat success、weighted cost 和 failure-to-recovery 曲线。

3. **Generalization study**
   - 至少两个 task；
   - 至少两个 frozen policies，或将 claim 明确限制为 GR00T-specific；
   - held-out onset、duration、camera corruption 和 task combination。

4. **预注册统计分析计划**
   - primary / secondary / exploratory endpoint；
   - power analysis；
   - hierarchical paired bootstrap 或 mixed-effects model；
   - multiple-comparison correction；
   - censored detection delay 的处理规则。

5. **实现信息防火墙和完整 artifact contract**
   - monitor context 不传完整 privileged `info`；
   - requested/executed action 分开保存；
   - 实现规格中列出的 intervention enum 和 chunk reset 语义；
   - 生成完整 manifest、calibration、policy state 和 audit 文件；
   - 非空输出目录原子拒绝覆盖。

## 六、建议的论文声称门槛

| 论文声称 | 最低证据要求 |
|---|---|
| 更早预警 | 相同 clean false-intervention/cost budget 下，对 temporal residual、FIPER-style、Foresight-style 等 baseline 的 paired held-out comparison |
| 更好机制诊断 | 无在线 fault label 的 held-out mechanism confusion matrix，以及未见 timing/duration/corruption 的泛化 |
| 更好校准 | 独立 clean calibration/test split、明确 sequential assumptions、coverage/reliability 和置信区间 |
| 更好恢复 | matched decision state 下对 fixed retry、re-query 和 oracle mechanism 的 intention-to-treat comparison |
| policy-agnostic | 至少两个 frozen policies 或跨 policy 的严格控制实验 |
| general RoboCasa | 至少两个 task 和多个 fault schedules；单 task 只能称 case study |

## 七、审稿式评分（当前版本）

| 维度 | 估计评分 | 说明 |
|---|---:|---|
| Novelty | 4/10 | 目前主要是已有组件的组合，核心不可替代性尚未证明 |
| Technical quality | 5/10 | 规范和工程约束较好，但 latent state、因果评测和 calibration 定义不完整 |
| Empirical support | 3--4/10 | 当前只有单 task/single policy clean provenance，尚无 recovery evidence |
| Clarity | 7/10 | 文档结构清楚、边界意识强，但“方法定义”和“工程门禁”混在一起 |
| Overall | 4/10 | Weak Reject；需要研究问题级别重构 |

## 最终结论

该项目**值得继续做**，但不应把“代码实现完成”误认为“顶会证据链完成”。当前最优先的工作不是继续堆叠 monitor feature，而是完成 Phase 0 identifiability pilot，并据此决定：

- 若机制可辨识且主动 probe 有净信息增益，再继续 fault-conditioned active recovery；
- 若机制不可辨识但 recovery 可改善，则收窄为 recovery-aware risk monitor；
- 若在控制成本后没有稳定恢复收益，则停止 diagnosis/recovery 主张，只保留 benchmark 或 risk-monitor 结果。

在上述关键证据补齐之前，论文标题、摘要和结论都不应使用“general-purpose”“policy-agnostic”或无条件的“fault diagnosis and recovery”表述。
