# 配置文件说明

标准 JSON 不允许写 `//` 注释，因此本目录采用两套文件：

- `*.json`：程序实际读取的配置，必须保持标准 JSON 格式。
- `*.example.jsonc`：供人阅读的中文注释版，不会被程序读取。

修改配置时，先查看对应的 `.example.jsonc`，再把需要调整的值写入同名 `.json`。

| 运行配置 | 中文注释版 | 用途 |
|---|---|---|
| `scheduler.json` | `example/scheduler.example.jsonc` | 调度、等待、倒退、服务时间和安全参数 |
| `initial-vehicles.json` | `example/initial-vehicles.example.jsonc` | 车辆编号、车型、初始位置、朝向和载荷 |
| `traffic-zones.json` | `example/traffic-zones.example.jsonc` | 恢复点和窄路交通区 |
| `robot-profiles.json` | `example/robot-profiles.example.jsonc` | 车辆尺寸、速度和加减速度 |

`example/task.example.jsonc` 是持续输入任务的格式示例。任务提交时必须使用标准 JSON，不能携带注释。

## 修改后的检查方法

在仓库根目录运行：

```powershell
python tools/build_repository.py
python tools/validate_repository.py
pytest -q
```

出现 `valid: true` 表示配置可用于仿真。当前安全参数没有最终确认，因此出现 `simulation-only` 告警属于预期行为。

自动任务分配与连续时间路径规划：

```powershell
python tools/simulate_planning.py scenarios/continuous-task-planning.json
```

结果写入 `runs/continuous-task-planning/`，其中 `planned-scenario.json` 可用于逐段检查车辆选择、路线、等待、取放货服务和放货后撤离。

滚动周期协调、Top-K 优先级和吞吐基准：

```powershell
python tools/simulate_dispatch.py scenarios/rolling-dispatch-benchmark.json
```

结果写入 `runs/rolling-dispatch-benchmark/`：

- `planning-summary.json`：每轮候选顺序、可行数量、词典序评分、安全提交边界和规划耗时。
- `benchmark.json`：拥堵策略与多个随机种子的吞吐、等待和安全对比。
- `planned-scenario.json`：被选中并用于确定性回放的完整计划。

调试单一策略时可添加 `--policy congestion` 或 `--policy random --seed 1`；只检查 Top-K 主场景而不运行完整基准时可添加 `--skip-benchmark`。

窄路原子前瞻、等待图监督和倒退恢复验收：

```powershell
python tools/validate_recovery.py scenarios/deadlock-recovery.json
```

结果写入 `runs/deadlock-recovery/`。`summary.json` 同时记录真实窄路入口外等待、两车等待环的沿当前边倒退，以及四车环无合法恢复路径时的安全停止。当前参考实现仅接受 `capacity=1`、禁止会车、同一时刻单方向通行的交通区。

RL 优先级策略需要单独安装训练依赖：

```powershell
python -m pip install -r requirements-rl.txt
python tools/train_priority_policy.py scenarios/interactive-multi-fleet.json `
  --state-source rolling --behavior-clone-epochs 4 `
  --steps 256 --candidate-count 1 --priority-prefix-count 2 `
  --output-dir runs/priority-policy-training
```

训练结果写入 `runs/priority-policy-training/priority-policy.pt` 和
`training-summary.json`。使用 checkpoint 运行单候选 RL 推理及三策略对照：

```powershell
python tools/simulate_dispatch.py scenarios/interactive-multi-fleet.json `
  --policy rl `
  --rl-checkpoint runs/priority-policy-training/priority-policy.pt `
  --rl-candidates 1 `
  --rl-allow-deviation `
  --output-dir runs/priority-policy-training/interactive-benchmark
```

训练和推理只在共享未来资源的局部冲突分量中运行。`--priority-prefix-count`
控制 RL 决定分量前部多少辆车，未被选择的尾部保留 congestion 顺序；
`--rl-candidates` 控制每个决策周期实际送入 SIPP 评估的 RL 候选数量。
checkpoint 缺失、推理异常、超时或输出非法时会使用拥堵启发式；合法 RL
排列若全部不可行，也会额外评估一次拥堵启发式候选。RL 只返回车辆优先级，
不能直接写资源预留、提交计划或绕过 `PlanValidator`。CPU 推理默认使用单线程
降低小 batch Transformer 的线程调度开销，可用 `MASP_RL_TORCH_THREADS` 覆盖。
未指定 `--rl-allow-deviation` 时，RL checkpoint 不会替换 congestion guardian，
适合安全回归；在通过独立任务分布验收前，开启偏离只应用于仿真实验。

同一局部 RL checkpoint 也可以接入在线仿真入口：

```powershell
python tools/simulate_online_dispatch.py scenarios/interactive-multi-fleet.json `
  --policy rl --rl-checkpoint runs/priority-policy-training/priority-policy.pt `
  --rl-candidates 1 --rl-allow-deviation
```

在线入口仍逐个候选执行 SIPP、预留和计划验证，并在显式 ACK 后才提交计划。

## 不要手工修改

`generated/` 目录下的统一地图、冲突资源和工位文件由工具生成，不应手工添加注释或修改内容，否则下次构建会被覆盖。
