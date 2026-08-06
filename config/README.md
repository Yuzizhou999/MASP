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
python tools/build_phase0.py
python tools/validate_phase0.py
pytest -q
```

出现 `valid: true` 表示配置可用于仿真。当前安全参数没有最终确认，因此出现 `simulation-only` 告警属于预期行为。

阶段 2 的自动任务分配与连续时间路径规划可运行：

```powershell
python tools/run_phase2.py scenarios/phase2-continuous-tasks.json
```

结果写入 `runs/phase2-continuous-tasks/`，其中 `planned-scenario.json` 可用于逐段检查车辆选择、路线、等待、取放货服务和放货后撤离。

阶段 3 的滚动周期协调、Top-K 优先级和吞吐基准可运行：

```powershell
python tools/run_phase3.py scenarios/phase3-rh-pp-benchmark.json
```

结果写入 `runs/phase3-rh-pp-benchmark/`：

- `planning-summary.json`：每轮候选顺序、可行数量、词典序评分、安全提交边界和规划耗时。
- `benchmark.json`：拥堵策略与多个随机种子的吞吐、等待和安全对比。
- `planned-scenario.json`：被选中并用于确定性回放的完整计划。

调试单一策略时可添加 `--policy congestion` 或 `--policy random --seed 1`；只检查 Top-K 主场景而不运行完整基准时可添加 `--skip-benchmark`。

阶段 4 的窄路原子前瞻、等待图监督和倒退恢复场景可运行：

```powershell
python tools/run_phase4.py scenarios/phase4-deadlock-recovery.json
```

结果写入 `runs/phase4-deadlock-recovery/`。`summary.json` 同时记录真实窄路入口外等待、两车等待环的沿当前边倒退，以及四车环无合法恢复路径时的安全停止。阶段 4 MVP 仅接受 `capacity=1`、禁止会车、同一时刻单方向通行的交通区；其他容量配置会在 Schema 或启动校验时被拒绝。

## 不要手工修改

`generated/` 目录下的统一地图、冲突资源和工位文件由工具生成，不应手工添加注释或修改内容，否则下次构建会被覆盖。
