# 仿真与调度场景

`explicit-single-vehicle.json` 使用显式计划验证仿真执行内核。场景写明车辆、任务以及每段动作的开始和结束时间；运行器根据地图自动补全道路冲突资源。

场景流程是：

1. `fork-001` 从 `fork:PP1171` 出发。
2. 沿三条有向边到达 `fork:AP1123`。
3. 用 5000 ms 完成取货，车辆由空载变为载货。
4. 沿三条有向边到达 `fork:AP2121`。
5. 用 5000 ms 完成放货，任务完成，车辆恢复空载。
6. 因为 AP 不允许驻留，车辆继续空载驶离并停到 `fork:PP1172`。

运行命令：

```powershell
python tools/simulate_explicit_plan.py scenarios/explicit-single-vehicle.json
```

结果默认写入 `runs/explicit-single-vehicle/`。该目录是运行产物，不纳入版本控制。

自动规划场景则使用任务分配器和连续时间 SIPP 生成 `plans`，不需要人工编写计划。

## 自动任务规划

`continuous-task-planning.json` 只定义两辆车和三个分时到达的任务。运行：

```powershell
python tools/simulate_planning.py scenarios/continuous-task-planning.json
```

规划器会输出 `planned-scenario.json`，其中可以查看自动选择的车辆、道路、等待区间、取放货服务和放货后的 PP/CP 撤离路线。

## 滚动调度基准

`rolling-dispatch-benchmark.json` 使用三辆 fork 车辆和五个分时任务，第一轮会让三辆车同时参与优先级协调。运行：

```powershell
python tools/simulate_dispatch.py scenarios/rolling-dispatch-benchmark.json
```

默认策略是 `top_k`。程序会额外运行一次拥堵优先基线和三个固定随机种子基线，并在 `runs/rolling-dispatch-benchmark/benchmark.json` 中检查：

1. 拥堵策略吞吐不低于随机策略均值。
2. 所有基线运行都没有资源预留冲突。
3. Top-K 和固定策略的规划 p95 均小于配置的规划周期。

`safeUntilMs` 可能晚于名义执行窗口，因为车辆不能停在 LM/AP；提交边界必须继续延伸到下一个允许等待的 PP/CP。

## 死锁监督与倒退恢复

`deadlock-recovery.json` 使用真实地图中的三组证据：

1. `jack:PP363 → shared:LM182 → shared:LM368 → jack:PP365` 窄路。对向车辆占用共享 `zone:zone-jack-pp363-pp365` 时，候选车必须在 PP363 等待，入口到出口的区域预留保持连续，内部 LM 不产生等待。
2. `fork:edge-323` 上的车辆被 `shared:LM1254` 阻塞时，从 99% 边进度沿当前边倒退 4.76388 m 到 `fork:PP1173`。地图没有该方向的反向边，因此该用例验证的是动态恢复倒退，而不是预定义倒向边。
3. `fork:LM1028 → LM1031 → LM2472 → LM2473 → LM1028` 四车环没有 5 m 内的合法恢复计划，监督器必须冻结相关资源并输出安全停止。

运行：

```powershell
python tools/validate_recovery.py scenarios/deadlock-recovery.json
```

该验收场景直接给出固定车辆意图和运行时 blocker 证据，不经过任务分配器，避免分配器交换任务后绕开目标窄路。

## 真实班次多车多任务回放

`realistic-multi-fleet.json` 使用完整的 6 辆 fork 与 8 辆 jack 车队，在 25 分钟回放窗口内分八个波次释放 32 个任务。16 个 fork 任务都会进入共享路网，16 个 jack 任务中有 8 个成对穿越共享走廊，最后一波会同时释放 4 个跨共享任务；其余 jack 任务保留为局部运输对照。该场景用于离线观察跨车种资源冲突、等待、预留和规划耗时，不是交互演示档。

首次运行建议跳过额外基线以缩短反馈时间：

```powershell
python tools/simulate_dispatch.py scenarios/realistic-multi-fleet.json --policy congestion --skip-benchmark
```

在线模拟会在任务发布时刻才把任务提交给调度器，并在每次计划 ACK 后动态注入执行事件：

```powershell
python tools/simulate_online_dispatch.py scenarios/realistic-multi-fleet.json --policy congestion
```

`interactive-multi-fleet.json` 是较快的 4 车交互档，使用三波共 6 个任务，适合先检查完整输出和事件时间线：

```powershell
python tools/simulate_dispatch.py scenarios/interactive-multi-fleet.json --policy congestion --skip-benchmark
```

对应的快速在线验证：

```powershell
python tools/simulate_online_dispatch.py scenarios/interactive-multi-fleet.json --policy congestion
```
