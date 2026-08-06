# 阶段 1 仿真场景

阶段 1 使用“显式计划”验证仿真执行内核。场景需要写明车辆、任务以及每一段动作的开始和结束时间，但不需要手工填写道路冲突资源，运行器会根据地图自动补全。

`phase1-single-vehicle.json` 的流程是：

1. `fork-001` 从 `fork:PP1171` 出发。
2. 沿三条有向边到达 `fork:AP1123`。
3. 用 5000 ms 完成取货，车辆由空载变为载货。
4. 沿三条有向边到达 `fork:AP2121`。
5. 用 5000 ms 完成放货，任务完成，车辆恢复空载。
6. 因为 AP 不允许驻留，车辆继续空载驶离并停到 `fork:PP1172`。

运行命令：

```powershell
python tools/run_phase1.py scenarios/phase1-single-vehicle.json
```

结果默认写入 `runs/phase1-single-vehicle/`。该目录是运行产物，不纳入版本控制。

阶段 2 会用任务分配器和连续时间 SIPP 自动产生这里的 `plans`，而不是继续要求人工编写计划。

## 阶段 2 自动规划

`phase2-continuous-tasks.json` 只定义两辆车和三个分时到达的任务。运行：

```powershell
python tools/run_phase2.py scenarios/phase2-continuous-tasks.json
```

规划器会输出 `planned-scenario.json`，其中可以查看自动选择的车辆、道路、等待区间、取放货服务和放货后的 PP/CP 撤离路线。

## 阶段 3 RH-PP 基准

`phase3-rh-pp-benchmark.json` 使用三辆 fork 车辆和五个分时任务，第一轮会让三辆车同时参与优先级协调。运行：

```powershell
python tools/run_phase3.py scenarios/phase3-rh-pp-benchmark.json
```

默认策略是 `top_k`。程序会额外运行一次拥堵优先基线和三个固定随机种子基线，并在 `runs/phase3-rh-pp-benchmark/benchmark.json` 中检查：

1. 拥堵策略吞吐不低于随机策略均值。
2. 所有基线运行都没有资源预留冲突。
3. Top-K 和固定策略的规划 p95 均小于配置的规划周期。

`safeUntilMs` 可能晚于名义执行窗口，因为车辆不能停在 LM/AP；提交边界必须继续延伸到下一个允许等待的 PP/CP。
