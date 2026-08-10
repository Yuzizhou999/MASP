# 调度场景可视化

`dispatch-dashboard.template.html` 是一个无外部依赖的单文件回放界面。构建脚本把仓库地图、规划路径、车辆/任务状态和事件日志内嵌到 HTML 中，因此生成文件可以直接用浏览器打开，也可以通过静态 HTTP 服务访问。

先生成一份运行数据（已有运行目录可以跳过）：

```powershell
python tools/simulate_dispatch.py scenarios/interactive-multi-fleet.json --policy congestion --skip-benchmark
```

构建界面：

```powershell
python tools/build_dispatch_dashboard.py `
  runs/interactive-multi-fleet/congestion
```

默认输出为 `runs/interactive-multi-fleet/congestion/dispatch-dashboard.html`。可用 `--output` 指定其他路径，`--map` 指定其他统一地图模型，`--scenario` 仅用于补充运行 manifest 中没有的场景字段。

优化后的在线压力结果可以直接沿用同一界面，并可传入基线运行目录显示规划性能对比：

```powershell
python tools/build_dispatch_dashboard.py `
  runs/online-dispatch-rl `
  --baseline-run runs/online-dispatch-baseline
```

打开生成的 HTML 后可以：

- 在地图上查看 Fork/Jack 车辆、共享路段和工位节点；
- 点击“放大地图”进入独立地图视图，按 Esc 或“关闭放大”恢复布局；
- 播放、暂停、拖动时间轴并调整 0.5x 到 4x 的速度；
- 点击车辆或任务查看车辆路线和当前计划段；
- 聚焦共享区域并查看当前占用共享路径的车辆数；
- 查看完成任务数、行驶/等待车辆数、吞吐、车辆状态、任务状态和事件时间线；
- 查看规划 P95、最慢周期、超时、路线组合、SIPP 尝试和 RL 实际参与次数。

如果浏览器限制了本地文件脚本，可以在仓库根目录运行：

```powershell
python -m http.server 8765 --directory runs/interactive-multi-fleet/congestion
```

然后访问 <http://127.0.0.1:8765/dispatch-dashboard.html>。运行目录属于实验产物，不纳入版本控制。
