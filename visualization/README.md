# 调度场景可视化

`dispatch-dashboard.template.html` 是一个无外部依赖的单文件回放界面。构建脚本把仓库地图、规划路径、车辆/任务状态和事件日志内嵌到 HTML 中，因此生成文件可以直接用浏览器打开，也可以通过静态 HTTP 服务访问。

先生成一份运行数据（已有运行目录可以跳过）：

```powershell
python tools/run_phase3.py scenarios/phase3-realistic-multi-fleet-interactive.json --policy congestion --skip-benchmark
```

构建界面：

```powershell
python tools/build_dispatch_dashboard.py `
  runs/phase3-realistic-multi-fleet-interactive/congestion
```

默认输出为 `runs/phase3-realistic-multi-fleet-interactive/congestion/dispatch-dashboard.html`。可用 `--output` 指定其他路径，`--map` 指定其他统一地图模型，`--scenario` 仅用于补充运行 manifest 中没有的场景字段。

打开生成的 HTML 后可以：

- 在地图上查看 Fork/Jack 车辆、共享路段和工位节点；
- 播放、暂停、拖动时间轴并调整 0.5x 到 4x 的速度；
- 点击车辆或任务查看车辆路线和当前计划段；
- 查看完成任务数、行驶/等待车辆数、吞吐、车辆状态、任务状态和事件时间线。

如果浏览器限制了本地文件脚本，可以在仓库根目录运行：

```powershell
python -m http.server 8765 --directory runs/phase3-realistic-multi-fleet-interactive/congestion
```

然后访问 <http://127.0.0.1:8765/dispatch-dashboard.html>。运行目录属于实验产物，不纳入版本控制。
