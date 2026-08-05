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

## 不要手工修改

`generated/` 目录下的统一地图、冲突资源和工位文件由工具生成，不应手工添加注释或修改内容，否则下次构建会被覆盖。
