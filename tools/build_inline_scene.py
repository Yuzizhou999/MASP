from __future__ import annotations

import argparse
from pathlib import Path


PLACEHOLDER = "__XIATE_MAP_MODEL_JSON__"

# 把地图数据"内联"进 HTML 预览页里
def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a generated map model into the scene fragment")
    parser.add_argument("template", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    template = args.template.read_text(encoding="utf-8")
    model = args.model.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"Missing placeholder {PLACEHOLDER}")

    output = template.replace(PLACEHOLDER, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
