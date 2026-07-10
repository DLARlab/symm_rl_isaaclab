# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the Go2 URDF variant with calf fixed-collision links folded into calves."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET


def _collision(xyz: str, rpy: str, length: str, radius: str) -> ET.Element:
    collision = ET.Element("collision")
    ET.SubElement(collision, "origin", {"xyz": xyz, "rpy": rpy})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "cylinder", {"length": length, "radius": radius})
    return collision


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    urdf_dir = repo_root / "source" / "isaaclab_assets" / "isaaclab_assets" / "robots" / "go2_symm" / "urdf"
    src_path = urdf_dir / "go2.urdf"
    dst_path = urdf_dir / "go2_calf_collapsed.urdf"

    tree = ET.parse(src_path)
    root = tree.getroot()
    remove_names: set[str] = set()

    for prefix in ("FL", "FR", "RL", "RR"):
        calf = root.find(f"link[@name='{prefix}_calf']")
        if calf is None:
            raise RuntimeError(f"Missing link: {prefix}_calf")

        calf.append(_collision("0.020000 0.000000 -0.148000", "0.000000 0.050000 0.000000", "0.065", "0.011"))
        calf.append(_collision("0.008013 0.000000 -0.187450", "0.000000 0.530000 0.000000", "0.03", "0.0155"))
        remove_names.update(
            {
                f"{prefix}_calflower",
                f"{prefix}_calflower1",
                f"{prefix}_calflower_joint",
                f"{prefix}_calflower1_joint",
            }
        )

    for elem in list(root):
        if elem.tag in {"link", "joint"} and elem.attrib.get("name") in remove_names:
            root.remove(elem)

    ET.indent(tree, space="  ")
    tree.write(dst_path, encoding="utf-8", xml_declaration=True)
    print(dst_path)


if __name__ == "__main__":
    main()
