# HiPHI 与 G1 对比 Replay

这个目录提供一个可直接运行的 replay 入口，用 MuJoCo 并排显示：

- 左侧（或运动方向的一侧）：HiPHI 原始 BVH 骨架和原始箱体轨迹；
- 另一侧：重定向后的 G1 和缩放后的箱体轨迹；
- G1 左右手的 thumb EE 代理点（青色/洋红色小球）。

入口脚本可以从任意工作目录执行，不再要求使用
`python -m omniretargeting.hiphi_compare_replay`。

## 快速开始

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/ubuntu/.venvs/omniretargeting/bin/python \
/home/ubuntu/hiphi/hiphi_compare_replay/replay.py \
--motion /home/ubuntu/hiphi/retargeting/outputs/dynamic_anchor_bimanual_no_shoulder_no_mirror_full_v3/motions/Bringing-carry_0001/motion_actor/motion_actor_retargeted.npz
```

脚本会根据 NPZ 路径里的 motion ID 自动寻找：

```text
/media/ubuntu/T9/HiPHI/data/Bringing/carry/<motion-id>/motion_actor.bvh
/media/ubuntu/T9/HiPHI/data/Bringing/carry/<motion-id>/metadata.json
```

并自动加载 NPZ 同目录下的：

```text
motion_actor_scaled_terrain.obj
motion_actor_scaled_objects/*.obj
motion_actor_scaled_objects/*_poses.json
```

## 播放控制

| 按键 | 功能 |
| --- | --- |
| `Space` | 暂停或继续 |
| `[` / `]` | 暂停并前退/前进一帧 |
| `0` | 回到当前 motion 的第 0 帧 |
| `P` / `N` | 上一条/下一条 motion |

当输入 NPZ 位于某个名为 `motions` 的目录下时，脚本会递归发现该目录中的
全部 `*_retargeted.npz`，排序后组成播放列表，因此可以用 `P/N` 连续视检。
切换时复用同一个 MuJoCo/GLFW 窗口，只替换动作和箱体轨迹，并从新动作的
第 0 帧开始；不会关闭后重新创建 viewer。为了兼容不同身高缩放后的 Box_H，
启动时会预扫描并去重箱体 mesh，动作本身仍在切换时按需加载。
HiPHI 序列若使用不同的 BVH 骨架层级（例如 51/55 joints），脚本会按播放列表
中的最大 joint/bone 数预留显示槽，切换时隐藏未使用槽，因此也无需重建窗口。

## 常用参数

```text
--motion PATH          必填，重定向输出 *_retargeted.npz
--hiphi-root PATH      HiPHI 根目录，默认 /media/ubuntu/T9/HiPHI
--source-motion PATH   显式指定第一条原始 BVH
--metadata PATH        显式指定第一条 metadata.json
--robot-model PATH     G1 MJCF/URDF 模型
--separation FLOAT     人与机器人初始间距，默认 2.2 m
--fps FLOAT            覆盖显示帧率
--hide-thumb-ee        隐藏左右 thumb EE 小球
--thumb-ee-inset FLOAT 仅将显示小球向掌根方向缩进，不改变 NPZ 或优化结果
```

查看完整帮助：

```bash
/home/ubuntu/.venvs/omniretargeting/bin/python \
/home/ubuntu/hiphi/hiphi_compare_replay/replay.py \
--help
```

## 环境与依赖

推荐直接使用现有环境：

```text
/home/ubuntu/.venvs/omniretargeting/bin/python
```

主要 Python 依赖为 `numpy`、`scipy`、`trimesh` 和 `mujoco`。MuJoCo viewer
需要可用的桌面显示和 OpenGL/GLFW 环境。

默认机器人模型为：

```text
<repo>/robot_models/unitree_g1/g1_29dof_holosoma_omnicontact_hand.xml
```

如果移动了模型或其 mesh 资源，请通过 `--robot-model` 指向能够独立加载的
MJCF/URDF 文件。

## 目录与可搬运性说明

这是一个放在 `/home/ubuntu/hiphi` 工程下的独立入口。单窗口 playlist 生命周期
由本目录的 `persistent_replay.py` 管理，同时复用：

```text
omniretargeting/hiphi_compare_replay.py
omniretargeting/data_sources/hiphi.py
omniretargeting/replay.py
omniretargeting/visualizer.py
```

这些模块提供路径推断、数据加载和 MuJoCo 场景拼装；本目录只维护同窗 `P/N`
切换所需的固定场景与播放状态。如果要把工具复制到仓库外，还需要一并打包
`replay.py`、`persistent_replay.py`、上述 Python 代码、G1 XML 和大约 30 MB 的
mesh 资源。

默认使用本工程内迁移后的实现：

```text
/home/ubuntu/hiphi/retargeting
```

如果仓库移动了，运行前指定：

```bash
export OMNIRETARGETING_ROOT=/new/path/to/omniretargeting
```

## 常见问题

- 找不到原始 BVH：检查 motion ID 是否与 HiPHI 目录名一致，或传入
  `--source-motion` 和 `--metadata`。
- `P/N` 没有切换：确认 NPZ 位于名为 `motions` 的祖先目录下。
- 找不到机器人 mesh：确认默认 G1 XML 引用的 mesh 目录存在，或使用
  `--robot-model`。
- viewer 无法打开：确认当前会话有图形显示权限；纯 SSH 环境通常需要 X11
  转发或本机桌面会话。
