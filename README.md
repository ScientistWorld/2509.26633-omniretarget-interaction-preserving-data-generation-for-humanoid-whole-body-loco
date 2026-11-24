# OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2509.26633"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2509.26633-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2509.26633](https://arxiv.org/abs/2509.26633) |
|:---|:---|

## Summary

OmniRetarget addresses humanoid motion retargeting for object and terrain interaction. The paper argues that preserving interaction geometry and enforcing hard kinematic constraints produces higher-quality reference trajectories than keypoint-only retargeting, reducing penetration and foot skating while maintaining contacts.

## Run it

```bash
pip install sciworld
sw research 2509.26633 --mode innovate --agent claude-code
sw evaluate 2509.26633
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

