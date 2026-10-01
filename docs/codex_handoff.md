# Codex handoff snapshot

Updated: 2026-10-01 (Asia/Singapore)

- Repository: `/home/briansyc/hri_minilab`
- Branch: `handover/minilab-1.3`
- Implementation HEAD before this documentation checkpoint: `d3e6373874f0dec9cfbe322203679738fa67e933` (`WIP: add validated target command parser`)
- Upstream: `dd40180f1121a66373d261e64a9a09eb69b1b2a7`, clean checkout
- Motion WIP commit: `f06f158dfe36278ad4647f0bc58c2de17fb612d5` (`WIP: add motion skills pending visual validation`)
- Parser WIP commit: `d3e6373874f0dec9cfbe322203679738fa67e933` (`WIP: add validated target command parser`)
- Environment: Conda `quadruped_mujoco`, Python 3.11.16 at `~/miniconda3/envs/quadruped_mujoco/bin/python`
- Committed milestones: platform setup/architecture; front camera `6a41a1b`; final scene `df6b075` (introduced `b1dfc3c`); perception `cd3c120`
- Latest full headless suite: `python -m unittest discover -s tests` — 25 tests passed.

## WIP commits on handover branch

- Motion files are committed in `f06f158`; status is **implemented / numerically tested / manual visual stability validation pending**.
- Parser files are committed in `d3e6373`; status is **implemented / unit tested / live LLM provider not connected / not connected to robot behavior**.
- Handover documentation is being finalized in the documentation commit that contains this snapshot.

Immediate next task: manually perform the eight native-viewer motion checks before declaring motion complete. See root `HANDOVER.md` for exact commands and technical context. Preserve the strict upstream, ground-truth, and LLM-validation constraints.
