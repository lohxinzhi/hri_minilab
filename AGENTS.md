# Instructions for future Codex sessions

1. Read `HANDOVER.md` first; it is the detailed source of truth.
2. Activate the `quadruped_mujoco` Conda environment before project Python work.
3. Inspect `git status` before changing files. Preserve all uncommitted work.
4. Treat motion as committed WIP, numerically tested but pending manual visual validation. Treat the parser as committed WIP, unit-tested but with no live LLM provider or behavior integration.
5. Do not edit `third_party/quadruped_mujoco/` unless integration makes it unavoidable and the user authorizes the specific change.
6. Never use `evaluation/ground_truth/` for perception, navigation, search, or steering.
7. Use only `dog_front_camera` as the Task 4 image source.
8. Run relevant tests before and after changes; do not open GUI tests unless the user requests/manual validation is intended.
9. Keep generated weights, caches, and `outputs/` out of source commits.
10. Preserve the strict LLM boundary: exact schema, local validation, and no direct control from free-form text.

Immediate continuation: have the teammate perform the eight pending native-viewer motion checks before motion is declared complete. Do not begin search/approach integration until the motion visual review is recorded.
