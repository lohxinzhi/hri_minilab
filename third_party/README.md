# Third-party platform provenance

The official simulator is [aoqianz/quadruped_mujoco](https://github.com/aoqianz/quadruped_mujoco). It was fetched locally under `third_party/quadruped_mujoco/`. The exact commit below is the pinned simulator version for this assignment.

The local checkout is ignored by Git. Upstream code is not currently redistributed by this repository; an explicit upstream redistribution licence has not been established. Preserve upstream licence, copyright, README, and attribution files in the local checkout.

This repository tracks provenance and will later track a setup/fetch script using the upstream URL and exact commit SHA. Our implementation remains outside `third_party/` wherever practical. If upstream modification becomes unavoidable, explicit patch files will be tracked under `patches/`. No fetch script or patches exist yet.

| Field | Value |
| --- | --- |
| Upstream repository URL | https://github.com/aoqianz/quadruped_mujoco |
| Exact commit SHA | `dd40180f1121a66373d261e64a9a09eb69b1b2a7` |
| Checkout date | 2026-09-30 (Asia/Singapore) |
| Expected local path | `third_party/quadruped_mujoco/` relative to project root |
| Licence / notice information | No top-level licence file found; `src/runtime_control/maps/BARKOUR_LICENSE.txt` exists. No general redistribution licence assumed. |

The checkout was clean after installation and headless verification. No upstream source files were edited. Editable installation generated only Git-ignored build metadata/caches. Future fetching must use the recorded SHA, not a moving branch tip.
