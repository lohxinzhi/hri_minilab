# Setup checklist

```text
Machine inspected        [x]
Miniconda installed      [x]
Repository structured    [x]
Upstream source pinned   [ ]
Conda environment ready  [x]
Dependencies installed   [ ]
Headless simulation      [ ]
Native viewer            [ ]
Keyboard controls        [ ]
Browser GUI              [ ]
Three terrain maps       [ ]
Three cameras            [ ]
Architecture inspected   [ ]
```

Repository scaffolding has been approved. The `quadruped_mujoco` Conda environment was created and verified after activation: Python 3.11.16 at `/home/briansyc/miniconda3/envs/quadruped_mujoco/bin/python`; `conda env list` confirmed it was active. No project dependencies have been installed. Visual and manual tests remain pending until explicitly confirmed by the user.

## Required Canvas submission videos

The final Canvas ZIP must contain all three demonstration videos, with their actual video extensions:

- `Video_Task2`
- `Video_Task3`
- `Video_Task4`

Generated videos may remain untracked by Git. The future packaging procedure must explicitly include these final videos even when ignored by Git, and fail if any is missing. Verify all three are present in the extracted ZIP. A Git-only archive is insufficient when the videos are untracked.

Selected evaluation CSVs, plots, screenshots, prompts, scene configurations, and required project-owned source assets must also be retained in the submission. No packaging procedure or demonstration video has been created yet.

Empty scaffold directories currently exist only on disk: Git does not track empty directories. No placeholder modules or files have been added.
