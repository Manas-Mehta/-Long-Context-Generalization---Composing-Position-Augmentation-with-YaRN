---
name: Always git push after making file changes
description: User is on HPC and relies on git pull to get updated scripts. Not pushing = HPC has stale files = wasted GPU runs.
type: feedback
---

Always run `git add` + `git commit` + `git push` immediately after editing any file that needs to be used on HPC. Do not wait until the end of a session.

**Why:** HPC has no direct access to local files. The user runs `git pull` on HPC to get updates. If changes are never pushed, HPC always runs the old version of scripts — partition settings, checkpoint paths, flags like `--max-samples` — none of it takes effect. This caused multiple failed eval submissions with the wrong partition, wrong checkpoint path, and missing `--max-samples 100` flag.

**How to apply:** After every Edit or Write to a SLURM script, training script, or eval script, immediately push. Don't batch it up. If you edit 3 files, push after all 3. State explicitly: "Pushing now so HPC has the latest."
