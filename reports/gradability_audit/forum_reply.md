Thanks @ryanholbrook for the clarification in the local-evaluation thread, and @JohnParkerson and @Bharat for the reproductions.

**Scope correction.** Given that the hidden tasks come from a different set of repositories and that gold patches validate at 100% on the private Docker scorer, the table in this notebook describes the **public development environment only** (public tasks, public `wheels/`, released tooling). It says nothing about the leaderboard, so please read it that way.

**Clarifications to our numbers:**
- The 41/129 per-repo counts come from swegemma's `verify_sample_task_airgapped` helper. On that path, FastAPI tasks fail to import `starlette`. Through the real Phase-2 path (`verify_task`, called as `Evaluator.evaluate_task` does), `starlette` installs fine. The FastAPI failures we reproduce there are test-only dependencies: `dirty_equals` for us, and others report `inline_snapshot` and `typing_inspection`.
- We re-ran the helper sweep after clearing `$TMPDIR/swegemma_sp_cache_v8`, the wheel-cache issue in thread 744029. The counts do not change, so they are not a cache artefact.
- The requests failures are not a missing module. @JohnParkerson's diagnosis matches what we see: on Python 3.13, pytest-httpbin's bundled certificate fails strict X.509 verification. Add the 3 version-gated tasks and `rich_3472` he lists, and the installed-package shadowing reported by @rdeschamps (tests importing the installed `requests`/`httpx` instead of `/workspace`).

**Remaining request, for local development only:** could the released setup include the test-only wheels (`dirty_equals`, `inline_snapshot`, `typing_inspection`, and a pytest-httpbin that works on 3.13) and make `/workspace` take precedence over installed copies of the task's own package? That would make gold-patch positive controls pass locally with the public tooling, so local comparisons become trustworthy. Thanks!
