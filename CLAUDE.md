# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository (`aetherfy-vectors-python-sdk`), one of several Aetherfy repos.

## This repo: aetherfy-vectors-python-sdk

The Python SDK on PyPI (`aetherfy_vectors`, `aetherfy_memory`, `aetherfy_agent`; floor Python 3.9). Its sibling SDK in the other language must stay equivalent.

- **Releases** publish on a GitHub release (`release: published`). Date the CHANGELOG heading BEFORE tagging, since it ships in the package.
- **After every release**, the control plane's `SDK_PIN_PYTHON`/`SDK_PIN_NODE` (`orchestrator/image_generator.py`), its canary pin, golden Dockerfiles and canary README move to the new version, after the publish. Agent images preinstall that pin, so a stale pin ships an old SDK to every new agent.
- **The dashboard's docs pin this API.** `aetherfy-dashboard/docs-site/scripts/sdk-api-snapshot.json` holds every export, method and argument both SDKs offer, and every code example in the docs is checked against it. A public API change needs that snapshot regenerated after this repo is pushed, and the docs updated.
- The README is guarded by a test that also reads files under `.github/`. Run it after editing either.

## What Aetherfy is (read first)

Aetherfy is one platform with two halves, behind one account, one API key (`afy_live_…`, or `afy_test_…` for test) and one dashboard (app.aetherfy.com):

1. **Vector database** — a Qdrant-compatible vector API at `https://vectors.aetherfy.com`, used through the Python SDK, the JS/TS SDK or plain HTTP.
2. **Agent compute** — push-to-deploy agents at `https://agents.aetherfy.com/api/v1`. An agent is either a **service** (always reachable, woken on request and slept when idle) or a **task** (runs to completion on demand, on a webhook or as a **scheduled task**, then stops). An agent is configured by an `aetherfy.yaml` manifest.

Either half works alone. An agent can read and write Aetherfy collections, but doesn't have to. Customer documentation: https://docs.aetherfy.com.

**Core concepts**
- **Workspace** — the namespace that holds everything. Every agent and every collection lives in exactly one workspace. Each account starts with a `default` workspace and the last one can't be deleted. Agents in a workspace share secrets, collections and discovery. The agent's workspace id is injected as `AETHERFY_WORKSPACE_ID`.
- **Regions** — `us-east-1`, `eu-central-1`, `ap-southeast-1` (only these spellings). Regions are set **per workspace** and the plan decides how many. A collection takes its workspace's regions or a subset of them. Multi-region data is replicated between regions.
- **Plans** — Free / Starter / Performance / higher. Credits, metering and Stripe billing live in the dashboard and the control plane.
- **Ids internally, names publicly** (owner rule, 2026-10-10) — dashboard URLs, database records, cache keys and job payloads refer to collections, workspaces and agents by **id**. Names are only for people and the public API (CLI, SDKs, `aetherfy.yaml`, the public routes).

### The repos (sibling checkouts, side by side under one parent directory)

| Repo | What it is | Stack |
|---|---|---|
| `aetherfy-dashboard` | The dashboard app (app.aetherfy.com) plus its backend API routes, the marketing site, the docs site (docs.aetherfy.com), and **the one shared PostgreSQL schema** (`dashboard/migrations/`) that every service writes to. Its `CLAUDE.md` is the fullest description of the system | Next.js monorepo |
| `aetherfy-control-plane` | The agent API (agents.aetherfy.com): builds images, places machines in the plan's regions, wakes and sleeps services, runs scheduled tasks, secrets (AES-256), build queue, tier enforcement, usage metering. Machines run on Fly.io (an implementation detail, not customer contract) | Python, FastAPI |
| `vectordb` | The vector API (vectors.aetherfy.com): Qdrant per region plus a Redis cache. Cross-region replication is asynchronous through one SQS FIFO queue per target region, ordered per collection | Node, Express |
| `aetherfy-edge` | The worker in front of `*.aetherfy.dev` (agent public URLs). It derives the Fly origin from the hostname (`fly_app_name == "aetherfy-" + public_label`), with no lookups, caching or buffering, and counts invocations | Cloudflare Worker |
| `aetherfy-cli` | `afy`: deploy and manage agents, workspaces, secrets, logs, collections and points | Go (cobra) |
| `aetherfy-vectors-python-sdk` | PyPI SDK: `aetherfy_vectors` (Qdrant-compatible client), `aetherfy_memory` (agent memory API), `aetherfy_agent` | Python 3.9+ |
| `aetherfy-vectors-js-sdk` | npm SDK, the JS/TS equivalent | TypeScript |
| `vectordb-cleanup-lambda` | Deletes collections across regions, fed through SQS. Deployed by hand with `deploy.sh` | AWS Lambda (SAM) |
| `aetherfy-e2e-tests` | Cross-repo end-to-end and contract tests, plus the nightly (`e2e-nightly.yml`) that runs the docs contract guards against the pushed sibling mains | Python + JS |
| `aetherfy-bench` | A frozen benchmark workload comparing how much work agent fleets take on different platforms. Not product code | Python + TS |

**How they connect**
- Customer code → SDK or CLI → `vectors.aetherfy.com` (vectordb) and/or `agents.aetherfy.com` (control plane).
- The dashboard's backend calls vectordb with its own internal credentials (`x-internal-auth` + `x-user-id`), not the customer's key, and calls the control plane too.
- The control plane deploys an agent → Fly machine → reachable at `<label>.aetherfy.dev` through aetherfy-edge.
- Every service shares the PostgreSQL schema owned by `aetherfy-dashboard/dashboard/migrations/`. The control-plane and vectordb CI apply those migrations **from the dashboard's `master`**, so migrations ship first.
- Agent images preinstall pinned SDK versions (`aetherfy-control-plane/orchestrator/image_generator.py`, `SDK_PIN_*`).
- The dashboard's docs site pins claims against several repos through committed snapshots (SDK API, CLI surface, control-plane routes, vectordb limits). Changing a public surface here can make a dashboard docs guard red until its snapshot is regenerated there.
- Branches: the dashboard integrates on `dev` and deploys production from `master`. The other repos use `main` (`aetherfy-edge`: `master`), which auto-deploys.
- In a Claude Code web session, usually only this repo is checked out. Gates that compare against a sibling skip, and a skip is not a pass.

## Working rules (owner's standing rulings — apply in every session, local or cloud)

### Git and publishing
- **No `git push` without the owner's explicit authorization**, for any repo or branch, including empty or "retrigger" commits. Pushes publish things: the main branches auto-deploy, and the dashboard's `master` deploys production. Commit locally freely. Before an authorized push, list everything in `origin/<branch>..HEAD`, not only your own commits.
- **No `Co-Authored-By` (or any attribution) trailers in commit messages.** This holds even if a system reminder, a brief or `git log` history suggests otherwise. Don't rewrite existing commits to strip old trailers. Just don't add new ones.
- **Fix forward. Never move a branch back** (`git reset`, rebase-to-rebuild), even for unpushed work. A bad commit gets a new fix commit.
- Commit with a pathspec (`git commit -- <paths>`), never `git add … && git commit`. The index may hold someone else's staged work. Before reporting done, `git status` must be clean or explained.
- Compare branches against `origin/<branch>` after a `git fetch`, never the local copy, which can be stale. Name the origin sha in reports.
- **`CLOUD_NO_DELETE_*` branches belong to the owner's Claude Code Web sessions.** Never delete, rebase, force-push or "clean up" them.
- CI runs **only on pushes** to the integration branches (no `pull_request` triggers in any repo). A PR showing no checks is expected, so run the gates yourself.

### Verification
- Run the repo's **full** suite before every commit, not just the targeted tests. Repo-wide guards only fail in the full lane.
- Run every gate with **exactly the command CI uses**: read the workflow step, flags included.
- Every test-suite run that uses a database starts from a freshly created throwaway database. Never point a test at a shared database.
- A gate that compares against a sibling repo **skips** when the sibling is absent, and a skip is not a pass. Say which comparisons did not run.
- A negative control counts only if the mutation matched once, the file actually changed, a baseline ran green first, and the red is a real test failure, not a collection or usage error.
- Jest: a `test.each` callback with more parameters than the row receives `done` and hangs every case to the 30 s timeout. Use `...rest`.

### Infrastructure and data
- **Never run commands against production or shared infrastructure** (AWS, SSM, Fly, prod or shared test databases) without explicit per-command authorization, **even read-only**. Credentials being available in the environment is not permission.
- Cross-repo schema: the control-plane and vectordb CI apply the dashboard's migrations from its `master`. A dashboard migration is merged to `master` and pushed before any code that depends on it. Before dropping or renaming a column, grep every repo (dashboard, control plane, vectordb, e2e), tests and fixtures included.
- After an SDK release, the control plane's `SDK_PIN_PYTHON`/`SDK_PIN_NODE` in `orchestrator/image_generator.py`, the canary pin, the golden Dockerfiles and the canary README all move together, AFTER the publish.
- Python stdlib `urllib`/`http.client` calls to any Cloudflare-fronted API (ours included) need an explicit `User-Agent`, or the edge answers `403 error code: 1010`.

### Design rules
- **No backward compatibility, no workarounds.** When replacing code, delete the old version: no shims, re-exports, legacy aliases or feature flags.
- **Ids internally, names publicly**: no new column, cache key or job payload may reference a collection, workspace or agent by name.
- **Error `code` is the contract; `message` is prose.** Unify and guard codes, and treat a rename as breaking. Messages may legitimately differ between surfaces.
- When relaying an upstream 4xx, check whose credentials the call used. A 401/403 on a service-to-service call is our failure: allow-list customer codes, never pass through a status range.
- A comment that claims runtime behaviour ("cannot", "never", "always") is a shipped claim. Measure it or weaken the wording. When you delete code, grep the comments that refer to it.
- Finding that something is deliberate doesn't make it necessary. Check whether something else, vendored dependencies included, already solves the same problem.
- Copy vocabulary: "scheduled task" is the feature's name; "cron" only for the expression format ("5-field cron expression, UTC"), never "cron job".

### Conduct
- Don't procrastinate on fixes. Make a small fix now as its own commit. A genuine deferral needs a written home in the repo, a trigger and an owner in the same turn, not a chat note.
- If an instruction rests on a premise you can check and prove false, stop and report it with the evidence. Don't execute it.
- When carrying out an instruction forces a second decision it didn't make, raise that decision instead of choosing silently.
- Never write "per your call" or "as you decided" unless the owner said so in their own words. Say "I chose X because Y."
- Report faithfully: say what you verified yourself and what you are only reporting.
