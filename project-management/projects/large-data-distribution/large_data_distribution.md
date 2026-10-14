# Large Test/Parity Data Distribution

## Problem
Test + parity fixtures under `tests/data/` are ~240 MB today and committed directly into
git. Planned additions (several hundred MB more) would bloat every clone and permanently
inflate git history. We want the data available to CI and to developers who need it, without
carrying it in the main repo's git objects.

Key constraint: git history **already** contains the current fixtures, so any "move it out"
option only stops *future* growth unless we also rewrite history (`git filter-repo` / BFG) —
a separate, disruptive step (rewrites SHAs, breaks open branches/PRs). Decide that separately.

## Options

| # | Option | How it works | Cost | Pros | Cons |
|---|--------|--------------|------|------|------|
| 1 | **GitHub Release assets** + fetch script | Tarball data, attach to a tagged Release; `scripts/fetch_test_data.py` downloads + checksums on demand | Free (public repo); ≤2 GB/file, no bandwidth charge | No new account/tooling; `gh release download` is trivial; versioned by tag | Manual re-upload on data change; assets are opaque blobs (no per-file diff/dedup) |
| 2 | **Git LFS** | Large files tracked by pointer; real bytes on LFS server | GitHub LFS bills storage + bandwidth past 1 GB/1 GB free | Transparent (`git clone` just works); per-file versioning | Quota is small and **paid**; bandwidth cost scales with CI clones; sharp edges on forks |
| 3 | **Separate data repo (submodule)** | `tests/data-large` is its own git repo, pinned by commit | Free unless *it* uses LFS | Clean separation; pinned + reproducible; main repo stays lean | Submodules are a known UX footgun (`--recursive`, detached HEAD); still git-history growth, just in the other repo |
| 4 | **External object storage** (S3 / Backblaze B2 / Cloudflare R2) + fetch script | Data in a bucket; script pulls by key + checksum | Storage + egress $ (R2 = free egress); needs credentials | Scales to any size; fast; CI-friendly | Ops overhead (bucket, keys, secrets in CI); not free; no built-in versioning |
| 5 | **Zenodo / OSF / Figshare** (research archive) | Upload dataset, get a DOI + stable URL; fetch script downloads | Free | Citable DOI, good for scientific reproducibility; versioned records; permanent | Upload/publish friction; large-file limits per record; not built for frequent churn |
| 6 | **DataLad / git-annex** | git tracks lightweight annex pointers; content lives on any remote (S3, http, etc.) | Free tooling; pay only for the remote you pick | Neuroimaging-native (BIDS/DataLad ecosystem); per-file versioning + pluggable remotes | Steepest learning curve; contributors must install datalad/git-annex |

## Recommendation (for discussion)
Start with **Option 1 (Release assets + a `fetch_test_data.py` script)** — lowest friction,
free, no new accounts, and CI just calls the script. Pin each Release tag + a SHA256 manifest
in-repo so fetches are reproducible. If churn becomes frequent or the data grows past what a
few tarballs can manage, graduate to **Option 6 (DataLad)** — it fits the neuroimaging
ecosystem and gives real per-file versioning over a cheap backend (R2/B2).

Whichever we pick, keep a small committed manifest (paths + SHA256 + source URL) so a fetch
is verifiable and CI fails loudly on drift. History rewrite to reclaim the existing ~240 MB
is a separate decision — do it only if/when clone size actually hurts.

## Status
- [ ] Decide primary option (default proposal: #1 now, #6 later)
- [ ] Prototype `scripts/fetch_test_data.py` (download + SHA256 verify + extract into `tests/data/`)
- [ ] Wire CI to fetch on demand; keep a tiny always-committed smoke subset for fast PR runs
- [ ] Decide separately whether to rewrite history to shrink existing clones
