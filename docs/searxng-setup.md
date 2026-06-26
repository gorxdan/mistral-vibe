# SearXNG Web Search

The `web_search` tool uses Mistral's web search by default. You can instead point it at a local [SearXNG](https://github.com/searxng/searxng) instance — a privacy-respecting metasearch engine that runs on your own machine — and let Chaton manage its lifecycle for you.

## Choosing a backend

During first-run onboarding, Chaton asks which web search backend to use:

- **Mistral web search** (default) — nothing to set up.
- **Local SearXNG** — Chaton records the SearXNG settings in your config. The container is started the next time you launch Chaton.

You can change this at any time by editing `[tools.web_search]` in your config (`~/.vibe/config.toml`, or a project `.vibe/config.toml`).

## Configuration

```toml
[tools.web_search]
searxng_url = "http://localhost:8888"   # set this to enable SearXNG
searxng_manage = true                    # let Chaton run the container
searxng_image = "searxng/searxng:latest"
searxng_container_name = "vibe-searxng"
searxng_port = 8888
searxng_autostart = true                 # start at session start if down
searxng_stop_on_exit = true              # stop on exit, only if Chaton started it
searxng_timeout = 30                      # per-request timeout (seconds)
searxng_health_timeout = 60              # total seconds to wait for a cold-starting container
searxng_disabled_engines = ["google"]    # fragile engines to disable in the managed container
```

`web_search` routes to SearXNG whenever `searxng_url` is set (or the `SEARXNG_URL` environment variable is present). With no SearXNG URL, it falls back to Mistral web search.

`searxng_timeout` caps a single search request; `searxng_health_timeout` is the separate, larger budget Chaton waits for a managed container to become healthy on a cold start. `searxng_disabled_engines` marks named engines as `disabled: true` in the managed container's `settings.yml` (then restarts it once). Commercial engines like `google`, `startpage`, `duckduckgo`, and `brave` are the most likely to rate-limit or CAPTCHA a self-hosted instance; disabling them shifts load to more tolerant engines.

## How lifecycle management works

When `searxng_manage` is enabled and a container engine — `docker` or `podman` — is on your `PATH`:

- **At session start**, if SearXNG is configured but not responding, Chaton starts it: it restarts an existing stopped `vibe-searxng` container, or creates one from `searxng_image` exposing `searxng_port`. If Chaton started it, it stops it again on exit (when `searxng_stop_on_exit` is true). A container that was already running — for example one you started yourself — is never claimed and never stopped.
- **During a search**, if the configured instance is unreachable, Chaton asks what to do:
  - **Start SearXNG** — launch the container and retry the search.
  - **Use Mistral this time** — run that one search through Mistral.
  - **Use Mistral, stop asking** — fall back to Mistral for the rest of the session.

When no container engine is available, Chaton cannot start SearXNG; it reports an actionable error with a copy-pasteable `run` command and falls back to Mistral.

## Managing SearXNG yourself

Set `searxng_manage = false` to use a SearXNG instance you run yourself (including a remote one). Chaton will query the configured `searxng_url` but will never start or stop a container.

To run SearXNG manually:

```bash
docker run -d --name vibe-searxng -p 127.0.0.1:8888:8080 searxng/searxng:latest
```

Bind to `127.0.0.1` (not the bare `-p 8888:8080`, which Docker exposes on `0.0.0.0` and so to your whole LAN). Chaton only ever talks to the instance over localhost.

SearXNG must allow the JSON response format, which Chaton uses to read results. Recent SearXNG images enable it by default; if you see empty results, ensure `json` is listed under `search.formats` in your SearXNG `settings.yml`.

## Rate limiting (the limiter) and remote instances

SearXNG ships a [limiter](https://docs.searxng.org/admin/searx.limiter.html) that uses [bot detection](https://docs.searxng.org/admin/searx.limiter.html) to rate-limit or block programmatic clients. Its job is to stop bots hammering the instance (which gets SearXNG itself blocked by upstream engines), and it is on by default on most public instances. Two things matter for Chaton:

- **Chaton identifies as a browser.** The limiter scores the default `python-httpx/*` User-Agent as a bot, so Chaton sends a browser User-Agent on both its readiness probe and its search requests. You do not need to configure anything for this.
- **Rate limits fall back, config errors don't.** If the instance returns `429 Too Many Requests` (the limiter throttling Chaton) or a `5xx` (overloaded), Chaton treats it as "down" and offers to fall back to Mistral web search for that request or the session. A `4xx` other than `429` (e.g. `404`) is treated as a configuration error and surfaced as a hard error rather than a silent fallback.

When pointing `searxng_url` at an instance you do not control (a public SearXNG), keep two caveats in mind:

1. **The limiter may still throttle you** under sustained use, regardless of User-Agent — the limiter also rate-limits by IP behaviour, not just headers. If searches frequently fall back to Mistral, run your own instance instead.
2. **JSON must be enabled.** Many public admins remove `json` from `search.formats` to deter automation; without it Chaton gets empty results or a `403`. Chaton can only patch `search.formats` on a container it manages, so a remote instance without JSON enabled will not work.

For a self-hosted single-user instance bound to localhost, the limiter adds little (there is only one client, seen as `127.0.0.1`), so leaving it off is reasonable; the value of `searxng_disabled_engines` is reducing rate-limiting from *upstream* engines, which the limiter does not address.

