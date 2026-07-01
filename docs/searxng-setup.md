# SearXNG Web Search

The `web_search` tool uses Mistral's web search by default. You can instead point it at a local [SearXNG](https://github.com/searxng/searxng) instance — a privacy-respecting metasearch engine that runs on your own machine — and let Mistral Vibe manage its lifecycle for you.

## Choosing a backend

During first-run onboarding, Mistral Vibe asks which web search backend to use:

- **Mistral web search** (default) — nothing to set up.
- **Local SearXNG** — Mistral Vibe records the SearXNG settings in your config. The container is started the next time you launch Mistral Vibe.

You can change this at any time by editing `[tools.web_search]` in your config (`~/.vibe/config.toml`, or a project `.vibe/config.toml`).

## Configuration

```toml
[tools.web_search]
searxng_url = "http://localhost:8888"   # set this to enable SearXNG
searxng_manage = true                    # let Mistral Vibe run the container
searxng_image = "searxng/searxng:latest"
searxng_container_name = "vibe-searxng"
searxng_port = 8888
searxng_autostart = true                 # start at session start if down
searxng_stop_on_exit = true              # stop on exit, only if Mistral Vibe started it
searxng_timeout = 30                      # per-request timeout (seconds)
searxng_health_timeout = 60              # total seconds to wait for a cold-starting container
# Force-enable these general-web engines in every managed container:
searxng_enabled_engines = ["bing", "duckduckgo", "startpage", "google", "qwant", "mojeek"]
searxng_disabled_engines = []             # engines to force-disable (overrides enabled_engines)
```

`web_search` routes to SearXNG whenever `searxng_url` is set (or the `SEARXNG_URL` environment variable is present). With no SearXNG URL, it falls back to Mistral web search.

`searxng_timeout` caps a single search request; `searxng_health_timeout` is the separate, larger budget Mistral Vibe waits for a managed container to become healthy on a cold start.

### Engine selection

The upstream SearXNG image ships most general-web engines (`google`, `bing`, `duckduckgo`, `startpage`, `qwant`, `mojeek`) as `disabled: true`, leaving only `brave` serving a plain query. Mistral Vibe's default `searxng_enabled_engines` force-enables a broad set in every container it manages so that when one engine rate-limits itself there are still others returning results — no single point of failure. The reconciliation runs once at session start (and after Mistral Vibe starts a fresh container): it removes `disabled: true` from each named engine in `settings.yml` and restarts once, then is a no-op on subsequent starts.

`searxng_disabled_engines` is the inverse — engines to force *off*. It overrides `searxng_enabled_engines`: an engine named in both lists ends disabled (an explicit disable beats the default enable). Set `searxng_enabled_engines = []` to opt out of force-enabling any engine and accept the upstream defaults.

## How lifecycle management works

When `searxng_manage` is enabled and a container engine — `docker` or `podman` — is on your `PATH`:

- **At session start**, if SearXNG is configured but not responding, Mistral Vibe starts it: it restarts an existing stopped `vibe-searxng` container, or creates one from `searxng_image` exposing `searxng_port`. If Mistral Vibe started it, it stops it again on exit (when `searxng_stop_on_exit` is true). A container that was already running — for example one you started yourself — is never claimed and never stopped.
- **During a search**, if the configured instance is unreachable, Mistral Vibe asks what to do:
  - **Start SearXNG** — launch the container and retry the search.
  - **Use Mistral this time** — run that one search through Mistral.
  - **Use Mistral, stop asking** — fall back to Mistral for the rest of the session.

When no container engine is available, Mistral Vibe cannot start SearXNG; it reports an actionable error with a copy-pasteable `run` command and falls back to Mistral.

## Managing SearXNG yourself

Set `searxng_manage = false` to use a SearXNG instance you run yourself (including a remote one). Mistral Vibe will query the configured `searxng_url` but will never start or stop a container.

To run SearXNG manually:

```bash
docker run -d --name vibe-searxng -p 127.0.0.1:8888:8080 searxng/searxng:latest
```

Bind to `127.0.0.1` (not the bare `-p 8888:8080`, which Docker exposes on `0.0.0.0` and so to your whole LAN). Mistral Vibe only ever talks to the instance over localhost.

SearXNG must allow the JSON response format, which Mistral Vibe uses to read results. Recent SearXNG images enable it by default; if you see empty results, ensure `json` is listed under `search.formats` in your SearXNG `settings.yml`.

## Rate limiting (the limiter) and remote instances

SearXNG ships a [limiter](https://docs.searxng.org/admin/searx.limiter.html) that uses [bot detection](https://docs.searxng.org/admin/searx.limiter.html) to rate-limit or block programmatic clients. Its job is to stop bots hammering the instance (which gets SearXNG itself blocked by upstream engines), and it is on by default on most public instances. Two things matter for Mistral Vibe:

- **Mistral Vibe identifies as a browser.** The limiter scores the default `python-httpx/*` User-Agent as a bot, so Mistral Vibe sends a browser User-Agent on both its readiness probe and its search requests. You do not need to configure anything for this.
- **Rate limits fall back, config errors don't.** If the instance returns `429 Too Many Requests` (the limiter throttling Mistral Vibe) or a `5xx` (overloaded), Mistral Vibe treats it as "down" and offers to fall back to Mistral web search for that request or the session. A `4xx` other than `429` (e.g. `404`) is treated as a configuration error and surfaced as a hard error rather than a silent fallback.

When pointing `searxng_url` at an instance you do not control (a public SearXNG), keep two caveats in mind:

1. **The limiter may still throttle you** under sustained use, regardless of User-Agent — the limiter also rate-limits by IP behaviour, not just headers. If searches frequently fall back to Mistral, run your own instance instead.
2. **JSON must be enabled.** Many public admins remove `json` from `search.formats` to deter automation; without it Mistral Vibe gets empty results or a `403`. Mistral Vibe can only patch `search.formats` on a container it manages, so a remote instance without JSON enabled will not work.

For a self-hosted single-user instance bound to localhost, the limiter adds little (there is only one client, seen as `127.0.0.1`), so leaving it off is reasonable; the value of `searxng_disabled_engines` is reducing rate-limiting from *upstream* engines, which the limiter does not address.
