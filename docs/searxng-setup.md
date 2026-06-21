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
searxng_timeout = 30                      # request + health-check timeout (seconds)
```

`web_search` routes to SearXNG whenever `searxng_url` is set (or the `SEARXNG_URL` environment variable is present). With no SearXNG URL, it falls back to Mistral web search.

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
docker run -d --name vibe-searxng -p 8888:8080 searxng/searxng:latest
```

SearXNG must allow the JSON response format, which Chaton uses to read results. Recent SearXNG images enable it by default; if you see empty results, ensure `json` is listed under `search.formats` in your SearXNG `settings.yml`.
