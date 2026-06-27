Use `skill` to load specialized skills that provide domain-specific instructions and workflows.

When to use: task matches an available skill | user names a skill (e.g., "use the review skill") | you need a skill's bundled workflows, templates, or scripts.

How it works: call `skill` with the skill's `name` from `<available_skills>`; it returns the full instructions plus a list of bundled files; follow them step by step — you are the executor.

Notes: bundled file paths are relative to the skill's base directory; each skill loads once per invocation — re-invoke to reload.
