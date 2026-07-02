Use `ask_user_question` to gather information when you need clarification, want to validate an assumption, or must choose between approaches. **Don't hesitate** — better to ask than guess wrong. Ask early, before doing significant work.

Each question: `question` text, a `header` chip (max 12 chars), and **2-4** `options` (an "Other" free-text option is added automatically). Put the recommended option first with "(Recommended)" in its label. Set `multi_select: true` when choices aren't mutually exclusive. You may pass **1-4** questions per call (multiple render as tabs).

Full field reference and a worked example: `tool-guides` skill.
