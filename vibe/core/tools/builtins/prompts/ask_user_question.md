Use `ask_user_question` to gather information when you need clarification, want to validate an assumption, or must choose between approaches. **Don't hesitate** — better to ask than guess wrong. Ask early, before doing significant work.

## Question fields
- `question`: the full question text — specific and clear.
- `header`: short chip label, **max 12 chars** (e.g. "Auth", "Database").
- `options`: **2-4** choices (an "Other" free-text option is added automatically). Each option has `label` (1-5 words) and `description` (the choice's meaning/tradeoff).
- `multi_select`: `true` if the user may pick several (default `false`) — use when choices aren't mutually exclusive.

You may pass **1-4** questions per call (multiple render as tabs).

## Tips
- Put the recommended option first and add "(Recommended)" to its label.
- Use descriptive headers; keep descriptions concise but informative about tradeoffs.

## Example
```json
{
  "questions": [{
    "question": "Which authentication method should we use?",
    "header": "Auth",
    "options": [
      {"label": "JWT tokens (Recommended)", "description": "Stateless, scalable, works well with APIs"},
      {"label": "Session cookies", "description": "Traditional, requires session storage"},
      {"label": "OAuth 2.0", "description": "Third-party auth, more complex setup"}
    ],
    "multi_select": false
  }]
}
```
