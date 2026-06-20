Use the `todo` tool to track a task list.

- `action: "read"` — view the current list.
- `action: "write"` with `todos` — replace the ENTIRE list (any item you omit is dropped, so resend everything you want to keep).

## Item schema
- `id`: unique string (e.g. "1", "task-a")
- `content`: task description
- `status`: `pending` | `in_progress` | `completed` | `cancelled`
- `priority`: `high` | `medium` | `low`

## When to use
Proactively for: multi-step tasks (3+ steps), non-trivial work needing planning, multiple user-provided tasks, tracking ongoing progress. Capture requirements as soon as you get them; mark a task `in_progress` BEFORE starting it and `completed` immediately after.

Skip for: single straightforward tasks, trivial operations (<3 steps), purely conversational requests.

## Rules
- Exactly ONE task `in_progress` at a time.
- Mark `completed` only when FULLY done — never if tests fail, work is partial, or errors are unresolved. When blocked, keep it `in_progress` and add a new task for the blocker.
- Create specific, actionable items; remove irrelevant ones entirely.

## Example
```json
{
  "action": "write",
  "todos": [
    {"id": "1", "content": "Add dark mode toggle", "status": "in_progress", "priority": "high"},
    {"id": "2", "content": "Run tests and verify build", "status": "pending", "priority": "medium"}
  ]
}
```
