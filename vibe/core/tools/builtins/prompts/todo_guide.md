# todo — item schema and example

## Item schema
- `id`: unique string (e.g. "1", "task-a")
- `content`: task description
- `status`: `pending` | `in_progress` | `completed` | `cancelled`
- `priority`: `high` | `medium` | `low`

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
