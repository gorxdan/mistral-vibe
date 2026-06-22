from __future__ import annotations

from typing import Annotated, Any

from pydantic import Discriminator, Tag

from vibe.core.nuage.models import (
    CustomTaskCanceled,
    CustomTaskCompleted,
    CustomTaskFailed,
    CustomTaskInProgress,
    CustomTaskStarted,
    CustomTaskTimedOut,
    WorkflowEventType,
    WorkflowExecutionCanceled,
    WorkflowExecutionCompleted,
    WorkflowExecutionFailed,
)


def _get_event_type_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        event_type_val = v.get("event_type", "")
        if isinstance(event_type_val, WorkflowEventType):
            return event_type_val.value
        return str(event_type_val)

    event_type_attr = getattr(v, "event_type", "")
    if isinstance(event_type_attr, WorkflowEventType):
        return event_type_attr.value
    return str(event_type_attr)


WorkflowEvent = Annotated[
    Annotated[
        WorkflowExecutionCompleted, Tag(WorkflowEventType.WORKFLOW_EXECUTION_COMPLETED)
    ]
    | Annotated[
        WorkflowExecutionFailed, Tag(WorkflowEventType.WORKFLOW_EXECUTION_FAILED)
    ]
    | Annotated[
        WorkflowExecutionCanceled, Tag(WorkflowEventType.WORKFLOW_EXECUTION_CANCELED)
    ]
    | Annotated[CustomTaskStarted, Tag(WorkflowEventType.CUSTOM_TASK_STARTED)]
    | Annotated[CustomTaskInProgress, Tag(WorkflowEventType.CUSTOM_TASK_IN_PROGRESS)]
    | Annotated[CustomTaskCompleted, Tag(WorkflowEventType.CUSTOM_TASK_COMPLETED)]
    | Annotated[CustomTaskFailed, Tag(WorkflowEventType.CUSTOM_TASK_FAILED)]
    | Annotated[CustomTaskTimedOut, Tag(WorkflowEventType.CUSTOM_TASK_TIMED_OUT)]
    | Annotated[CustomTaskCanceled, Tag(WorkflowEventType.CUSTOM_TASK_CANCELED)],
    Discriminator(_get_event_type_discriminator),
]
