from __future__ import annotations

from datetime import datetime
from enum import StrEnum
import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, Tag


class WorkflowEventType(StrEnum):
    WORKFLOW_EXECUTION_COMPLETED = "WORKFLOW_EXECUTION_COMPLETED"
    WORKFLOW_EXECUTION_FAILED = "WORKFLOW_EXECUTION_FAILED"
    WORKFLOW_EXECUTION_CANCELED = "WORKFLOW_EXECUTION_CANCELED"
    CUSTOM_TASK_STARTED = "CUSTOM_TASK_STARTED"
    CUSTOM_TASK_IN_PROGRESS = "CUSTOM_TASK_IN_PROGRESS"
    CUSTOM_TASK_COMPLETED = "CUSTOM_TASK_COMPLETED"
    CUSTOM_TASK_FAILED = "CUSTOM_TASK_FAILED"
    CUSTOM_TASK_TIMED_OUT = "CUSTOM_TASK_TIMED_OUT"
    CUSTOM_TASK_CANCELED = "CUSTOM_TASK_CANCELED"


class WorkflowExecutionStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    TERMINATED = "TERMINATED"
    CONTINUED_AS_NEW = "CONTINUED_AS_NEW"
    TIMED_OUT = "TIMED_OUT"


class JSONPatchBase(BaseModel):
    path: str
    value: Any = None


class JSONPatchAdd(JSONPatchBase):
    op: Literal["add"] = "add"


class JSONPatchReplace(JSONPatchBase):
    op: Literal["replace"] = "replace"


class JSONPatchRemove(JSONPatchBase):
    op: Literal["remove"] = "remove"


class JSONPatchAppend(JSONPatchBase):
    op: Literal["append"] = "append"
    value: str = ""


JSONPatch = Annotated[
    Annotated[JSONPatchAppend, Tag("append")]
    | Annotated[JSONPatchAdd, Tag("add")]
    | Annotated[JSONPatchReplace, Tag("replace")]
    | Annotated[JSONPatchRemove, Tag("remove")],
    Discriminator("op"),
]


class JSONPatchPayload(BaseModel):
    type: Literal["json_patch"] = "json_patch"
    value: list[JSONPatch] = Field(default_factory=list)


class JSONPayload(BaseModel):
    type: Literal["json"] = "json"
    value: Any = None


Payload = Annotated[
    Annotated[JSONPayload, Tag("json")]
    | Annotated[JSONPatchPayload, Tag("json_patch")],
    Discriminator("type"),
]


class Failure(BaseModel):
    message: str


class BaseEvent(BaseModel):
    event_id: str
    event_timestamp: int = 0
    root_workflow_exec_id: str = ""
    parent_workflow_exec_id: str | None = None
    workflow_exec_id: str = ""
    workflow_run_id: str = ""
    workflow_name: str = ""


class WorkflowExecutionFailedAttributes(BaseModel):
    task_id: str = ""
    failure: Failure


class WorkflowExecutionCanceledAttributes(BaseModel):
    task_id: str = ""
    reason: str | None = None


class WorkflowExecutionCompletedAttributes(BaseModel):
    task_id: str = ""
    result: JSONPayload = Field(default_factory=lambda: JSONPayload(value=None))


class CustomTaskStartedAttributes(BaseModel):
    custom_task_id: str
    custom_task_type: str
    payload: JSONPayload = Field(default_factory=lambda: JSONPayload(value=None))


class CustomTaskInProgressAttributes(BaseModel):
    custom_task_id: str
    custom_task_type: str
    payload: Payload


class CustomTaskCompletedAttributes(BaseModel):
    custom_task_id: str
    custom_task_type: str
    payload: JSONPayload = Field(default_factory=lambda: JSONPayload(value=None))


class CustomTaskFailedAttributes(BaseModel):
    custom_task_id: str
    custom_task_type: str
    failure: Failure


class CustomTaskTimedOutAttributes(BaseModel):
    custom_task_id: str
    custom_task_type: str
    timeout_type: str | None = None


class CustomTaskCanceledAttributes(BaseModel):
    custom_task_id: str
    custom_task_type: str
    reason: str | None = None


class WorkflowExecutionCompleted(BaseEvent):
    event_type: Literal[WorkflowEventType.WORKFLOW_EXECUTION_COMPLETED] = (
        WorkflowEventType.WORKFLOW_EXECUTION_COMPLETED
    )
    attributes: WorkflowExecutionCompletedAttributes


class WorkflowExecutionFailed(BaseEvent):
    event_type: Literal[WorkflowEventType.WORKFLOW_EXECUTION_FAILED] = (
        WorkflowEventType.WORKFLOW_EXECUTION_FAILED
    )
    attributes: WorkflowExecutionFailedAttributes


class WorkflowExecutionCanceled(BaseEvent):
    event_type: Literal[WorkflowEventType.WORKFLOW_EXECUTION_CANCELED] = (
        WorkflowEventType.WORKFLOW_EXECUTION_CANCELED
    )
    attributes: WorkflowExecutionCanceledAttributes


class CustomTaskStarted(BaseEvent):
    event_type: Literal[WorkflowEventType.CUSTOM_TASK_STARTED] = (
        WorkflowEventType.CUSTOM_TASK_STARTED
    )
    attributes: CustomTaskStartedAttributes


class CustomTaskInProgress(BaseEvent):
    event_type: Literal[WorkflowEventType.CUSTOM_TASK_IN_PROGRESS] = (
        WorkflowEventType.CUSTOM_TASK_IN_PROGRESS
    )
    attributes: CustomTaskInProgressAttributes


class CustomTaskCompleted(BaseEvent):
    event_type: Literal[WorkflowEventType.CUSTOM_TASK_COMPLETED] = (
        WorkflowEventType.CUSTOM_TASK_COMPLETED
    )
    attributes: CustomTaskCompletedAttributes


class CustomTaskFailed(BaseEvent):
    event_type: Literal[WorkflowEventType.CUSTOM_TASK_FAILED] = (
        WorkflowEventType.CUSTOM_TASK_FAILED
    )
    attributes: CustomTaskFailedAttributes


class CustomTaskTimedOut(BaseEvent):
    event_type: Literal[WorkflowEventType.CUSTOM_TASK_TIMED_OUT] = (
        WorkflowEventType.CUSTOM_TASK_TIMED_OUT
    )
    attributes: CustomTaskTimedOutAttributes


class CustomTaskCanceled(BaseEvent):
    event_type: Literal[WorkflowEventType.CUSTOM_TASK_CANCELED] = (
        WorkflowEventType.CUSTOM_TASK_CANCELED
    )
    attributes: CustomTaskCanceledAttributes


class WorkflowExecutionWithoutResultResponse(BaseModel):
    workflow_name: str
    execution_id: str
    parent_execution_id: str | None = None
    root_execution_id: str = ""
    status: WorkflowExecutionStatus | None = None
    start_time: datetime
    end_time: datetime | None = None
    total_duration_ms: int | None = None


class WorkflowExecutionListResponse(BaseModel):
    executions: list[WorkflowExecutionWithoutResultResponse] = Field(
        default_factory=list
    )
    next_page_token: str | None = None


class SignalWorkflowResponse(BaseModel):
    message: str = "Signal accepted"


class UpdateWorkflowResponse(BaseModel):
    update_name: str = ""
    result: Any = None


class StreamEventWorkflowContext(BaseModel):
    namespace: str = ""
    workflow_name: str = ""
    workflow_exec_id: str = ""
    parent_workflow_exec_id: str | None = None
    root_workflow_exec_id: str | None = None


class StreamEvent(BaseModel):
    stream: str = ""
    timestamp_unix_nano: int = Field(default_factory=time.time_ns)
    data: Any
    workflow_context: StreamEventWorkflowContext = Field(
        default_factory=StreamEventWorkflowContext
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    broker_sequence: int | None = None


class StreamEventsQueryParams(BaseModel):
    workflow_exec_id: str = ""
    start_seq: int = 0
