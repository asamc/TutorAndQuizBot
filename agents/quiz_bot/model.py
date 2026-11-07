"""Domain model for the Quiz Agent.

The split that matters is Question versus QuestionView. A Question holds the
answer key and stays on the server side; a QuestionView is the only form that is
ever handed to the UI. Keeping the answer out of the view structurally, rather
than relying on the UI to hide it, is the defence for REF-5 in
docs/threat-model.md section 3.2: there is nothing to leak.
"""

from dataclasses import dataclass, field
from enum import Enum


class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    OPEN_ENDED = "open_ended"


class Mode(str, Enum):
    RANDOM = "random"
    TOPIC = "topic"


@dataclass(frozen=True)
class Citation:
    """Where a question's answer comes from. Built from chunk metadata, never
    from model output, so the model cannot invent a source."""

    document_name: str
    page_number: int
    heading: str | None = None

    def __str__(self) -> str:
        location = f"p. {self.page_number}"
        if self.heading:
            location += f": {self.heading}"
        return f"{self.document_name}, {location}"


@dataclass(frozen=True)
class QuestionView:
    """What the user sees before answering. No answer, no explanation."""

    id: str
    type: QuestionType
    stem: str
    options: list[str] | None  # None for open-ended questions


@dataclass
class Question:
    """A full question, including its answer key. Server side only."""

    id: str
    type: QuestionType
    stem: str
    options: list[str] | None
    answer: str  # for multiple choice, the exact text of the correct option
    explanation: str
    citation: Citation

    def view(self) -> QuestionView:
        options = list(self.options) if self.options is not None else None
        return QuestionView(self.id, self.type, self.stem, options)


@dataclass
class GradeResult:
    question_id: str
    correct: bool
    score: float  # 0.0 to 1.0; open-ended grading may give partial credit
    correct_answer: str
    feedback: str
    citation: Citation


@dataclass
class QuizSession:
    id: str
    questions: list[Question]
    results: dict[str, GradeResult] = field(default_factory=dict)  # by question id
