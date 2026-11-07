"""Quiz sessions: the interface the UI calls.

    start_quiz(library, mode, topic, n) -> (session_id, [QuestionView, ...])
    submit(session_id, question_id, answer) -> GradeResult

Full questions, answer keys included, live only in _SESSIONS on the server side.
The UI receives QuestionViews and, after each answer, a GradeResult. See
docs/architecure.md section 3.3.

Sessions are held in memory for the prototype, so they end when the process
does. That is acceptable for a single-user local app.
"""

import uuid

from agents.quiz_bot.documents import Library
from agents.quiz_bot.generator import Ask, generate_mc
from agents.quiz_bot.model import (
    GradeResult,
    Mode,
    QuestionType,
    QuestionView,
    QuizSession,
)

_SESSIONS: dict[str, QuizSession] = {}


class QuizError(Exception):
    """A request the Quiz Agent cannot serve. The message is display-safe."""


def start_quiz(
    library: Library,
    mode: Mode,
    topic: str | None = None,
    n: int = 5,
    ask: Ask | None = None,
) -> tuple[str, list[QuestionView]]:
    """Generate a quiz of up to n questions from the user's documents and
    return its views.

    Fewer than n questions come back if the material runs out or some slides
    fail generation. Zero questions is an error.
    """
    if not library.chunks:
        raise QuizError("Upload a course document first.")
    if mode is Mode.TOPIC:
        if not topic or not topic.strip():
            raise QuizError("Choose a topic for a topic quiz.")
        chunks = library.search(topic, n)
        if not chunks:
            raise QuizError("No course material matches that topic.")
    else:
        chunks = library.random_chunks(n)

    questions = [q for q in (generate_mc(c, ask) for c in chunks) if q is not None]
    if not questions:
        raise QuizError("Could not generate questions. Please try again.")

    session = QuizSession(id=uuid.uuid4().hex, questions=questions)
    _SESSIONS[session.id] = session
    return session.id, [q.view() for q in questions]


def submit(session_id: str, question_id: str, answer: str) -> GradeResult:
    """Grade one answer. For multiple choice, answer is the text of the chosen
    option.

    Multiple choice is graded in code, not by the model: it is exact, and an
    answer cannot argue its way into a better grade.

    An already-graded question returns its original result, so resubmitting
    cannot change a grade.
    """
    session = _SESSIONS.get(session_id)
    if session is None:
        raise QuizError("Quiz session not found.")
    question = next((q for q in session.questions if q.id == question_id), None)
    if question is None:
        raise QuizError("Question not found in this quiz.")
    if question_id in session.results:
        return session.results[question_id]

    if question.type is not QuestionType.MULTIPLE_CHOICE:
        raise QuizError("This question type cannot be graded yet.")
    if answer not in (question.options or []):
        raise QuizError("Choose one of the listed options.")

    correct = answer == question.answer
    verdict = "Correct." if correct else f"Incorrect. The answer is: {question.answer}."
    result = GradeResult(
        question_id=question.id,
        correct=correct,
        score=1.0 if correct else 0.0,
        correct_answer=question.answer,
        feedback=f"{verdict} {question.explanation}",
        citation=question.citation,
    )
    session.results[question_id] = result
    return result
