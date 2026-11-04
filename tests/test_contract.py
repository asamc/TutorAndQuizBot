"""The ingest types must match the contract published to the UI.

docs/ingest-interface.md section 3 defines Stage, Outcome, Progress, Result and
Flag, and the UI is built against that document. ingest/model.py transcribes
them. If the two drift, the contract is worthless and the drift is silent --
nothing else in the project would notice.

So rather than duplicating the expected values here, these tests parse the
Python block out of the document and compare it against the live classes. The
document stays the single source of truth.
"""

import ast
import pathlib
import re

import pytest

from ingest.model import Flag, Outcome, Progress, Result, Stage

DOC_PATH = pathlib.Path(__file__).resolve().parents[1] / "docs" / "ingest-interface.md"

# Types the document is expected to define, split by how they are compared.
EXPECTED_ENUMS = {"Stage", "Outcome"}
EXPECTED_DATACLASSES = {"Progress", "Result", "Flag"}


def _published_source() -> str:
    """The first ```python block in the interface doc."""
    blocks = re.findall(r"```python\n(.*?)```", DOC_PATH.read_text(), re.S)
    assert blocks, f"no python block found in {DOC_PATH.name}"
    return blocks[0]


def _parse_published() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (enum values, dataclass field names) as declared in the doc.

    A class is read as an enum when its body assigns string constants, and as a
    dataclass when its body carries annotated fields.
    """
    enums: dict[str, list[str]] = {}
    fields: dict[str, list[str]] = {}

    for node in ast.parse(_published_source()).body:
        if not isinstance(node, ast.ClassDef):
            continue
        values = [
            s.value.value
            for s in node.body
            if isinstance(s, ast.Assign) and isinstance(s.value, ast.Constant)
        ]
        names = [s.target.id for s in node.body if isinstance(s, ast.AnnAssign)]
        if values:
            enums[node.name] = values
        if names:
            fields[node.name] = names

    return enums, fields


DOC_ENUMS, DOC_FIELDS = _parse_published()


def test_doc_still_defines_every_contract_type():
    """Guard the guard.

    If the doc is restructured so a type disappears from the code block, the
    comparisons below would pass by comparing nothing. This fails first instead.
    """
    assert set(DOC_ENUMS) == EXPECTED_ENUMS
    assert set(DOC_FIELDS) == EXPECTED_DATACLASSES


@pytest.mark.parametrize("name, cls", [("Stage", Stage), ("Outcome", Outcome)])
def test_enum_members_match_doc(name, cls):
    """Same members, same string values, same order."""
    assert [member.value for member in cls] == DOC_ENUMS[name]


@pytest.mark.parametrize(
    "name, cls", [("Progress", Progress), ("Result", Result), ("Flag", Flag)]
)
def test_dataclass_fields_match_doc(name, cls):
    """Same field names, same order. Order matters: these are constructed
    positionally in the doc's examples."""
    assert list(cls.__dataclass_fields__) == DOC_FIELDS[name]
