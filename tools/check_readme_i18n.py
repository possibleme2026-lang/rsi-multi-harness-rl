# Copyright 2026 The rsi-multi-harness-rl Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check that the two README files stay in sync.

The failure this guards against is specific: someone edits the English README,
forgets the Chinese one, and the two drift until a reader cannot tell which is
current. Structure is checked, not prose, because the two files are deliberately
different languages.

What must match: section count and nesting, code-block count, table-row count, the
external links, and the content of machine-readable blocks (bibtex, shell commands).
What may differ: every human-readable string.

Run with ``--write`` is not offered: this script has nothing to write. A drift is a
real drift and has to be fixed in one of the two files by hand.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGLISH = REPO_ROOT / "README.md"
CHINESE = REPO_ROOT / "README.zh-CN.md"

HEADING = re.compile(r"^(#{2,6})\s+(.*)$", re.MULTILINE)
CODE_FENCE = re.compile(r"^```(\w*)", re.MULTILINE)
TABLE_ROW = re.compile(r"^\|", re.MULTILINE)
EXTERNAL_LINK = re.compile(r"https?://[^\s)\"'<>]+")
INTERNAL_LINK = re.compile(r"\]\(((?!https?://)[^)#][^)]*)\)")
FENCED_BLOCK = re.compile(r"^```(\w*)\n(.*?)^```$", re.MULTILINE | re.DOTALL)

#: Blocks whose executable content must be identical in both files.
#: ``python`` is here because a code sample is not prose: translating a comment is fine,
#: changing an API call is drift. ``bash`` and ``toml`` likewise.
MACHINE_READABLE_LANGUAGES = ("bibtex", "bash", "toml", "python")

#: Languages whose blocks may carry translated comments. The code must match, but a ``#``
#: line is read by a human and is expected to be in the file's own language. Comparing the
#: blocks whole would force one README to carry comments in the other's language, which is
#: worse than the drift this script exists to catch.
COMMENT_PREFIXES = {"bash": "#", "toml": "#", "python": "#"}


class Drift(Exception):
    """The two README files have diverged."""


def read(path: Path) -> str:
    """Read a README, failing loudly when it is missing."""
    if not path.is_file():
        raise Drift(f"{path.relative_to(REPO_ROOT)} does not exist")
    return path.read_text(encoding="utf-8")


def heading_shape(text: str) -> list[tuple[int, int]]:
    """Return each heading's depth and its position in sequence.

    Titles are compared by depth only, never by text: the files are in different
    languages, so comparing text would fail on every heading.
    """
    return [(len(match.group(1)), index) for index, match in enumerate(HEADING.finditer(text))]


def compare_scalar(label: str, english: int, chinese: int) -> list[str]:
    """Compare two counts and describe the difference."""
    if english == chinese:
        return []
    return [f"{label}: README.md has {english}, README.zh-CN.md has {chinese}"]


def check_headings(english: str, chinese: str) -> list[str]:
    """Section nesting must match, so neither file is missing a section."""
    en_shape = heading_shape(english)
    zh_shape = heading_shape(chinese)
    if len(en_shape) != len(zh_shape):
        return [f"heading count: README.md has {len(en_shape)}, README.zh-CN.md has {len(zh_shape)}"]
    problems: list[str] = []
    for index, ((en_depth, _), (zh_depth, _)) in enumerate(zip(en_shape, zh_shape, strict=True)):
        if en_depth != zh_depth:
            problems.append(f"heading #{index + 1}: depth {en_depth} in README.md, {zh_depth} in README.zh-CN.md")
    return problems


def check_external_links(english: str, chinese: str) -> list[str]:
    """Both files must cite the same sources, so neither omits a reference."""
    en_links = {link.rstrip(".,;") for link in EXTERNAL_LINK.findall(english)}
    zh_links = {link.rstrip(".,;") for link in EXTERNAL_LINK.findall(chinese)}
    problems: list[str] = []
    for link in sorted(en_links - zh_links):
        problems.append(f"link only in README.md: {link}")
    for link in sorted(zh_links - en_links):
        problems.append(f"link only in README.zh-CN.md: {link}")
    return problems


def check_internal_links(english: str, chinese: str) -> list[str]:
    """Every relative link in either file must resolve on disk."""
    problems: list[str] = []
    for path, text in ((ENGLISH, english), (CHINESE, chinese)):
        for target in INTERNAL_LINK.findall(text):
            resolved = (REPO_ROOT / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                problems.append(f"{path.name} links to a missing path: {target}")
    return problems


def strip_comments(language: str, body: str) -> str:
    """Drop comments from a block, keeping the text that actually executes.

    Handles both a whole-line comment and a trailing one, because a shell example
    conventionally annotates a command on the same line. Only the executable text is
    compared; the annotation is prose and belongs in the file's own language.

    Only for languages where ``#`` is prose. A bibtex block keeps everything: its ``%`` is
    rare enough that treating it as a comment would hide more than it reveals.
    """
    prefix = COMMENT_PREFIXES.get(language)
    if prefix is None:
        return body
    kept: list[str] = []
    for line in body.splitlines():
        code = line.split(prefix, 1)[0].rstrip() if prefix in line else line.rstrip()
        if code:
            kept.append(code)
    return "\n".join(kept)


def check_machine_readable_blocks(english: str, chinese: str) -> list[str]:
    """Blocks that are not prose must be identical, so citations cannot drift."""
    problems: list[str] = []
    for language in MACHINE_READABLE_LANGUAGES:
        en_blocks = [
            strip_comments(language, body) for lang, body in FENCED_BLOCK.findall(english) if lang == language
        ]
        zh_blocks = [
            strip_comments(language, body) for lang, body in FENCED_BLOCK.findall(chinese) if lang == language
        ]
        if len(en_blocks) != len(zh_blocks):
            problems.append(
                f"{language} block count: README.md has {len(en_blocks)}, README.zh-CN.md has {len(zh_blocks)}"
            )
            continue
        for index, (en_body, zh_body) in enumerate(zip(en_blocks, zh_blocks, strict=True)):
            if en_body != zh_body:
                problems.append(f"{language} block #{index + 1} differs between the two files")
    return problems


def check_language_switch(english: str, chinese: str) -> list[str]:
    """Each file must link to the other, or a reader is stranded."""
    problems: list[str] = []
    if "README.zh-CN.md" not in english:
        problems.append("README.md does not link to README.zh-CN.md")
    if "README.md" not in chinese:
        problems.append("README.zh-CN.md does not link to README.md")
    return problems


def main() -> int:
    """Run every check and report all drift at once."""
    try:
        english = read(ENGLISH)
        chinese = read(CHINESE)
    except Drift as exc:
        print(f"FAIL: {exc}")
        return 1

    problems: list[str] = []
    problems.extend(check_headings(english, chinese))
    problems.extend(compare_scalar("code blocks", len(CODE_FENCE.findall(english)), len(CODE_FENCE.findall(chinese))))
    problems.extend(compare_scalar("table rows", len(TABLE_ROW.findall(english)), len(TABLE_ROW.findall(chinese))))
    problems.extend(check_external_links(english, chinese))
    problems.extend(check_internal_links(english, chinese))
    problems.extend(check_machine_readable_blocks(english, chinese))
    problems.extend(check_language_switch(english, chinese))

    if problems:
        print("README i18n drift detected:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nFix by updating whichever file is behind. Structure and machine-readable")
        print("blocks must match; prose is expected to differ.")
        return 1

    print("README.md and README.zh-CN.md are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
