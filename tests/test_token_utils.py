from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_book_batch_writer.token_utils import total_tokens_from_message


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            SimpleNamespace(
                usage_metadata={
                    "input_tokens": 120,
                    "output_tokens": 80,
                    "total_tokens": 200,
                },
                response_metadata={},
            ),
            200,
        ),
        (
            SimpleNamespace(
                usage_metadata=None,
                response_metadata={
                    "usage": {"input_tokens": 120, "output_tokens": 80}
                },
            ),
            200,
        ),
        (
            SimpleNamespace(
                usage_metadata=None,
                response_metadata={
                    "usage_metadata": {
                        "prompt_token_count": 150,
                        "candidates_token_count": 50,
                        "total_token_count": 200,
                    }
                },
            ),
            200,
        ),
        (
            SimpleNamespace(
                usage_metadata=None,
                response_metadata={
                    "prompt_eval_count": 100,
                    "eval_count": 75,
                },
            ),
            175,
        ),
    ],
)
def test_total_tokens_from_provider_metadata(message, expected: int) -> None:
    assert total_tokens_from_message(message) == expected
