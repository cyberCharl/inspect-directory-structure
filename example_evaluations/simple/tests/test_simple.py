"""Template evaluation tests.

We recommend you test your evaluation in the following ways:

- Test that the dataset is created correctly.
- Test that the evaluation runs successfully end-to-end in a sandbox and that a correct
  submission is marked as correct.
- Test that the evaluation runs successfully end-to-end in a sandbox and that an
  incorrect submission is marked as incorrect.

End-to-end tests should ideally be run using a mocked language model. If you need to use a real language model, you should mark the test with `calls_llm` (see below).

If the evaluation has a custom scorer, you should write unit tests for it. Add any other tests you see fit.

You should use the following pytest markers to categorize your tests:

@pytest.mark.slow: For tests that take a long time to run. Includes sandbox tests.
@pytest.mark.calls_llm: For tests that call a real language model.
"""

import pytest
from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import EvalLog
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelName,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import CORRECT, INCORRECT, Target, Value
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from ..simple import (
    create_dataset,
    my_scorer,
    simple_addition_variant,
    simple_multiplication_variant,
)


def test_create_dataset():
    # Test the dataset creation code works across all variants
    variants = ["multiplication", "addition"]
    for variant in variants:
        dataset = create_dataset(0, variant, 10)
        assert len(dataset) == 10

    # test that, if created with the same seed, the datasets are the same, and if created with different seeds, they are different
    dataset1 = create_dataset(0, "multiplication", 10)
    dataset2 = create_dataset(0, "multiplication", 10)
    dataset3 = create_dataset(1, "multiplication", 10)
    assert dataset1 == dataset2
    assert dataset1 != dataset3


def create_correct_task_state(target: str) -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=1,
        epoch=1,
        input="",
        messages=[],
        output=ModelOutput(
            choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=target))]
        ),
    )


def create_incorrect_task_state(target: str) -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=1,
        epoch=1,
        input="",
        messages=[],
        output=ModelOutput(
            choices=[
                ChatCompletionChoice(message=ChatMessageAssistant(content="incorrect"))
            ]
        ),
    )


async def test_scorer():
    # Test that the template task's scorer works correctly.
    scorer = my_scorer()
    dataset_1 = create_dataset(0, "multiplication", 10)
    for sample in dataset_1:
        assert isinstance(sample.target, str)  # to satisfy mypy

        # test a correct answer gets a correct score
        state = create_correct_task_state(sample.target)
        score = await scorer(state, Target(target=sample.target))
        assert score.value == CORRECT

        # test an incorrect answer gets an incorrect score
        state = create_incorrect_task_state(sample.target)
        score = await scorer(state, Target(target=sample.target))
        assert score.value == INCORRECT


def check_all_samples_have_score_value(logs: list[EvalLog], expected_value: Value):
    for log in logs:
        samples = log.samples
        assert samples is not None
        for sample in samples:
            assert sample.score is not None
            assert sample.score.value == expected_value


@pytest.mark.slow
def test_template_task_e2e_success():
    # Test the whole task runs end-to-end (for the first sample) and that a correct submission is marked as correct.
    variant_to_task_fn = {
        "multiplication": simple_multiplication_variant,
        "addition": simple_addition_variant,
    }
    for variant in ["multiplication", "addition"]:
        seed = 0
        dataset = create_dataset(seed, variant, 10)
        task = variant_to_task_fn[variant](seed=seed)
        task.dataset = MemoryDataset(dataset[:1])

        model = get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput(
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessageAssistant(
                                role="assistant",
                                source="generate",
                                content=str(dataset[0].target),
                                # TODO: this will break if ppl change the name of the submit answer tool - is it possible to make this more robust?
                                tool_calls=[
                                    ToolCall(
                                        id="my_tool_call_1",
                                        function="submit",
                                        arguments={"answer": str(dataset[0].target)},
                                        type="function",
                                    )
                                ],
                            )
                        )
                    ]
                ),
            ],
        )

        logs = eval(
            tasks=[task],
            model=model,
            limit=1,
        )
        check_all_samples_have_score_value(logs, CORRECT)


@pytest.mark.slow
def test_template_task_e2e_failure():
    # Test the whole task runs end-to-end (for the first sample) and that an incorrect submission is marked as incorrect.
    variant_to_task_fn = {
        "multiplication": simple_multiplication_variant,
        "addition": simple_addition_variant,
    }
    for variant in ["multiplication", "addition"]:
        seed = 0
        dataset = create_dataset(seed, variant, 10)
        task = variant_to_task_fn[variant](seed=seed)
        task.dataset = MemoryDataset(dataset[:1])

        model = get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput(
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessageAssistant(
                                role="assistant",
                                source="generate",
                                content="incorrect",
                                # TODO: this will break if ppl change the name of the submit answer tool - is it possible to make this more robust?
                                tool_calls=[
                                    ToolCall(
                                        id="my_tool_call_1",
                                        function="submit",
                                        arguments={"answer": "incorrect"},
                                        type="function",
                                    )
                                ],
                            )
                        )
                    ]
                ),
            ],
        )

        logs = eval(
            tasks=[task],
            model=model,
            limit=1,
        )
        check_all_samples_have_score_value(logs, INCORRECT)
