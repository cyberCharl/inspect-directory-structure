"""This is an example evaluation which uses web browsing tools to answer multi-hop questions (with some example questions taken from hotpotqa dev set - https://hotpotqa.github.io).

It includes definitions for the evaluation, dataset, prompt, and scorer.
"""

from hashlib import sha256
from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact, model_graded_qa
from inspect_ai.solver import Solver, basic_agent
from inspect_ai.tool import web_browser

TASK_PROMPT = """Your task is to answer multi-hop questions as accurately as possible.

You will be given access to a web browser tool to help you find the answers, with which you can navigate to different web pages and extract information.

You will be asked a multi hop question, and you should use your best judgement to answer it best with access to the tools you have available.

You may take multiple rounds of action to perform research. Once you have found the answer, you should use the `submit` tool to submit your answer. Do not include any extraneous information in your answer, only the answer itself. Good luck!

The question is:
{question}"""

QUESTION_ANSWER_PAIRS = [
    (
        "Are both Superdrag and Collective Soul rock bands?",
        "Yes",
    ),
    (
        "Are Pago Pago International Airport and Hoonah Airport both on American territory?",
        "Yes",
    ),
    (
        "At which university does the biographer of John Clare teach English Literature?",
        "University of Oxford",
    ),
    (
        "Are Mike Bryan and Ray Ruffels both tennis players?",
        "Yes",
    ),
    (
        "What was the fort renamed in which the English abandoned to move to a fort located atop State Street Hill?",
        "Fort Albany",
    ),
]


def hash_string(data: str) -> str:
    return sha256(data.encode()).hexdigest()


def create_dataset() -> list[Sample]:
    samples = []
    for question, answer in QUESTION_ANSWER_PAIRS:
        samples.append(
            Sample(
                id=hash_string(question),
                input=TASK_PROMPT.format(question=question),
                target=answer,
            )
        )
    return samples


def default_solver() -> Solver:
    # the below is the solver we will use if no task-specific solver is provided
    return basic_agent(
        tools=[*web_browser()],
    )


@task
def web_multihop_question_answering(
    solver: Solver | None = None,
    scorer_type: Literal["exact_match", "model_graded_qa"] = "model_graded_qa",
) -> Task:
    solver = solver or default_solver()
    return Task(
        dataset=create_dataset(),
        # Instead of defining a custom scorer, we'll use inspect's `model_graded_qa` scorer for this task.
        # This gives a grading model access to the model output and target, and asks it to assign a CORRECT/INCORRECT score based on whether the output matches the target.
        # This allows matching of the model output to the target without having to worry about the exact format of the output, and is generally less prone to bias or error than other types of model grading (e.g. freeform grading).
        # We also make the scorer configurable for testing purposes.
        scorer=model_graded_qa() if scorer_type == "model_graded_qa" else exact(),
        sandbox=("docker", str(Path(__file__).parent / "compose.yaml")),
        plan=solver,
        max_messages=10,
    )
