"""This is a template for how to structure your Inspect evaluation. It includes the evaluation, dataset, prompt, and scorer. If you're in doubt about how to build a evaluation, refer to this template - we'll try to keep it as up-to-date as possible.

The only mandatory part of this template is to define a `@task` function for each evaluation variant/category you want to create. The rest of the code / structure is optional, but we recommend you use it as a guide.
"""

from hashlib import sha256
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    CORRECT,
    # INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
)
from inspect_ai.solver import (  # Plan, generate, use_tools
    Solver,
    TaskState,
    basic_agent,
)
from inspect_ai.tool import bash, python
from inspect_ai.util import sandbox

NEW_PROMPT = """Your objective is to generate a nested directory hierarchy according to the rules specified below.

**Core Requirements:**

1.  **Target Location:** All directories must be created inside the `/home/agent/output/` directory. You should ensure this base path exists before creating subdirectories.
2.  **Hierarchy Depth:** The directory tree must extend to a precise depth of {n}.
3.  **Directory Branching:** Every non-leaf directory (i.e., directories at depths 0 to {n}-1, relative to `/home/agent/output/`) must branch into exactly {n} subdirectories.
4.  **Directory Naming Scheme:**
    *   All created directories must have names consisting of a single lowercase letter from the English alphabet ('a'-'z').
    *   Directory names must be unique across the entire generated structure.
    *   Assign names sequentially using letters from 'a' onwards. The sequence follows a recursive, depth-first pattern:
        *   Create the first required directory at the current level using the next available letter.
        *   Recursively create the entire subtree required within that new directory.
        *   Once the subtree is complete, create the second required directory at the current level using the *next* available letter, and recurse.
        *   Continue this process until all {n} directories at the current level are created.
    *   Once the letter 'z' is reached, start with double letters e.g. 'aa', 'ab'
5.  **Leaf Nodes:** Directories located at the final depth ({n}) must remain empty.
6.  **Input Constraint:** The value of {n} will be an integer, assume 1 <= {n} <= 5.

**Final Action:**
Upon successful creation of the complete directory structure matching all requirements, submit the text 'submit'.

**Illustrative Example (n=2):**
The expected structure within `/home/agent/output/`:
/home/agent/output/
├── a/  (Created first at depth 1)
│   ├── b/  (Created first within 'a/' at depth 2, empty)
│   └── c/  (Created second within 'a/' at depth 2, empty)
└── d/  (Created second at depth 1, after 'a/' and its contents are done)
    ├── e/  (Created first within 'd/' at depth 2, empty)
    └── f/  (Created second within 'd/' at depth 2, empty)"""

agent = basic_agent(tools=[bash()])

load_dotenv()


def hash_string(data: str) -> str:
    return sha256(data.encode()).hexdigest()


def create_directory_sample(n: int) -> Sample:
    prompt = NEW_PROMPT.format(n=n)
    return Sample(
        id=hash_string(prompt),
        input=prompt,
    )


def create_dataset(num_samples: int) -> list[Sample]:
    samples = []
    for i in range(3, num_samples + 3):
        samples.append(create_directory_sample(i))

    return samples


def default_solver() -> Solver:
    # the below is the solver we will use if no task-specific solver is provided
    return basic_agent(
        tools=[bash(timeout=60), python(timeout=60)],
        # the submit_description field can be used to provide a hint to the solver about what the expected output format is
        submit_description="Submit an answer for evaluation.",
    )


@scorer(metrics=[accuracy()])
def my_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        # There are predefined scorers in inspect_ai.scorer which might be more appropriate for this task, but we define a custom one here for illustrative purposes. In practice, you would want to use something like inspect's built-in `exact()` scorer which does this plus some extra normalization.

        # if state.output.completion.strip() == target.text:
        #     return Score(value=CORRECT, explanation="Answer is correct.")
        # return Score(
        #     value=INCORRECT,
        #     explanation=f"Expected answer {target.text}, got {state.output.completion}",
        # )
        output = await sandbox().exec(["tree"], cwd="home/agent/output")
        print(output)

        return Score(value=CORRECT, explanation=output.stdout)

    return score


def _create_task(
    solver: Solver | None = None,
    num_samples: int = 2,
) -> Task:
    solver = solver or default_solver()
    return Task(
        dataset=create_dataset(num_samples=num_samples),
        scorer=my_scorer(),
        sandbox=("docker", str(Path(__file__).parent / "compose.yaml")),
        solver=solver,
        message_limit=5,
    )


@task
def my_task(solver: Solver | None = None) -> Task:
    # Although it's not mandatory, we recommend you write your tasks with a `solver` function so that the solver can be easily swapped out.
    # We include a separate `_create_task` function as, in more complex evaluations, you may want a single `_create_task` function that can be used by multiple `@task` functions to create different variants of the task.
    return _create_task(solver)
