"""This is a template for how to structure your Inspect evaluation. It includes the evaluation, dataset, prompt, and scorer. If you're in doubt about how to build a evaluation, refer to this template - we'll try to keep it as up-to-date as possible.

The only mandatory part of this template is to define a `@task` function for each evaluation variant/category you want to create. The rest of the code / structure is optional, but we recommend you use it as a guide.
"""

import re
from collections import defaultdict
from hashlib import sha256
from pathlib import Path  # Keep Path if needed elsewhere, though not for direct access

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
from inspect_ai.tool import (
    bash,
    python,
)
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
    return Sample(id=hash_string(prompt), input=prompt, target=str(n))


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
def old_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        # There are predefined scorers in inspect_ai.scorer which might be more appropriate for this task, but we define a custom one here for illustrative purposes. In practice, you would want to use something like inspect's built-in `exact()` scorer which does this plus some extra normalization.

        # if state.output.completion.strip() == target.text:
        #     return Score(value=CORRECT, explanation="Answer is correct.")
        # return Score(
        #     value=INCORRECT,
        #     explanation=f"Expected answer {target.text}, got {state.output.completion}",
        # )
        output = await sandbox().exec(["tree"], cwd="/home/agent/output")
        print(output)

        return Score(value=CORRECT, explanation=output.stdout)

    return score


def calculate_expected_dirs(n):
    """Calculates the total expected number of directories for depth n."""
    if n == 0:
        return 1  # Just the root '.'
    # Sum of n^k for k from 0 to n (inclusive)
    # Root (n^0=1) + level 1 (n^1=n) + level 2 (n^2) + ... + level n (n^n)
    # No, the structure is: root + n dirs at level 1 + n*n dirs at level 2... + n*n...*n (n times) at level n
    # Total dirs = 1 (root) + n + n^2 + ... + n^n
    total = 1
    current_level_count = 1
    for _ in range(n):
        current_level_count *= n
        total += current_level_count
    return total


@scorer(metrics=[accuracy()])
def directory_tree_scorer():
    async def check_directory_structure_tree(state: TaskState, target: Target) -> Score:
        """Checks if the agent created the correct directory structure by running 'tree' in the sandbox and parsing its output."""
        try:
            expected_n = int(target.text)
            if expected_n < 1:  # Task definition implies n >= 1
                return Score(
                    value=0.0,
                    explanation=f"Invalid 'n' value: {expected_n}. Must be >= 1.",
                )
        except (ValueError, TypeError):
            return Score(
                value=0.0, explanation=f"Could not parse 'n' from target: {target.text}"
            )

        output_dir = "/home/agent/output"  # Standard output directory

        try:
            # Execute tree command in the sandbox
            # Use -d for directories only, makes parsing slightly cleaner
            # Use -L n+1 to limit depth shown, prevents overly large output for failed states? Maybe not necessary.
            exec_result = await sandbox().exec(
                # ["tree", "-d", "-L", str(expected_n + 1)], # -d shows only dirs, -L limits depth
                ["tree"],
                cwd=output_dir,
            )

            if not exec_result.success:
                # Check if tree is installed. If stderr indicates "command not found", provide specific feedback.
                if (
                    "command not found" in exec_result.stderr.lower()
                    or "no such file" in exec_result.stderr.lower()
                ):
                    return Score(
                        value=0.0,
                        explanation=f"Scoring error: 'tree' command not found in sandbox. Stderr: {exec_result.stderr}",
                    )
                # Handle case where /home/agent/output doesn't exist
                if "No such file or directory" in exec_result.stderr:
                    return Score(
                        value=0.0,
                        explanation=f"Scoring error: Output directory '{output_dir}' not found by 'tree'. Stderr: {exec_result.stderr}",
                    )
                # General execution failure
                return Score(
                    value=0.0,
                    explanation=f"Scoring error: 'tree' command failed with return code {exec_result.returncode}. Stderr: {exec_result.stderr}",
                )

            tree_output = exec_result.stdout
            if not tree_output or not tree_output.strip():
                # Handle case where the directory exists but is empty
                if expected_n > 0:
                    return Score(
                        value=0.0,
                        explanation="Directory structure is empty, but expected depth {expected_n}.",
                    )
                else:  # This case (n=0) isn't strictly per the brief, but logically means empty dir is correct.
                    # The brief says n >= 1, so maybe this isn't needed. If n=0 was allowed:
                    # return Score(value=1.0, explanation="Directory structure is empty, which is correct for n=0.")
                    return Score(
                        value=0.0,
                        explanation=f"Directory structure is empty, but expected depth {expected_n}.",
                    )  # Align with n>=1

            print(tree_output)

            # --- Start Parsing ---
            lines = tree_output.strip().split("\n")
            # ... skip summary line parsing for now if needed ...

            structure = defaultdict(list)
            nodes = {}
            max_depth_found = 0
            depth_stack = {0: "ROOT"}
            nodes["ROOT"] = (".", 0, ".")

            # Try the more robust ASCII regex:
            line_regex = re.compile(r"^([|`\- \t]*)(\S.*)$")  # Use ASCII-safe chars
            # Original regex commented out:
            # line_regex = re.compile(r"^([│└├─\s]*)(\S.*)$")

            print(
                f"\n--- DEBUG: Parsing Tree Output (expecting n={expected_n}) ---"
            )  # DEBUG START

            processed_lines = 0
            for i, line in enumerate(lines):  # Use enumerate to get line number
                line = line.rstrip()
                print(f"DEBUG Line {i}: Raw='{line}'")  # DEBUG Raw Line

                if line == "." or "directories" in line or "files" in line:
                    print(f"DEBUG Line {i}: Skipping (root/summary)")  # DEBUG Skip
                    continue

                match = line_regex.match(line)
                if not match:
                    # If the regex fails, this is a major clue!
                    print(f"DEBUG Line {i}: !!! REGEX FAILED !!!")  # DEBUG Regex Fail
                    continue

                prefix = match.group(1)
                name = match.group(2)

                name_start_index = len(prefix)
                # Calculate depth robustly: count levels of indentation markers
                # Simplest reliable way often remains index // 4 for standard tree
                current_depth = name_start_index // 4

                print(
                    f"DEBUG Line {i}: Prefix='{prefix}' (len={len(prefix)}), Name='{name}', Index={name_start_index}, Calculated Depth={current_depth}"
                )  # DEBUG PARSING

                if (
                    current_depth <= 0 and name != "."
                ):  # Allow depth 0 only for root '.' which we skip anyway
                    print(
                        f"DEBUG Line {i}: Skipping line with depth <= 0"
                    )  # DEBUG Skip Depth 0
                    continue

                # --- Check if max_depth_found is updated ---
                old_max_depth = max_depth_found
                max_depth_found = max(max_depth_found, current_depth)
                if old_max_depth != max_depth_found:
                    print(
                        f"DEBUG Line {i}: *** Max depth updated to {max_depth_found} ***"
                    )  # DEBUG Max Depth Change

                processed_lines += 1

                # ... (rest of the loop: parent finding, structure building) ...
                # You can add more prints here if needed for parent/stack debugging

            print("--- DEBUG: Parsing Complete ---")  # DEBUG END LOOP
            print(f"DEBUG: Final max_depth_found determined by loop: {max_depth_found}")
            print(
                f"DEBUG: Total lines processed (excluding root/summary): {processed_lines}"
            )

            # --- Validation ---
            # 1. Check maximum depth
            print(
                f"DEBUG: Comparing final max_depth_found ({max_depth_found}) with expected_n ({expected_n})"
            )
            if max_depth_found != expected_n:
                return Score(
                    value=0.0,
                    explanation=f"[Debugged] Incorrect depth: Maximum depth found is {max_depth_found}, expected {expected_n}.",
                )  # Add tag

            # 2. Check total directory count (reported vs expected based on n)
            # We count processed lines + 1 (for the root '.')
            actual_dirs_parsed = (
                processed_lines + 1
            )  # +1 for the root '.' which isn't in the loop
            # Note: tree reports dirs *including* '.', so reported_dirs should match actual_dirs_parsed
            # if actual_dirs_parsed != reported_dirs:
            #     # This might indicate a parsing issue or unexpected tree output format
            #     print(
            #         f"Warning: Parsed {actual_dirs_parsed} dirs, but tree summary reported {reported_dirs}."
            #     )
            #     # Decide whether to fail or trust the parsing
            #     # Let's trust our parsing for structure checks, but maybe flag this.

            expected_total_dirs = calculate_expected_dirs(expected_n)
            if actual_dirs_parsed != expected_total_dirs:
                return Score(
                    value=0.0,
                    explanation=f"Incorrect total directory count: Found {actual_dirs_parsed}, expected {expected_total_dirs} for n={expected_n}.",
                )

            # 3. Validate branching factor recursively/iteratively using the built structure
            for node_id, (depth, path_str) in nodes.items():
                if node_id == "ROOT":  # Skip virtual root unless n=0 edge case needed
                    children_count = len(structure.get(node_id, []))
                    if depth < expected_n and children_count != expected_n:
                        return Score(
                            value=0.0,
                            explanation=f"Incorrect branching at root ('.') (depth {depth}): Found {children_count} children, expected {expected_n}.",
                        )
                    continue  # Root checks done

                children_count = len(structure.get(node_id, []))

                if depth < expected_n:
                    if children_count != expected_n:
                        return Score(
                            value=0.0,
                            explanation=f"Incorrect branching at '{path_str}' (depth {depth}): Found {children_count} children, expected {expected_n}.",
                        )
                elif depth == expected_n:
                    if children_count != 0:
                        return Score(
                            value=0.0,
                            explanation=f"Incorrect leaf node at '{path_str}' (depth {depth}): Found {children_count} children, expected 0 (should be empty).",
                        )
                # else depth > expected_n: # This case is already caught by max_depth check

            # If all checks passed
            return Score(
                value=1.0,
                explanation=f"Correct directory structure found for n={expected_n} (Depth: {max_depth_found}, Branching: {expected_n}).",
            )

        except Exception as e:
            # Catch potential errors during parsing or sandbox execution
            import traceback

            print(f"Scoring Error: {e}\n{traceback.format_exc()}")  # Log for debugging
            return Score(
                value=0.0, explanation=f"Scoring error: {type(e).__name__}: {e}"
            )

    return check_directory_structure_tree


def _create_task(
    solver: Solver | None = None,
    num_samples: int = 1,
) -> Task:
    solver = solver or default_solver()
    return Task(
        dataset=create_dataset(num_samples=num_samples),
        scorer=directory_tree_scorer(),
        sandbox=("docker", str(Path(__file__).parent / "compose.yaml")),
        solver=solver,
        message_limit=10,
    )


@task
def my_task(solver: Solver | None = None) -> Task:
    # Although it's not mandatory, we recommend you write your tasks with a `solver` function so that the solver can be easily swapped out.
    # We include a separate `_create_task` function as, in more complex evaluations, you may want a single `_create_task` function that can be used by multiple `@task` functions to create different variants of the task.
    return _create_task(solver)
