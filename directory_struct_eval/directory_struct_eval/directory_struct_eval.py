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
    for i in range(1, num_samples + 1):
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
    if n < 0:
        return 0  # Just the root '.'
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
            if expected_n < 1:
                return Score(
                    value=0.0,
                    explanation=f"Invalid 'n' value: {expected_n}. Must be >= 1.",
                )
        except (ValueError, TypeError):
            return Score(
                value=0.0, explanation=f"Could not parse 'n' from target: {target.text}"
            )

        output_dir = "/home/agent/output"

        try:
            exec_result = await sandbox().exec(
                ["tree"],
                cwd=output_dir,
            )

            if not exec_result.success:
                stderr_lower = exec_result.stderr.lower()
                if (
                    "command not found" in stderr_lower
                    or "no such file" in stderr_lower
                ):
                    return Score(
                        value=0.0,
                        explanation=f"Scoring error: 'tree' command not found or failed in sandbox. Stderr: {exec_result.stderr}",
                    )
                if "no such file or directory" in stderr_lower:
                    # Check if it's the command itself or the directory
                    if output_dir in exec_result.stderr:
                        return Score(
                            value=0.0,
                            explanation=f"Scoring error: Output directory '{output_dir}' not found by 'tree'. Stderr: {exec_result.stderr}",
                        )
                    else:  # Likely tree command itself not found
                        return Score(
                            value=0.0,
                            explanation=f"Scoring error: 'tree' command not found in sandbox PATH. Stderr: {exec_result.stderr}",
                        )

                return Score(
                    value=0.0,
                    explanation=f"Scoring error: 'tree' command failed with return code {exec_result.returncode}. Stderr: {exec_result.stderr}",
                )

            tree_output = exec_result.stdout
            if not tree_output or not tree_output.strip():
                # Handle case where the directory exists but is empty
                # An empty dir has 1 entry in `tree -d` output: "." and a summary "0 directories"
                # If tree_output is truly empty or only whitespace, something else is wrong.
                # Let's check based on lines after splitting.
                lines_check = tree_output.strip().split("\n")
                if len(lines_check) <= 1:  # Only '.' or empty
                    if expected_n == 0:  # Although brief implies n>=1
                        return Score(
                            value=1.0,
                            explanation="Directory structure is empty, correct for n=0.",
                        )
                    else:
                        return Score(
                            value=0.0,
                            explanation=f"Directory structure appears empty or invalid, expected depth {expected_n}.",
                        )
                # If not empty, proceed with parsing

            print("--- Tree Output ---")
            print(tree_output)
            print("-------------------")

            # --- Start Parsing ---
            lines = tree_output.strip().split("\n")
            if not lines:
                return Score(
                    value=0.0, explanation="Empty 'tree' output after stripping."
                )

            # --- Parse Summary Line --- (Important for file check if not using -d)
            summary_line = lines[-1]
            reported_dirs = 0
            reported_files = 0  # Assume 0 if using -d, parse otherwise
            match_summary = re.match(
                r"(\d+)\s+directories?(?:,\s+(\d+)\s+files?)?", summary_line
            )
            if match_summary:
                reported_dirs = int(match_summary.group(1))
                if match_summary.group(2):  # If files group exists
                    reported_files = int(match_summary.group(2))
                # If using `tree -d`, reported_files should ideally be 0 or absent
                if reported_files != 0:
                    return Score(
                        value=0.0,
                        explanation=f"Incorrect structure: Found {reported_files} files, expected 0.",
                    )
            else:
                # Handle cases where summary line might be missing (e.g., very shallow tree or error)
                print(f"Warning: Could not parse tree summary line: '{summary_line}'")

            # --- Structure Parsing ---
            structure = defaultdict(list)
            nodes = {}
            max_depth_found = 0
            depth_stack = {0: "ROOT"}
            nodes["ROOT"] = (".", 0, ".")  # Store name, depth, path

            line_regex = re.compile(
                r"^([|`\- \t]*)(\S.*)$"
            )  # ASCII-safe prefix, then name

            print(f"\n--- DEBUG: Parsing Tree (expecting n={expected_n}) ---")

            processed_lines_count = 0  # Count actual directory lines parsed
            for i, line in enumerate(lines):
                line = line.rstrip()
                print(f"DEBUG Line {i}: Raw='{line}'")

                # Skip empty lines
                if not line:
                    print(f"DEBUG Line {i}: Skipping empty line.")
                    continue

                # Skip root indicator and summary line explicitly
                if line == "." or "directories" in line or "files" in line:
                    print(f"DEBUG Line {i}: Skipping (root/summary)")
                    continue

                match = line_regex.match(line)
                if not match:
                    print(f"DEBUG Line {i}: !!! REGEX FAILED !!!")
                    continue  # Skip lines that don't match the expected format

                prefix = match.group(1)
                name = match.group(2)

                name_start_index = len(prefix)
                current_depth = (
                    name_start_index // 4
                )  # Standard tree indent calculation

                print(
                    f"DEBUG Line {i}: Prefix='{prefix}' (len={len(prefix)}), Name='{name}', Index={name_start_index}, Calculated Depth={current_depth}"
                )

                if current_depth <= 0:  # Should only be '.' which is skipped above
                    print(f"DEBUG Line {i}: Skipping line with depth <= 0")
                    continue

                max_depth_found = max(max_depth_found, current_depth)
                processed_lines_count += 1

                parent_depth = current_depth - 1
                parent_id = depth_stack.get(parent_depth)
                if parent_id is None:
                    # This indicates an illogical tree structure in the output
                    return Score(
                        value=0.0,
                        explanation=f"Parsing error: Could not find parent for node '{name}' at depth {current_depth}. Tree structure likely invalid. Line: '{line}'",
                    )

                parent_name, parent_actual_depth, parent_path = nodes[parent_id]
                current_path = f"{parent_path}/{name}" if parent_path != "." else name
                node_id = current_path  # Use path as unique ID

                # Store node info (name, depth, path)
                nodes[node_id] = (name, current_depth, current_path)

                # Add to parent's children list
                structure[parent_id].append(node_id)

                # Update the stack for the current depth
                depth_stack[current_depth] = node_id
                # Clean up stack for deeper levels if tree goes back up
                keys_to_remove = [d for d in depth_stack if d > current_depth]
                for d in keys_to_remove:
                    del depth_stack[d]

            print("--- DEBUG: Parsing Complete ---")
            print(f"DEBUG: Final max_depth_found determined by loop: {max_depth_found}")
            print(f"DEBUG: Total directory lines processed: {processed_lines_count}")
            # Compare parsed lines count (+1 for root) with reported count
            # The root '.' isn't processed in the loop, so add 1
            actual_dirs_parsed = processed_lines_count + 1
            if reported_dirs > 0 and actual_dirs_parsed != reported_dirs:
                print(
                    f"Warning: Parsed {actual_dirs_parsed} dirs (incl. root), but tree summary reported {reported_dirs}."
                )

            # --- Validation ---
            # 1. Check maximum depth
            print(
                f"DEBUG: Comparing final max_depth_found ({max_depth_found}) with expected_n ({expected_n})"
            )
            if max_depth_found != expected_n:
                return Score(
                    value=0.0,
                    explanation=f"Incorrect depth: Maximum depth found is {max_depth_found}, expected {expected_n}.",
                )

            # 2. Check total directory count against theoretical expectation
            expected_total_dirs = calculate_expected_dirs(expected_n)
            # Use actual_dirs_parsed which counts '.' + all processed lines
            if actual_dirs_parsed != expected_total_dirs:
                return Score(
                    value=0.0,
                    explanation=f"Incorrect total directory count: Found {actual_dirs_parsed} (parsed), expected {expected_total_dirs} for n={expected_n}.",
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
                explanation=f"Correct directory structure found for n={expected_n} (Depth: {max_depth_found}, Branching: {expected_n}, Dirs: {actual_dirs_parsed}).",
            )

        except Exception as e:
            import traceback

            print(f"Scoring Error: {e}\n{traceback.format_exc()}")
            return Score(
                value=0.0, explanation=f"Scoring error: {type(e).__name__}: {e}"
            )

    return check_directory_structure_tree


def _create_task(
    solver: Solver | None = None,
    num_samples: int = 3,
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
