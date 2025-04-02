import os


# Use os.walk within the sandbox/accessible filesystem
def check_directory_structure_walk(
    base_output_path, base_path_str, base_path_len, expected_n, max_depth_found
) -> tuple[bool, int, str]:
    structure_correct = True
    explanation = "Checks passed."  # Default success message

    for current_dir_path, subdir_names, _ in os.walk(str(base_output_path)):
        current_path_parts = os.path.normpath(current_dir_path).split(os.sep)
        if current_path_parts[-1] == "":
            current_path_parts.pop()
        current_depth = len(current_path_parts) - base_path_len

        if current_depth < 0:
            continue

        max_depth_found = max(max_depth_found, current_depth)

        if current_depth < expected_n:
            num_subdirs = len(subdir_names)
            if num_subdirs != expected_n:
                structure_correct = False
                explanation = f"Incorrect branching: Dir '{os.path.relpath(current_dir_path, base_path_str)}' (depth {current_depth}) has {num_subdirs} subdirs, expected {expected_n}."
                break
        elif current_depth == expected_n:
            num_subdirs = len(subdir_names)
            if num_subdirs != 0:
                structure_correct = False
                explanation = f"Incorrect leaf: Dir '{os.path.relpath(current_dir_path, base_path_str)}' (depth {current_depth}) not empty, has {num_subdirs} subdirs."
                break
        elif current_depth > expected_n:
            structure_correct = False
            explanation = f"Depth exceeded: Dir '{os.path.relpath(current_dir_path, base_path_str)}' found at depth {current_depth}, max allowed is {expected_n}."
            break

    return structure_correct, max_depth_found, explanation
