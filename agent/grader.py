def grade_answer(answer, rubric: dict = None) -> bool:
    """
    Checks if answer is a valid list with expected structure.
    """
    # Accept both list (direct agent output) and dict formats
    if isinstance(answer, list):
        # Validate list format: should have 4 elements
        if len(answer) != 4:
            return False
        # Validate types: [int, str, float, str]
        if not (isinstance(answer[0], int) and 
                isinstance(answer[1], str) and 
                isinstance(answer[2], (int, float)) and 
                isinstance(answer[3], str) and 
                answer[3].startswith("data:image/png;base64,")):
            return False
        return True
    
    elif isinstance(answer, dict):
        # Original dict validation logic
        if rubric and 'required_keys' in rubric:
            for key in rubric['required_keys']:
                if key not in answer or answer[key] is None:
                    return False
        return True
    
    return False
