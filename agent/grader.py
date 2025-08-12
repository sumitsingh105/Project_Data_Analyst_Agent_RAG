
# grader.py
from agent.agent import validate_generic_results   # ← add this line

def grade_answer(answer, rubric: dict = None) -> bool:
    """Enhanced generic validation"""
    # Accept both list and dict formats
    if isinstance(answer, list):
        # Use the new generic validation
        is_valid, _ = validate_generic_results(answer)
        return is_valid
    
    elif isinstance(answer, dict):
        # Keep existing dict validation
        if rubric and 'required_keys' in rubric:
            for key in rubric['required_keys']:
                if key not in answer or answer[key] is None:
                    return False
        return True
    
    return False
