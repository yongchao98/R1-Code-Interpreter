#!/usr/bin/env python3

import re

def extract_equation_with_GPT4_gpqa(response: str) -> str:
    
    answer_patterns = [
        r'(?:Answer|ANSWER):\s*([A-D])',
        r'(?:The answer is|answer is)\s*([A-D])',
        r'\b([A-D])\s*(?:is (?:the )?correct|is (?:the )?answer)',
        r'(?:^|\n)\s*([A-D])\s*(?:\.|$)',
        r'(?:option|choice)\s*([A-D])',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    
    return None

def validate_solution_gpqa(extracted_answer: str, solution_data: dict) -> tuple:
    if extracted_answer is None:
        return False, "No answer extracted from response"
    
    correct_answer = solution_data['correct_answer'].strip()
    
    if extracted_answer == correct_answer:
        return True, f"Correct! Answer {extracted_answer} matches expected {correct_answer}"
    else:
        return False, f"Incorrect. Answer {extracted_answer} does not match expected {correct_answer}"

if __name__ == "__main__":
    # Test answer extraction
    test_responses = [
        'The answer is A',
        'Answer: B',
        'I think the correct answer is C.',
        'Looking at the options, D is correct.',
        'The answer should be A because...'
    ]

    print('Testing answer extraction:')
    for response in test_responses:
        extracted = extract_equation_with_GPT4_gpqa(response)
        print(f'Response: "{response}" -> Extracted: {extracted}')

    # Test validation
    print('\nTesting validation:')
    sample_puzzle = {
        'question': 'Test question?',
        'correct_answer': 'A',
        'incorrect_answers': ['B', 'C', 'D']
    }

    test_cases = ['A', 'B', None, 'X']
    for answer in test_cases:
        result, message = validate_solution_gpqa(answer, sample_puzzle)
        print(f'Answer: {answer} -> Result: {result}, Message: {message}')
    
    print('\nGPQA functions test completed successfully!')