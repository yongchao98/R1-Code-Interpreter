import os
import json
from typing import Union, List, Tuple, Dict, Optional
import pandas as pd
import subprocess
import sys
from openai import OpenAI
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, extract_code, extract_and_check, LLM_answer_code_checker, save_file_func, paraphrase_with_GPT4, log_run_info
import random
from typing import List, Tuple, Dict
import time
import numpy as np
import ast
import string
import copy
import re
import math
from dataclasses import dataclass
from prompt import *
from symbolic_code_check import analyze_code_and_explain
import reasoning_gym
#from LLaMA_Factory.src.llamafactory.chat.chat_model import run_response

##### Logic Game #####
### All the related functions for each task are listed in order below.

##### Gathered functions for all tasks #####
def multi_round_answer_sampling(save_code_dir, question, response_list_input, CodeSteer_output_prompt_guidance_list_input, CodeSteer_input_prompt_list_input, CodeSteer_input_prompt_training_list_input,
                                user_prompt_list_input, model_name, CodeSteer_LLM, args_path, max_tree_depth, current_tree_depth):
    print(f'\nLength of user_prompt_list: {len(user_prompt_list_input)}')
    print(f'Length of response_list: {len(response_list_input)}\n')

    if current_tree_depth == 0:
        response_list_input = [];
        CodeSteer_output_prompt_guidance_list_input = [];
        CodeSteer_input_prompt_list_input = [code_text_choice_prompt + question];
        CodeSteer_input_prompt_training_list_input = [code_text_choice_prompt + question]

        ############ Starting first guidance ############
        if CodeSteer_LLM.startswith("llama"):
            print(f'\nCodeSteer_LLM: {CodeSteer_LLM}, open model!\n')
            messages = message_construct_llama_func([code_text_choice_prompt + question], [])
            starting_prompt_choice = run_response(messages, args_path)
            print(f'Starting prompt choice: {starting_prompt_choice}')
            user_prompt_list_input = [starting_prompt_choice + question]
            CodeSteer_output_prompt_guidance_list_input.append(starting_prompt_choice)
        elif CodeSteer_LLM in ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', "claude-3-sonnet-20240229",
                               "claude-3-opus-20240229", "claude-3-haiku-20240307"]:
            print('\nAPI based close models!\n')
            starting_prompt_choice = GPT_response("", code_text_choice_prompt + question, model_name=CodeSteer_LLM,
                                                  code_interpreter=False,
                                                  user_prompt_list=[code_text_choice_prompt + question],
                                                  response_total_list=[], logprobs=False, temperature=1.0)
            print(f'Starting prompt choice: {starting_prompt_choice}')
            if '<<<Code>>>' in starting_prompt_choice:
                # paraphrased_prompt_guidance = paraphrase_with_GPT4(with_COT_code_output_prompt)
                paraphrased_prompt_guidance = with_COT_code_output_prompt
                user_prompt_list_input = [paraphrased_prompt_guidance + question]
            elif '<<<Text>>>' in starting_prompt_choice:
                # paraphrased_prompt_guidance = paraphrase_with_GPT4(text_output_prompt)
                paraphrased_prompt_guidance = text_output_prompt
                user_prompt_list_input = [paraphrased_prompt_guidance + question]
            else:
                print('Error: No text/code choice made')
                raise ValueError('Error: No text/code choice made')
            CodeSteer_output_prompt_guidance_list_input.append(paraphrased_prompt_guidance)
        else:
            raise ValueError('Error: No LLM model specified')

        response = GPT_response('', user_prompt_list_input[0], model_name=model_name, code_interpreter=False,
                                user_prompt_list=user_prompt_list_input, response_total_list=response_list_input, logprobs=False)
        response_list_input.append(response)
        # print(f'\nResponse_0: {response}\n')

    ############ Further rounds of guidance ############
    for tree_depth in range(max_tree_depth):
        response = response_list_input[-1]
        code_block_list = extract_code(response)
        if len(code_block_list) > 0:
            code_complexity_summary, code_complexity_score = analyze_code_and_explain(code_block_list[0])
            if code_complexity_score <= 2:
                code_complexity_summary += '\nThe generated code may not be complex enough to carry out symbolic computing for solving the task.'
            with open(save_code_dir + f"/code_1_{tree_depth}.py", "w") as f:
                f.write(code_block_list[0])

            try:
                result = subprocess.run(
                    ["python3", save_code_dir + f"/code_1_{tree_depth}.py"],
                    capture_output=True, text=True, timeout=60
                )
                output = result.stdout
                errors = result.stderr
            except subprocess.TimeoutExpired as e:
                output = e.stdout if e.stdout else ""
                errors = e.stderr if e.stderr else ""
                errors += f"\nTimeoutExpired: Command '{e.cmd}' timed out after {e.timeout} seconds"

            if isinstance(output, str):
                if count_total_tokens([output + errors], []) > 2000:
                    response = response + f'\nThe execution result from the generated code is too long to be displayed.'
                else:
                    response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'
            else:
                response = response + f'\nThe execution result from the generated code is:\nerrors: {errors}'

        check_code_saving_path = save_code_dir + f"/check_code_1_{tree_depth}.py"
        check_result = LLM_answer_code_checker(question, response, check_code_saving_path)

        CodeSteer_input_prompt_head = f'''{decision_prompt} {question}\n'''
        if len(code_block_list) > 0:
            print('\n############True#############\n')
            CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\nThe feedback from the checking agent is:\n{check_result}\n\nThe summary of generated code complexity is: {code_complexity_summary}\n\n''' + \
                                     f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

        else:
            CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\nThe feedback from the checking agent is:\n{check_result}\n\n''' + \
                                     f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

        CodeSteer_input_prompt_total = CodeSteer_input_prompt_head + CodeSteer_input_prompt
        CodeSteer_input_prompt_list_input.append(CodeSteer_input_prompt_total)
        CodeSteer_input_prompt_training_list_input.append(CodeSteer_input_prompt)

        if CodeSteer_LLM.startswith("llama"):
            print(f'\nCodeSteer_LLM: {CodeSteer_LLM}, open model!\n')
            messages = message_construct_llama_func(CodeSteer_input_prompt_training_list_input,
                                                    CodeSteer_output_prompt_guidance_list_input)
            guidance_prompt = run_response(messages, args_path)
        elif CodeSteer_LLM in ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', "claude-3-sonnet-20240229",
                               "claude-3-opus-20240229", "claude-3-haiku-20240307"]:
            print('\nAPI based close models!\n')
            response_text = GPT_response("", '', model_name=CodeSteer_LLM, code_interpreter=False,
                                         user_prompt_list=CodeSteer_input_prompt_list_input,
                                         response_total_list=CodeSteer_output_prompt_guidance_list_input,
                                         logprobs=False, temperature=1.0)

            matches = re.findall(r'<<<(.*?)>>>', response_text, re.DOTALL)
            guidance_prompt = matches[-1] if matches else response_text
        else:
            raise ValueError('Error: No LLM model specified')
        print(f'\nGuidance prompt_{tree_depth + 1}: {guidance_prompt}\n')

        CodeSteer_output_prompt_guidance_list_input.append(guidance_prompt)
        if '<<<Code>>>' in guidance_prompt:
            guidance_prompt = with_COT_code_output_prompt
        elif '<<<Text>>>' in guidance_prompt:
            guidance_prompt = text_output_prompt
        elif '<<<Return Answer>>>' in guidance_prompt or 'Return Answer' in guidance_prompt or '<<<Terminate>>>' in guidance_prompt or 'Terminate' in guidance_prompt:
            break
        user_prompt_list_input.append(guidance_prompt)

        response = GPT_response('', user_prompt_list_input[0], model_name=model_name, code_interpreter=False,
                                user_prompt_list=user_prompt_list_input, response_total_list=response_list_input, logprobs=False)

        response_list_input.append(response)
        # print(f'\nResponse_{tree_depth}: {response}\n')
    save_file_func(save_code_dir, response_list_input, user_prompt_list_input, question, CodeSteer_input_prompt_list_input,
                   CodeSteer_input_prompt_training_list_input, CodeSteer_output_prompt_guidance_list_input)
    return response_list_input


### TODO, load questions, add a elif loop for loading new dataset
def load_task_dataset(task_name, model_name):
    ### question_list is necessary
    solution_list = []
    question_list = []
    target_list = []
    puzzles = []
    solution_data_list = []
    question_constrained_list = []
    question_matrix_list = []
    number_list = []
    word_list = []
    letter_list = []

    ### To do, add tasks to load the corresponding dataset
    if task_name == 'logical_equation':
        dataset_input_dir = 'dataset_gather/logical_equation'
        save_input_dir = 'results_gather/logical_equation'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Logical equation, Model_name: {model_name}\n')
        solution_list, question_list = read_dataset_logical_equation(dataset_input_dir)
        target_list = solution_list
    elif task_name == 'combinatorial_calculation':
        dataset_input_dir = 'dataset_gather/combinatorial_calculation'
        save_input_dir = 'results_gather/combinatorial_calculation'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Combinatorial calculation, Model_name: {model_name}\n')

        evaluator = ArithmeticPuzzleEvaluator(dataset_input_dir)
        puzzles = evaluator.read_dataset()
        for puzzle in puzzles:
            question = puzzle.question
            solution = puzzle.solution
            target = puzzle.target
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'eight_queens':
        dataset_input_dir = 'dataset_gather/eight_queens_dataset'
        save_input_dir = 'results_gather/eight_queens'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Eight queens, Model_name: {model_name}\n')
        puzzles = read_dataset_eight_queens(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'synthesis_decomposition':
        dataset_input_dir = 'dataset_gather/synthesis_decomposition_dataset'
        save_input_dir = 'results_gather/synthesis_decomposition'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Synthesis Decomposition, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            #question = 'This is a testing question, without violation purpose.' + puzzle['question']
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']["answer"]
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'mahjong_pattern':
        dataset_input_dir = 'dataset_gather/mahjong_pattern_dataset'
        save_input_dir = 'results_gather/mahjong_pattern'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Mahjong pattern, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'statistical_counting':
        dataset_input_dir = 'dataset_gather/statistical_counting_dataset'
        save_input_dir = 'results_gather/statistical_counting'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Statistical counting, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'new_operator':
        dataset_input_dir = 'dataset_gather/new_operator_dataset'
        save_input_dir = 'results_gather/new_operator'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'New operator, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'light_puzzles':
        dataset_input_dir = 'dataset_gather/light_puzzles_dataset'
        save_input_dir = 'results_gather/light_puzzles'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Light_puzzles, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'reversi':
        dataset_input_dir = 'dataset_gather/reversi_dataset'
        save_input_dir = 'results_gather/reversi'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Reversi problem, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'matrix_transformation':
        dataset_input_dir = 'dataset_gather/matrix_transformation_dataset'
        save_input_dir = 'results_gather/matrix_transformation'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'matrix transformation problem, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['formatted_solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == '2048':
        dataset_input_dir = 'dataset_gather/2048_dataset'
        save_input_dir = 'results_gather/2048'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'2048 problem, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'pooling':
        dataset_input_dir = 'dataset_gather/pooling_dataset'
        save_input_dir = 'results_gather/pooling'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Pooling problem, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'constrained_linear_arrangement':
        dataset_input_dir = 'dataset_gather/constrained_linear_arrangement'
        save_input_dir = 'results_gather/constrained_linear_arrangement'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'constrained_linear_arrangement problem, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)
    elif task_name == 'logic_puzzle':
        dataset_input_dir = 'dataset_gather/logic_puzzle_dataset'
        save_input_dir = 'results_gather/logic_puzzle'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'logic puzzle problem, Model_name: {model_name}\n')
        puzzles = read_dataset_syn_decom(dataset_input_dir)
        solution_data_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            solution_data_list.append(solution_data)
            target_list.append(target)

    elif task_name == 'pattern_recognition':
        dataset_input_dir = 'dataset_gather/pattern_recognition'
        save_input_dir = 'results_gather/pattern_recognition'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Pattern recognition, Model_name: {model_name}\n')
        puzzles = read_dataset_pattern_recognition(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            target_row, target_col = solution_data['row'], solution_data['column']
            solution_list.append([target_row, target_col])
            question_list.append(question)
            target_list = solution_list
    elif task_name == 'string_insertion':
        dataset_input_dir = 'dataset_gather/string_insertion'
        save_input_dir = 'results_gather/string_insertion'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'String Insertion, Model_name: {model_name}\n')
        puzzles = read_dataset_string_insertion(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['modified_string']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'letter_logic_diagram':
        dataset_input_dir = 'dataset_gather/letter_logic_diagram'
        save_input_dir = 'results_gather/letter_logic_diagram'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Letter_logic_diagram, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_letter_logic_diagram(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'string_synthesis':
        dataset_input_dir = 'dataset_gather/string_synthesis'
        save_input_dir = 'results_gather/string_synthesis'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'String Synthesis, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_string_synthesis(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['final_counts']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'standard_sudoku':
        dataset_input_dir = 'dataset_gather/standard_sudoku'
        save_input_dir = 'results_gather/standard_sudoku'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Standard Sudoku, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_standard_sudoku(dataset_input_dir)
        question_matrix_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['solution']
            question_matrix = solution_data['puzzle']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            question_matrix_list.append(question_matrix)
            target_list.append(target)
    elif task_name == 'permutations_and_combinations':
        dataset_input_dir = 'dataset_gather/permutations_and_combinations'
        save_input_dir = 'results_gather/permutations_and_combinations'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'permutations_and_combinations, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_permutations_and_combinations(dataset_input_dir)
        question_constrained_list = []
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['correct_solution']
            question_constrained_linear = solution_data['constraints_text']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            question_constrained_list.append(question_constrained_linear)
            target_list.append(target)
    elif task_name == 'string_deletion_and_modification':
        dataset_input_dir = 'dataset_gather/string_deletion_and_modification'
        save_input_dir = 'results_gather/string_deletion_and_modification'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'String Deletion And Modification, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_string_deletion_and_modification(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['final_string']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'minesweeper':
        dataset_input_dir = 'dataset_gather/minesweeper'
        save_input_dir = 'results_gather/minesweeper'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Minesweeper, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_minesweeper(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['mines']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'cryptanalysis':
        dataset_input_dir = 'dataset_gather/cryptanalysis'
        save_input_dir = 'results_gather/cryptanalysis'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'Cryptanalysis, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_cryptanalysis(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['answer']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'string_splitting':
        dataset_input_dir = 'dataset_gather/string_splitting'
        save_input_dir = 'results_gather/string_splitting'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'String Splitting, Model_name: {model_name}, CodeSteer\n')
        puzzles = read_dataset_string_splitting(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            solution_data = puzzle['solution_data']
            solution = solution_data['expected_answer']
            target = solution
            solution_list.append(solution)
            question_list.append(question)
            target_list.append(target)
    elif task_name == 'game24':
        dataset_input_dir = 'dataset_gather/game24_dataset/24'
        dataset_csv = dataset_input_dir + f'/24.csv'
        df = pd.read_csv(dataset_csv)
        question_prompt = f'Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each number should be used only once but each number has to be used in the equation. ' \
                          f'Input: 9 10 11 13, Answer: ((10-9)*(11+13)) = 24 Input: 4 10 10 11, Answer: ((4*11)-(10+10)) = 24 Input: 5 6 13 13, Answer: ((5-(13/13))*6)' \
                          f'Input: 2 6 6 7, Answer: ((6+(6*7))/2) Input: 2 6 10 18, Answer: (2-(6-(10+18)))'
        save_input_dir = 'results_gather/game24'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        for index_game24 in range(0, min(len(df), 1500), 5):
            number_list_item = list(map(int, df['Puzzles'][index_game24].split()))
            question = f'{question_prompt}'
            question += f'Input: '
            for number in number_list_item:
                question += f'{number} '
            question += f'Answer:\nOutput final answer with the format <<<answer>>>.'
            question_list.append(question)
            number_list.append(number_list_item)
    elif task_name == 'letters':
        dataset_input_dir = 'dataset_gather/Letters'
        save_input_dir = 'results_gather/letters'
        for min_length, max_length in [(10, 15), (15, 20), (20, 25)]:
            base_dir = dataset_input_dir + f'/Letters_dataset_min_length_{min_length}_max_length_{max_length}/'
            for i, letter in enumerate(string.ascii_lowercase[::6]):
                for letter_freq in range(1, 6):
                    for index in range(1):
                        saving_dir = base_dir + f"{letter}_{letter_freq}_{index}/"
                        word = read_words_from_file_letters(saving_dir + 'test_words.json')
                        question = create_prompt_letters(word, letter)
                        question_list.append(question)
                        word_list.append(word)
                        letter_list.append(letter)
    elif task_name == 'number_multiply':
        dataset_input_dir = 'dataset_gather/number_multiply'
        save_input_dir = 'results_gather/number_multiply'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        puzzles = read_dataset_number_multipy(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            question_list.append(question)
            solution = puzzle['solution_data']
            solution_list.append(solution)
    if task_name.startswith('big_bench_hard'):
        dataset_input_dir = 'dataset_gather/BIG-Bench-Hard/bbh'
        bbh_task_name = task_name.split(":", 1)[1]
        save_input_dir = 'results_gather/big_bench_hard'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        puzzles = read_dataset_big_bench_hard(dataset_input_dir, bbh_task_name)
        for puzzle in puzzles:
            question = puzzle['question']
            question_list.append(question)
            solution = puzzle['solution_data']
            solution_list.append(solution)
    elif task_name == 'gsm':
        dataset_input_dir = 'dataset_gather/gsmhardv2.jsonl'
        save_input_dir = 'results_gather/gsm'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        puzzles = read_dataset_gsm(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            question_list.append(question)
            solution = puzzle['solution_data']
            solution_list.append(solution)
    elif task_name == 'math_geometry':
        dataset_input_dir = 'dataset_gather/MATH/train/geometry'
        save_input_dir = 'results_gather/math_geometry'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        puzzles = read_dataset_math_geometry(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            question_list.append(question)
            solution = puzzle['solution_data']
            solution_list.append(solution)
    elif task_name == 'math_counting_and_probability':
        dataset_input_dir = 'dataset_gather/MATH/train/counting_and_probability'
        save_input_dir = 'results_gather/math_counting_and_probability'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        puzzles = read_dataset_math_counting_and_probability(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            question_list.append(question)
            solution = puzzle['solution_data']
            solution_list.append(solution)
    elif task_name == 'BoxNet_v2':
        dataset_input_dir = 'dataset_gather/BoxNet1_v2_dataset'
        save_input_dir = 'results_gather/BoxNet_v2'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        grid_sizes = [(2, 6, 3, 5), (2, 8, 3, 5), (3, 6, 4, 6), (4, 8, 4, 6), (4, 6, 4, 6), (2, 8, 4, 6), (5, 5, 4, 8)]
        for iteration_num in range(20):
            for rows, cols, num_box_low, num_box_high in grid_sizes:
                samples = read_samples_BoxNet_V2(dataset_input_dir + f'/question_samples_{rows}_{cols}_{iteration_num}.json')
                sample = samples[0]
                puzzles.append(sample)
                question = generate_prompt_BoxNet_V2(sample)
                question_list.append(question)
    elif task_name == 'BoxLift':
        dataset_input_dir = 'dataset_gather/BoxLift_dataset'
        save_input_dir = 'results_gather/BoxLift'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        puzzles = read_dataset_boxlift(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            question_list.append(question)
            solution = puzzle['solution_data']
            solution_list.append(solution)
    elif task_name == 'Blocksworld':
        dataset_input_dir = 'dataset_gather/Blocksworld_dataset'
        save_input_dir = 'results_gather/Blocksworld'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        puzzles = read_dataset_blocksworld(dataset_input_dir)
        for puzzle in puzzles:
            question = puzzle['question']
            question_list.append(question)
            solution = puzzle['solution_data']
            solution_list.append(solution)
    elif task_name == 'Gridworld':
        dataset_input_dir = 'dataset_gather/GridWorld1_dataset'
        save_input_dir = 'results_gather/GridWorld1'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        grid_sizes = [(4, 4, 4, 5), (4, 5, 4, 5), (5, 5, 4, 5), (5, 5, 6, 8), (5, 6, 7, 10), (6, 6, 7, 10), (6, 7, 7, 10)]
        for iteration_num in range(20):
            for rows, cols, num_goals, num_obstacles in grid_sizes:
                sample = read_samples_gridworld(dataset_input_dir + f'/gridworld_sample_{rows}x{cols}_{iteration_num + 1}.json')
                puzzles.append(sample)
                question = generate_prompt_gridworld(sample)
                question_list.append(question)
    elif task_name.startswith('reasoning_gym_'):
        dataset_name = task_name.split('reasoning_gym_')[1]
        save_input_dir = os.path.join('results_gather', 'reasoning_gym', dataset_name)

        dataset_path= os.path.join('dataset_gather', 'reasoning_gym', f'{dataset_name}.csv')
        data_list = pd.read_csv(dataset_path)
        question_list = data_list['question'].tolist()
        solution_list = data_list['full_data'].tolist()

    elif task_name == 'gpqa':
        dataset_input_dir = 'dataset_gather/gpqa'
        save_input_dir = 'results_gather/gpqa'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        print(f'GPQA, Model_name: {model_name}\n')
        puzzles = read_dataset_gpqa(dataset_input_dir)
        for puzzle in puzzles:
            # Format as multiple choice question
            formatted_question = format_gpqa_question(puzzle)
            question_list.append(formatted_question)
            # Use the letter answer (A, B, C, D) instead of text answer
            solution_list.append(puzzle['correct_letter'])

    return solution_list, question_list, target_list, puzzles, solution_data_list, question_constrained_list, question_matrix_list, number_list, word_list, letter_list, save_input_dir

### TODO: add elif, add a new extract_equation_with_GPT_combi_calcu fucntion with a different name.
def verify_solution_func_gather(i, task_name, response, save_code_dir, question, solution, target, puzzles, solution_data_list, solution_list, question_constrained_list, question_matrix_list, number_list_item, word, letter):
    # Verify solution based on task type
    ### Unchanged
    original_response = response

    code_block_list = extract_code(response)
    for index, code_string in enumerate(code_block_list):
        with open(save_code_dir + f"/code_1_{index}.py", "w") as f:
            f.write(code_string)

    # Test the generated code
    if not os.path.exists(save_code_dir + f"/code_1_0.py"):
        pass
    else:
        try:
            result = subprocess.run(
                ["python3", "-c", f"exec(open('{save_code_dir}/code_1_0.py').read()); print(result)"],
                capture_output=True, text=True, timeout=60)
            if result.stdout == '':
                result = subprocess.run(
                    ["python3", "-c", f"exec(open('{save_code_dir}/code_1_0.py').read()); print(Answer)"],
                    capture_output=True, text=True, timeout=60)
            if result.stdout == '':
                result = subprocess.run(
                    ["python3", "-c", f"exec(open('{save_code_dir}/code_1_0.py').read()); print(answer)"],
                    capture_output=True, text=True, timeout=60)

            # if '<<<' in result.stdout and '>>>' in result.stdout:
            response = result.stdout
            errors = result.stderr
            response = response[:10000]; errors = errors[:10000]
        except Exception as e:
            pass

    ### To do, add tasks to extract the solution from the generated response and verify the correctness, here we check both response and original_response
    ### Take care the names of variables to be the same as other tasks since we need to save several variables in the following process.
    if task_name == 'logical_equation':
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_logical_equation(response)
        extracted_text_1, _ = extract_and_check(output_1)
        extracted_text_1 = extracted_text_1.strip()
        try:
            solution_1 = eval(extracted_text_1)
        except:
            solution_1 = []

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_logical_equation(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        extracted_text_2 = extracted_text_2.strip()
        try:
            solution_2 = eval(extracted_text_2)
        except:
            solution_2 = []

        constraints = [line.split('. ')[1] for line in question.split('\n')
                       if line.strip() and line[0].isdigit()]

        True_false_result_1 = verify_solution_logical_equation(solution, solution_1, constraints)
        True_false_result_2 = verify_solution_logical_equation(solution, solution_2, constraints)
    elif task_name == 'combinatorial_calculation':
        dataset_input_dir = 'dataset_gather/combinatorial_calculation'
        evaluator = ArithmeticPuzzleEvaluator(dataset_input_dir)
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_combi_calcu(response)
        try:
            match = re.search(r'<<<\[(.*?)\]>>>', output_1)
            solution_1 = match.group(1)
        except:
            solution_1 = ''
        extracted_text_1 = solution_1
        parsed_solution_1 = [elem.strip().strip('"\'') for elem in solution_1.split(',')]
        puzzle = puzzles[i]
        True_false_result_1, message_1 = evaluator.verify_solution(puzzle, parsed_solution_1)
        print('Feedback1: ', message_1)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_combi_calcu(original_response)
        try:
            match = re.search(r'<<<\[(.*?)\]>>>', output_2)
            solution_2 = match.group(1)
        except:
            solution_2 = ''
        extracted_text_2 = solution_2
        parsed_solution_2 = [elem.strip().strip('"\'') for elem in solution_2.split(',')]
        True_false_result_2, message_2 = evaluator.verify_solution(puzzle, parsed_solution_2)
        print('Feedback2: ', message_2)
    elif task_name == 'eight_queens':
        solution_data = solution_data_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_eight_queens(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_eight_queens(extracted_text_1, solution_data)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_eight_queens(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_eight_queens(extracted_text_2, solution_data)
    elif task_name == 'synthesis_decomposition':
        solution_data = solution_data_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_syn_decom(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_syn_decom(extracted_text_1, solution_data,
                                                                     solution_data['complexity'])

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_syn_decom(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_syn_decom(extracted_text_2, solution_data,
                                                                     solution_data['complexity'])
    elif task_name == 'mahjong_pattern':
        solution_data = solution_data_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_mahjong(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_mahjong(extracted_text_1, solution_data,
                                                                   solution_data['complexity'])

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_mahjong(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_mahjong(extracted_text_2, solution_data,
                                                                   solution_data['complexity'])
    elif task_name == 'statistical_counting':
        solution_data = solution_data_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_stat_counting(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_stat_counting(extracted_text_1, solution_data,
                                                                         solution_data['complexity'])

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_stat_counting(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_stat_counting(extracted_text_2, solution_data,
                                                                         solution_data['complexity'])
    elif task_name == 'new_operator':
        solution_data = solution_data_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_new_op(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_new_op(extracted_text_1, solution_data,
                                                                  solution_data['complexity'])

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_new_op(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_new_op(extracted_text_2, solution_data,
                                                                  solution_data['complexity'])
    elif task_name == 'light_puzzles':
        solution_data = solution_data_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_light(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_light(extracted_text_1, solution_data,
                                                                 solution_data['complexity'])

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_light(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_light(extracted_text_2, solution_data,
                                                                 solution_data['complexity'])
    elif task_name == 'reversi':
        # import pdb; pdb.set_trace()
        solution_data = solution_data_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_reversi(response.replace('\n', ''))
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_reversi(extracted_text_1, solution_data,
                                                                   solution_data['complexity'])

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_reversi(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_reversi(extracted_text_2, solution_data,
                                                                   solution_data['complexity'])
    elif task_name == 'matrix_transformation':
        # import pdb; pdb.set_trace()
        solution_data = solution_data_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_matrix_trans(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_matrix_trans(extracted_text_1, solution_data,
                                                                        solution_data['complexity'])

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_matrix_trans(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_matrix_trans(extracted_text_2, solution_data,
                                                                        solution_data['complexity'])
    elif task_name == '2048':
        # import pdb; pdb.set_trace()
        solution_data = solution_data_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_2048(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_2048(extracted_text_1, solution_data,
                                                                solution_data['complexity'])

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_2048(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_2048(extracted_text_2, solution_data,
                                                                solution_data['complexity'])
    elif task_name == 'pooling':
        # import pdb; pdb.set_trace()
        solution_data = solution_data_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_pooling(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_pooling(extracted_text_1, solution_data,
                                                                   solution_data['complexity'])

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_pooling(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_pooling(extracted_text_2, solution_data,
                                                                   solution_data['complexity'])
    elif task_name == 'constrained_linear_arrangement':
        # import pdb; pdb.set_trace()
        solution_data = solution_data_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_constrained(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_constrained(extracted_text_1, solution_data,
                                                                       solution_data['complexity'])

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_constrained(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_constrained(extracted_text_2, solution_data,
                                                                       solution_data['complexity'])
    elif task_name == 'logic_puzzle':
        # import pdb; pdb.set_trace()
        solution_data = solution_data_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_logic_puzzle(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_logic_puzzle(extracted_text_1, solution_data,
                                                                        solution_data['complexity'])

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_logic_puzzle(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_logic_puzzle(extracted_text_2, solution_data,
                                                                        solution_data['complexity'])

    elif task_name == 'pattern_recognition':
        solution_data = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_pattern_recognition(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_pattern_recognition(extracted_text_1, solution_data)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_pattern_recognition(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_pattern_recognition(extracted_text_2, solution_data)
    elif task_name == 'string_insertion':
        solution_data = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_string_insertion(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_string_insertion(extracted_text_1, solution_data)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_string_insertion(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_string_insertion(extracted_text_2, solution_data)
    elif task_name == 'letter_logic_diagram':
        solution_data = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_letter_logic_diagram(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_letter_logic_diagram(extracted_text_1, solution_data)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_letter_logic_diagram(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_letter_logic_diagram(extracted_text_2, solution_data)
    elif task_name == 'string_synthesis':
        solution_data = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_string_synthesis(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        # print(f'response: {response}')
        True_false_result_1, message_1 = validate_solution_string_synthesis(extracted_text_1, solution_data)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_string_synthesis(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        # print(f'original_response: {original_response}')
        True_false_result_2, message_2 = validate_solution_string_synthesis(extracted_text_2, solution_data)
    elif task_name == 'standard_sudoku':
        question_data = question_matrix_list[i]
        solution_data = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_standard_sudoku(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        # print(f'response: {response}')
        True_false_result_1, message_1 = validate_solution_standard_sudoku(extracted_text_1, question_data)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_standard_sudoku(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        # print(f'original_response: {original_response}')
        True_false_result_2, message_2 = validate_solution_standard_sudoku(extracted_text_2, question_data)
    elif task_name == 'permutations_and_combinations':
        question_data = question_constrained_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_permutations_and_combinations(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        # print(f'response: {response}')
        True_false_result_1, message_1 = validate_solution_permutations_and_combinations(extracted_text_1,
                                                                                         question_data)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_permutations_and_combinations(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_permutations_and_combinations(extracted_text_2,
                                                                                         question_data)
    elif task_name == 'string_deletion_and_modification':
        solution_data = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_string_deletion_and_modification(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_string_deletion_and_modification(extracted_text_1,
                                                                                            solution_data)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_string_deletion_and_modification(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        # print(f'original_response: {original_response}')
        True_false_result_2, message_2 = validate_solution_string_deletion_and_modification(extracted_text_2,
                                                                                            solution_data)
    elif task_name == 'minesweeper':
        solution_data = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_minesweeper(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_minesweeper(extracted_text_1, solution_data)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_minesweeper(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, message_2 = validate_solution_minesweeper(extracted_text_2, solution_data)
    elif task_name == 'cryptanalysis':
        solution_data = solution_list[i]

        extracted_text_1, _ = extract_and_check(response)
        iteration_num_1 = 0
        while extracted_text_1 == '' and iteration_num_1 < 3:
            iteration_num_1 += 1
            #token_len_response = count_total_tokens([response], [])
            #print(f'token_len_response: {token_len_response}')
            extracted_text_1 = extract_equation_with_GPT4_cryptanalysis(response)

        extracted_text_2, _ = extract_and_check(original_response)
        iteration_num_2 = 0
        while extracted_text_2 == '' and iteration_num_2 < 3:
            iteration_num_2 += 1
            extracted_text_2 = extract_equation_with_GPT4_cryptanalysis(original_response)
        solution_1 = extracted_text_1
        solution_2 = extracted_text_2
        print(f'extracted_text_1: {extracted_text_1}')
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_1, message_1 = validate_solution_cryptanalysis(extracted_text_1, solution_data)
        True_false_result_2, message_2 = validate_solution_cryptanalysis(extracted_text_2, solution_data)
    elif task_name == 'string_splitting':
        solution_data = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_string_splitting(response)
        extracted_text_1, _ = extract_and_check(output_1)
        solution_1 = extracted_text_1
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, message_1 = validate_solution_string_splitting(extracted_text_1, solution_data)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_string_splitting(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        solution_2 = extracted_text_2
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        # print(f'original_response: {original_response}')
        True_false_result_2, message_2 = validate_solution_string_splitting(extracted_text_2, solution_data)
    elif task_name == 'game24':
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_game24(response)
        extracted_text_1, _ = extract_and_check(output_1)
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1 = validate_solution_game24(number_list_item, extracted_text_1)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_game24(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2 = validate_solution_game24(number_list_item, extracted_text_2)
        solution_1 = extracted_text_1; solution_2 = extracted_text_2
    elif task_name == 'letters':
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_letters(response)
        extracted_text_1, _ = extract_and_check(output_1)
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')
        True_false_result_1, explanation_1 = evaluate_response_letters(word, letter, extracted_text_1)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_letters(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')
        True_false_result_2, explanation_2 = evaluate_response_letters(word, letter, extracted_text_2)

        solution_1 = extracted_text_1;
        solution_2 = extracted_text_2
    elif task_name == 'number_multiply':
        solution_data = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_number_multiply(response)
        extracted_text_1, _ = extract_and_check(output_1)
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_number_multiply(original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')

        True_false_result_1 = str(solution_data) in extracted_text_1 or str(
            format_number_with_commas(int(solution_data))) in extracted_text_1
        True_false_result_2 = str(solution_data) in extracted_text_2 or str(
            format_number_with_commas(int(solution_data))) in extracted_text_2

        solution_1 = extracted_text_1;
        solution_2 = extracted_text_2
    elif task_name.startswith('big_bench_hard'):
        solution_data = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = is_equiv_func_big_bench_hard(question, solution_data, response)
        extracted_text_1, _ = extract_and_check(output_1)
        extracted_text_1 = extracted_text_1.strip()
        print(f'extracted_text_1: {extracted_text_1}')

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = is_equiv_func_big_bench_hard(question, solution_data, original_response)
        extracted_text_2, _ = extract_and_check(output_2)
        extracted_text_2 = extracted_text_2.strip()
        print(f'extracted_text_2: {extracted_text_2}')

        True_false_result_1 = 'Correct' in extracted_text_1 or 'correct' in extracted_text_1
        True_false_result_2 = 'Correct' in extracted_text_2 or 'correct' in extracted_text_2

        solution_1 = extracted_text_1;
        solution_2 = extracted_text_2
    elif task_name == 'gsm':
        solution_data = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_gsm(response)
        extracted_text_1, _ = extract_and_check(output_1)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_gsm(original_response)
        extracted_text_2, _ = extract_and_check(output_2)

        True_false_result_1 = is_equiv_func_gsm(solution_data, extracted_text_1)
        True_false_result_1, _ = extract_and_check(True_false_result_1)
        True_false_result_1 = True_false_result_1 == 'True'
        True_false_result_2 = is_equiv_func_gsm(solution_data, extracted_text_2)
        True_false_result_2, _ = extract_and_check(True_false_result_2)
        True_false_result_2 = True_false_result_2 == 'True'

        solution_1 = extracted_text_1;
        solution_2 = extracted_text_2
    elif task_name == 'math_geometry':
        target_answer = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_math_geometry(response)
        extracted_text_1, _ = extract_and_check(output_1)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_math_geometry(original_response)
        extracted_text_2, _ = extract_and_check(output_2)

        True_false_result_1 = is_equiv_func_math_geometry(target_answer, extracted_text_1)
        True_false_result_1, _ = extract_and_check(True_false_result_1)
        True_false_result_1 = True_false_result_1 == 'True'
        True_false_result_2 = is_equiv_func_math_geometry(target_answer, extracted_text_2)
        True_false_result_2, _ = extract_and_check(True_false_result_2)
        True_false_result_2 = True_false_result_2 == 'True'

        solution_1 = extracted_text_1;
        solution_2 = extracted_text_2
    elif task_name == 'math_counting_and_probability':
        target_answer = solution_list[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_math_counting_and_probability(response)
        extracted_text_1, _ = extract_and_check(output_1)

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_math_counting_and_probability(original_response)
        extracted_text_2, _ = extract_and_check(output_2)

        True_false_result_1 = is_equiv_func_math_counting_and_probability(target_answer, extracted_text_1)
        True_false_result_1, _ = extract_and_check(True_false_result_1)
        True_false_result_1 = True_false_result_1 == 'True'
        True_false_result_2 = is_equiv_func_math_counting_and_probability(target_answer, extracted_text_2)
        True_false_result_2, _ = extract_and_check(True_false_result_2)
        True_false_result_2 = True_false_result_2 == 'True'

        solution_1 = extracted_text_1;
        solution_2 = extracted_text_2
    elif task_name == 'BoxNet_v2':
        sample = puzzles[i]

        extracted_text_1, _ = extract_and_check(response)
        iteration_num_1 = 0
        while extracted_text_1 == '' and iteration_num_1 < 3:
            iteration_num_1 += 1
            extracted_text_1 = extract_equation_with_GPT4_BoxNet_V2(response)

        extracted_text_2, _ = extract_and_check(original_response)
        iteration_num_2 = 0
        while extracted_text_2 == '' and iteration_num_2 < 3:
            iteration_num_2 += 1
            extracted_text_2 = extract_equation_with_GPT4_BoxNet_V2(original_response)

        True_false_result_1, message_1 = check_llm_response_BoxNet_V2(extracted_text_1, sample)
        True_false_result_2, message_2 = check_llm_response_BoxNet_V2(extracted_text_2, sample)
        print(f'\nMessage from response: {message_1}')
        print(f'\nMessage from original_response: {message_2}')

        solution_1 = extracted_text_1;
        solution_2 = extracted_text_2
    elif task_name == 'BoxLift':
        solution_data = solution_list[i]
        boxes, lifters, estimated_steps = solution_data['boxes'], solution_data['lifters'], solution_data['estimated_steps']

        extracted_text_1, _ = extract_and_check(response)
        iteration_num_1 = 0
        while extracted_text_1 == '' and iteration_num_1 < 3:
            iteration_num_1 += 1
            extracted_text_1 = extract_equation_with_GPT4_boxlift(response)

        extracted_text_2, _ = extract_and_check(original_response)
        iteration_num_2 = 0
        while extracted_text_2 == '' and iteration_num_2 < 3:
            iteration_num_2 += 1
            extracted_text_2 = extract_equation_with_GPT4_boxlift(original_response)

        True_false_result_1, remaining1, success_failure_list1 = verify_solution_boxlift(boxes, lifters, extracted_text_1, estimated_steps)
        True_false_result_2, remaining2, success_failure_list2 = verify_solution_boxlift(boxes, lifters, extracted_text_2, estimated_steps)

        solution_1 = extracted_text_1;
        solution_2 = extracted_text_2
    elif task_name == 'Blocksworld':
        solution_data = puzzles[i]['solution_data']
        initial_state, goal_state = solution_data['initial_state'], solution_data['goal_state']

        extracted_text_1, _ = extract_and_check(response)
        iteration_num_1 = 0
        while extracted_text_1 == '' and iteration_num_1 < 3:
            iteration_num_1 += 1
            extracted_text_1 = extract_equation_with_GPT4_blocksworld(response)

        extracted_text_2, _ = extract_and_check(original_response)
        iteration_num_2 = 0
        while extracted_text_2 == '' and iteration_num_2 < 3:
            iteration_num_2 += 1
            extracted_text_2 = extract_equation_with_GPT4_blocksworld(original_response)

        True_false_result_1, message_1 = validate_response_blocksworld(initial_state, goal_state, extracted_text_1)
        True_false_result_2, message_2 = validate_response_blocksworld(initial_state, goal_state, extracted_text_2)

        solution_1 = extracted_text_1;
        solution_2 = extracted_text_2
    elif task_name == 'Gridworld':
        sample = puzzles[i]

        extracted_text_1, _ = extract_and_check(response)
        iteration_num_1 = 0
        while extracted_text_1 == '' and iteration_num_1 < 3:
            iteration_num_1 += 1
            extracted_text_1 = extract_equation_with_GPT4_gridworld(response)

        extracted_text_2, _ = extract_and_check(original_response)
        iteration_num_2 = 0
        while extracted_text_2 == '' and iteration_num_2 < 3:
            iteration_num_2 += 1
            extracted_text_2 = extract_equation_with_GPT4_gridworld(original_response)

        True_false_result_1, message_1 = check_llm_response_gridworld(extracted_text_1, sample)
        True_false_result_2, message_2 = check_llm_response_gridworld(extracted_text_2, sample)
        print(f'\nMessage from response: {message_1}')
        print(f'\nMessage from original_response: {message_2}')

        solution_1 = extracted_text_1;
        solution_2 = extracted_text_2
    elif task_name.startswith('reasoning_gym_'):
        dataset_name = task_name.split('reasoning_gym_')[1]
        solution = json.loads(solution)
        target = solution['answer']

        extracted_text_1, _ = extract_and_check(response)
        iteration_num_1 = 0
        while extracted_text_1 == '' and iteration_num_1 < 3:
            iteration_num_1 += 1
            extracted_text_1 = extract_equation_with_GPT4_reasoning_gym(response, dataset_name)

        extracted_text_2, _ = extract_and_check(original_response)
        iteration_num_2 = 0
        while extracted_text_2 == '' and iteration_num_2 < 3:
            iteration_num_2 += 1
            extracted_text_2 = extract_equation_with_GPT4_reasoning_gym(original_response, dataset_name)

        True_false_result_1 = validate_solution_reasoning_gym(dataset_name, extracted_text_1, solution)
        True_false_result_2 = validate_solution_reasoning_gym(dataset_name, extracted_text_2, solution)
        solution_1 = extracted_text_1
        solution_2 = extracted_text_2
        print(f'\nresponse: {response}')
        print(f'\nextracted_text_1: {extracted_text_1}')
        print(f'\noriginal_response: {original_response}')
        print(f'\nextracted_text_2: {extracted_text_2}')
    elif task_name == 'gpqa':
        solution_data = puzzles[i]
        output_1 = None;
        iteration_num_1 = 0
        while output_1 == None and iteration_num_1 < 3:
            iteration_num_1 += 1
            output_1 = extract_equation_with_GPT4_gpqa_with_mapping(response, solution_data.get('option_mapping', {}))

        output_2 = None;
        iteration_num_2 = 0
        while output_2 == None and iteration_num_2 < 3:
            iteration_num_2 += 1
            output_2 = extract_equation_with_GPT4_gpqa_with_mapping(original_response, solution_data.get('option_mapping', {}))

        True_false_result_1, message_1 = validate_solution_gpqa(output_1, solution_data)
        True_false_result_2, message_2 = validate_solution_gpqa(output_2, solution_data)
        print(f'\nMessage from response: {message_1}')
        print(f'\nMessage from original_response: {message_2}')

        solution_1 = output_1; solution_2 = output_2
        extracted_text_1 = str(output_1); extracted_text_2 = str(output_2)

        print(f'True_false_result from response: {True_false_result_1}')
        print(f'True_false_result from original_response: {True_false_result_2}')
        print(f'target_solution: {solution}')
        print(f'extracted_answer from response: {solution_1}')
        print(f'extracted_answer from original_response: {solution_2}')
        with open(save_code_dir + f"/True_false_result_1.txt", "w") as f:
            f.write(str(True_false_result_1))
        with open(save_code_dir + f"/True_false_result_2.txt", "w") as f:
            f.write(str(True_false_result_2))
        with open(save_code_dir + f"/extracted_answer_1.txt", "w") as f:
            f.write(extracted_text_1)
        with open(save_code_dir + f"/extracted_answer_2.txt", "w") as f:
            f.write(extracted_text_2)

        if True_false_result_1 == False and True_false_result_2 == False:
            print('False')
            with open(save_code_dir + f"/success_failure.txt", "w") as f:
                f.write('False')
        elif True_false_result_1 == True or True_false_result_2 == True:
            print('True')
            with open(save_code_dir + f"/success_failure.txt", "w") as f:
                f.write('True')
        else:
            raise ValueError('Error: No True_false_result found')

        return True_false_result_1, True_false_result_2


    print(f'True_false_result from response: {True_false_result_1}')
    print(f'True_false_result from original_response: {True_false_result_2}')
    print(f'target_solution: {solution}')
    print(f'target_answer: {target}')
    print(f'extracted_text from response: {solution_1}')
    print(f'extracted_text from original_response: {solution_2}')
    with open(save_code_dir + f"/True_false_result_1.txt", "w") as f:
        f.write(str(True_false_result_1))
    with open(save_code_dir + f"/True_false_result_2.txt", "w") as f:
        f.write(str(True_false_result_2))
    with open(save_code_dir + f"/extracted_answer_1.txt", "w") as f:
        f.write(extracted_text_1)
    with open(save_code_dir + f"/extracted_answer_2.txt", "w") as f:
        f.write(extracted_text_2)

    if True_false_result_1 == False and True_false_result_2 == False:
        print('False')
        with open(save_code_dir + f"/success_failure.txt", "w") as f:
            f.write('False')
    elif True_false_result_1 == True or True_false_result_2 == True:
        print('True')
        with open(save_code_dir + f"/success_failure.txt", "w") as f:
            f.write('True')
    else:
        raise ValueError('Error: No True_false_result found')

    return True_false_result_1, True_false_result_2


def extract_equation_with_GPT4_gpqa(response: str) -> str:
    import re
    
    # First try to extract from the standardized <<<answer>>> format
    standardized_pattern = r'<<<\s*([A-D])\s*>>>'
    matches = re.findall(standardized_pattern, response, re.IGNORECASE | re.MULTILINE)
    if matches:
        return matches[-1].upper()
    
    # Fallback to original patterns for backward compatibility
    answer_patterns = [
        r'(?:Answer|ANSWER):\s*([A-D])',
        r'(?:The answer is|answer is)\s*([A-D])',
        r'\b([A-D])\s*(?:is (?:the )?correct|is (?:the )?answer)',
        r'(?:^|\n)\s*([A-D])\s*(?:\.|$)',
        r'(?:option|choice)\s*([A-D])',
        r'\b([A-D])\)\s*\S+',  # Matches "C) Blue", "A) Red", "B) +∞", etc.
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    
    return None

def extract_equation_with_GPT4_gpqa_with_mapping(response: str, option_mapping: dict) -> str:
    # First try to extract letter directly using the new standardized format
    letter_answer = extract_equation_with_GPT4_gpqa(response)
    if letter_answer:
        return letter_answer
    
    # If no letter found, try to match answer text to options
    import re
    
    # Try to extract from standardized format first (may contain full text)
    standardized_pattern = r'<<<\s*(.+?)\s*>>>'
    matches = re.findall(standardized_pattern, response, re.IGNORECASE | re.MULTILINE)
    if matches:
        answer_text = matches[-1].strip()
        # Check if it's already a letter
        if len(answer_text) == 1 and answer_text.upper() in ['A', 'B', 'C', 'D']:
            return answer_text.upper()
        # Try to find which option matches this text
        for letter, option_text in option_mapping.items():
            if answer_text in option_text or option_text in answer_text:
                return letter
    
    # Fallback to original text extraction patterns
    text_patterns = [
        r'(?:Answer|ANSWER):\s*(.+?)(?:\n|$)',
        r'(?:The answer is|answer is)\s*(.+?)(?:\n|$)',
        r'(?:Therefore|Thus),?\s+(?:the answer is\s*)?(.+?)(?:\n|$)',
    ]
    
    for pattern in text_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            answer_text = matches[-1].strip(' .*')
            # Try to find which option matches this text
            for letter, option_text in option_mapping.items():
                if answer_text in option_text or option_text in answer_text:
                    return letter
    
    return None

def validate_solution_gpqa(extracted_answer: str, solution_data: dict) -> tuple[bool, str]:
    if extracted_answer is None:
        return False, "No answer extracted from response"
    
    # Check if we have letter mapping from formatted question
    if 'correct_letter' in solution_data:
        correct_letter = solution_data['correct_letter']
        if extracted_answer == correct_letter:
            return True, f"Correct! Answer {extracted_answer} matches expected {correct_letter}"
        else:
            return False, f"Incorrect. Answer {extracted_answer} does not match expected {correct_letter}"
    else:
        # Fallback to original text matching
        correct_answer = solution_data['correct_answer'].strip()
        if extracted_answer == correct_answer:
            return True, f"Correct! Answer {extracted_answer} matches expected {correct_answer}"
        else:
            return False, f"Incorrect. Answer {extracted_answer} does not match expected {correct_answer}"
##### Combinatorial Calculation #####
@dataclass
class PuzzleInstance:
    question: str
    numbers: List[int]
    target: int
    solution: List[str]
    complexity: int
    sample_id: int

class ArithmeticPuzzleEvaluator:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        
    def read_dataset(self) -> List[PuzzleInstance]:
        """Read all puzzle instances from the dataset directory in sample_i order"""
        puzzles = []
        sample_id = 0
        
        while True:
            sample_dir = os.path.join(self.dataset_dir, f'sample_{sample_id}')
            if not os.path.exists(sample_dir):
                break
                
            try:
                # Read question
                with open(os.path.join(sample_dir, 'question.txt'), 'r') as f:
                    question = f.read().strip()
                
                # Read solution
                with open(os.path.join(sample_dir, 'solution.json'), 'r') as f:
                    solution_data = json.load(f)
                
                puzzle = PuzzleInstance(
                    question=question,
                    numbers=solution_data['numbers'],
                    target=solution_data['target'],
                    solution=solution_data['solution'],
                    complexity=solution_data['complexity'],
                    sample_id=sample_id
                )
                puzzles.append(puzzle)
                sample_id += 1
                
            except Exception as e:
                print(f"Error reading sample_{sample_id}: {e}")
                break
        
        return puzzles

    def evaluate_expression(self, expression: List[str]) -> Optional[float]:
        """Evaluate an arithmetic expression represented as a list of strings"""
        try:
            # Join the expression list into a string
            expr_str = ''.join(expression)
            # Calculate the result
            result = eval(expr_str)
            # Check if result is effectively an integer
            if isinstance(result, (int, float)) and math.isclose(result, round(result), rel_tol=1e-9):
                return round(result)
            return result
        except:
            return None

    def parse_llm_answer(self, answer: str) -> Optional[List[str]]:
        """Parse the LLM's answer from format <<<[symbols]>>> into a list of strings"""
        try:
            # Extract content between <<< and >>>
            match = re.search(r'<<<\[(.*?)\]>>>', answer)
            if not match:
                return None
                
            content = match.group(1)
            # Split by commas and clean up each element
            elements = [elem.strip().strip('"\'') for elem in content.split(',')]
            return elements
            
        except Exception as e:
            print(f"Error parsing LLM answer: {e}")
            return None

    def verify_solution(self, puzzle: PuzzleInstance, parsed_solution) -> Tuple[bool, str]:
        """
        Verify if the LLM's solution is correct for the arithmetic puzzle.

        Args:
            puzzle: PuzzleInstance containing the original problem and solution
            llm_answer: String response from the LLM in format <<<[elements]>>>

        Returns:
            Tuple of (is_correct: bool, message: str)
        """
        # Extract numbers and operations
        numbers = []
        operations = []

        for elem in parsed_solution:
            elem = elem.strip()
            # Try to convert to number
            try:
                num = int(elem)
                numbers.append(num)
            except ValueError:
                # If not a number, must be an operation or parenthesis
                if elem in ['+', '-', '*', '/', '(', ')']:
                    operations.append(elem)
                else:
                    return False, f"Invalid symbol found: {elem}"

        # Check numbers
        if sorted(numbers) != sorted(puzzle.numbers):
            return False, f"Numbers in solution {sorted(numbers)} don't match required numbers {sorted(puzzle.numbers)}"

        # Check operations are valid
        valid_ops = set(['+', '-', '*', '/', '(', ')'])
        if not all(op in valid_ops for op in operations):
            invalid_ops = [op for op in operations if op not in valid_ops]
            return False, f"Invalid operations found: {invalid_ops}"

        # Check parentheses are balanced
        paren_count = 0
        for op in operations:
            if op == '(':
                paren_count += 1
            elif op == ')':
                paren_count -= 1
            if paren_count < 0:  # Closing before opening
                return False, "Unbalanced parentheses - closing before opening"
        if paren_count != 0:  # Unequal number of open/close
            return False, "Unbalanced parentheses - unclosed parentheses"

        # Evaluate the expression
        try:
            result = self.evaluate_expression(parsed_solution)
            if result is None:
                return False, "Could not evaluate expression"

            if not math.isclose(result, puzzle.target, rel_tol=1e-9):
                return False, f"Expression evaluates to {result}, target is {puzzle.target}"

        except Exception as e:
            return False, f"Error evaluating expression: {str(e)}"

        # If we got here, everything checked out
        return True, "Correct solution"

def extract_equation_with_GPT4_combi_calcu(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<list of integers and symbols>>>, like <<<[(, 6, /, 1, ), +, 4]>>>, or <<<[6, +, 3, +, 1]>>>, or <<<[9, *, (, 6, -, 9, /, 4, )]>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer empty list <<<[0]>>. Symbols and numbers cannot be connected into one item, items like (6, 3), (4 are totally wrong. If finded, correct it like correcting wrong [(6, +, 3)] to correct [(, 6, +, 3, )].\n' \
             'Also complete expression like (9 + 9) + (4 + 6) should be corrected to list of items [(, 9, +, 9, ), +, (, 4, +, 6, )]\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

##### Logical Equation #####
def read_dataset_logical_equation(dataset_input_dir):
    question_list = []; solution_list = []
    for num_letters, num_constraints, values in [
        (9, 7, [1, 3, 4, 9, 16, 27, 36, 80, 121]),
        (9, 8, [3, 6, 9, 20, 32, 36, 80, 121, 120]),
        (11, 10, [3, 9, 16, 27, 36, 48, 75, 80, 121, 150, 225]),
        (11, 11, [3, 9, 16, 20, 39, 48, 75, 80, 121, 150, 225]),
        (13, 10, [1, 2, 3, 5, 7, 16, 15, 24, 10, 45, 28, 36, 50]),
        (13, 11, [2, 3, 5, 7, 16, 15, 24, 10, 45, 28, 36, 50, 96]),
        (13, 12, [2, 3, 5, 7, 16, 15, 24, 10, 45, 28, 36, 50, 96]),
        (15, 12, [2, 3, 5, 7, 16, 15, 24, 10, 45, 28, 36, 50, 78, 90, 100]),
        (15, 13, [1, 3, 5, 7, 16, 15, 24, 12, 45, 34, 36, 56, 78, 95, 100]),
        (15, 14, [1, 3, 5, 7, 16, 15, 24, 12, 45, 34, 36, 56, 78, 95, 100]),
    ]:
        for i in range(15):
            base_dir = os.path.join(dataset_input_dir, f'sample_{num_letters}_{num_constraints}_{i}')
            with open(os.path.join(base_dir, f'question.txt'), 'r') as f:
                question = f.read()
                question_list.append(question)
            with open(os.path.join(base_dir, f'solution.json'), 'r') as f:
                solution = json.load(f)
                solution_list.append(solution)

    # Create list of indices
    indices = list(range(len(solution_list)))
    # Shuffle the indices
    #random.shuffle(indices)

    # Create new lists using the shuffled indices
    shuffled_solutions = [solution_list[i] for i in indices]
    shuffled_questions = [question_list[i] for i in indices]
    return shuffled_solutions, shuffled_questions

def extract_equation_with_GPT4_logical_equation(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<list of integers>>>, like <<<[1, 4, 20, 6]>>>, or <<<[20, 43, 12, 50, 64]>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer empty list <<<[]>>.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def verify_solution_logical_equation(example_solution: List[int], proposed_solution: List[int], constraints: List[str]) -> bool:
    """Verify if the proposed solution satisfies all constraints"""

    if len(proposed_solution) != len(example_solution):
        return False
    if None in proposed_solution:
        return False
    #if all(isinstance(item, int) for item in proposed_solution):
    #    return False
    letters = [chr(65 + i) for i in range(len(proposed_solution))]
    solution_dict = {letters[i]: proposed_solution[i] for i in range(len(proposed_solution))}

    print(f'\nexample_solution: {example_solution}')
    print(f'proposed_solution: {proposed_solution}\n')
    for constraint in constraints:
        print(constraint)
    print('\n')

    def parse_term(term: str) -> float:
        if term[0].isalpha():  # Single letter
            return solution_dict[term]
        else:  # Term like '3B'
            multiplier = float(term[:-1])
            letter = term[-1]
            return multiplier * solution_dict[letter]

    for constraint in constraints:
        parts = constraint.split()

        if len(parts) == 3:  # A = B, A > B, A < B
            left_val = parse_term(parts[0])
            right_val = parse_term(parts[2])

            if parts[1] == "=":
                if left_val != right_val:
                    return False
            elif parts[1] == ">":
                if left_val <= right_val:
                    return False
            elif parts[1] == "<":
                if left_val >= right_val:
                    return False

        elif len(parts) == 5:  # A + B = 5, B - D = 1
            left_val = parse_term(parts[0])
            right_val = parse_term(parts[2])
            result = float(parts[4])

            if parts[1] == "+":
                if left_val + right_val != result:
                    return False
            elif parts[1] == "-":
                if left_val - right_val != result:
                    return False
    for item in proposed_solution:
        if int(item) not in example_solution:
            #print(f"Item {item} from proposed not in example")
            return False
    # Check if all numbers in proposed solution are valid
    #return sorted(proposed_solution) == sorted(example_solution)
    return True

##### Eight Queens #####
def validate_solution_eight_queens(response: str, solution_data: Dict) -> Tuple[bool, str]:
        # Parse response into queen positions
        positions = response.split(',')
        queens = []
        try:
            for pos in positions:
                row, col = map(int, pos.strip().split())
                queens.append((row, col))
        except:
            pass

        # Check number of queens
        if len(queens) != 8:
            return False, f"Wrong number of queens: {len(queens)}/8"

        # Check for blocked positions
        blocked = solution_data['blocked_positions']
        for queen in queens:
            if list(queen) in blocked:
                return False, f"Queen placed on blocked position: {queen}"

        # Check for conflicts
        for i, q1 in enumerate(queens):
            for q2 in queens[i + 1:]:
                if (q1[0] == q2[0] or  # Same row
                        q1[1] == q2[1] or  # Same column
                        abs(q1[0] - q2[0]) == abs(q1[1] - q2[1])):  # Same diagonal
                    return False, f"Conflicting queens: {q1} and {q2}"

        return True, "Valid solution"


def read_dataset_eight_queens(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_eight_queens(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<position pairs of integers>>>, like <<<0 3, 1 0, 2 4>>>, or <<<2 0, 4 3, 1 2, 5 0, 6 4>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<0 0>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation


##### Synthesis and decomposition #####
def validate_solution_syn_decom(response: str, solution: Dict[str, Union[List[str], List[List[str]]]],
                                complexity: int) -> Tuple[bool, str]:
    # import pdb; pdb.set_trace()
    # pattern = r'<<<\s*\[(.*?)\]\s*>>>'
    # match = re.search(pattern, response, re.DOTALL)
    # if not match:
    #     return False, f"Wrong response fromat"

    # answer_str = match.group(1)
    # Split by comma and clean up each element
    answer_parts = [part.strip() for part in response.split(',')]
    # import pdb; pdb.set_trace()

    # # Check required fields
    # if 'answer' not in llm_solution or 'process' not in llm_solution:
    #     return False, f"JSON needs to include answer and process fields"

    # # Check answer format
    # if not isinstance(llm_solution['answer'], list):
    #     return False, f"answer field should contain a list"

    # # Check process format
    # if not isinstance(llm_solution['process'], list):
    #     return False, f"process field should contain a list"

    # Check number of elements based on complexity
    expected_length = 5 if complexity > 4 else (4 if complexity > 2 else 3)
    if len(answer_parts) != expected_length:
        return False, f"number of agricultural product does not match the question"

    # Check if answer matches
    if complexity > 2:
        if answer_parts != solution['solution']['answer'][:expected_length]:
            return False, f"answer is not correct"
    else:
        if answer_parts[:2] != solution['solution']['answer'][:2] or answer_parts[-1] != solution['solution']['answer'][
            -2]:
            return False, f"answer is not correct"

    # # Check if process steps are valid
    # for step in llm_solution['process']:
    #     if not isinstance(step, list) or len(step) != expected_length:
    #         return False, f"process is not valid"

    return True, "Valid solution"

def read_dataset_syn_decom(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_syn_decom(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<list of values of crops>>>, like <<<3, 1, 2>>>, or <<<1, 3, 1, 2, 0>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<0>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation


##### Mahjong-type #####
def validate_solution_mahjong(response: str, solution: Dict[str, Union[List[str], List[List[str]]]], complexity: int) -> \
Tuple[bool, str]:
    # import pdb; pdb.set_trace()
    # answer_parts = [part.strip() for part in response.split(',')]
    if response == '':
        return False, f"answer cannot be empty"

    elif int(response) != solution['solution']:
        return False, f"answer is not correct"

    return True, "Valid solution"


def read_dataset_mahjong(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles


def extract_equation_with_GPT4_mahjong(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<number in final round>>>, like <<<2>>>, or <<<0>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<-1>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: '

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                            user_prompt_list=[prompt + response], response_total_list=[],
                                            logprobs=False)
    return extract_equation


##### Statistical counting #####
def validate_solution_stat_counting(response: str, solution: Dict[str, Union[List[str], List[List[str]]]],
                                    complexity: int) -> Tuple[bool, str]:
    # import pdb; pdb.set_trace()
    # answer_parts = [part.strip() for part in response.split(',')]
    if response == '':
        return False, f"answer cannot be empty"

    elif int(response) != solution['solution']:
        return False, f"answer is not correct"

    return True, "Valid solution"


def read_dataset_stat_counting(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_stat_counting(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<final score>>>, like <<<2>>>, or <<<5>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<0>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: '

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                            user_prompt_list=[prompt + response], response_total_list=[],
                                            logprobs=False)
    return extract_equation


##### new operator #####
def validate_solution_new_op(response: str, solution: Dict[str, Union[List[str], List[List[str]]]], complexity: int) -> \
Tuple[bool, str]:
    # import pdb; pdb.set_trace()
    # answer_parts = [part.strip() for part in response.split(',')]
    try:
        if response == '':
            return False, f"answer cannot be empty"

        elif float(response) != float(solution['solution']):
            return False, f"answer is not correct"
    except:
        return False, f"answer cannot be convert to float"

    return True, "Valid solution"


def read_dataset_new_op(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles


def extract_equation_with_GPT4_new_op(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<final result>>>, like <<<2>>>, or <<<5>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<0>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: '

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                            user_prompt_list=[prompt + response], response_total_list=[],
                                            logprobs=False)
    return extract_equation


##### light out #####
def validate_solution_light(response: str, solution: Dict[str, Union[List[str], List[List[str]]]], complexity: int) -> \
Tuple[bool, str]:
    try:
        answer_parts = [int(part.strip()) for part in response.split(',')]
        # import pdb; pdb.set_trace()

        if len(answer_parts) != solution['n'] ** 2:
            return False, f"number of lights in light network is not correct"

        # Check if answer matches
        if answer_parts != solution['solution']:
            return False, f"answer is not correct"
    except:
        return False, f"answer format is not correct"

    return True, "Valid solution"


def read_dataset_light(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles


def extract_equation_with_GPT4_light(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<final light network>>>, like <<<1, 1, 0, 0>>>, or <<<1, 0, 1, 0, 1, 1, 1, 1, 0>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<0>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: '

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                            user_prompt_list=[prompt + response], response_total_list=[],
                                            logprobs=False)
    return extract_equation


##### reversi #####
def validate_solution_reversi(response: str, solution: Dict[str, Union[List[str], List[List[str]]]], complexity: int) -> \
Tuple[bool, str]:
    try:
        # import pdb; pdb.set_trace()
        answer_parts = [part.strip() for part in response.split(',')]
        solution_answer = [part.strip() for part in solution['solution'].split(',')]

        if len(answer_parts) != solution['board_size'] ** 2:
            return False, f"number of position in the board is not correct"

        # Check if answer matches
        if answer_parts != solution_answer:
            return False, f"answer is not correct"
    except:
        return False, f"answer format is not correct"

    return True, "Valid solution"


def read_dataset_reversi(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles


def extract_equation_with_GPT4_reversi(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<final board situations>>>, like <<<*, *, *, *, 1, 1, 0, 0, *, *, *, *, 1, 1, 0, 0>>>, or <<<1, 0, 1, 0, 1, 1, 1, 1, 0>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Ignore any newline character in the answer. Note that if you find no final answer is answered, then directly answer <<<0>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: '

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                            user_prompt_list=[prompt + response], response_total_list=[],
                                            logprobs=False)
    return extract_equation


##### matrix transformation #####
def validate_solution_matrix_trans(response: str, solution: Dict[str, Union[List[str], List[List[str]]]],
                                   complexity: int) -> Tuple[bool, str]:
    try:
        answer_parts = [part.strip() for part in response.split(',')]
        solution = [part.strip() for part in solution['formatted_solution'].split(',')]
        # import pdb; pdb.set_trace()

        if len(answer_parts) != len(solution):
            return False, f"number of position in the board is not correct"

        # Check if answer matches
        if answer_parts != solution:
            return False, f"answer is not correct"
    except:
        return False, f"answer format is not correct"

    return True, "Valid solution"


def read_dataset_matrix_trans(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles


def extract_equation_with_GPT4_matrix_trans(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<final board situations>>>, like <<<A, B, C, D>>>, or <<<A, B, C, D, E, F, H, I, G>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<0>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: '

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                            user_prompt_list=[prompt + response], response_total_list=[],
                                            logprobs=False)
    return extract_equation


##### 2048 #####
def validate_solution_2048(response: str, solution: Dict[str, Union[List[str], List[List[str]]]], complexity: int) -> \
Tuple[bool, str]:
    try:
        # import pdb; pdb.set_trace()
        answer_parts = [part.strip() for part in response.split(',')]
        solution = [part.strip() for part in solution['solution'].replace('\n', ',').split(',')]
        # import pdb; pdb.set_trace()

        if len(answer_parts) != len(solution):
            return False, f"number of position in the board is not correct"

        # Check if answer matches
        if answer_parts != solution:
            return False, f"answer is not correct"
    except:
        return False, f"answer format is not correct"

    return True, "Valid solution"


def read_dataset_2048(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles


def extract_equation_with_GPT4_2048(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<final board layout>>>, like <<<2, 2, 0, 8>>>, or <<<2, 4, 8, 16, 2, 4, 0, 0, 0>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<0>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: '

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                            user_prompt_list=[prompt + response], response_total_list=[],
                                            logprobs=False)
    return extract_equation


##### pooling #####
def validate_solution_pooling(response: str, solution: Dict[str, Union[List[str], List[List[str]]]], complexity: int) -> \
Tuple[bool, str]:
    try:
        # import pdb; pdb.set_trace()
        answer_parts = [int(part.strip()) for part in response.split(',')]
        solution = np.concatenate(solution['solution']).tolist()
        # import pdb; pdb.set_trace()

        if len(answer_parts) != len(solution):
            return False, f"number of position in the board is not correct"

        # Check if answer matches
        if answer_parts != solution:
            return False, f"answer is not correct"
    except:
        return False, f"answer format is not correct"

    return True, "Valid solution"


def read_dataset_pooling(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles


def extract_equation_with_GPT4_pooling(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<pooling result>>>, like <<<2, 2, 0, 8>>>, or <<<2, 4, 8, 16, 2, 4, 0, 0, 0>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<0>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: '

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                            user_prompt_list=[prompt + response], response_total_list=[],
                                            logprobs=False)
    return extract_equation


##### Constrained linear arrangement #####
def validate_solution_constrained(response: str, solution: Dict[str, Union[List[str], List[List[str]]]],
                                  complexity: int) -> Tuple[bool, str]:
    try:
        # import pdb; pdb.set_trace()
        answer_parts = [part.strip() for part in response.split(',')]
        solution_answer = solution['solution']  # [part.strip() for part in solution['solution'].split(',')]

        if len(answer_parts) != len(solution_answer):
            return False, f"number of played cards needs to be same as number of rounds"

        elements = {
            "pieces": ["A", "B", "C", "D", "E"],  # Metal, Wood, Water, Fire, Earth
            "generates": {"A": "B", "B": "C", "C": "D", "D": "E", "E": "A"},
            "overcomes": {"A": "C", "C": "E", "E": "B", "B": "D", "D": "A"}
        }
        stick = {
            "pieces": ["A", "B", "C", "D"],  # Stick, Tiger, Chicken, Worm
            "beats": {"A": "B", "B": "C", "C": "D", "D": "A"}
        }
        animal = {
            "pieces": ["A", "B", "C", "D", "E", "F", "G", "H"],  # Elephant to Mouse
            "special": {"H": "A"},  # Mouse beats Elephant
            "hierarchy": {"A": 8, "B": 7, "C": 6, "D": 5, "E": 4, "F": 3, "G": 2, "H": 1}
        }
        if solution['game_type'] == "elements":
            for i, result in enumerate(solution['results']):
                player_move = solution['player_moves'][i]
                if result == 'draw':
                    if answer_parts[i] == elements['generates'][player_move] or answer_parts[i] == \
                            elements['overcomes'][player_move]:
                        return False, f"answered card cannot result in a draw for round {i + 1}"
                if result == 'win':
                    if not answer_parts[i] == solution_answer[i]:
                        return False, f"answered card cannot result in a win for round {i + 1}"
                if result == 'loss':
                    if not answer_parts[i] == solution_answer[i]:
                        return False, f"answered card cannot result in a loss for round {i + 1}"
        elif solution['game_type'] == "stick_game":
            for i, result in enumerate(solution['results']):
                player_move = solution['player_moves'][i]
                if result == 'draw':
                    if answer_parts[i] == stick['beats'][player_move] or player_move == stick['beats'][answer_parts[i]]:
                        return False, f"answered card cannot result in a draw for round {i + 1}"
                if result == 'win':
                    if not answer_parts[i] == solution_answer[i]:
                        return False, f"answered card cannot result in a win for round {i + 1}"
                if result == 'loss':
                    if not answer_parts[i] == solution_answer[i]:
                        return False, f"answered card cannot result in a loss for round {i + 1}"
        elif solution['game_type'] == "animal":
            if len(answer_parts) != len(set(answer_parts)):
                return False, f"answered card cannot repeat"
            for i, result in enumerate(solution['results']):
                player_move = solution['player_moves'][i]
                if result == 'draw':
                    if not answer_parts[i] == player_move:
                        return False, f"answered card cannot result in a draw for round {i + 1}"
                if result == 'win':
                    if not (animal["hierarchy"][player_move] > animal["hierarchy"][answer_parts[i]] or (
                            player_move == "H" and answer_parts[i] == "A")):
                        return False, f"answered card cannot result in a win for round {i + 1}"
                if result == 'loss':
                    if not (animal["hierarchy"][player_move] < animal["hierarchy"][answer_parts[i]] or (
                            player_move == "A" and answer_parts[i] == "H")):
                        return False, f"answered card cannot result in a loss for round {i + 1}"
    except:
        return False, f"answer format is not correct"

    return True, "Valid solution"


def read_dataset_constrained(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break
    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_constrained(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<opponent played card result>>>, like <<<A, B, C>>>, or <<<C, D, E, F>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<0>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: '

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                    user_prompt_list=[prompt + response], response_total_list=[],
                                    logprobs=False)
    return extract_equation


##### Logic puzzle  #####
def validate_solution_logic_puzzle(response: str, solution_data: Dict, complexity: int) -> Tuple[bool, str]:
    # Parse response into queen positions
    positions = response.split(',')
    numbers = []
    try:
        for pos in positions:
            row, col = map(int, pos.strip().split())
            numbers.append((row, col))
    except:
        pass
    # import pdb; pdb.set_trace()
    solution = solution_data['solution']
    grid = solution_data['grid']
    # Check number of queens
    if len(numbers) != len(solution):
        return False, f"Wrong number of selected number positions"

    for row in range(len(grid)):
        summation = 0
        product = 1
        for column in range(len(grid)):
            if (row, column) in numbers:
                summation += grid[row][column]
                product *= grid[row][column]
        if solution_data['constraints']['type'] == 'sum_le_4':
            if summation > 4:
                return False, f"the constraint of sum of the selected numbers in each row should be less than or equal to 4 is not fulfilled"
        if solution_data['constraints']['type'] == 'product_gt_0':
            if product <= 0:
                return False, f"the constraint of product of the selected numbers in each row should be greater than 0 is not fulfilled"

    for column in range(len(grid)):
        summation = 0
        product = 1
        for row in range(len(grid)):
            if (row, column) in numbers:
                summation += grid[row][column]
                product *= grid[row][column]
        if solution_data['constraints']['type'] == 'sum_le_4':
            if summation > 4:
                return False, f"the constraint of sum of the selected numbers in each column should be less than or equal to 4 is not fulfilled"
        if solution_data['constraints']['type'] == 'product_gt_0':
            if product <= 0:
                return False, f"the constraint of product of the selected numbers in each column should be greater than 0 is not fulfilled"

    return True, "Valid solution"


def read_dataset_logic_puzzle(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break
    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_logic_puzzle(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<selected position pairs>>>, like <<<0 3, 1 0, 2 4>>>, or <<<2 0, 4 3, 1 2, 5 0, 6 4>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<0 0>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: '
    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                    user_prompt_list=[prompt + response], response_total_list=[],
                                    logprobs=False)
    return extract_equation


#####Pattern Recognition#######
def read_dataset_pattern_recognition(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles


def extract_equation_with_GPT4_pattern_recognition(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<one position pairs of integers (row, column)>>>, like <<<1,3>>>,.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<0,0>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def validate_solution_pattern_recognition(response: str, solution_data: list) -> Tuple[bool, str]:
    try:
        # Parse response into position
        pos = response.split(',')
        row, col = pos[0].strip(), pos[1].strip()
        print(row, col)

        row1, col1 = solution_data[0], solution_data[1]
        print(row1, col1)

        # Check for conflicts
        if not(int(row) == int(row1) and int(col) == int(col1)):
            return False, f"Conflicting position: {(row, col)} and {(row1, col1)}"

        return True, "Valid solution"
    except:
        return False, f"answer format is not correct"

#####String Insertion#######
def read_dataset_string_insertion(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_string_insertion(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<a string>>>, like <<<ABCDEACE>>>,.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def validate_solution_string_insertion(response: str, solution_data: str) -> Tuple[bool, str]:
    try:
        # Parse response into position
        str1 = str(response.strip())
        print(str1)

        str2 = solution_data
        print(str2)

        # Check for conflicts
        if str1 != str2:
            return False, f"Conflicting string: {str1} and {str2}"

        return True, "Valid solution"
    except:
        return False, f"answer format is not correct"


#####Letter Logic Diagram#######
def read_dataset_letter_logic_diagram(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break
    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_letter_logic_diagram(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<a 7*7 matrix>>>, like <<<[[a,b,c,d,e,f,g],[b,.......], ......]>>>,.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<[]>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def validate_solution_letter_logic_diagram(
        response: str,
        solution_data: List[List[str]]
) -> Tuple[bool, str]:
    try:
        # 1. Safely parse the response string into a Python object
        inner_str = response.strip('[]')
        sub_strs = inner_str.split('],[')

        parsed_response = []
        for part in sub_strs:
            clean_part = part.strip('[]')
            items = clean_part.split(',')
            items = [x.strip() for x in items]
            parsed_response.append(items)
        print(parsed_response)
        print(solution_data)
        # 2. Validate the data structure
        # Check if parsed_response is a list of length 7
        if len(parsed_response) != 7:
            return False, "The response data is not a list of length 7 (rows)."

        # For each row, check if it is a list of length 7 and contains single-character strings
        for i, row in enumerate(parsed_response):
            if len(row) != 7:
                return False, f"Row {i + 1} is not a list of length 7."
            for j, item in enumerate(row):
                if parsed_response[i][j] != solution_data[i][j]:
                    return False, "The answer does not match the correct solution."

        return True, "Correct answer."
    except:
        return False, f"answer format is not correct"


#####String Synthesis#######
def read_dataset_string_synthesis(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_string_synthesis(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<< a string of the number of certain block>>>, like <<<101010000>>>,.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def validate_solution_string_synthesis(
        response: str,
        solution_data: List[int]
) -> Tuple[bool, str]:

    try:
        print("response_list"+response)
        if response == "":
            return False, f"Empty list"
        response = response.replace("[", "").replace("]", "").replace("'", "").replace(",", "").replace(" ", "")

        try:
            response_list = [int(x) for x in response]
        except ValueError as e:
            return False, f"Conversion error: {e}"

        print(response_list)
        print(solution_data)

        if response_list == solution_data:
            return True, "Valid solution"
        else:
            return False, f"Conflicting list: {response_list} and {solution_data}"
    except:
        return False, f"answer format is not correct"

#####Standard Sudoku#######
def read_dataset_standard_sudoku(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_standard_sudoku(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<< a 9*9 matrix >>>, like <<<[[1,2,3,4,5,6,7,8,9],[2,...],...,[]]>>>,.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def check_sudoku_solution(
        puzzle: List[List[int]],
        solution_candidate: List[List[int]]
) -> bool:
    # Check givens
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != 0 and puzzle[r][c] != solution_candidate[r][c]:
                return False
    # Check rows
    for r in range(9):
        row_vals = set()
        for c in range(9):
            val = solution_candidate[r][c]
            if val < 1 or val > 9 or val in row_vals:
                return False
            row_vals.add(val)
    # Check columns
    for c in range(9):
        col_vals = set()
        for r in range(9):
            val = solution_candidate[r][c]
            if val < 1 or val > 9 or val in col_vals:
                return False
            col_vals.add(val)
    # Check 3x3 subgrids
    for sub_row in range(0, 9, 3):
        for sub_col in range(0, 9, 3):
            box_vals = set()
            for r in range(sub_row, sub_row + 3):
                for c in range(sub_col, sub_col + 3):
                    val = solution_candidate[r][c]
                    if val in box_vals:
                        return False
                    box_vals.add(val)
    return True

def  validate_solution_standard_sudoku(
        response: str,
        question_data: List[List[int]]
) -> Tuple[bool, str]:

    try:
        if response == "":
            return False, "False answer."
        try:
            parsed_response = json.loads(response)
            is_correct = check_sudoku_solution(question_data, parsed_response)
        except:
            return False, "False answer."

        if is_correct:
            return True, "Correct answer."
        else:
            return False, "False answer."
    except:
        return False, f"answer format is not correct"

#####permutations_and_combinations#######
def read_dataset_permutations_and_combinations(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_permutations_and_combinations(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<< a list of strings >>>, like <<<["A", "B", "C", "D", "E"]>>>,.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<[]>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def parse_constraints(constraints_text: str) -> List[Tuple]:
    """
    Parse each line of constraints_text into a structured list of constraints.
    We handle 5 types of constraints, matching how they're generated in the puzzle code:

      1) ("fixed_position", item, pos)
         e.g. "Book A must be placed in position 3."

      2) ("left_of", item1, item2)
         e.g. "Book A must be to the left of book B."

      3) ("not_in_position", item, pos)
         e.g. "Book C cannot be placed in position 2."

      4) ("right_of", item1, item2)
         e.g. "Book D must be to the right of book A."

      5) ("adjacent_to", item1, item2)
         e.g. "Book E must be adjacent to book F."

    Returns: a list of tuples like [("left_of", "A", "B"), ("fixed_position", "C", 4), ...]
    """

    # Split the text by newlines
    lines = constraints_text.strip().split('\n')
    parsed_constraints = []

    # Regex patterns for each known constraint type
    # 1) fixed_position
    #    "Book X must be placed in position Y."
    pattern_fixed_position = re.compile(
        r'Book\s+([A-Za-z])\s+must\s+be\s+placed\s+in\s+position\s+(\d+)'
    )

    # 2) left_of
    #    "Book X must be to the left of book Y."
    pattern_left_of = re.compile(
        r'Book\s+([A-Za-z])\s+must\s+be\s+to\s+the\s+left\s+of\s+book\s+([A-Za-z])'
    )

    # 3) not_in_position
    #    "Book X cannot be placed in position Y."
    pattern_not_in_pos = re.compile(
        r'Book\s+([A-Za-z])\s+cannot\s+be\s+placed\s+in\s+position\s+(\d+)'
    )

    # 4) right_of
    #    "Book X must be to the right of book Y."
    pattern_right_of = re.compile(
        r'Book\s+([A-Za-z])\s+must\s+be\s+to\s+the\s+right\s+of\s+book\s+([A-Za-z])'
    )

    # 5) adjacent_to
    #    "Book X must be adjacent to book Y."
    pattern_adjacent_to = re.compile(
        r'Book\s+([A-Za-z])\s+must\s+be\s+adjacent\s+to\s+book\s+([A-Za-z])'
    )

    for line in lines:
        line = line.strip()

        # Try each pattern in turn:
        m_fp = pattern_fixed_position.search(line)
        if m_fp:
            item = m_fp.group(1)
            pos = int(m_fp.group(2))
            parsed_constraints.append(("fixed_position", item, pos))
            continue

        m_lo = pattern_left_of.search(line)
        if m_lo:
            item1 = m_lo.group(1)
            item2 = m_lo.group(2)
            parsed_constraints.append(("left_of", item1, item2))
            continue

        m_nip = pattern_not_in_pos.search(line)
        if m_nip:
            item = m_nip.group(1)
            pos = int(m_nip.group(2))
            parsed_constraints.append(("not_in_position", item, pos))
            continue

        m_ro = pattern_right_of.search(line)
        if m_ro:
            item1 = m_ro.group(1)
            item2 = m_ro.group(2)
            parsed_constraints.append(("right_of", item1, item2))
            continue

        m_adj = pattern_adjacent_to.search(line)
        if m_adj:
            item1 = m_adj.group(1)
            item2 = m_adj.group(2)
            parsed_constraints.append(("adjacent_to", item1, item2))
            continue

        # If no known pattern matched:
        print(f"[Warning] Unrecognized constraint format:\n  {line}")

    return parsed_constraints


def check_constraints(arrangement: List[str], constraints: List[Tuple]) -> Tuple[bool, str]:
    """
    Given a final arrangement (e.g. ["A","B","E","C","D"])
    and a list of constraint tuples, verify each constraint.

    Return:
      (True, "OK") if all constraints are satisfied
      (False, <reason>) if any constraint is violated
    """

    # Build a map { item: position_index }, using 1-based indexing
    position_map = {book: i+1 for i, book in enumerate(arrangement)}

    for constraint in constraints:
        ctype = constraint[0]

        if ctype == "fixed_position":
            # e.g. ("fixed_position", "E", 3)
            _, item, pos = constraint
            actual = position_map.get(item, -1)
            if actual != pos:
                return False, (
                    f"Constraint violated: Book {item} must be placed in position {pos}, "
                    f"but is at position {actual}."
                )

        elif ctype == "left_of":
            # e.g. ("left_of", "A", "C")
            _, item1, item2 = constraint
            if position_map.get(item1, 99999) >= position_map.get(item2, -99999):
                return False, (
                    f"Constraint violated: Book {item1} must be to the left of book {item2}, "
                    f"but positions are {item1}={position_map[item1]}, {item2}={position_map[item2]}."
                )

        elif ctype == "not_in_position":
            # e.g. ("not_in_position", "C", 1)
            _, item, pos = constraint
            if position_map.get(item, -1) == pos:
                return False, (
                    f"Constraint violated: Book {item} cannot be placed in position {pos}, "
                    f"but it is at position {pos}."
                )

        elif ctype == "right_of":
            # e.g. ("right_of", "D", "A")
            _, item1, item2 = constraint
            if position_map.get(item1, -1) <= position_map.get(item2, 99999):
                return False, (
                    f"Constraint violated: Book {item1} must be to the right of book {item2}, "
                    f"but positions are {item1}={position_map[item1]}, {item2}={position_map[item2]}."
                )

        elif ctype == "adjacent_to":
            # e.g. ("adjacent_to", "A", "B")
            _, item1, item2 = constraint
            pos_diff = abs(position_map.get(item1, -1) - position_map.get(item2, -1))
            if pos_diff != 1:
                return False, (
                    f"Constraint violated: Book {item1} must be adjacent to book {item2}, "
                    f"positions are {item1}={position_map[item1]}, {item2}={position_map[item2]}."
                )

        else:
            return False, f"Unknown constraint type: {constraint}"

    # If no constraint was violated
    return True, "OK"


def check_solution(constraints_text: str, correct_solution: List[str]) -> Tuple[bool, str]:
    """
    Entry point to validate a solution against puzzle constraints described in text.
    1) parse the constraints
    2) check them one by one on the provided correct_solution

    Returns (True, "OK") if all constraints are satisfied,
    otherwise (False, error_reason).
    """
    constraints = parse_constraints(constraints_text)
    valid, reason = check_constraints(correct_solution, constraints)
    return valid, reason
def  validate_solution_permutations_and_combinations(
        response: str,
        question_data: str
) -> Tuple[bool, str]:

    try:
        print("response" + response)
        print("question_data" + question_data)
        if response == "[]" or response == "":
            return False, "Empty answer"

        response_solution = json.loads(response)
        result, message = check_solution(question_data, response_solution)

        if result:
            return True, message
        else:
            return False, message
    except:
        return False, f"answer format is not correct"


#####String Deletion And Modification#######
def read_dataset_string_deletion_and_modification(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_string_deletion_and_modification(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<< a string>>>, like <<bbbababab>>>,.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def  validate_solution_string_deletion_and_modification(
        response: str,
        solution_data: str
) -> Tuple[bool, str]:
    try:
        print("response" + response)
        print("solution_data" + solution_data)
        if response == "[]" or response == "":
            return False, "Empty answer"

        if str(response) == str(solution_data):
            return True, "Correct Answer"
        else:
            return False, "False"
    except:
        return False, f"answer format is not correct"


#####Minesweeper#######
def read_dataset_minesweeper(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_minesweeper(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<< a matrix >>>, like <<<[[1,2], [1,3], [2,1], ...]>>>,.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<[]>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def  validate_solution_minesweeper(
        response: str,
        solution_data: List[List[int]]
) -> Tuple[bool, str]:
    print("response" + response)
    print("solution_data")
    print(solution_data)

    try:
        if response == "[]" or response == "":
            return False, "Empty answer"

        try:
            list_data = json.loads(response)
        except json.JSONDecodeError as e:
            return False, "JSON Error"

        print(list_data)

        list_set = set(map(tuple, list_data))  # 将每个坐标转换为元组，再转换为集合
        solution_set = set(map(tuple, solution_data))

        if list_set == solution_set:
            print("The coordinates are the same.")
            return True, "Correct Answer"
        else:
            print("The coordinates are different.")
            return False, "False"
    except:
        return False, f"answer format is not correct"

#####String Deletion And Modification#######
def read_dataset_string_deletion_and_modification(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_string_deletion_and_modification(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<< a string>>>, like <<bbbababab>>>,.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def  validate_solution_string_deletion_and_modification(
        response: str,
        solution_data: str
) -> Tuple[bool, str]:
    try:
        print("response" + response)
        print("solution_data" + solution_data)
        if response == "[]" or response == "":
            return False, "Empty answer"

        if str(response) == str(solution_data):
            return True, "Correct Answer"
        else:
            return False, "False"
    except:
        return False, f"answer format is not correct"


#####Cryptanalysis#######
def read_dataset_cryptanalysis(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_cryptanalysis(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<< a string >>>, like <<<AC10>>>,.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def  validate_solution_cryptanalysis(
        response: str,
        solution_data: List[str]
) -> Tuple[bool, str]:
    print("response: " + response)
    print(f'solution_data: {solution_data}')

    input_prompt_equiv_func = r'Evaluate whether the following list pair has the same values.' \
                              r'Neglect the format difference and the extra text. The order of the numbers matter! Different order same numbers are still False!' \
                              r'The examples are: ([\'5\', \'6\', Z, V], [5, 6, Z, V], True), ([5, 6, Z, V], [6, 5, Z, V], False), (, [6, 5, Z, V], False), (<<<>>>, [6, 5, Z, V], False)' \
                              r'(51XH, [5, 1, X, H], True), (87YS, [\'7\', \'8\', \'Y\', \'S\'], False)' \
                              r', ([\'3\', \'7\', \'W\', \'E\'], [\'0\', \'7\', \'W\', \'E\'], False)' \
                              r'In the end of your response, answer <<<True>>> or <<<False>>>'

    input_prompt_equiv_func = input_prompt_equiv_func + f'\n({solution_data}, {response}), Your answer:'
    check_response = GPT_response('Your are a helpful checker for list comparison.', input_prompt_equiv_func,
                            model_name='gpt-4o',
                            code_interpreter=False, user_prompt_list=[input_prompt_equiv_func],
                            response_total_list=[], logprobs=False)
    while 'True' not in check_response and 'False' not in check_response:
        check_response = GPT_response('Your are a helpful checker for list comparison.', input_prompt_equiv_func,
                                      model_name='gpt-4o',
                                      code_interpreter=False, user_prompt_list=[input_prompt_equiv_func],
                                      response_total_list=[], logprobs=False)
    if 'True' in check_response:
        return True, ''
    else:
        return False, ''

#####String Splitting#######
def read_dataset_string_splitting(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0

    while True:
        sample_dir = os.path.join(dataset_dir, f'sample_{sample_id}')
        if not os.path.exists(sample_dir):
            print(f'No sample_{sample_dir} found')
            break

        try:
            # Read question and solution
            question_path = os.path.join(sample_dir, 'question.txt')
            solution_path = os.path.join(sample_dir, 'solution.json')

            with open(question_path, 'r') as f:
                question = f.read().strip()

            with open(solution_path, 'r') as f:
                solution_data = json.load(f)

            puzzles.append({
                'sample_id': sample_id,
                'question': question,
                'solution_data': solution_data
            })
            sample_id += 1

        except Exception as e:
            print(f"Error reading sample_{sample_id}: {e}")
            break

    shuffled_puzzles = puzzles.copy()  # Create a copy to avoid modifying original
    #random.shuffle(shuffled_puzzles)
    return shuffled_puzzles

def extract_equation_with_GPT4_string_splitting(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<< a string representing the outcome in the order of machines A, B, C, ... parts X, Y, Z, ...>>>, like <<<112...>>>,.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<>>>. If the initial answer follows wrong format, then correct it.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def  validate_solution_string_splitting(
        response: str,
        solution_data: List[str]
) -> Tuple[bool, str]:
    print("response" + response)
    print("solution_data")
    print(solution_data)
    try:
        if response == "[]" or response == "":
            return False, "Empty answer"
        solution_str = ''.join(solution_data)
        print(solution_str)

        if response == solution_str:
            print("The strings are the same.")
            return True, "Correct Answer"
        else:
            print("The strings are different.")
            return False, "False"
    except:
        return False, f"answer format is not correct"

##### Game24 #####
def extract_equation_with_GPT4_game24(response):
    prompt = 'Your task is to extract the equation from the given answer by another LLM:\n' \
             'Note that the equation should include four numbers be in the form like ((11 * 8) + 8) / 4, ((3 * 5) - 12) * 8, ((7 - 4) * 11) - 9 = 24,' \
             '((6 / 3) * 7) + 10, (37 - (29 - 16)), ((19 + 18) - 13) = 24\n No other symbols or texts should be included. If included, remove it.' \
             'Here is the reponse, return your answer with the format <<<equation>>>, like <<<((7 - 4) * 11) - 9 = 24>>>. ' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list=[prompt + response], response_total_list=[], logprobs=False)
    return extract_equation

def validate_solution_game24(number_list, extracted_text):
    if "=" not in extracted_text:
        extracted_text = extracted_text + " = 24"

    # Split the extracted_text into left and right parts
    equation_part_list = extracted_text.split("=")
    if len(equation_part_list) == 2:
        left_side = equation_part_list[0]
        right_side = equation_part_list[1]
    elif len(equation_part_list) > 2:
        left_side = equation_part_list[0]
        right_side = equation_part_list[-1]
    #left_side, right_side = extracted_text.split("=")

    # Evaluate the right side
    try:
        right_value = eval(right_side.strip())
        if right_value != 24:
            return False
    except:
        return False

    # Prepare the number list as a multiset
    number_multiset = sorted(number_list)

    # Extract numbers and operators from the left side using regex
    # Supports LaTeX expressions as well
    left_side_numbers = re.findall(r'\d+', left_side)
    left_side_numbers = list(map(int, left_side_numbers))
    left_side_numbers_sorted = sorted(left_side_numbers)

    if number_multiset != left_side_numbers_sorted:
        return False

    # Replace LaTeX syntax with Python syntax for evaluation
    left_side = left_side.replace(r'\times', '*').replace(r'\div', '/')
    left_side = re.sub(r'\\frac{(\d+)}{(\d+)}', r'(\1/\2)', left_side)

    # Evaluate the left side
    try:
        left_value = eval(left_side.strip())
        if left_value != 24:
            return False
    except:
        return False
    return True

##### Letters #####
def evaluate_response_letters(word, target_letter, llm_response):
    """
    Evaluate the LLM's response for correctness.

    :param word: The test word
    :param target_letter: The letter that was counted
    :param llm_response: The response from the LLM
    :return: A tuple (is_correct, explanation)
    """
    # Extract count and positions from LLM response
    match = re.search(r"Count: (\d+), Positions: \[([\d, ]+)\]", llm_response)
    if not match:
        return False, "Response format is incorrect"

    llm_count = int(match.group(1))
    llm_positions = [int(pos) for pos in match.group(2).split(',')]

    # Calculate correct count and positions
    correct_count = word.count(target_letter)
    correct_positions = [i + 1 for i, letter in enumerate(word) if letter == target_letter]

    if llm_count != correct_count:
        return False, f"Incorrect count. Expected {correct_count}, got {llm_count}"

    if set(llm_positions) != set(correct_positions):
        return False, f"Incorrect positions. Expected {correct_positions}, got {llm_positions}"

    return True, "Correct response"

def extract_equation_with_GPT4_letters(response):
    prompt = 'Your task is to extract the final answer from the given answer by another LLM:\n' \
             'Note that the final answer should follow strictly the format like Count: 5, Positions: [2, 4, 13, 17, 22], Count: 1, Positions: [5], ' \
             'Count: 4, Positions: [3, 11, 18, 24] \n' \
             'Here is the response, return your answer with the format <<<final answer>>>, like <<<Count: 4, Positions: [3, 11, 18, 24]>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer are answered, then directly answer <<<Count: 0, Positions: []>>>.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs=False)
    return extract_equation

def create_prompt_letters(word, target_letter):
    prompt = f"How many '{target_letter}'s are in the word '{word}' and what are their positions? The position in the word counts from 1 not 0.\n"
    prompt += "Surround the answer with <<<content>>>. "
    prompt += f"Please respond in the format: <<<Count: X, Positions: [Y, Z, ...]>>>, such as <<<Count: 2, Positions: [1, 3]>>>.\n" \
              f"Your answer:\n"
    return prompt

def read_words_from_file_letters(filename):
    """
    Read a list of words from a JSON file.

    :param filename: Name of the file to read from
    :return: List of words
    """
    with open(filename, 'r') as f:
        words = json.load(f)
    #print(f"Words read from {filename}")
    return words

#####Number Multipy#######
def read_dataset_number_multipy(dataset_dir: str) -> List[Dict]:
    """Read all puzzle instances from the dataset directory in sample_i order"""
    puzzles = []
    sample_id = 0
    # count_total_sample = 0

    for digit_num_list in [[1, 2, 4], [1, 3, 4], [1, 2, 2, 4]]:
        dir_digit_name = f'digit'
        for digit_num in digit_num_list:
            dir_digit_name += f'_{digit_num}'
            dataset_base_dir = os.path.join(dataset_dir, f'{dir_digit_name}')
        for i in range(0, 20):
            # count_total_sample += 1
            dataset_base_dir_sample = os.path.join(dataset_base_dir, f'sample_{i}')
            generated_num_list = read_value_list(dataset_base_dir_sample + f"/input_value_list.txt")
            target_answer = read_answer(dataset_base_dir_sample + f"/target_answer.txt")

            equation_prompt = f'{generated_num_list[0]}'
            for generated_num in generated_num_list[1:]:
                equation_prompt += f'*{generated_num}'
            question = f'What is the result of ' + equation_prompt + '?'
            puzzles.append({
                'digit_num': digit_num,
                'sample_id': sample_id,
                'question': question,
                'solution_data': target_answer
            })
    # print("count_total_sample:" + str(count_total_sample))
    return puzzles

def extract_equation_with_GPT4_number_multiply(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<number>>>, like <<<43243.4>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<No answer found>>>.\n' \
             'If there is equation in the answer but no final numbers, do not calculate the number by yourself.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs=False)
    return extract_equation

def validate_solution_number_multiply(llm_response, target_answer):
    if int(llm_response) == target_answer:
        return True
    return False
def format_number_with_commas(number):
    return f"{number:,}"
def read_value_list(file_path):
    with open(file_path, 'r') as f:
        value_list = f.read()
    return ast.literal_eval(value_list)

def read_answer(file_path):
    with open(file_path, 'r') as f:
        answer = f.read()
    return int(answer)

#####Big Bench Hard#######
def read_dataset_big_bench_hard(dataset_dir: str, bbh_task_name:str) -> List[Dict]:
    puzzles = []
    DATA_PATH = dataset_dir + f'/{bbh_task_name}.json'
    question_json_list = []
    with open(DATA_PATH, 'r') as file:
        for line in file:
            question_json_list.append(json.loads(line))

    for i in range(0, len(question_json_list[0]['examples'])):
        # print(f'Sample num: {i} in {env_name}, total is: {len(question_json_list[0]["examples"])}')
        data = question_json_list[0]['examples'][i]
        question = data['input'] + f'\n' + f'\nOutput final answer with the format <<<answer>>>.'
        target_answer = data['target']

        puzzles.append({
            'example_data': data,
            'question': question,
            'solution_data': target_answer
        })

    return puzzles
def is_equiv_func_big_bench_hard(question, target_answer, response):
    input_prompt_equiv_func = r'Evaluate whether the answer given by another LLM is correct. ' \
                              r'I will give you the question, the answer from another LLM, and the target answer. ' \
                              r'Then you need to extract the final answer from the tested LLM response and evaluate the correctness of the answer. ' \
                              r'In the end of your response, answer with the list of both extracted answer and Correct/Wrong judgement surrounded by <<<>>>. ' \
                              r'The examples are: <<<[No clear answer, Wrong]>>>, <<<[Yes, Wrong]>>>, <<<[No, Wrong]>>>, <<<[(C) 03/07/2017, Correct]>>>, <<<[(F) 01/24/1947, Wrong]>>>, ' \
                              r'<<<[(D) The motorcyle is the oldest, Wrong]>>>, <<<[(F) The quail is the fourth from the left, Correct]>>>, ' \
                              r'<<<[True, Correct]>>>, <<<[True, Wrong]>>>, <<<[False, Correct]>>>. ' \
                              r'<<<[]]]]))), Correct]>>>, <<<[]]>>>, Wrong]>>>, <<<[))))]]]>, Correct]>>>. ' \
                              f'\nMost of the time, the final answer is displayed at the end of the LLM response. However, if there is no clear answer, just answer <<<[No clear answer, Wrong]>>>. ' \
                              f'Do not change or summarize the answer by yourself. Just compare tested LLM answer with the target answer. Especially whether the options are matching! ' \
                              f'For instance, A and D options are not matching, which is judged as wrong! ' \
                              f'Now the question is: {question}; the tested answer is: {response}; the target answer is: {target_answer}. Your evaluation answer: '

    response = GPT_response('', input_prompt_equiv_func, model_name='gpt-4o',
                            code_interpreter=False, user_prompt_list = [input_prompt_equiv_func], response_total_list = [], logprobs = False)
    return response



#####Big Bench Hard#######
def read_dataset_gsm(dataset_dir: str) -> List[Dict]:
    puzzles = []
    question_json_list = []
    with open(dataset_dir, 'r') as file:
        for line in file:
            question_json_list.append(json.loads(line))
    for i in range(1, len(question_json_list), 25):
        data = question_json_list[i]
        target_answer = data['target']
        question = data[
                       'input'] + f'\n' + f'\nOutput final answer with the format <<<answer>>> such as <<<123.42>>>, <<<125.0>>>, <<<-9867>>>.\nYour answer: '
        puzzles.append({
            'example_data': data,
            'question': question,
            'solution_data': target_answer
        })
    return puzzles
def extract_equation_with_GPT4_gsm(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<list>>>, like <<<43243.4>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<No answer found>>>.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def is_equiv_func_gsm(target_answer, extracted_text):
    input_prompt_equiv_func = r'Evaluate whether the following numerical pair has the same values.' \
                              r'Neglect the format difference and the extra text like units and names and equations.' \
                              r'The value can be regarded as the same if they are < 1e-3 relative difference.' \
                              r'The examples are: ("12", "12.0", True), ("5*sqrt(13)", "15.97112779602377", False),' \
                              r'("10\text{ inches}", "10.0", True), ("42", "41.99999999999998", True), ("frac{63}{64}", "0.984375", True),' \
                              r'("frac{5\sqrt{5}}{3}", "5\sqrt{5}/3", True), (\tfrac34, "3/4", True), ("frac{1033}{4}+30\sqrt{3}", "169.0", False), ("AB=12+12\sqrt{3}", "12(\sqrt{3} + 1)", True),' \
                              r'((18, -18), (18, -18), True). ' \
                              r'In the end of your response, answer <<<True>>> or <<<False>>>'
    input_prompt_equiv_func = input_prompt_equiv_func + f'\n({target_answer}, {extracted_text}), Your answer:'
    response = GPT_response('Your are a helpful checker for math expressions.', input_prompt_equiv_func, model_name='gpt-4o',
                            code_interpreter=False, user_prompt_list = [input_prompt_equiv_func], response_total_list = [], logprobs = False)
    return response

#####Math Geometry#######
def read_dataset_math_geometry(dataset_dir: str) -> List[Dict]:
    puzzles = []
    for i in range(0, 1000):
        problem_path = dataset_dir + f'/{i}.json'
        if os.path.exists(problem_path):
            with open(problem_path, 'r') as file:
                data = json.load(file)
            question = data[
                           'problem'] + f'\n' + f'\nOutput final answer with the format <<<answer>>> such as <<<123.42>>>, <<<125.0>>>, <<<-9867>>>.\nYour answer: '
            target = data['solution']
            target_answer = remove_boxed(last_boxed_only_string(target))
            puzzles.append({
                'example_data': data,
                'question': question,
                'solution_data': target_answer
            })

    return puzzles

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def extract_equation_with_GPT4_math_geometry(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<list>>>, like <<<43243.4>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<No answer found>>>.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def is_equiv_func_math_geometry(target_answer, extracted_text):
    input_prompt_equiv_func = r'Evaluate whether the following numerical pair has the same values.' \
                              r'Neglect the format difference and the extra text like units and names and equations.' \
                              r'The value can be regarded as the same if they are < 1e-3 relative difference.' \
                              r'The examples are: ("12", "12.0", True), ("5*sqrt(13)", "15.97112779602377", False),' \
                              r'("10\text{ inches}", "10.0", True), ("42", "41.99999999999998", True), ("frac{63}{64}", "0.984375", True),' \
                              r'("frac{5\sqrt{5}}{3}", "5\sqrt{5}/3", True), (\tfrac34, "3/4", True), ("frac{1033}{4}+30\sqrt{3}", "169.0", False), ("AB=12+12\sqrt{3}", "12(\sqrt{3} + 1)", True),' \
                              r'((18, -18), (18, -18), True). ' \
                              r'In the end of your response, answer <<<True>>> or <<<False>>>'
    input_prompt_equiv_func = input_prompt_equiv_func + f'\n({target_answer}, {extracted_text}), Your answer:'
    response = GPT_response('Your are a helpful checker for math expressions.', input_prompt_equiv_func, model_name='gpt-4o',
                            code_interpreter=False, user_prompt_list = [input_prompt_equiv_func], response_total_list = [], logprobs = False)
    return response

#####Math Counting and Probability#######
def read_dataset_math_counting_and_probability(dataset_dir: str) -> List[Dict]:
    puzzles = []
    for i in range(0, 1000):
        problem_path = dataset_dir + f'/{i}.json'
        if os.path.exists(problem_path):
            with open(problem_path, 'r') as file:
                data = json.load(file)
            question = data[
                           'problem'] + f'\n' + f'\nOutput final answer with the format <<<answer>>> such as <<<123.42>>>, <<<125.0>>>, <<<-9867>>>.\nYour answer: '
            target = data['solution']
            target_answer = remove_boxed(last_boxed_only_string(target))
            puzzles.append({
                'example_data': data,
                'question': question,
                'solution_data': target_answer
            })
    return puzzles

def extract_equation_with_GPT4_math_counting_and_probability(response):
    prompt = 'Your task is to extract the final numerical answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<list>>>, like <<<43243.4>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<No answer found>>>.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs = False)
    return extract_equation

def is_equiv_func_math_counting_and_probability(target_answer, extracted_text):
    input_prompt_equiv_func = r'Evaluate whether the following numerical pair has the same values.' \
                              r'Neglect the format difference and the extra text like units and names and equations.' \
                              r'The value can be regarded as the same if they are < 1e-3 relative difference.' \
                              r'The examples are: ("12", "12.0", True), ("5*sqrt(13)", "15.97112779602377", False),' \
                              r'("10\text{ inches}", "10.0", True), ("42", "41.99999999999998", True), ("frac{63}{64}", "0.984375", True),' \
                              r'("frac{5\sqrt{5}}{3}", "5\sqrt{5}/3", True), (\tfrac34, "3/4", True), ("frac{1033}{4}+30\sqrt{3}", "169.0", False), ("AB=12+12\sqrt{3}", "12(\sqrt{3} + 1)", True),' \
                              r'((18, -18), (18, -18), True). ' \
                              r'In the end of your response, answer <<<True>>> or <<<False>>>'
    input_prompt_equiv_func = input_prompt_equiv_func + f'\n({target_answer}, {extracted_text}), Your answer:'
    response = GPT_response('Your are a helpful checker for math expressions.', input_prompt_equiv_func, model_name='gpt-4o',
                            code_interpreter=False, user_prompt_list = [input_prompt_equiv_func], response_total_list = [], logprobs = False)
    return response


#####BoxNet_V2#######
def read_samples_BoxNet_V2(filename="question_samples.json"):
    """Read and return samples from the saved file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    with open(filename, "r") as f:
        samples = json.load(f)
    return samples

########################################
# 2. Prompt Generation for LLM Testing
########################################

def generate_prompt_BoxNet_V2(sample):
    prompt = f"""You are given the following planning problem:

Grid dimensions: {sample['grid']['rows']} x {sample['grid']['cols']}
Cells: {sample['grid']['cells']}
Adjacency: {json.dumps(sample['grid']['adjacency'], indent=2)}

There is one robot arm in each cell: {sample['arms']}

Boxes: {sample['boxes']}
Initial state: {json.dumps(sample['initial_state'], indent=2)}
Goal locations: {json.dumps(sample['goal_locations'], indent=2)}

Task: Generate a plan as a JSON-formatted list representing the states at successive time steps.
Each state is a dictionary mapping each box to its current cell location.
The plan must satisfy the following:
  1. The first state equals the initial state.
  2. The final state equals the goal state (i.e. each box is located in the same cell as its goal).
  3. Between successive states, a box may either remain in its current cell or move to an adjacent cell (as defined in the adjacency list).
  4. Each cell contains only one arm. Hence, in each cell at most one box can be moved at a time to the adjacent cell.
  5. If a box is at its goal cell, no further action needed for this box. Just keeping it at the goal cell.
  6. Represent each cell state as its current cell location.

In the end of your answer return a list of states and surround it with <<<>>>, such as
<<<[{{"box1": "C1,2", "box2": "C2,3"}}, {{"box1": "C1,3", "box2": "C2,3"}}, ...]>>>.

Your answer:
"""
    return prompt

########################################
# 3. LLM Response Extraction and Correctness Check
########################################

def extract_json_from_response_BoxNet_V2(response):
    """
    Extract the JSON output that is surrounded by markers <<< and >>>.
    Returns the JSON string (without the markers).
    """
    pattern = r"<<<(.*?)>>>"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        raise ValueError("Could not find markers <<< and >>> in the response.")
    json_str = matches[0].strip()
    return json_str

def extract_equation_with_GPT4_BoxNet_V2(response):
    prompt = ('Your task is to extract the final answer from the given answer by another LLM:\n'
              'The final answer should be in the format <<<answer>>>, like <<<[{{"box1": "C1,2", "box2": "C2,3"}}, {{"box1": "C1,3", "box2": "C2,3"}}, ...]>>>.\n'
              'Return only the answer in that format.\n'
              'Input text: ')
    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                    user_prompt_list=[prompt + response], response_total_list=[], logprobs=False)
    return extract_equation


def check_plan_legality_BoxNet_V2(plan, sample):
    """
    Check that for each pair of consecutive states, every box moves legally.
    A legal move is defined as:
      - If a box does not change its position, that's allowed.
      - If a box moves from one cell to another, the new cell must be adjacent to the previous cell.
      - In a single transition, at most one box may move from the same source cell.
    Returns (True, message) if all transitions are legal; otherwise (False, error message).
    """
    adjacency = sample["grid"]["adjacency"]
    goal_locations = sample["goal_locations"]
    boxes = sample["boxes"]

    for step_idx in range(len(plan) - 1):
        state_curr = plan[step_idx]
        state_next = plan[step_idx + 1]

        # Dictionary to track how many boxes are moved from each cell in this step.
        moved_from_counts = {}

        for box in boxes:
            pos_curr = state_curr.get(box)
            pos_next = state_next.get(box)

            # Once a box is marked "goal", it should remain there.
            if pos_curr == "goal":
                if pos_next != "goal":
                    return False, f"Box '{box}' was marked 'goal' in step {step_idx} but changed in step {step_idx + 1}."

            # Check if the box moved (including moves to "goal")
            if pos_curr != pos_next:
                # Increment the counter for the source cell.
                moved_from_counts[pos_curr] = moved_from_counts.get(pos_curr, 0) + 1

                # Check specific move conditions.
                if pos_next == "goal":
                    if pos_curr != goal_locations[box]:
                        return False, (f"Box '{box}' moved to 'goal' in step {step_idx + 1} but was in {pos_curr}, "
                                       f"not in its goal cell {goal_locations[box]}.")
                else:
                    if pos_curr not in adjacency:
                        return False, f"Invalid current cell {pos_curr} for box '{box}' in step {step_idx}."
                    if pos_next not in adjacency.get(pos_curr, []):
                        return False, (f"Box '{box}' moved from {pos_curr} to {pos_next} in step {step_idx + 1}, "
                                       "which are not adjacent.")

        # Verify that at most one box moved from each cell in this transition.
        for cell, count in moved_from_counts.items():
            if count > 1:
                return False, f"More than one box moved from cell {cell} in step {step_idx + 1}."

    return True, "All moves are legal."


def check_llm_response_BoxNet_V2(response, sample):
    """
    Check the correctness of the LLM response.

    Steps:
      1. Extract the JSON output from between <<< and >>>.
      2. Parse the JSON into a plan (a list of states).
      3. Verify that:
         - The plan is a list with at least 2 steps.
         - Each step is a dictionary containing all boxes.
         - The first step equals the initial state.
         - The final step equals the goal state (each box is at its goal cell or marked as 'goal').
         - Every transition between consecutive states is legal.
    Returns a tuple (is_correct, message).
    """
    try:
        json_str = extract_json_from_response(response)
        plan = json.loads(json_str)
        print('1')
    except Exception as e:
        try:
            plan = json.loads(response)
            print('2')
        except Exception as e2:
            try:
                modify_response = extract_equation_with_GPT4(response)
                json_str = extract_json_from_response(modify_response)
                plan = json.loads(json_str)
                print('3')
            except:
                plan = []

    print(f'plan: {plan}')
    print(f'Length of plan: {len(plan)}')

    if not isinstance(plan, list):
        return False, "The extracted JSON is not a list of states."
    if len(plan) < 2:
        return False, f"The plan should have at least 2 steps, but got {len(plan)} steps."

    required_boxes = set(sample["boxes"])
    for i, state in enumerate(plan):
        if not isinstance(state, dict):
            return False, f"Step {i} is not a dictionary."
        if set(state.keys()) != required_boxes:
            return False, (f"Step {i} does not contain the correct boxes. Expected: {required_boxes}, got: {set(state.keys())}")

    if plan[0] != sample["initial_state"]:
        return False, "The first step does not match the initial state."

    if plan[-1] != sample["goal_locations"]:
        return False, f"The final state does not match the goal state. Expected: {sample['goal_locations']}, got: {plan[-1]}"

    legal, message = check_plan_legality_BoxNet_V2(plan, sample)
    if not legal:
        return False, message

    return True, "The LLM response is correctly formatted and all moves are legal."


######BoxLift######
def read_dataset_boxlift(dataset_dir: str) -> List[Dict]:
    puzzles = []
    for iteration_num in range(20):
        for num_boxes, num_lifters, min_box_weight, max_box_weight, min_lifter_capacity, max_lifter_capacity in \
            [(8, 3, 10, 100, 40, 80), (12, 4, 20, 200, 30, 120), (16, 5, 30, 300, 40, 160), (20, 6, 40, 400, 50, 200), (24, 6, 40, 400, 50, 200),
             (8, 4, 10, 100, 40, 80), (12, 5, 20, 200, 30, 120), (16, 6, 30, 300, 40, 160), (20, 7, 40, 400, 50, 200), (24, 7, 40, 400, 50, 200)]:
            print(f'\n\nNum_boxes = {num_boxes}, Num_lifters = {num_lifters}, Iteration_num = {iteration_num}')

            boxes, lifters = read_test_case(dataset_dir + f'/BoxLift_{num_boxes}_{num_lifters}/BoxLift{iteration_num}/BoxLift.json')
            print(f"Initial boxes: {boxes}")
            print(f"Initial lifters: {lifters}")

            estimated_steps = estimate_steps(boxes, lifters)
            print(f"Estimated number of steps: {estimated_steps}")
            question = create_prompt_boxlift(boxes, lifters, estimated_steps)
            puzzles.append({
                'solution_data': {
                    'boxes': boxes,
                    'lifters': lifters,
                    'estimated_steps': estimated_steps,
                },
                'question': question
            })

    return puzzles

def read_test_case(filename: str) -> Tuple[List[int], List[int]]:
    """
    Read the test case (boxes and lifters) from a JSON file.

    :param filename: Name of the file to read from.
    :return: A tuple containing a list of box weights and a list of lifter capacities.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["boxes"], data["lifters"]

def estimate_steps(boxes: List[int], lifters: List[int]) -> int:
    remaining_boxes = sorted(boxes, reverse=True)  # Sort boxes in descending order
    steps = 0

    while remaining_boxes:
        steps += 1
        available_lifters = lifters.copy()

        i = 0
        while i < len(remaining_boxes) and available_lifters:
            box = remaining_boxes[i]
            combined_strength = sum(available_lifters)

            if combined_strength >= box:
                # Lift the box using as many lifters as needed
                lift_strength = 0
                used_lifters = []
                for j, lifter in enumerate(available_lifters):
                    lift_strength += lifter
                    used_lifters.append(j)
                    if lift_strength >= box:
                        break

                # Remove the used lifters and the lifted box
                for j in reversed(used_lifters):
                    available_lifters.pop(j)
                remaining_boxes.pop(i)
            else:
                i += 1  # Move to the next box if we can't lift this one

    return steps

def create_prompt_boxlift(boxes: List[int], lifters: List[int], estimated_steps) -> str:
    prompt = f"""Task: BoxLift

You are given a list of boxes with the following weights: {boxes}
And a list of lifters with the following maximum lifting capacities: {lifters}

Your task is to assign the lifters to lift all the boxes in multiple steps, following these rules:
1. Multiple boxes can be lifted in each step.
2. Each lifter can only lift one box at a time.
3. Each lifting agent can be used only once in each step.
4. Multiple lifters can combine together to lift one box if the box is too heavy for a single lifter.
5. Try to lift all the boxes using the minimum number of steps possible.
6. You need to lift all the boxes in less than or equal to {estimated_steps} steps.

Please provide your solution in the following format:
Step 1: [(Box weight, [Lifter indices]), (Box weight, [Lifter indices]), ...]
Step 2: [(Box weight, [Lifter indices]), (Box weight, [Lifter indices]), ...]
...

For example:
Step 1: [(50, [0, 2]), (30, [1]), (20, [3])]
This means in Step 1, lifters 0 and 2 are lifting a box weighing 50, lifter 1 is lifting a box weighing 30, and lifter 3 is lifting a box weighing 20.

Surround the answer with <<<content>>>.

For example, <<<Step 1: [(50, [0, 2]), (30, [1]), (20, [3])]\nStep 2: [(40, [0, 1]), (20, [2]), (20, [3])]\nStep 3:...>>>

Ensure all boxes are lifted and provide the most efficient solution possible.

Your answer:\n
"""
    return prompt

def extract_equation_with_GPT4_boxlift(response):
    prompt = 'Your task is to extract the final answer from the given answer by another LLM:\n' \
             'Note that the equation should be in the form like <<<answer>>>, <<<Step 1: [(185, [0, 1]), (108, [0, 1])]\nStep 2: [(184, [0, 1]), (75, [0, 1])]\nStep 3: [(174, [0, 1]), (70, [0, 1])]\nStep 4: [(171, [0, 1]), (63, [0]), (34, [0])]\nStep 5: [(157, [0, 1]), (32, [0]), (31, [0])]>>>, \n' \
             'Here is the reponse, return your answer with the format <<<equation>>>, like <<<Step 1: [(185, [0, 1]), (108, [0, 1])]\nStep 2: [(184, [0, 1]), (75, [0, 1])]>>>. ' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list=[prompt + response], response_total_list=[], logprobs=False)
    return extract_equation

def verify_solution_boxlift(boxes: List[int], lifters: List[int], solution: str, estimated_steps) -> Tuple[bool, List[int]]:
    remaining_boxes = boxes.copy()
    success_failure_list = []

    steps = solution.split("Step")[1:]  # Split the solution into steps
    if len(steps) > estimated_steps:
        success_failure = 'Too many steps'
        success_failure_list.append(success_failure)
        #return False, remaining_boxes, success_failure

    for index in range(min(estimated_steps, len(steps))):
        step = steps[index]
        used_lifters = set()
        try:
            assignments = eval(step.split(":")[1].strip())
            for box_weight, lifter_indices in assignments:
                # Check if the box weight is valid
                if box_weight not in remaining_boxes:
                    success_failure = 'Invalid box weight'
                    success_failure_list.append(success_failure)
                    #return False, remaining_boxes, success_failure

                elif any(index >= len(lifters) for index in lifter_indices):
                    success_failure = 'Invalid lifter index'
                    success_failure_list.append(success_failure)
                    #return False, remaining_boxes, success_failure

                # Check if lifters are used only once per step
                elif any(index in used_lifters for index in lifter_indices):
                    success_failure = 'Lifter used more than once'
                    success_failure_list.append(success_failure)
                    #return False, remaining_boxes, success_failure

                # Check if lifters can lift the box
                elif sum(lifters[i] for i in lifter_indices) < box_weight:
                    success_failure = 'Insufficient lifter strength'
                    success_failure_list.append(success_failure)
                    # return False, remaining_boxes, success_failure
                    #pass
                else:
                    remaining_boxes.remove(box_weight)
                    used_lifters.update(lifter_indices)
        except:
            success_failure = 'Invalid format'
            success_failure_list.append(success_failure)
            return False, remaining_boxes, success_failure

    return len(remaining_boxes) == 0, remaining_boxes, success_failure_list

#####Blocksworld#######
def read_dataset_blocksworld(dataset_dir: str) -> List[Dict]:
    puzzles = []

    for index in range(20):
        for num_blocks, initial_stacks, goal_stacks in [
        (5, 3, 3), (5, 4, 3), (6, 3, 3), (6, 4, 3), (7, 3, 3), (7, 4, 3), (8, 3, 3), (8, 4, 3), (9, 3, 3), (9, 4, 3),
                (10, 3, 3), (10, 4, 3), (11, 3, 3), (11, 4, 3)
    ]:
            dataset_base_dir_sample = os.path.join(dataset_dir,
                                                   f"{num_blocks}_{initial_stacks}_{goal_stacks}_{index}/")
            # Read states from file
            initial_state, goal_state = read_state_from_file(dataset_base_dir_sample + f"blocksworld_task.txt")
            print(
                f'num_blocks: {num_blocks}, initial_stacks: {initial_stacks}, goal_stacks: {goal_stacks}, index: {index}')
            # Generate prompt from the read states
            question = state_to_prompt(initial_state, goal_state)
            puzzles.append({
                'solution_data': {
                    'initial_state': initial_state,
                    'goal_state': goal_state,
                },
                'question': question
            })

    return puzzles

def extract_equation_with_GPT4_blocksworld(response):
    prompt = 'Your task is to extract the final answer of the given answer by another LLM:\n' \
             'Here is the response, return your answer with the format <<<list>>>, like <<<Yes>>>, <<<No>>>.\n' \
             'If the input text does not have <<<>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'Note that if you find no final answer is answered, then directly answer <<<No answer found>>>.\n' \
             'Input text: ' \

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False, user_prompt_list = [prompt + response], response_total_list = [], logprobs=False)
    return extract_equation


def read_state_from_file(filename: str):
    """
    Read the initial and goal states from a text file.
    """
    initial_state = {}
    goal_state = {}
    current_state = initial_state

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line == "Initial State:":
            current_state = initial_state
        elif line == "Goal State:":
            current_state = goal_state
        elif line:
            parts = line.split(': ')
            stack = parts[0] if len(parts) > 1 else parts[0][:-1]
            blocks = parts[1].split() if len(parts) > 1 else []
            current_state[stack] = blocks

    return initial_state, goal_state

def state_to_prompt(state, goal):
    """
    Convert a Blocksworld state to a prompt description for the LLM.
    """
    prompt = "Blocksworld Task:\n\nInitial State:\n"
    for stack, blocks in state.items():
        prompt += f"{stack}: {' '.join(blocks)}\n"

    prompt += "\nGoal State:\n"
    for stack, blocks in goal.items():
        prompt += f"{stack}: {' '.join(blocks)}\n"

    prompt += "\nPlease provide a series of moves to reach the goal state. " \
              "You can only move one block at a time. And that box should be the top box of the stack. " \
              "Note that from the left to the right in each stack is the order from the bottom to the top of boxes. " \
              "For example, in stack A B C D, A is the bottom box and D is the top box so that you can only move D in this case. "
    prompt += "***Be careful that you can only pick up the top box in each stack. Check this rule before your move!***. "
    prompt += "\nEach move should be in the format: 'Move [block] from [source] to [destination]'. "
    prompt += "You cannot create new stacks but only move among the existing stacks. "
    prompt += "Separate each move with a newline. Surround the answer with <<<content>>>. "
    prompt += "Answer with the required format like the example: <<<Move B from 2 to table\nMove A from 1 to 2\nMove C from 3 to 1\nMove D from 3 to 2\nMove B from 1 to 2>>>\n"
    prompt += "Each action should be separated by separate line. Your answer: \n"

    return prompt

def validate_response_blocksworld(initial_state, goal_state, response):
    """
    Validate the LLM's response and check if it reaches the goal state.
    """
    current_state = {stack: blocks.copy() for stack, blocks in initial_state.items()}
    #print('current_state:', current_state)
    moves = response.strip().split('\n')
    #print('goal state:', goal_state)
    for move in moves:
        parts = move.split()
        if len(parts) != 6 or parts[0] != "Move" or parts[2] != "from" or parts[4] != "to":
            return False, f"Invalid move format: {move}"

        block, source, destination = parts[1], parts[3], parts[5]
        if 'stack' not in source:
            source = 'stack' + source
        if 'stack' not in destination:
            destination = 'stack' + destination

        if source not in current_state or destination not in current_state:
            return False, f"Invalid source or destination stack: {move}"

        if not current_state[source] or current_state[source][-1] != block:
            return False, f"Invalid move: {move}. Block {block} is not at the top of the source stack."

        # Move the block
        moved_block = current_state[source].pop()
        current_state[destination].append(moved_block)
        #print('current_state:', current_state)

    def compare_states(state1, state2):
        state1_non_empty = {k: v for k, v in state1.items() if v}
        state2_non_empty = {k: v for k, v in state2.items() if v}
        return state1_non_empty == state2_non_empty

    # Check if the final state matches the goal state
    if compare_states(current_state, goal_state):
        return True, "Goal state reached successfully!"
    else:
        return False, "The final state does not match the goal state."


#####Gridworld#######
def generate_prompt_gridworld(sample: dict) -> str:
    """
    Given a gridworld sample, generate a prompt instructing the LLM
    to output a valid path in JSON format enclosed in <<<>>>.
    """
    rows = sample['grid']['rows']
    cols = sample['grid']['cols']
    obstacles = sample['obstacles']
    goals = sample['goals']
    initial = sample['initial_robot']
    adjacency_str = json.dumps(sample['grid']['adjacency'], indent=2)

    prompt = f"""
You are given the following Gridworld planning problem:

Grid dimensions: {rows} x {cols}
Obstacles: {obstacles}
Goals: {goals}
Initial robot position: {initial}
Adjacency:
{adjacency_str}

Task:
- The robot must start at {initial}.
- The robot must visit all goals at least once (in any order).
- The robot must NOT pass through any obstacle cells.
- At each step, the robot can move to an adjacent cell (up, down, left, or right).
- Output your plan as a JSON list of robot positions (cells), from the initial position
  to the final position after all goals have been visited.
- The first position in your list must be the initial position.
- Enclose your final JSON list in <<< >>>. For example:
  <<<["C1,1", "C2,1", "C2,2", ...]>>>

Now provide your plan (a valid path):
"""
    return prompt.strip()


def read_dataset_gpqa(dataset_dir: str) -> List[Dict]:
    import pandas as pd
    puzzles = []
    
    gpqa_files = ['gpqa_main.csv']
    
    for file_name in gpqa_files:
        file_path = os.path.join(dataset_dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            for _, row in df.iterrows():
                if pd.isna(row.get('Question', '')) or pd.isna(row.get('Correct Answer', '')):
                    continue
                puzzle = {
                    'question': str(row['Question']).strip(),
                    'correct_answer': str(row['Correct Answer']).strip(),
                    'incorrect_answers': [
                        str(row['Incorrect Answer 1']).strip(),
                        str(row['Incorrect Answer 2']).strip(),
                        str(row['Incorrect Answer 3']).strip()
                    ],
                    'explanation': str(row['Explanation']).strip(),
                    'subdomain': str(row.get('Subdomain', '')).strip(),
                    'file_source': file_name
                }
                puzzles.append(puzzle)
    
    return puzzles

def format_gpqa_question(puzzle: dict) -> str:
    import random
    
    # Create list of all options
    options = [puzzle['correct_answer']] + puzzle['incorrect_answers']
    
    # Randomize order and assign letters
    random.shuffle(options)
    
    # Find which letter corresponds to correct answer
    correct_letter = None
    for i, option in enumerate(options):
        if option == puzzle['correct_answer']:
            correct_letter = chr(ord('A') + i)
            break
    
    # Store the correct letter mapping in the puzzle for validation
    puzzle['correct_letter'] = correct_letter
    puzzle['option_mapping'] = {chr(ord('A') + i): option for i, option in enumerate(options)}
    
    # Format the question
    formatted_question = f"{puzzle['question']}\n\n"
    for i, option in enumerate(options):
        letter = chr(ord('A') + i)
        formatted_question += f"{letter}) {option}\n"
    
    # Add standardized answer format instruction
    formatted_question += "\nOnce you feel you are ready for the final answer, directly return the answer with the format <<<answer content>>> at the end of your response, e.g. <<<C>>>, <<<A>>>"
    
    return formatted_question.strip()

#####################################################
# 3. Extracting and Checking LLM Responses
#####################################################

def extract_json_from_response_gridworld(response: str) -> str:
    """
    Extract the JSON string between <<< and >>> from the LLM response.
    Raises an error if not found.
    """
    pattern = r"<<<(.*?)>>>"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        return '[]'
        raise ValueError("Could not find JSON enclosed by <<< and >>> in the response.")
    return matches[0].strip()

def extract_equation_with_GPT4_gridworld(response):
    prompt = ('Your task is to extract the final answer from the given answer by another LLM:\n'
              'The final answer should be in the format <<<answer>>>, like <<<["C2,1","C3,1", ...]>>>.\n'
              'Return only the answer in that format.\n'
              'Input text: ')
    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                    user_prompt_list=[prompt + response], response_total_list=[], logprobs=False)
    return extract_equation

def check_gridworld_plan_legality(plan: List[str],
                                  sample: dict) -> Tuple[bool, str]:
    """
    Check if the given path 'plan' is valid:
      1. The path starts at sample['initial_robot'].
      2. The path only moves in adjacency steps (up, down, left, right).
      3. The path never enters an obstacle.
      4. The path visits all goals at least once.
    Returns (True, "OK") if valid, or (False, "Error message") otherwise.
    """
    adjacency = sample['grid']['adjacency']
    obstacles = set(sample['obstacles'])
    goals = set(sample['goals'])
    initial = sample['initial_robot']

    # 1. Check start
    if not plan:
        return False, "Empty plan."
    if plan[0] != initial:
        return False, f"Plan does not start at initial position {initial}."

    # 2. Check adjacency and obstacle avoidance
    for i in range(len(plan) - 1):
        curr = plan[i]
        nxt = plan[i + 1]
        if nxt not in adjacency[curr]:
            return False, f"Invalid move from {curr} to {nxt} (not adjacent)."
        if nxt in obstacles:
            return False, f"Path enters an obstacle cell {nxt}."

    # 3. Check that all goals are visited at least once
    visited_goals = set(plan).intersection(goals)
    if visited_goals != goals:
        return False, f"Not all goals visited. Visited {visited_goals}, needed {goals}."

    return True, "OK"


def check_llm_response_gridworld(response: str, sample: dict) -> Tuple[bool, str]:
    """
    1. Extract the JSON list from the LLM response.
    2. Parse it as a list of grid cells (strings).
    3. Check the plan for legality.
    """
    print('response:', response)
    try:
        # First try to parse the entire response
        plan = ast.literal_eval(response)
    except (ValueError, SyntaxError):
        # If that fails, extract the JSON part and try again
        json_str = extract_json_from_response_gridworld(response)
        try:
            plan = ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            return False, "Extracted content is not valid JSON."

    if not isinstance(plan, list):
        return False, "The extracted JSON is not a list."
    if any(not isinstance(pos, str) for pos in plan):
        return False, "Some elements in the path list are not strings (cells)."

    is_valid, message = check_gridworld_plan_legality(plan, sample)
    return is_valid, message

def read_samples_gridworld(filename="gridworld_sample_6x7_7.json"):
    """
    Read a Gridworld sample from the specified JSON file and return it as a Python dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    with open(filename, "r") as f:
        sample = json.load(f)

    return sample


##### Reasoning Gym #####
def extract_equation_with_GPT4_reasoning_gym(response, dataset_name):
    pure_answer = ''
    notes = ''

    if dataset_name in ['arc_agi']:
        pure_answer = 'an integer grid like "2 6 6 1\n6 3 0 1\n1 0 2 4\n9 3 8 0", the number of rows and columns might be very large'
        notes = """# Notes
1. Return any integer grid you find, whether or not it is enclosed within <<<>>>. 
2. Do not return <<<>>> if input text is not empty. 
3. Do not return answer like <<<final answer>>> or <<<answer>>>.\n"""
    elif dataset_name == 'basic_arithmetic':
        pure_answer = 'a number'
    elif dataset_name == 'binary_matrix':
        pure_answer = 'an integer grid like "3 2 1\n2 1 0\n3 5 6" or a 2D list (matrix) of integers like [[3, 2, 1], [2, 1, 0], [3, 5, 6]], where both the number of rows and columns can be large'
        notes = 'Do not return answer like <<<final answer>>> or <<<answer>>>.\n'
    #     elif dataset_name == 'boxnet':
    #         pure_answer = """
    # a json str like
    # [
    #     {"Agent[0.5, 1.5]": "move(box_red, square[1.5, 1.5])"},
    #     {"Agent[1.5, 0.5]": "move(box_blue, square[2.5, 0.5])"},
    #     {"Agent[2.5, 1.5]": "move(box_green, square[2.5, 0.5])"}
    # ]
    # """
    elif dataset_name == 'caesar_cipher':
        pure_answer = 'decrypted Caesar cipher text'
    elif dataset_name == 'calendar_arithmetic':
        pure_answer = 'day of the week, integer or Yes/No'
    elif dataset_name == 'circuit_logic':
        pure_answer = '0 or 1'
    elif dataset_name == 'codeio':
        pure_answer = ('a JSON object or a str in JSON format, or a str in ```json\n<str>\n```, or a single str\n'
                       "For example: {'value': 12}, or ```json\n{'a': 1, 'bb': np.float32(3.2)}\n```, or \"Correct\"")
        notes = f'If the input text does not have <<<final answer>>>, but you find {pure_answer}, add <<<>>> around it and return your answer. For a str in ```json\n<str>\n```, only return str without ```json and ```.\n'
    elif dataset_name == 'color_cube_rotation':
        pure_answer = 'name of a color'
        notes = 'Do not return answer like <<<final answer>>> or <<<answer>>>.\n'
    elif dataset_name == 'complex_arithmetic':
        pure_answer = 'a complex number (which may contain only a real or imaginary part)'
    elif dataset_name in ['count_bits', 'count_primes']:
        pure_answer = 'an integer'
    elif dataset_name == 'countdown':
        pure_answer = 'a mathematical formula'
    elif dataset_name == 'course_schedule':
        pure_answer = 'a boolean value (True or False)'
        notes = 'Do not return answer like <<<final answer>>> or <<<answer>>>.\n'
    elif dataset_name == 'cryptarithm':
        pure_answer = 'a comma separated mapping from letters to digits, for example A=1,B=2,C=3'
    # elif dataset_name in ['decimal_arithmetic', 'decimal_chain_sum']:
    elif dataset_name in ['decimal_arithmetic']:
        pure_answer = 'a float number'
    elif dataset_name == 'dice':
        pure_answer = 'a fraction (e.g. 7/10, 161/302)'
    elif dataset_name == 'emoji_mystery':
        pure_answer = 'a sentence in natural language'
        notes = 'Return any sentence you find enclosed with <<<>>>.\n'
    elif dataset_name == 'family_relationships':
        pure_answer = 'a single word that represents a family relationship, e.g., father, wife, mother-in-law'
    elif dataset_name == 'figlet_font':
        pure_answer = 'a word'
        notes = 'Return any word you find enclosed with <<<>>>.\n'
    elif dataset_name == 'fraction_simplification':
        pure_answer = 'a fraction (e.g. 7/10, $\\frac{261}{316}$, $\\dfrac{1062}{2579}$)'
    elif dataset_name == 'futoshiki':
        pure_answer = """n x n Futoshiki puzzle, e.g.
4   3   2   6   5   1

1   4   5   2   3   6

3   1   4   5   6 > 2
            ∨       ∧
5   6   1   4   2   3

6 > 2   3   1   4   5
    ∧                
2   5   6   3   1   4
"""
    elif dataset_name == 'game_of_life':
        pure_answer = """grid in JSON format, e.g.
[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
"""
    elif dataset_name == 'game_of_life_halting':
        pure_answer = 'a boolean value (True or False)'
        notes = 'Do not return answer like <<<final answer>>> or <<<answer>>>.\n'
    elif dataset_name == 'gcd':
        pure_answer = 'an integer'
    elif dataset_name == 'graph_color':
        pure_answer = 'JSON map of vertices to colors'
    elif dataset_name == 'group_anagrams':
        pure_answer = "a list of lists of strings, e.g. [['eat', 'tea'], ['tan', 'nat'], ['apple']]"
        notes = 'If the input contains two identical lists—either as separate lists or one wrapped in triple brackets—return only the first list.'
        # notes = ("Return any list of lists of strings you find enclosed with <<<>>>.\n"
        #          "When extracting, do not omit any brackets from either the inner or outer lists. For example, "
        #          "for <<<[['eat', 'tea'], ['tan', 'nat']]>>>, you should return [['eat', 'tea'], ['tan', 'nat']], "
        #          "but do not return [['eat', 'tea'], ['tan', 'nat'] (omitted the second bracket of outer list) or "
        #          "['eat', 'tea'], ['tan', 'nat']] (omitted the first bracket of outer list).\n")
    elif dataset_name == 'gsm_symbolic':
        pure_answer = 'an integer or a float'
    elif dataset_name == 'intermediate_integration':
        pure_answer = 'a mathematical formula'
    elif dataset_name == 'isomorphic_strings':
        pure_answer = 'a boolean value (True or False)'
        notes = 'Do not return answer like <<<final answer>>> or <<<answer>>>.\n'
    elif dataset_name == 'jugs':
        pure_answer = 'a JSON-parsable list'
    elif dataset_name == 'knights_knaves':
        pure_answer = 'a string'
    elif dataset_name in ['largest_island', 'lcm', 'leg_counting', 'letter_counting']:
        pure_answer = 'an integer'
    elif dataset_name == 'letter_jumble':
        pure_answer = 'a sentence'
        notes = '1. Return any sentence you find enclosed with <<<>>>.\n2. Do not alter the sequence of letters in the sentence found within the input text.'
    elif dataset_name == 'list_functions':
        pure_answer = 'a list of integer/integers, e.g. [12], [60, 25, 327]'
        notes = 'Do not return answer like <<<final answer>>> or <<<answer>>>.\n'
    elif dataset_name == 'mahjong_puzzle':
        pure_answer = 'one of the following: "Peng", "Chi", or "Pass" (without quotes)'
        notes = 'Return any word you find enclosed with <<<>>>.\n'
        # notes = 'If there are more than one "<<<Peng>>>", "<<<Chi>>>", or "<<<Pass>>>" (without quotes), return <<<>>>.\n'
    elif dataset_name == 'manipulate_matrix':
        pure_answer = 'an integer matrix, e.g. 1 3 3\n1 6 2, the number of rows and columns might be very large'
        notes = 'Return any matrix you find enclosed within <<<>>>. Do not return answer like <<<final answer>>> or <<<answer>>>.\n'
    elif dataset_name == 'maze':
        pure_answer = 'an integer'
    elif dataset_name == 'mini_sudoku':
        pure_answer = 'a 4x4 integer matrix, e.g. 1 2 3 4\n2 3 4 1\n3 4 1 2\n4 1 2 3'
    elif dataset_name == 'modulo_grid':
        pure_answer = """a grid composed of '✅' and '❌', like this:
❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌
❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅
❌❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌
❌✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌
✅❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌
❌❌❌❌❌❌❌❌✅❌❌❌❌❌❌❌❌❌❌❌
"""
        notes = "Return any grid you find composed of '✅' and '❌' or any grid you find enclosed within <<<>>>. Do not return answer like <<<final answer>>> or <<<answer>>>.\n"
    elif dataset_name == 'n_queens':
        pure_answer = """a grid composed of 'Q' and '_', like this:
_ _ Q _ _ _ _ _
_ _ _ _ _ Q _ _
_ Q _ _ _ _ _ _
_ _ _ _ Q _ _ _
_ _ _ _ _ _ _ Q
Q _ _ _ _ _ _ _
_ _ _ _ _ _ Q _
_ _ _ Q _ _ _ _
"""
        notes = "Return any grid you find composed of 'Q' and '_' or any grid you find enclosed within <<<>>>. Do not return answer like <<<final answer>>> or <<<answer>>>.\n"
    elif dataset_name == 'needle_haystack':
        pure_answer = 'a name (string)'
    elif dataset_name == 'number_filtering':
        pure_answer = 'a list of numbers (integers and floats)'
    elif dataset_name == 'number_format':
        pure_answer = 'a number'
    elif dataset_name == 'number_sequence':
        pure_answer = 'an integer'
        notes = 'Do not return answer like <<<final answer>>> or <<<answer>>>.\n'
    elif dataset_name == 'number_sorting':
        pure_answer = 'a list of numbers'
        notes = 'Do not return answer like <<<final answer>>> or <<<answer>>>.\n'
    elif dataset_name == 'palindrome_generation':
        pure_answer = 'a string which is a palindrome'
    elif dataset_name == 'palindrome_partitioning':
        pure_answer = """A list of lists of strings, where each string is a palindrome, e.g. <<<[["x", "h", "x", "w", "k", "l", "k", "i"], ["x", "h", "x", "w", "klk", "i"], ["xhx", "w", "k", "l", "k", "i"], ["xhx", "w", "klk", "i"]]>>>. The list may be very long."""
        notes = 'If you find any list enclosed within <<<>>>, return it, and do not return <<<>>>. The list may be very long.\n'
    elif dataset_name == 'polynomial_equations':
        pure_answer = 'a decimal number or comma-separated decimal numbers'
    elif dataset_name == 'polynomial_multiplication':
        pure_answer = 'a polynomial, e.g. -1220*x**4 + 4130*x**3 - 73*x**2 + 12506*x + 2'
    elif dataset_name == 'pool_matrix':
        pure_answer = """an integer matrix or a float matrix, e.g. 
5 7
3 9
0 8
, or
[[5 7], [3 9], [0 8]]
, or
5.0 3.25 3.75 2.5 7.0
4.0 3.25 3.75 4.75 6.0
, or
[[5.0 3.25 3.75 2.5 7.0], [4.0 3.25 3.75 4.75 6.0]]
, the number of rows and columns might be very large
"""
        notes = 'Preserve the original format of the matrix in the input text.\n'
    # elif dataset_name == 'power_function':
    #     pure_answer = 'a float'
    elif dataset_name == 'prime_factorization':
        pure_answer = 'a mathematical formula'
    elif dataset_name == 'products':
        pure_answer = 'an integer'
    elif dataset_name == 'puzzle24':
        pure_answer = 'a mathematical formula'
    elif dataset_name == 'quantum_lock':
        pure_answer = "a sequence of letters separated by '→', for example: A → B → C"
    elif dataset_name == 'ransom_note':
        pure_answer = 'a boolean value (True or False)'
        notes = 'Return any boolean value you find enclosed with <<<>>>. Do not return answer like <<<final answer>>> or <<<answer>>>.\n'
    elif dataset_name == 'rearc':
        pure_answer = 'an integer grid, e.g. 1 3 2\n1 1 0\n7 6 3\n2 5 1\n5 9 2, the number of rows and columns might be very large.'
        notes = """# Notes 
1. Return any integer grid you find, whether or not it is enclosed within <<<>>>. 
2. Do not return <<<>>> if input text is not empty. 
3. Do not return answer like <<<final answer>>> or <<<answer>>>.\n"""
    elif dataset_name == 'rectangle_count':
        pure_answer = 'an integer'
    elif dataset_name == 'rotate_matrix':
        pure_answer = 'an integer grid, e.g. 1 3 2\n1 1 0\n7 6 3\n2 5 1\n5 9 2, [[1, 3, 2], [1, 1, 0], [7, 6, 3], [2, 5, 1], [5, 9, 2]], the number of rows and columns might be very large.'
        notes = """# Notes
1. Return any integer grid you find, whether or not it is enclosed within <<<>>>. 
2. Do not return <<<>>> if input text is not empty. 
3. Do not return answer like <<<final answer>>> or <<<answer>>>.\n"""
    elif dataset_name == 'rotten_oranges':
        pure_answer = 'an integer'
    elif dataset_name == 'rubiks_cube':
        pure_answer = 'a string formatted in Singmaster notation'
        notes = 'Do not return answer like <<<final answer>>> or <<<answer>>>.\n'
    elif dataset_name == 'rush_hour':
        pure_answer = 'A sequence of letter-integer deltas, each formatted as a letter or letters followed by a signed integer, e.g. J+2 AA+1, or F+1 K+1 M-1 C+3 H+2'
    elif dataset_name == 'self_reference':
        pure_answer = 'an integer'
    # elif dataset_name == 'sentence_reordering':
    #     pure_answer = 'A complete or incomplete sentence (the word order of this sentence might not be quite right)'
    elif dataset_name == 'shortest_path':
        pure_answer = 'A sequence composed of the four strings "left", "right", "up", "down", or the single string "infeasible", e.g. "up up", "left up down right right up", "infeasible"'
        # notes = 'Return any sequence you find, whether or not it is enclosed within <<<>>>.\n'
    elif dataset_name in ['simple_equations', 'simple_geometry']:
        pure_answer = 'a number'
    elif dataset_name == 'simple_integration':
        pure_answer = 'a mathematical formula, e.g. -5*x**19/2 - 13*x**7 + 6*x + C'
    elif dataset_name == 'sokoban':
        pure_answer = 'a string of characters, e.g. URLUURD, UDRRRRRLUUURLURRDUULLLLULDRDRDLUUURRRDRRULLLL'
    elif dataset_name == 'spell_backward':
        pure_answer = 'a string'
    elif dataset_name == 'spiral_matrix':
        pure_answer = 'a space-separated list of integers, e.g. 7 4 8 2 1 4 1 2 7 9 0 1 8 2 2 9 9 9 6 4 6 0 8 5 9 1 1 9 4 6 0 7 9 5 3 8 4 9 2 5 0 5 1 1 2 1 0 2 7 2 6 1 9 9 3 7 6 9 4 3 2 2 6 9'
    elif dataset_name in ['string_insertion', 'string_manipulation']:
        pure_answer = 'a string'
    elif dataset_name == 'string_splitting':
        pure_answer = 'a space-separated list of integers, e.g. 1 1 2 1 0 2'
    elif dataset_name == 'string_synthesis':
        pure_answer = 'a space-separated list of integers, e.g. 2 2 0 1 0 0 2 1 0'
    elif dataset_name == 'sudoku':
        pure_answer = """a 9x9 grid with numbers separated by spaces, and rows separated by newlines, e.g.
1 5 4 2 9 7 3 8 6
7 3 6 8 1 4 9 2 5
8 2 9 3 6 5 4 7 1
2 8 5 1 7 3 6 4 9
4 6 1 5 2 9 7 3 8
3 9 7 4 8 6 5 1 2
5 7 2 6 3 8 1 9 4
6 1 3 9 4 2 8 5 7
9 4 8 7 5 1 2 6 3
"""
    elif dataset_name == 'syllogism':
        pure_answer = 'A string with the value either "Yes" or "No"'
    elif dataset_name == 'time_intervals':
        pure_answer = 'A string representing a time duration, such as "2 days, 17:15", "07:21:16", "03:29:15.532", "12 days"'
    elif dataset_name == 'tower_of_hanoi':
        pure_answer = """A sequence of steps describing disk moves (potentially containing a very large number of steps), e.g.
Move disk 1 from Peg 2 to Peg 1
Move disk 2 from Peg 2 to Peg 3
Move disk 1 from Peg 1 to Peg 3
Move disk 3 from Peg 2 to Peg 1
Move disk 1 from Peg 3 to Peg 2
Move disk 2 from Peg 3 to Peg 1
Move disk 1 from Peg 2 to Peg 1
Move disk 4 from Peg 2 to Peg 3
Move disk 1 from Peg 1 to Peg 3
Move disk 2 from Peg 1 to Peg 2
Move disk 1 from Peg 3 to Peg 2
Move disk 3 from Peg 1 to Peg 3
Move disk 1 from Peg 2 to Peg 1
Move disk 2 from Peg 2 to Peg 3
Move disk 1 from Peg 1 to Peg 3
"""
        notes = """
1. Return any sequence of steps describing disk moves you find, whether or not it is enclosed within <<<>>>.
2. Do not return <<<>>> if input text is not empty.\n
"""
    elif dataset_name == 'tsumego':
        pure_answer = 'coordinates, e.g. D3, H12, K6'
    elif dataset_name == 'word_ladder':
        pure_answer = 'a comma-separated sequence of uppercase letters without spaces, e.g. CELL,CALL,PALL,PALS,PATS,PITS,ZITS,ZITI'
    elif dataset_name == 'word_sequence_reversal':
        pure_answer = 'a comma-separated list of words with a space after the comma, e.g. "separated, were, place, way, The, condense"'
    elif dataset_name == 'word_sorting':
        pure_answer = 'a comma-separated list of words, e.g. "000, 365, exclaimed, _mode, observation, projectors, telephones, turning"'
    elif dataset_name == 'zebra_puzzles':
        pure_answer = "A string that represents a person’s name"

    pure_answer_prompt = f'The **final answer** is in the format: {pure_answer}\n' if len(pure_answer) > 0 else ''

    prompt = 'Your task is to extract the final answer from the given answer by another LLM:\n' \
             + pure_answer_prompt + \
             'Note that the final answer should follow strictly the format like <<<final answer>>>\n' \
             'If the input text contains a final answer in the format like <<<final answer>>>, return your answer with the format <<<final answer>>>.\n' \
             'If the input text does not have <<<final answer>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
             'If the input text does not have <<<final answer>>> and is not the pure answer, directly return <<<>>>.\n' \
             'Note that if you find no **final answer** are answered, then directly answer <<<>>>.\n' \
             'If the input text only contains code without **final answer**, do not run/analyze/understand the code in the input text to get an answer. You should directly answer <<<>>>.\n' \
             'Do not generate an answer that is not explicitly contained within the input text.\n' \
             + notes + \
             'Input text: '

    # if dataset_name == 'binary_matrix':
    #     prompt = 'Your task is to extract the final answer from the given answer by another LLM:\n' \
    #              + pure_answer_prompt + \
    #              'Note that the final answer should follow strictly the format like <<<final answer>>>\n' \
    #              'If the input text contains a final answer in the format like <<<final answer>>>, return your answer with the format <<<final answer>>>.\n' \
    #              'If the input text does not have <<<final answer>>> and is already the pure answer, add <<<>>> and return your answer.\n' \
    #              'If the input text does not have <<<final answer>>> and is not the pure answer, directly return <<<>>>.\n' \
    #              'If the input text has string like "<<<answer>>>" followed by the final answer, add <<<>>> to the final answer and return it.\n' \
    #              'Note that if you find no final answer are answered, then directly answer <<<>>>. Do not run the code in the input text or give an irrelevant answer.\n' \
    #              'Input text: '

    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                    user_prompt_list=[prompt + response], response_total_list=[], logprobs=False)
    return extract_equation

def validate_solution_reasoning_gym(dataset_name, answer, full_data):
    if dataset_name in ['arc_agi', 'rearc']:
        full_data['metadata']['output'] = tuple(tuple(inner) for inner in full_data['metadata']['output'])
    # elif dataset_name == 'binary_matrix':
    #     answer = str_to_list_of_lists(answer)
    #     print(f'binary_matrix answer: {answer}')
    #     print("\n".join(" ".join(str(x) for x in row) for row in answer))
    elif dataset_name in ['codeio', 'group_anagrams', 'palindrome_partitioning']:
        if dataset_name in ['group_anagrams', 'palindrome_partitioning']:
            if answer.endswith("']") or answer.endswith('"]'):
                answer += ']'
        if dataset_name in ['palindrome_partitioning']:
            if answer.endswith("']]]") or answer.endswith('"]]]'):
                answer = answer[:-1]
        try:
            answer_literal_eval = ast.literal_eval(answer)
            if isinstance(answer_literal_eval, dict) or isinstance(answer_literal_eval, list):
                answer = json.dumps(answer_literal_eval)
        except:
            pass
    elif dataset_name == 'prime_factorization':
        if not answer or answer.strip() == '':
            return False

        for factor in answer.split("×"):
            if not factor or factor.strip() == '':
                return False
    elif dataset_name in ['simple_equations', 'simple_geometry']:
        answer = convert_str_if_integer(answer)


    data = reasoning_gym.create_dataset(dataset_name, size=1, seed=1)
    score = data.score_answer(answer=answer, entry=full_data)
    print(f"answer: {answer}, full_data: {full_data['answer']}, score: {score}")

    if dataset_name == 'binary_matrix' and abs(score - 0.1) < 1e-6:
        return True

    if abs(score - 1.0) < 1e-6:
        return True
    else:
        return False

def str_to_list_of_lists(s: str) -> list[list[int]] | str:
    """
    Convert a string representation of a 2D list into an actual list of lists of integers.
    Strips whitespace and uses safe parsing.
    """
    s = s.strip()  # Remove leading/trailing whitespace
    try:
        result = ast.literal_eval(s)  # Safely parse the string
        if isinstance(result, list) and all(isinstance(row, list) for row in result):
            return result
        else:
            print(f'str_to_list_of_lists failed for {s}')
            return s
    except Exception as e:
        print(f'str_to_list_of_lists failed for {s}')
        return s

def convert_str_if_integer(s):
    try:
        num = float(s)
        if num.is_integer():
            return str(int(num))
        else:
            return s  # keep as-is if not integer
    except ValueError:
        return s  # keep as-is if not a number