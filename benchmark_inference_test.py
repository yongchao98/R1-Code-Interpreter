import json
import re
import pandas as pd
import os
import subprocess
import sys
from openai import OpenAI
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, extract_code, extract_and_check, LLM_answer_code_checker, save_file_func, paraphrase_with_GPT4
import random
import math
import json
from typing import List, Tuple, Dict
import time
import numpy as np
from prompt import *
from argparse import ArgumentParser
#from add_reasoning_gym_dataset import available_datasets

from benchmark.run_R1_code_interpreter_data_syn import run_logic_game_baselines

if __name__ == '__main__':
    # gpt-4o, gpt-4o-mini, gpt-3.5-turbo for OpenAi API

    def log_run_info(log_file, run_info):
        with open(log_file, 'a') as f:
            f.write(run_info + "\n")

    infer_base_path = '/path-to-R1-Code-Interpreter/R1-Code-Interpreter/LLaMA-Factory/examples/inference/' # To be filled

    for model_name, LLM_path in [['gpt-4o', '']]:
        # [['gpt-4o', ''], ['R1_CI_14B', 'R1_CI_14B.yaml'], ['R1_CI_7B', 'R1_CI_7B.yaml'], ['R1_CI_3B', 'R1_CI_3B.yaml']]
        args_path = os.path.join(infer_base_path, LLM_path)

        gather_save_input_dir = 'results_gather'
        start_index = 0
        max_sample_num = 100
        max_round_num = 5

        # List of all environment names
        env_name_list1 = ['pooling', 'reversi', 'light_puzzles', 'new_operator', 'mahjong_pattern',
                          'statistical_counting', 'synthesis_decomposition', '2048', 'matrix_transformation']
        env_name_list2 = ['pattern_recognition', 'constrained_linear_arrangement', 'string_synthesis', 'logic_puzzle',
                          'string_insertion', 'letter_logic_diagram', 'standard_sudoku',
                          'string_deletion_and_modification']
        env_name_list3 = ['string_splitting', 'permutations_and_combinations', 'logical_equation',
                          'combinatorial_calculation', 'cryptanalysis', 'BoxLift', 'Blocksworld', 'gsm',
                          'math_geometry']
        env_name_list4 = ['eight_queens', 'game24', 'letters', 'number_multiply', 'math_counting_and_probability',
                          'BoxNet_v2', 'Gridworld']

        ### 27 tasks from BigBench Hard
        env_name_big_bench_hard = ['big_bench_hard:' + task for task in
                                   ['date_understanding', 'web_of_lies', 'disambiguation_qa', 'formal_fallacies',
                                    'geometric_shapes',
                                    'logical_deduction_seven_objects', 'navigate', 'dyck_languages',
                                    'boolean_expressions',
                                    'causal_judgement',
                                    'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_three_objects',
                                    'movie_recommendation',
                                    'multistep_arithmetic_two', 'object_counting', 'penguins_in_a_table',
                                    'word_sorting',
                                    'tracking_shuffled_objects_three_objects',
                                    'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects',
                                    'temporal_sequences',
                                    'sports_understanding', 'snarks', 'salient_translation_error_detection',
                                    'ruin_names',
                                    'reasoning_about_colored_objects']]

        # Divide based on the modulo pattern
        list_1 = ['codeio', 'color_cube_rotation', 'complex_arithmetic', 'count_primes', 'countdown', 'course_schedule',
                  'dice',
                  'emoji_mystery', 'family_relationships', 'fraction_simplification', 'futoshiki', 'game_of_life',
                  'gcd',
                  'graph_color', 'group_anagrams', 'isomorphic_strings', 'jugs', 'knight_swap', 'largest_island', 'lcm',
                  'leg_counting', 'list_functions', 'manipulate_matrix', 'maze', 'needle_haystack', 'number_filtering',
                  'number_format', 'number_sorting', 'palindrome_generation', 'palindrome_partitioning',
                  'polynomial_multiplication', 'pool_matrix', 'propositional_logic', 'quantum_lock', 'ransom_note',
                  'rectangle_count', 'rotate_matrix', 'rotten_oranges', 'rush_hour', 'self_reference', 'shortest_path',
                  'simple_geometry', 'simple_integration', 'sokoban', 'spiral_matrix', 'string_manipulation',
                  'syllogism',
                  'tower_of_hanoi', 'tsumego', 'word_ladder', 'zebra_puzzles'] + ['ab', 'acre', 'advanced_geometry',
                                                                                  'aiw', 'arc_1d', 'arc_agi',
                                                                                  'base_conversion',
                                                                                  'basic_arithmetic', 'bf',
                                                                                  'binary_alternation']

        list_2 = ['count_bits', 'decimal_arithmetic', 'figlet_font', 'game_of_life_halting', 'intermediate_integration',
                  'knights_knaves', 'letter_jumble', 'modulo_grid', 'number_sequence', 'polynomial_equations',
                  'products',
                  'rearc', 'rubiks_cube', 'simple_equations', 'spell_backward', 'time_intervals',
                  'word_sequence_reversal'] + ['binary_matrix', 'bitwise_arithmetic', 'caesar_cipher',
                                               'calendar_arithmetic', 'chain_sum',
                                               'circuit_logic']

        reasoning_gym_datasets_train = ['reasoning_gym_' + task for task in list_1]
        reasoning_gym_datasets_test = ['reasoning_gym_' + task for task in list_2]

        task_list_train = env_name_list1 + env_name_list2 + env_name_list3 + env_name_big_bench_hard[
                                                                             :20] + reasoning_gym_datasets_train
        task_list_test = env_name_list4 + env_name_big_bench_hard[20:] + reasoning_gym_datasets_test

        base_path = 'results_gather'
        runtime_list = []
        #baseline_method_name: 1_only_ques, All_code_CoT, All_text, R1_code_interpreter_test, R1_code_interpreter_data_syn_1, R1_code_interpreter_data_syn_hint, R1_code_interpreter_data_syn_code,
        # CodeSteer, CodeSteer_wo_self-answer_checker, Code_Interpreter
        for baseline_method_name in ['All_text']:
            # for task_name in task_list_train + task_list_test:
            for task_name in ['gpqa']:
                print(f'task_name: {task_name}')
                start_time = time.time()
                run_logic_game_baselines(task_name, gather_save_input_dir, model_name, baseline_method_name, args_path, start_index, max_sample_num, max_round_num)
                end_time = time.time()
                runtime = end_time - start_time
                runtime_list.append(runtime/max_sample_num)

                output_path = base_path + f'/Cost_runtime_gather_{model_name}_{baseline_method_name}.txt'
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'a') as f:
                    f.write(f"{runtime/max_sample_num}\n")
                print(f'Mean time cost: {np.mean(runtime_list)}')
                print(f'\nDataset saved to: {output_path}')
