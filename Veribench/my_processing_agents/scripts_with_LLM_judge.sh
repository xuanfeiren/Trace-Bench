#!/bin/bash

# Easy_set category task IDs (41 tasks total)
# Run all easy_set tasks (10-50)
uv run python my_processing_agents/solution_PS_withLLMjudge.py \
                    --task_idx 10 \
                    --num_steps 100 \
                    --num_threads 30 \
                    --log_frequency 1 \
                    --test_frequency 1 \
                    --num_candidates 3 \
                    --algorithm PS \
                    --epsilon 0.02 \
                    --with_llm_judge \
                    --use_wandb \
                    --project_name "LLM_judge_veribench-PS-num_candidates-2" 

# uv run python my_processing_agents/solution_PS_withLLMjudge.py \
#                     --task_idx 20 \
#                     --num_steps 11 \
#                     --num_threads 30 \
#                     --log_frequency 1 \
#                     --test_frequency 1 \
#                     --num_candidates 5 \
#                     --algorithm PS_epsNet_Summarizer \
#                     --epsilon 0 \
#                     --epsilon_for_summarizer 0.02 \
#                     --with_llm_judge \
#                     --use_wandb \
#                     --project_name "LLM_judge_veribench-PS-epsNet0.02-only-Summarizer-num_candidates-5" 
uv run python my_processing_agents/solution_PS_withLLMjudge.py \
        --task_idx 10 \
        --num_steps 20 \
        --num_threads 30 \
        --log_frequency 1 \
        --test_frequency 1 \
        --num_candidates 5 \
        --algorithm PS_epsNet_Summarizer \
        --epsilon 0.02 \
        --epsilon_for_summarizer 0.02 \
        --with_llm_judge \
        --use_wandb \
        --project_name "(2)LLM_judge_veribench-PS_eps0.02Net_Summarizer-num_candidates-5" 

uv run python my_processing_agents/solution_PS_withLLMjudge.py \
        --task_idx 11 \
        --num_steps 20 \
        --num_threads 30 \
        --log_frequency 1 \
        --test_frequency 1 \
        --num_candidates 5 \
        --algorithm PS_epsNet_Summarizer \
        --epsilon 0.02 \
        --epsilon_for_summarizer 0.02 \
        --with_llm_judge \
        --use_wandb \
        --project_name "(2)LLM_judge_veribench-PS_eps0.02Net_Summarizer-num_candidates-5"
        
# PS only summarizer
# cd Veribench
# for task_idx in {41..50}; do
#     uv run python my_processing_agents/solution_PS_withLLMjudge.py \
#                     --task_idx $task_idx \
#                     --num_steps 11 \
#                     --num_threads 30 \
#                     --log_frequency 1 \
#                     --test_frequency 1 \
#                     --num_candidates 5 \
#                     --algorithm PS_epsNet_Summarizer \
#                     --epsilon 0 \
#                     --epsilon_for_summarizer 0.02 \
#                     --with_llm_judge \
#                     --use_wandb \
#                     --project_name "LLM_judge_veribench-PS-epsNet0.02-only-Summarizer-num_candidates-5" 
# done

# for task_idx in {41..50}; do
#     uv run python my_processing_agents/solution_PS_withLLMjudge.py \
#                     --task_idx $task_idx \
#                     --num_steps 11 \
#                     --num_threads 30 \
#                     --log_frequency 1 \
#                     --test_frequency 1 \
#                     --num_candidates 5 \
#                     --algorithm PS \
#                     --with_llm_judge \
#                     --use_wandb \
#                     --project_name "LLM_judge_veribench-PS-num_candidates-5" 
# done


# for task_idx in {10..50}; do
#     uv run python my_processing_agents/solution_PS_withLLMjudge.py \
#                     --task_idx $task_idx \
#                     --num_steps 50 \
#                     --num_threads 10 \
#                     --log_frequency 1 \
#                     --test_frequency 1 \
#                     --num_candidates 3 \
#                     --num_proposals 1 \
#                     --algorithm PS_epsNet_Summarizer \
#                     --epsilon 0.02 \
#                     --with_llm_judge \
#                     --use_wandb \
#                     --project_name "LLM_judge_veribench-PS-eps0.02Net-Summarizer-num_candidates-3-num_proposals-1" 
# done


echo "All easy_set tasks completed!"
## dspy
uv run python my_processing_agents/solution_dspy_with_LLMjudge.py \
    --task_idx 10 \
    --save_results \
    --run_num 4
for task_idx in {49..50}; do    
    uv run python my_processing_agents/solution_dspy_with_LLMjudge.py \
    --task_idx $task_idx \
    --save_results
done

## OpenEvolve
uv run python my_processing_agents/solution_openevolve_with_LLMjudge.py \
                --task_idx 10 \
                --save_results \
                --run_num 3
for task_idx in {11..20}; do
    uv run python my_processing_agents/solution_openevolve_with_LLMjudge.py \
                --task_idx $task_idx \
                --save_results
done
for task_idx in {21..30}; do
    uv run python my_processing_agents/solution_openevolve_with_LLMjudge.py \
                --task_idx $task_idx \
                --save_results
done
for task_idx in {31..40}; do
    uv run python my_processing_agents/solution_openevolve_with_LLMjudge.py \
                --task_idx $task_idx \
                --save_results
done
for task_idx in {41..50}; do 
   uv run python my_processing_agents/solution_openevolve_with_LLMjudge.py \
                --task_idx $task_idx \
                --save_results
done



uv run python my_processing_agents/solution_GEPA_with_LLMjudge.py --task_idx $task_idx --save_results --log_dir gepa_running_log/results_llm_judge/gepa --max_metric_calls 150


uv run python my_processing_agents/solution_GEPA_with_LLMjudge.py --task_idx 10 --save_results --log_dir results_llm_judge/gepa_1 --max_metric_calls 20 --save_name gepa_1


# GEPA
for task_idx in {10..30}; do
    uv run python my_processing_agents/solution_GEPA_with_LLMjudge.py --task_idx $task_idx --save_results --log_dir results_llm_judge/gepa_1 --max_metric_calls 50 --save_name gepa_2
done

for task_idx in {21..30}; do
    uv run python my_processing_agents/solution_GEPA_with_LLMjudge.py --task_idx $task_idx --save_results --log_dir results_llm_judge/gepa_1 --max_metric_calls 50 --save_name gepa_1
done

for task_idx in {31..50}; do
    uv run python my_processing_agents/solution_GEPA_with_LLMjudge.py --task_idx $task_idx --save_results --log_dir results_llm_judge/gepa_1 --max_metric_calls 50 --save_name gepa_2
done

for task_idx in {41..50}; do
    uv run python my_processing_agents/solution_GEPA_with_LLMjudge.py --task_idx $task_idx --save_results --log_dir results_llm_judge/gepa_1 --max_metric_calls 50 --save_name gepa_1
done

# Updated GEPA

for task_idx in {11..50}; do
    uv run python my_processing_agents/solution_GEPA_direct_with_LLMjudge.py \
        --task_idx $task_idx \
        --max_metric_calls 50 \
        --save_results \
        --run 1 \
        --log_dir results_llm_judge/gepa \
        --save_name gepa_1
done