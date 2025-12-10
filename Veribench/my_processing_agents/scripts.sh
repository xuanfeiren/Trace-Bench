# Basic training
python optimize_veribench_agent.py --num_train_samples 10 --num_steps 5

# With WandB logging
python optimize_veribench_agent.py --num_train_samples 20 --use_wandb --project_name veribench-experiment

python my_processing_agents/optimize_veribench_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --num_candidates 5 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 101 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --num_eval_samples 1 \
        --test_frequency 1 \
        --log_frequency 1 \
        --num_proposals 1 \
        --project_name "veribench-debug" \
        --run_name "debug-num_candidates-5" \
        --optoprime_version v2 \
        --use_wandb
# PS
python my_processing_agents/optimize_veribench_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --num_candidates 5 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 101 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --num_eval_samples 1 \
        --test_frequency 1 \
        --log_frequency 1 \
        --num_proposals 1 \
        --project_name "veribench" \
        --run_name "PS" \
        --optoprime_version v2 \
        --use_wandb



# PS + EpsnetPS
python my_processing_agents/optimize_veribench_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --num_candidates 5 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 101 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --num_eval_samples 1 \
        --test_frequency 1 \
        --log_frequency 1 \
        --num_proposals 1 \
        --project_name "veribench-debug" \
        --run_name "PS+EpsnetPS" \
        --optoprime_version v2 \
        --use_wandb \
        --ablation \
        --epsnetPS 

# PS + EpsnetPS + Summarizer
python my_processing_agents/optimize_veribench_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --num_candidates 5 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 101 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --num_eval_samples 1 \
        --test_frequency 1 \
        --log_frequency 1 \
        --num_proposals 1 \
        --project_name "veribench-debug" \
        --run_name "PS+EpsnetPS+Summarizer" \
        --optoprime_version v2 \
        --use_wandb \
        --ablation \
        --epsnetPS \
        --use_summarizer

python my_processing_agents/optimize_veribench_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --num_candidates 5 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 101 \
        --num_threads 50 \
        --memory_update_frequency 0 \
        --num_eval_samples 1 \
        --test_frequency 1 \
        --log_frequency 1 \
        --num_proposals 1 \
        --project_name "veribench-debug" \
        --run_name "PS+EPSNet+Summarizer" \
        --optoprime_version v2 \
        --use_wandb \
        --ablation \
        --epsnetPS \
        --use_summarizer


python my_processing_agents/optimize_veribench_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --num_candidates 1 \
        --batch_size 1 \
        --num_batches 10 \
        --num_steps 21 \
        --num_threads 50 \
        --memory_update_frequency 0 \
        --num_eval_samples 1 \
        --test_frequency 1 \
        --log_frequency 1 \
        --num_proposals 1 \
        --project_name "veribench-10-tasks" \
        --run_name "PS-candidates-1-batch_size-1-num_batches-10-claude-3-5-sonnet" \
        --optoprime_version v2 \
        --use_wandb \
        --ablation

python my_processing_agents/solution_PS.py \
        --task_idx 0 \
        --num_steps 10 \
        --num_threads 30 \
        --log_frequency 1 \
        --test_frequency 1 \
        --num_candidates 5
