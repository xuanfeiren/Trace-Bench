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