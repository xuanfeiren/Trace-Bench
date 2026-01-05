uv run python trace_search.py --num_candidates 10 --num_proposals 5 --num_threads 20  --log_frequency 1 --test_frequency 1 --num_steps 500 --algorithm PS --use_wandb --project_name "circle-packing" --run_name "circle-packing"

uv run python trace_search_center.py --num_candidates 10 --num_proposals 3 --num_threads 30  --log_frequency 1 --test_frequency 1 --num_steps 100 --algorithm PS --use_wandb --project_name "circle-packing" --run_name "centers-search"

uv run python trace_center_search.py --num_candidates 5 --num_proposals 1 --num_threads 20  --log_frequency 1 --test_frequency 1 --num_steps 100 --algorithm PS --use_wandb --project_name "circle-packing-results" --run_name "centers-search-PS"

uv run python trace_center_function_search.py --num_candidates 5 --num_proposals 1 --num_threads 20  --log_frequency 1 --test_frequency 1 --num_steps 100 --use_wandb --project_name "circle-packing" --run_name "centers-function-search-claude-3.5-sonnet"

uv run python trace_center_search.py --num_candidates 5 --num_proposals 1 --num_threads 20  --log_frequency 1 --test_frequency 1 --num_steps 100 --algorithm PS_epsNet_Summarizer 


# wandb 

uv run python trace_center_search.py --num_candidates 5 --num_proposals 1 --num_threads 20  --log_frequency 1 --test_frequency 1 --num_steps 50 --algorithm PS --use_wandb --project_name "circle-packing-results" --run_name "centers-search-PS"

uv run python trace_center_search.py --num_candidates 5 --num_proposals 1 --num_threads 20  --log_frequency 1 --test_frequency 1 --num_steps 50 --algorithm PS_epsNet --epsilon 0.1 --use_wandb --project_name "circle-packing-results" --run_name "centers-search-PS_epsNet-epsilon-0.1"

uv run python trace_center_search.py --num_candidates 5 --num_proposals 1 --num_threads 20  --log_frequency 1 --test_frequency 1 --num_steps 50 --algorithm PS_Summarizer --use_wandb --project_name "circle-packing-results" --run_name "centers-search-PS_Summarizer"


uv run python trace_center_search.py --num_candidates 5 --num_proposals 1 --num_threads 20  --log_frequency 1 --test_frequency 1 --num_steps 50 --epsilon 0.1 --algorithm PS_epsNet_Summarizer --use_wandb --project_name "circle-packing-results" --run_name "centers-search-PS_epsNet_Summarizer-epsilon-0.1"

# wandb function optimization

uv run python trace_center_function_search.py --num_candidates 5 --num_proposals 1 --num_threads 20  --log_frequency 1 --test_frequency 1 --num_steps 50 --algorithm PS --use_wandb --project_name "circle-packing-results" --run_name "function-search-PS"

uv run python trace_center_function_search.py --num_candidates 5 --num_proposals 1 --num_threads 20  --log_frequency 1 --test_frequency 1 --num_steps 50 --algorithm PS_epsNet --epsilon 0.1 --use_wandb --project_name "circle-packing-results" --run_name "function-search-PS_epsNet-epsilon-0.1"


uv run python trace_center_function_search.py --num_candidates 5 --num_proposals 1 --num_threads 20  --log_frequency 1 --test_frequency 1 --num_steps 50 --epsilon 0.1 --algorithm PS_epsNet_Summarizer --use_wandb --project_name "circle-packing-results" --run_name "function-search-PS_epsNet_Summarizer-epsilon-0.1"

# mimic openevolve

uv run python trace_center_function_search.py --num_candidates 5 --num_proposals 1 --num_threads 20  --log_frequency 1 --test_frequency 1 --num_steps 50 --algorithm PS