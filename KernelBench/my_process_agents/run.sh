modal run my_process_agents/kernel_PS_modal.py --task-idx 3 --num-steps 30 --num-candidates 3 --num-threads 20 --num-proposals 1 --log-frequency 1 --test-frequency 1 --algorithm-name PS --use-wandb --project-name kernelbench --run-name task_3-PS

# debug
modal run my_process_agents/kernel_PS_modal.py --task-idx 3 --num-steps 3 --num-candidates 3 --num-threads 10 --num-proposals 1 --log-frequency 1 --test-frequency 1 --algorithm-name PS 

# notes
# task 0,1,3,4: hard
# task 2: easy

for task_id in {4..15}; do
    modal run my_process_agents/kernel_PS_modal.py --task-idx $task_id --num-steps 2 --num-candidates 2 --num-threads 10 --num-proposals 1 --log-frequency 1 --test-frequency 1 --algorithm-name PS --use-wandb --project-name kernelbench --run-name matrix_$task_id-PS
done

# try task 13
modal run my_process_agents/kernel_PS_modal.py --task-idx 13 --num-steps 50 --num-candidates 1 --num-threads 10 --num-proposals 1 --log-frequency 1 --test-frequency 1 --algorithm-name PS --use-wandb --project-name kernelbench --run-name matrix_13-PS-gemini-2.0-flash