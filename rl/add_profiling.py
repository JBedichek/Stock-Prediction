#!/usr/bin/env python3
"""
Script to add profiling sections to train_actor_critic.py

This adds detailed profiling to track time spent in each operation.
"""

def add_profiling_to_train_episode():
    """Add profiling sections to train_episode_vectorized method."""

    # Read the file
    with open('rl/train_actor_critic.py', 'r') as f:
        lines = f.readlines()

    # Find key sections and add profiling
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 1. Stock selection (around line 743)
        if '            for date in unique_dates:' in line and 'OPTIMIZED: Use pre-computed' in lines[i+1]:
            new_lines.append('            with self.profiler.profile(\'stock_selection\'):\n')
            new_lines.append(line)
            # Find the end of this for loop (next unindented or less indented line)
            i += 1
            indent_level = len(line) - len(line.lstrip())
            while i < len(lines):
                current_line = lines[i]
                current_indent = len(current_line) - len(current_line.lstrip())
                if current_line.strip() and current_indent <= indent_level:
                    break
                new_lines.append(current_line)
                i += 1
            continue

        # 2. State creation current (around line 820)
        elif '            # Get states for each environment (OPTIMIZED: batched GPU transfer' in line:
            new_lines.append(line)
            new_lines.append('            with self.profiler.profile(\'state_creation_current\'):\n')
            i += 1
            # Add next 5 lines (the function call)
            for _ in range(5):
                if i < len(lines):
                    new_lines.append(lines[i])
                    i += 1
            continue

        # 3. Action selection (around line 828)
        elif '            # Batch action selection (SINGLE GPU forward pass' in line:
            new_lines.append(line)
            new_lines.append('            # Use decaying epsilon for exploration\n')
            i += 1
            if i < len(lines):
                i += 1  # Skip the epsilon comment line
            new_lines.append('            with self.profiler.profile(\'action_selection\'):\n')
            # Add the results = ... line
            for _ in range(7):
                if i < len(lines):
                    new_lines.append(lines[i])
                    i += 1
            continue

        # 4. Environment step (around line 856)
        elif '            # Step all environments' in line:
            new_lines.append(line)
            new_lines.append('            with self.profiler.profile(\'environment_step\'):\n')
            i += 1
            # Add next 3 lines
            for _ in range(3):
                if i < len(lines):
                    new_lines.append(lines[i])
                    i += 1
            continue

        # 5. Transition storage loop (around line 861)
        elif '            # Store transitions and update statistics' in line:
            new_lines.append(line)
            new_lines.append('            with self.profiler.profile(\'transition_storage\'):\n')
            i += 1
            # Find the end of this for loop
            indent_level = len(lines[i]) - len(lines[i].lstrip())
            while i < len(lines):
                current_line = lines[i]
                # Check if we've exited the for loop
                if current_line.strip().startswith('# Update positions') or current_line.strip().startswith('positions_list ='):
                    break
                new_lines.append(current_line)
                i += 1
            continue

        # 6. Critic training (around line 1027)
        elif '                # Train critic' in line and 'freeze_critic_episodes' in lines[i+1]:
            new_lines.append(line)
            i += 1
            new_lines.append(lines[i])  # if self.episode >= freeze_critic_episodes:
            i += 1
            new_lines.append('                    with self.profiler.profile(\'critic_training\'):\n')
            # Add next line (critic_loss, ...)
            for _ in range(1):
                if i < len(lines):
                    new_lines.append(lines[i])
                    i += 1
            continue

        # 7. Actor training (around line 1056)
        elif '                # Train actor (less frequently)' in line:
            new_lines.append(line)
            i += 1
            new_lines.append(lines[i])  # if update_idx % ...
            i += 1
            new_lines.append('                    with self.profiler.profile(\'actor_training\'):\n')
            # Add next line (actor_loss, ...)
            for _ in range(1):
                if i < len(lines):
                    new_lines.append(lines[i])
                    i += 1
            continue

        # 8. Target network update (around line 1068)
        elif '                # Update both target critics' in line:
            new_lines.append(line)
            i += 1
            new_lines.append('                with self.profiler.profile(\'target_network_update\'):\n')
            # Add next line
            for _ in range(1):
                if i < len(lines):
                    new_lines.append(lines[i])
                    i += 1
            continue

        # 9. Record steps in the main loop
        elif '            self.global_step += 1' in line:
            new_lines.append(line)
            new_lines.append('            self.profiler.record_step()\n')
            i += 1
            continue

        else:
            new_lines.append(line)
            i += 1

    # Write back
    with open('rl/train_actor_critic.py', 'w') as f:
        f.writelines(new_lines)

    print("✅ Added profiling sections to train_episode_vectorized()")

if __name__ == '__main__':
    add_profiling_to_train_episode()
