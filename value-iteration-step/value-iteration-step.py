def value_iteration_step(values, transitions, rewards, gamma):
    num_states = len(values)
    num_actions = len(transitions[0])
    
    new_values = []
    
    for s in range(num_states):
        action_values = []
        
        for a in range(num_actions):
            expected_future = 0.0
            
            for s_next in range(num_states):
                expected_future += (
                    transitions[s][a][s_next] * values[s_next]
                )
            
            q_sa = rewards[s][a] + gamma * expected_future
            action_values.append(q_sa)
        
        new_values.append(max(action_values))
    
    return new_values