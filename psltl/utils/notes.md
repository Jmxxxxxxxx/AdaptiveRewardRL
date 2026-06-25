1. Entry point
   run.py receives command-line arguments and starts Learner.

2. Environment construction
   get_ltl_env loads the automaton and wraps the original environment.

3. Product state
   observation = MDP state + automaton state.

4. Automaton update
   label = env.get_events()
   next_q = atm.delta(curr_q, label)

5. Reward
   progress reward = max(d(prev_q) - d(curr_q), 0)

6. Adaptive update
   if recent success rate < 10% at update interval, update distance function.

7. Training
   DQN learns on the LTL-augmented environment.