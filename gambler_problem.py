"""
A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips.
With heads he wins the stake he has bet on the flip and loses on tails.
Game ends when either he wins the goal of $100 or runs out of money.
On each flip  he can decide on what integer dollars to bet. We will try to formulate this
problem as an MDP.

"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

GOAL = 100
# all states from 0 to goal state
STATES = np.arange(GOAL+1)

P_HEAD = 0.50

def plot_from_ex():
	state_val = np.zeros(GOAL+1)
	state_val[GOAL] = 1.
	sweep_history = []

	# value iteration
	while True:
		old_state_val = state_val.copy()
		sweep_history.append(old_state_val)

		for state in STATES[1:GOAL]:
			actions = np.arange(min(state, GOAL-state)+1)
			action_returns = []

			for action in actions:
				action_returns.append(
					P_HEAD*state_val[state + action]+(1-P_HEAD)*state_val[state - action])
			next_val = np.max(action_returns)
			state_val[state] = next_val
		delta = abs(state_val-old_state_val).max()
		if delta < 1e-7:
			sweep_history.append(state_val)
			break

	# compute optimal policy
	policy = np.zeros(GOAL+1)

	for state in STATES[1:GOAL]:
		actions = np.arange(min(state, GOAL-state)+1)
		action_returns = []

		for action in actions:
			action_returns.append(
				P_HEAD*state_val[state + action]+(1-P_HEAD)*state_val[state - action])

		# rounding off to match the plot from the book
		# based on https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
		policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]

	plt.figure(figsize=(12,12))
	plt.subplot(2, 1, 1)

	for sweep, state_val in enumerate(sweep_history):
		plt.plot(state_val, label='sweep {}'.format(sweep))
	plt.xlabel('Capital')
	plt.ylabel('Value Estimates')
	plt.legend(loc='best')

	plt.subplot(2, 1, 2)
	plt.scatter(STATES, policy)
	plt.xlabel('Capital')
	plt.ylabel('Final Policy (stake)')
	plt.savefig('plot.png')
	plt.close()

if __name__ == '__main__':
	plot_from_ex()
