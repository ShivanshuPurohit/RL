"""
A 4x4 gridworld with top-left and bottom-right states unavailable, and rest
other states being S = {1, 2, ..., 14} and actions A = {up, down, left, right}
which deterministically cause the corresponding state transitions except for when
they would take the agent off the grid. 
The reward r(s, a, s') = -1 for all s, a, s'. Our goal is to find the optimal 
policy for this MDP.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np

GRIDSIZE = 4
# actions up, down, left, right
ACTIONS = [np.array([-1, 0]), 
		   np.array([1, 0]),
		   np.array([0, 1]),
		   np.array([0, -1])]

P_ACTION = 0.25

def is_terminal(state):
	x, y = state
	return(x==0 and y==0) or (x==GRIDSIZE-1 and y==GRIDSIZE-1)

def step(state, action):
	if is_terminal(state):
		return state, 0
	next_state = (np.array(state)+action).tolist()
	x, y = next_state
	# prevent agent from falling out
	if x < 0 or x >= GRIDSIZE or y < 0 or y >= GRIDSIZE:
		next_state = state

	reward = -1
	return next_state, reward


def plot(image):
	fig, ax = plt.subplots()
	ax.set_axis_off()
	table = Table(ax, bbox=[0, 0, 1, 1])

	n_row, n_col = image.shape
	height, width = 1. / n_row, 1. / n_col

	for (i, j), val in np.ndenumerate(image):
		table.add_cell(i, j, width, height, text=val,
						loc='center')

	for i in range(len(image)):
		table.add_cell(i, -1, width, height, text=i+1, loc='right')
		table.add_cell(-1, i, width, height/2, text=i+1, loc='center')

	ax.add_table(table)


def find_state_val(inplace=True, discount=1.):
	new_state_values = np.zeros((GRIDSIZE, GRIDSIZE))
	iteration = 0
	while True:
		if inplace:
			state_values = new_state_values
		else:
			state_values = new_state_values.copy()
		old_state_values = state_values.copy()

		for i in range(GRIDSIZE):
			for j in range(GRIDSIZE):
				value = 0
				for action in ACTIONS:
					(next_i, next_j), reward = step([i, j], action)
					value += P_ACTION * (reward + discount * state_values[next_i, next_j])
				new_state_values[i, j] = value

		max_delta_value = abs(old_state_values - new_state_values).max()
		if max_delta_value < 1e-4:
			break

		iteration += 1

	return new_state_values, iteration

def plot_fig_from_ex():
	_, asynchronous_iteration = find_state_val(inplace=True)
	values, synchronous_iteration = find_state_val(inplace=False)
	plot(np.round(values, decimals=2))
	print('In-place: {} iterations'.format(asynchronous_iteration))
	print('Synchronous: {} iterations'.format(synchronous_iteration))
	plt.savefig('fig 4_1.png')
	plt.close()


if __name__ == '__main__':
	plot_fig_from_ex()
