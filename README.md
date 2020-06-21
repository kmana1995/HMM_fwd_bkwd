# Read Me
State probabilities, transition probabilities, and estimates of the forward path of a hidden markovian process using forward-backward and viterbi algorithm.

The baum-welch algorithm makes use of the forward/backward algorithm to estimate the state parameters (mu and sigma parameters of each state in this continous application), and the transition probabilities (probability of transitioning from hidden state i to hidden state j) using expectation maximization. These defined features of an HMM can then be used to estimate the state and generate predictions at a given timestep using the prior information of the previous and future step.

The viterbi algorithm predicts the most likely forward pass given the state parameters and transition probabilities defined using the baum-welch algorithm highlighted above.


Baum-Welch Algorithm:
https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

Viterbi Algorithm:
https://en.wikipedia.org/wiki/Viterbi_algorithm

Article of applications:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.123.1016&rep=rep1&type=pdf
