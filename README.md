# Read Me

Hidden markov models attempt to identify and predict the impact of underlying "hidden" states on the observable output of a process. Once the elements of the underlying states are accurately estimated, one can more effectively estimate the given state of a process or the likelihood of a future output (prediction). Hidden markov models have applications across continuous and discrete observations, this application assumes continuous outputs with normal structures of the hidden states.

State probabilities, transition probabilities, estimates of the forward path, and OOB predictions of a hidden markovian process are produced. This model includes application of the baum-welch (forward-backward) and viterbi algorithm. Testing of this code has been mainly focused on time series forecasting applications.

The baum-welch algorithm makes use of the forward/backward algorithm to estimate the state parameters (mu and sigma parameters of each state in this continous application), and the transition probabilities (probability of transitioning from hidden state i to hidden state j) using expectation-maximization. These defined features of an HMM can then be used to estimate the state and generate predictions at a given timestep using the prior information of the previous and future step.

The viterbi algorithm predicts the most likely forward pass given the state parameters and transition probabilities defined using the baum-welch algorithm highlighted above. The viterbi algorithm is especially useful when applied to time series since it does not make unrealistic use of future observations, only the prior observation.


Baum-Welch Algorithm:
https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

Viterbi Algorithm:
https://en.wikipedia.org/wiki/Viterbi_algorithm

Article of applications:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.123.1016&rep=rep1&type=pdf
