# CS50ai
CS50’s Introduction to Artificial Intelligence with Python
https://cs50.harvard.edu/ai/2020/
---
Course successfully completed: https://certificates.cs50.io/7c8622ec-0f9e-477f-87f6-46ac6960feae.pdf?size=letter
---
## # week 0: Search
---
lecture notes: https://cs50.harvard.edu/ai/2020/notes/0/

Finding a solution to a problem, like a navigator app that finds the best route from your origin to the destination, or like playing a game and figuring out the next move.

### Uniform Search

- __Depth First Search (DFS)__:  
explores a search path by traversing as far as possible along each branch before backtracking
- __Breadth First Search (BFS)__:  
opposite of DFS, explores a search path by systematically visiting all the vertices at the same level before moving to the next level

### Informed Search

- __Greedy Best First Search__:  
explores a graph by prioritizing the nodes that appear to be closest to the goal based on a heuristic function, aiming to reach the goal quickly without considering the entire search space; the efficiency of the greedy best-first algorithm depends on how good the heuristic function is
- __A $^* $ Search__:  
combines elements of breadth-first search and greedy best-first search by considering the cost of reaching a node from the start node and the estimated cost to reach the goal, using a heuristic function, in order to find the optimal path from the start to the goal node in a graph or a search space.
### Adversarial Search:  
search where the algorithm faces an opponent that tries to achieve the opposite goal (like tic-tac-toe)  

- __Minimax__:  
decision-making algorithm which aims to find the optimal move in a two-player adversarial game by considering the possible outcomes and minimizing the maximum potential loss, assuming the opponent plays optimally and alternates turns.
- __Alpha-Beta Pruning__:  
A way to optimize Minimax; skips some of the recursive computations that are decidedly unfavorable. It eliminates the evaluation of certain branches by maintaining lower and upper bounds (alpha and beta) on the possible values of nodes, thus avoiding the exploration of paths that are known to be irrelevant to the final decision, improving the efficiency of the search.
- __Depth-Limited Minimax__:  
considers only a pre-defined number of moves before it stops, without ever getting to a terminal state. However, this doesn’t allow for getting a precise value for each action, since the end of the hypothetical games has not been reached. To deal with this problem, Depth-limited Minimax relies on an evaluation function that estimates the expected utility of the game from a given state, or, in other words, assigns values to states.

### # week 0 Projects:
- Degrees: Breadth-First Search (BFS)
- Tictactoe: Adversarial Search - Minimax
---

## # week 1: Knowledge
---

lecture notes: https://cs50.harvard.edu/ai/2020/notes/1/

Representing information and drawing inferences from it.

### Propositional Logic:
- __Symbols__: most often letters (P, Q, R) that are used to represent a proposition
- __Logical Connectives__:
 logical symbols that connect propositional symbols in order to reason in a more complex way about the "given" world.
    - not (¬) inverses the truth value of the proposition
    - and (∧) connects two different propositions
    - inclusive or (∨) is true as as long as either of its arguments is true
    - implication (→) represents a structure: “if P then Q.” 
    - Biconditional (↔) implicates that “if P → Q" also "Q → P" applies  

    &nbsp;
    
    |   P   |   Q   |  P∧Q  |  P∨Q  |  P→Q  |  P↔Q  |
    |:-----:|:------|:-----:|:-----:|:-----:|:-----:|
    | false | false | false | false | true  | true  |
    | false | true  | false |  true | true  | false |
    | true  | false | false |  true | false | false |
    | true  | true  |  true |  true | true  | true  |
    
- __Model__: assignment of a truth value to every proposition
- __Knowledge Base (KB)__: a set of sentences known by a knowledge-based agent, which have the form of propositional logic sentences that can be used to make additional inferences about the "given world".
- __Entailment__ (⊨): describes the relationship between statements that hold true when one statement logically follows from one or more statements. If α ⊨ β (α entails β), then in any world where α is true, β is true, too.

### Inference:
process of deriving new sentences from old ones.  

```python
from logic import *

# Create new classes representing each proposition.
rain = Symbol("rain")  # It is raining.
hagrid = Symbol("hagrid")  # Harry visited Hagrid
dumbledore = Symbol("dumbledore")  # Harry visited Dumbledore

# Save sentences into the KB
knowledge = And(  
    Implication(Not(rain), hagrid),
    Or(hagrid, dumbledore),
    Not(And(hagrid, dumbledore)), 
    dumbledore  
    )
```
### # week 1 Projects:
- Knights: propositional logic to solve a puzzle
- Minesweeper: knowledge-base/ inferences
---
## # week 2: Uncertainty
---
lecture notes: https://cs50.harvard.edu/ai/2020/notes/2/

Dealing with uncertain events using probability.
- Unconditional Probability:  the degree of belief in a proposition in the absence of any other evidence, e.g. the result of rolling a die is not dependent on previous events.
- Conditional Probability:  degree of belief in a proposition given some evidence that has already been revealed. partial information is used to make educated guesses about the future. To use this information, which affects the probability that the event occurs in the future, we rely on conditional probability.
- Bayes' Rule: mathematical formula used to calculate the probability of an event occurring given the probability of another event occurring. It is expressed as P(A|B) = P(B|A) * P(A) / P(B)

- Joint Probability: Joint probability is the probability of two events occurring together. It is expressed in case of independent events as P(A,B)= P(A)⋅P(B) and for dependent events as P(A,B) = P(A) * P(A|B)
- Bayesian Networks: a data structure that represents the dependencies among random variables. Bayesian networks have the following properties:
    - They are directed graphs.
    - Each node on the graph represent a random variable.
    - An arrow from X to Y represents that X is a parent of Y (the probability distribution of Y depends on the value of X)
    - Each node X has probability distribution P(X | Parents(X)).

- Markov Model: assumption that the current state depends on only a finite fixed number of previous states, which makes the task manageable but the result will get more rough. Often the model is only based on the information of the one last event (e.g. predicting tomorrow’s weather based on today’s weather)
- Hidden Markov Model:
a type of a Markov model for a system with hidden states that generate some observed event. Sometimes the AI has some measurement of the world but no access to the precise state of the world. In these cases, the state of the world is called the hidden state and whatever data the AI has access to are the observations. 

### # week 2 Projects:
- PageRank: sampling pages from a Markov Chain and by iteratively applying the PageRank formula
- Heredity: Joint Probability
---
## # week 3: Optimization
---
lecture notes: https://cs50.harvard.edu/ai/2020/notes/3/

Finding not only a correct way to solve a problem, but a better—or the best—way to solve it.

- Local Search
    - Hill Climbing
    - Simulated Annealing
- Linear Programming
- Constraint Satisfaction
- Node Consistency
- Arc Consistency
- Backtracking Search: search algorithm that takes into account the structure of a constraint satisfaction search problem. In general, it is a recursive function that attempts to continue assigning values as long as they satisfy the constraints. If constraints are violated, it tries a different assignment.
    - interleaving backtracking search with inference
    - Minimum Remaining Values (MRV): heuristic to choose the next variable to assign a value to during the search process while satisfying a set of constraints and to improve the efficiency of the backtracking search by selecting the variable that has the fewest remaining options for assignment.

### # week 3 Project:
- Crossword: Backtracking Search with MRV heuristic
---
## # week 4: Learning/ Machine Learning
---

lecture notes: https://cs50.harvard.edu/ai/2020/notes/4/

Improving performance based on access to data and experience. For example, your email is able to distinguish spam from non-spam mail based on past experience.

- Supervised Learning (data with labels)
    - Nearest-Neighbor Classification
    - Perceptron Learning
    - Support Vector Machines
    - Regression
    - Loss Functions
    - Overfitting
    - Regularization
    - scikit-learn
- Reinforcement Learning
    - Markov Decision Processes
    - Q-Learning
- Unsupervised Learning
    - k-means Clustering

### # week 4 Projects:
- Shopping: nearest-neighbor Classification (sklearn)
- Nim: Q-learning

---
## # week 5: Neural Networks
---
lecture notes: https://cs50.harvard.edu/ai/2020/notes/5/

A program structure inspired by the human brain that is able to perform tasks effectively.

- Activation Functions
- Neural Network Structure
- Gradient Descent
- Multilayer Neural Networks
- Backpropagation
- Overfitting
- TensorFlow
- Computer Vision
- Image Convolution
- Convolutional Neural Networks
- Recurrent Neural Networks

### # week 5 Project:
- Traffic: Convolutional Neuronal Networks (sklearn, tensorflow.keras  )

#### Project Notes: Convolutional Neuronal Networks 
#### traffic.py - tensorflow keras sequential CNN model - composition and observations

- first approaches with only one convolutional and one maxpooling layer had very poor accuracy
- different pooling layer sizes (larger) increased accuracy, but only a bit
- adding a second convolutional and maxpooling layer increased testing accuracy significantly
- doubling the filter of the second convolutional layer increased testing accuracy even more 
- adding a third convolutional (again doubling the filter) and maxpooling layer produced very good testing accuracy of 0.976
- removing the dropout layer increases the training accuracy even more (0.99), but testing accuracy decreases ( maybe overfitted?)
- changing the activation function to sigmoid (output) increases calculation time, but results are not really better
- adding more hidden layers did not further increase the already good accuracy

-> final model has several convolutional layers with doubling filter sizes and relu activation functions and several maxpooling layers (alternating), flattening,
one hidden layer (x nodes) with dropout layer and finally the output layer with xxxx activation function

```python
# Create a convolutional neural network (CNN)
model = tf.keras.models.Sequential([

    # Convolutional layer. Learn 32 filters using a 3x3 kernel
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ),
    
    # ... several max-pooling, convolutional, flatten, hidden layers not shown here
    
    # Add an output layer with output units for all categories
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="xxxx")
])
```
---
## # week 6: Language
---

lecture notes: https://cs50.harvard.edu/ai/2020/notes/6/

Processing natural language, which is produced and understood by humans.

- Syntax and Semantics
- Context-Free Grammar
- nltk library
- n-grams
- Tokenization
- Markov Models
- Bag-of-Words Model
- Naive Bayes
- Information Retrieval
    - tf-idf library
    - Information Extraction
- Word Net DB
- Word Representation
    - word2vec

### # week 6 Projects:
- Parser: context-free grammar (nltk)
- Questions: Tokenization, idfs, Question Answering (nltk)


