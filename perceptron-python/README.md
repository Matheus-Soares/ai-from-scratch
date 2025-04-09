## Perceptron in Python

This project implements a Perceptron from scratch in Python, aiming to learn the basic concepts of artificial intelligence and machine learning.

### Description

The Perceptron is a fundamental algorithm in machine learning, primarily used for binary classification tasks. It is a type of linear classifier that attempts to find a way to separates data points into two classes. The algorithm works iteratively by adjusting the weights of the input features based on the errors it makes during training.

The main limitation of the Perceptron is that it can only solve linearly separable problems, meaning it cannot handle datasets where the classes cannot be separated by a straight line (or hyperplane in higher dimensions). If the data is not linearly separable, the algorithm will fail to converge to a solution. Despite this limitation, the Perceptron is an important building block in understanding more complex machine learning models.

### Project Structure

- `core.py`: Contains the Perceptron implementation.
- `visualize.py`: Script to train the Perceptron and visualize the decision boundary.
- `README.md`: Project documentation.

### Dependencies

Make sure you have the following libraries installed:

- `numpy`
- `matplotlib`

You can install them using the following command:

```bash
pip install numpy matplotlib
```

### How to Run

1. Clone this repository:

```bash
git clone https://github.com/Matheus-Soares/ai-from-scratch.git
cd ai-from-scratch
```

2. Run the `visualize.py` script to train the Perceptron and visualize the decision boundary:

```bash
python perceptron-python/visualize.py
```

### Example Output

The script will display a plot with the data points and the decision boundary learned by the Perceptron.
