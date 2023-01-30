# Initialize weights with random numbers, given matrix size
initweights(x, y) = rand(y, x)

# Heaviside step (activation) function
activate(z) = z < 0 ? 0 : 1

# The function that sums the dot product of given inputs and weights
∑(x, w) = sum(.*(x, w))

# Predict the output with given input and weights
predict(x, w) = activate(∑(w, x))
