# Initialize weights with random numbers, given matrix size
initweights(x, y) = rand(y, x)

# Heaviside step function
step(z) = z < 0 ? 0 : 1

# Generic activation function
activate(f, z) = f(z)

# The function that sums the dot product of given inputs and weights
∑(x, w) = sum(x .* w)

# Predict the output with given input and weights
predict(x, w, f=step) = activate(f, ∑(w, x))

