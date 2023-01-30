# Initialize weights with random numbers and return as a Vector.
initweights(x, y = 1) = vec(rand(y, x))

# Heaviside step function
step(z, θ = 0) = z < θ ? 0 : 1

# Generic activation function
activate(f, z) = f(z)

# The function that sums the dot product of given inputs and weights
∑(x, w) = sum(x .* w)

# Predict the output with given input and weights
predict(x, w, f=step) = activate(f, ∑(w, x))

