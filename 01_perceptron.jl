# Initialize weights with random numbers and return as a Vector.
initweights(x, y=1) = vec(rand(y, x))

# Heaviside step function
θ(z, treshold=0) = z < treshold ? 0 : 1

# Generic activation function
activate(z, f) = f(z)

# The function that sums the dot product of given inputs and weights
∑(x, w) = sum(x .* w)

# Predict the output with given input and weights, default activation function
# is θ (Heaviside)
predict(x, w, f=θ) = activate(∑(w, x), f)

# Train the Perceptron by adjusting weights based on desired output
function train(x::Matrix{Float64}, labels::Vector{Float64}, α=0.1)
    rows = size(x)[1] # Number of rows in the dataset
    inputsize = size(x)[2] # Number of columns in one row (inputs)
    w = initweights(inputsize)

    for (i, _) ∈ enumerate(1:1:rows)
        y = labels[i] # The desired output for this row (label)
        data = x[i, :] # Data in current row (vector)

        ŷ = predict(data, w) # Prediction with the current weights
       
        w += α * (ŷ - y) * data # Update the weights if necessary (ŷ - y != 0)
    end
    w # Return the weights
end
