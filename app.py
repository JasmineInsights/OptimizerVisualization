import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re

st.sidebar.image("inomaticslogo.jpg", use_column_width=True)

# Initialize session state for storing history, iteration count, and velocity
if "history" not in st.session_state:
    st.session_state.history = []
if "iteration" not in st.session_state:
    st.session_state.iteration = 0
if "velocity" not in st.session_state:
    st.session_state.velocity = 0

# Function for gradient calculation
def gradient(func, x):
    return (func(x + 1e-5) - func(x)) / 1e-5  # Numerical derivative approximation

# Sidebar for user input
st.sidebar.title("Optimizer Visualizer")

# Function input
function_str = st.sidebar.text_input("Enter the function (in terms of x):", "x^2 + 2*x + 1")
start_point = st.sidebar.number_input("Enter the starting point:", value=0.0)
learning_rate = st.sidebar.number_input("Enter the learning rate:", value=0.1, step=0.01)
optimizer = st.sidebar.selectbox(
    "Select Optimizer", 
    ("Gradient Descent", "SGD+Momentum", "NAG", "Adagrad", "Adadelta", "RMSprop")
)

# Replace common symbols with Python-friendly syntax
function_str = re.sub(r'\^', '**', function_str)
function_str = re.sub(r'([0-9])([a-zA-Z])', r'\1*\2', function_str)  # 2x -> 2*x
function_str = re.sub(r'([a-zA-Z])([0-9])', r'\1*\2', function_str)  # x2 -> x*2

# Setup button to initialize history and velocity
if st.sidebar.button("Set Up"):
    st.session_state.history = [start_point]
    st.session_state.iteration = 0
    st.session_state.velocity = 0

# Button to perform the next iteration
if st.sidebar.button("Next Iteration"):
    if st.session_state.history:
        try:
            # Convert string function to Python function
            func = lambda x: eval(function_str)
            
            # Get the current point and gradient
            current_x = st.session_state.history[-1]
            grad = gradient(func, current_x)

            if optimizer == "Gradient Descent":
                next_x = current_x - learning_rate * grad

            elif optimizer == "SGD+Momentum":
                momentum = 0.9  # Momentum factor
                st.session_state.velocity = momentum * st.session_state.velocity - learning_rate * grad
                next_x = current_x + st.session_state.velocity

            elif optimizer == "NAG":
                momentum = 0.9  # Momentum factor
                look_ahead_x = current_x + momentum * st.session_state.velocity
                st.session_state.velocity = momentum * st.session_state.velocity - learning_rate * gradient(func, look_ahead_x)
                next_x = current_x + st.session_state.velocity

            elif optimizer == "Adagrad":
                epsilon = 1e-8
                st.session_state.velocity += grad**2
                next_x = current_x - (learning_rate / (np.sqrt(st.session_state.velocity) + epsilon)) * grad

            elif optimizer == "Adadelta":
                decay_rate = 0.9
                epsilon = 1e-8
                st.session_state.velocity = decay_rate * st.session_state.velocity + (1 - decay_rate) * grad**2
                next_x = current_x - (learning_rate / (np.sqrt(st.session_state.velocity) + epsilon)) * grad

            elif optimizer == "RMSprop":
                decay_rate = 0.9
                epsilon = 1e-8
                st.session_state.velocity = decay_rate * st.session_state.velocity + (1 - decay_rate) * grad**2
                next_x = current_x - (learning_rate / (np.sqrt(st.session_state.velocity) + epsilon)) * grad

            # Append next point to history
            st.session_state.history.append(next_x)
            st.session_state.iteration += 1

            # Plot the function and the gradient descent path
            x_vals = np.linspace(min(st.session_state.history) - 10, max(st.session_state.history) + 10, 500)
            y_vals = func(x_vals)

            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, label=f'{function_str}', color='blue')
            plt.scatter(st.session_state.history, [func(x) for x in st.session_state.history], color='red', s=50)
            plt.plot(st.session_state.history, [func(x) for x in st.session_state.history], linestyle='--', color='gray')

            # Plot the slope (tangent line) at the current point
            tangent_x_vals = np.linspace(current_x - 10, current_x + 10, 100)
            tangent_y_vals = func(current_x) + grad * (tangent_x_vals - current_x)
            plt.plot(tangent_x_vals, tangent_y_vals, color='orange', label=f"Slope at x={current_x:.2f}")

            plt.xlim(min(x_vals), max(x_vals))  # Set the x-axis limits to the full range
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title(f"Iteration {st.session_state.iteration} - {optimizer}")
            plt.legend()
            st.pyplot(plt)

        except Exception as e:
            st.sidebar.error(f"Error in function: {e}")
    else:
        st.sidebar.error("Please click 'Set Up' first to initialize the starting point.")

# Display current point and iteration
if st.session_state.history:
    st.sidebar.write(f"**Current Point:** {st.session_state.history[-1]:.4f}")
    st.sidebar.write(f"**Iteration:** {st.session_state.iteration}")
