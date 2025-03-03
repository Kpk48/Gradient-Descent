import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(24)

# Define true parameters: y = slope_true * x + intercept_true + noise
slope_true, intercept_true = 2.8, 0.9

# Generate synthetic dataset
x_vals = np.linspace(-20, 20, 200)
noise_vals = np.random.normal(0, 2, x_vals.shape)
y_vals = slope_true * x_vals + intercept_true + noise_vals

def compute_mse(actual, predicted):
    """Calculate Mean Squared Error (MSE)."""
    return np.mean((actual - predicted) ** 2)

def linear_search_optimization(x, y, fixed_intercept=intercept_true, slope_range=(-5, 5), points=65):
    """Perform linear search to find the best slope that minimizes MSE."""
    slope_candidates = np.linspace(slope_range[0], slope_range[1], points)
    mse_values = [compute_mse(y, slope * x + fixed_intercept) for slope in slope_candidates]
    
    optimal_slope = slope_candidates[np.argmin(mse_values)]
    lowest_mse = min(mse_values)
    
    return slope_candidates, mse_values, optimal_slope, lowest_mse

# Execute linear search
slope_candidates, mse_values, best_slope_ls, min_mse_ls = linear_search_optimization(x_vals, y_vals)

def plot_linear_search(slope_vals, mse_vals, optimal_slope, min_mse):
    """Visualize the results of linear search."""
    plt.figure(figsize=(10, 6))
    plt.plot(slope_vals, mse_vals, color='grey', label='MSE vs Slope')
    plt.scatter(optimal_slope, min_mse, color='red', label=f'Optimal Slope: {optimal_slope:.4f}')
    plt.xlabel('Slope')
    plt.ylabel('MSE')
    plt.title('Linear Search Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_linear_search(slope_candidates, mse_values, best_slope_ls, min_mse_ls)

def gradient_descent(x, y, lr=0.001, epochs=1000, init_slope=-5, init_intercept=5):
    """Optimize slope and intercept using gradient descent."""
    slope, intercept = init_slope, init_intercept
    n = len(x)
    
    slope_history, mse_history = [], []
    
    for _ in range(epochs):
        predictions = slope * x + intercept
        
        # Compute gradients
        grad_slope = (-2 / n) * np.sum(x * (y - predictions))
        grad_intercept = (-2 / n) * np.sum(y - predictions)
        
        # Update parameters
        slope -= lr * grad_slope
        intercept -= lr * grad_intercept
        
        # Store values for visualization
        slope_history.append(slope)
        mse_history.append(compute_mse(y, predictions))
    
    return slope, intercept, slope_history, mse_history

# Execute gradient descent
slope_gd, intercept_gd, slope_steps, mse_steps = gradient_descent(x_vals, y_vals)

def plot_gradient_descent(slope_vals, mse_vals, slope_steps, mse_steps):
    """Plot gradient descent progress alongside linear search results."""
    mse_gd_values = [compute_mse(y_vals, slope * x_vals + intercept_gd) for slope in slope_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(slope_vals, mse_gd_values, color='grey', label='MSE vs Slope (Gradient Descent)')
    plt.scatter(slope_steps, [compute_mse(y_vals, slope * x_vals + intercept_gd) for slope in slope_steps],
                color='red', label='Gradient Steps', alpha=0.8, marker='o')
    plt.scatter(slope_steps[-1], compute_mse(y_vals, slope_steps[-1] * x_vals + intercept_gd),
                color='red', label=f'Final Slope: {slope_steps[-1]:.4f}', zorder=3)
    plt.xlabel('Slope')
    plt.ylabel('MSE')
    plt.title('Gradient Descent Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_gradient_descent(slope_candidates, mse_values, slope_steps, mse_steps)

# Display results
print(f"True Parameters: Slope = {slope_true}, Intercept = {intercept_true}")
print(f"Linear Search: Best Slope = {best_slope_ls:.4f}, Min MSE = {min_mse_ls:.4f}")
print(f"Gradient Descent: Final Slope = {slope_gd:.4f}, Min MSE = {min(mse_steps):.4f}")
