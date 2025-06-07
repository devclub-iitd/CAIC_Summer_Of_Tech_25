import numpy as np
import matplotlib.pyplot as plt
import LR

X=[]
Y=[]
for i in range(2,1000):
    temp=[]
    noise1 = np.random.normal(loc=500, scale=2)
    noise2 = np.random.normal(loc=500, scale=2)
    noise3 = np.random.normal(loc=500, scale=2)
    temp.append(i-1+noise1)
    temp.append(i+noise2)
    temp.append(i+1+noise3)
    X.append(temp)
    Y.append(i)
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)

plt.figure(figsize=(10, 6))
plt.plot(X[:, 0], Y, 'o-', label='Feature 1 vs Y')
plt.plot(X[:, 1], Y, 's-', label='Feature 2 vs Y')
plt.plot(X[:, 2], Y, '^-', label='Feature 3 vs Y')
plt.xlabel('Feature values')
plt.ylabel('Y values')
plt.title('Plot of each feature in X against Y')
plt.legend()
plt.grid(True)
plt.show()

# Manual train-test split (80% train, 20% test)
split_index = int(0.4 * len(X))
X_train = np.concatenate((X[:split_index], X[2*split_index:]), axis=0)
Y_train = np.concatenate((Y[:split_index], Y[2*split_index:]), axis=0)
X_test = X[split_index:2*split_index]
Y_test = Y[split_index:2*split_index]

Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)
X_min= np.min(X,axis=0)
X_max= np.max(X,axis=0)
Y_min= np.min(Y,axis=0)
Y_max= np.max(Y,axis=0)
# Train the model
# cant give give keyword name directly
weights, bias, costs,w_hist,b_hist = LR.linearRegression(X_train, Y_train, lr=0.01, lambda_=0.01,X_min=None,X_max=None,Y_min=None,Y_max=None,iter=1000,loss='MSE')
print("Final Weights:\n", weights)
print("Final Bias:\n", bias)
print(w_hist)
print(b_hist)

# eval
X_test_norm = (X_test - X_min) / (X_max - X_min + 1e-6)
predictions_norm= np.dot(X_test_norm, weights) + bias
predictions = predictions_norm * (Y_max - Y_min + 1e-6) + Y_min

test_error = np.mean((predictions - Y_test) ** 2)
print("Mean Squared Error on Test Data:", test_error)

from mpl_toolkits.mplot3d import Axes3D
"""
# Use first 3 features as axes
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot actual test data
ax.scatter(X_test[:, 0], X_test[:, 1], Y_test.flatten(), color='blue', label='Actual', s=50)

# Plot predicted data
ax.scatter(X_test[:, 0], X_test[:, 1], predictions.flatten(), color='red', label='Predicted', s=50)

ax.set_xlabel('Feature 1 (i-1)')
ax.set_ylabel('Feature 2 (i)')
ax.set_zlabel('Y / Predicted Y')
ax.set_title('3D Plot: Actual vs Predicted Values')
ax.legend()
plt.tight_layout()
plt.show()"""

import plotly.graph_objects as go
import numpy as np

# Flatten weights and bias
w1, w2, w3 = weights.flatten()
b = bias.flatten()[0]

# Define grid for x1 and x2
x1_vals = np.linspace(np.min(X_test[:, 0]), np.max(X_test[:, 0]), 20)
x2_vals = np.linspace(np.min(X_test[:, 1]), np.max(X_test[:, 1]), 20)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

# Compute x3 based on data structure
x3_grid = x2_grid + 1

# Compute predicted y values on grid
y_pred_grid = w1 * x1_grid + w2 * x2_grid + w3 * x3_grid + b

# Create the figure
fig = go.Figure()

# Add surface plot for the model plane
fig.add_trace(go.Surface(
    x=x1_grid,
    y=x2_grid,
    z=y_pred_grid,
    colorscale='Viridis',
    opacity=0.7,
    name='Learned Regression Surface',
    showlegend=True
))

# Add scatter points for actual data
fig.add_trace(go.Scatter3d(
    x=X_test[:, 0],
    y=X_test[:, 1],
    z=Y_test.flatten(),
    mode='markers',
    marker=dict(color='red', size=5),
    name='Actual Test Data',
    showlegend=True
))

# Update layout with legend box
fig.update_layout(
    title='Learned Regression Plane vs. Actual Test Data',
    scene=dict(
        xaxis_title='Feature 1 (i-1)',
        yaxis_title='Feature 2 (i)',
        zaxis_title='Y'
    ),
    legend=dict(
        title='Legend',
        x=0.02,
        y=0.98,
        bgcolor='rgba(255,255,255,0.7)',
        bordercolor='black',
        borderwidth=1
    ),
    width=800,
    height=600
)

fig.show()


