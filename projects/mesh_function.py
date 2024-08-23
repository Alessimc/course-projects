import numpy as np


def mesh_function(f, t):
    mesh_point_values = np.zeros(len(t))
    for i in range(len(t)):
        mesh_point_values[i] = f(t[i])

    return mesh_point_values    

def func(t):
    return np.where((0 <= t) & (t <= 3), np.exp(-t),
                    np.where((3 < t) & (t <= 4), np.exp(-3*t), 0))

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)

# plotting
import plotly.graph_objs as go
N = 40
n = np.arange(0, N+1)
dt = 0.1
t = n*dt

fig = go.Figure()

fig.add_trace(go.Scatter(x=t, y=func(t), mode='lines', name='f(t) = sin(t)'))

fig.update_layout(
    title=r'$\text{Plot of }f(t) = \begin{cases} '
          r'\exp(-t) & \text{if } 0 \leq t \leq 3, \\'
          r'\exp(-3t) & \text{if } 3 < t \leq 4, \\'
          r'\end{cases}$',
    xaxis_title='t',
    yaxis_title='f(t)',
    template='plotly_dark',
    font=dict(size=16)
)

# fig.show()