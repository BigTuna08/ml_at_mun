

from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.preprocessing import scale

data = load_breast_cancer()
type(data)

df = pd.DataFrame(data.data)
df.columns = data.feature_names
df



df = pd.DataFrame(scale(data.data))
df.columns = data.feature_names
df


from plotly import graph_objects as go

go.Figure(data=go.Heatmap(z=df.values)).show(renderer="svg")