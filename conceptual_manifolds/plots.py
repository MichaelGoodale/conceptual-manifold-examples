import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import torch

def generate_plot(stuff, adjs, 
        draw_distribution = None,
        line_from_to = None,
        line_categories = None,
        N_lines = 1,
        ):
    with torch.no_grad():
        n = 5000
        names = stuff.keys()
        df = pd.DataFrame(torch.cat([stuff[k].sample(n).detach() for k in names]))
        df['category'] = sum([[s]*n for s in names], [])

        surface_df = pd.DataFrame(torch.cat([stuff[k].surface(n).detach() for k in names]))
        surface_df['category'] = sum([[s]*n for s in names], [])

        sns.set(rc={'figure.figsize':(12,8)})

        if draw_distribution is None:
            draw_distribution = names

        sns.scatterplot(data=df[df.category.isin(draw_distribution)], x=0, y=1, hue='category', alpha=.05,
                   hue_order=names, legend=False)
        sns.scatterplot(data=surface_df, x=0, y=1, hue='category', hue_order=names)

        resolution = 500

        if line_from_to is not None:
            for i in range(N_lines):
                A = torch.tensor(df.loc[df['category'] == line_from_to[0], (0,1)].sample(1).to_numpy())
                B = torch.tensor(df.loc[df['category'] == line_from_to[1], (0,1)].sample(1).to_numpy())

                lines = []
                for line in line_categories:
                   lines.append(stuff[line].get_draw_paths(A, B, resolution=resolution).view(-1, 2).T)

                geodesic_df = pd.DataFrame(torch.cat(lines, dim=-1).T)
                geodesic_df['category'] = sum([[cat]*resolution for cat in line_categories], [])
                sns.lineplot(data=geodesic_df, x=0, y=1, hue='category', hue_order=names,
                             linewidth=4,
                             legend=False, estimator=None,sort=False)

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.show()
