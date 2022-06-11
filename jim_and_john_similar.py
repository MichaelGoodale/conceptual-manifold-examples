
import torch
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm

import conceptual_manifolds.EModels as EModels
import conceptual_manifolds.language as lg
from conceptual_manifolds.train import train_model
from conceptual_manifolds.plots import generate_plot

D = 4
W = 4

stuff = {
    'jim': EModels.EPredicate(2, depth=D, width=W),
    'john': EModels.EPredicate(2, depth=D, width=W),
    'people': EModels.EPredicate(2, depth=D, width=W),
    'guitarist': EModels.EPredicate(2, depth=D, width=W),
}
adjs = {
    'good' : EModels.EAdjective(2, depth=D, width=W),
}
                 
def get_props(stuff, adjs):
    prop = []    
    prop.append(lg.All(stuff['guitarist'], stuff['people']))

    prop.append(lg.MuSample(stuff['jim'], lambda x: lg.Not(stuff['john'](x))))
    prop.append(lg.MuSample(stuff['john'], lambda x: lg.Not(stuff['jim'](x))))

    prop.append(lg.Similar(stuff['john'], stuff['jim'], stuff['people']))
    prop.append(lg.Not(lg.Similar(stuff['john'], stuff['jim'], stuff['guitarist'])))

    prop.append(lg.MuSample(stuff['jim'], stuff['people']))
    prop.append(lg.MuSample(stuff['john'], stuff['people']))

    prop.append(lg.MuSample(stuff['jim'], stuff['guitarist']))
    prop.append(lg.MuSample(stuff['john'], stuff['guitarist']))

    return prop

stuff, adjs = train_model(stuff, adjs, get_props)
generate_plot(stuff, adjs, draw_distribution=[], line_from_to=('john', 'jim'), line_categories=['people', 'guitarist'])
