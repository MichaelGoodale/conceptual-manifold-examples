
import torch
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm

import conceptual_manifolds.EModels as EModels
import conceptual_manifolds.language as language 
from conceptual_manifolds.train import train_model
from conceptual_manifolds.plots import generate_plot

D = 2
W = 8

stuff = {
    'jim': EModels.EPredicate(2, depth=D, width=W),
    'john': EModels.EPredicate(2, depth=D, width=W),
    'people': EModels.EPredicate(2, depth=D, width=W),
    'fighter': EModels.EPredicate(2, depth=D, width=W),
}
adjs = {
    'good' : EModels.EAdjective(2, depth=D, width=W),
}
                 
def get_props(stuff, adjs):
    prop = []    
    prop.append(language.All(stuff['fighter'], stuff['people']))

    prop.append(language.MuSample(stuff['jim'], lambda x: language.Not(stuff['john'](x))))
    prop.append(language.MuSample(stuff['john'], lambda x: language.Not(stuff['jim'](x))))

    prop.append(language.MoreThan(adjs['good'], stuff['jim'], stuff['john'], stuff['fighter'].transform))
    prop.append(language.MoreThan(adjs['good'], stuff['john'], stuff['jim'], stuff['people'].transform))

    prop.append(language.MuSample(stuff['jim'], stuff['people']))
    prop.append(language.MuSample(stuff['john'], stuff['people']))
    prop.append(language.MuSample(stuff['jim'], stuff['fighter']))
    prop.append(language.MuSample(stuff['john'], stuff['fighter']))
    return prop

stuff, adjs = train_model(stuff, adjs, get_props)
generate_plot(stuff, adjs, draw_distribution=[], line_from_to=('john', 'jim'), line_categories=['people', 'fighter'])
