import torch
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm

import conceptual_manifolds.EModels as EModels
import conceptual_manifolds.language as lg
from conceptual_manifolds.train import train_model
from conceptual_manifolds.plots import generate_plot

D = 4
W = 50


stuff = {
    "lion": EModels.EPredicate(2, depth=D, width=W),
    'has_mane': EModels.EPredicate(2, depth=D, width=W),
    'give_live_birth': EModels.EPredicate(2, depth=D, width=W),
    'male_lion': EModels.EPredicate(2, depth=D, width=W),
    'female_lion': EModels.EPredicate(2, depth=D, width=W),
}


adjs = {
}

def get_props(stuff, adjs):
    prop = []
    prop.append(lg.All(stuff['male_lion'], stuff['lion']))
    prop.append(lg.All(stuff['female_lion'], stuff['lion']))

    prop.append(lg.All(stuff['female_lion'], lambda x: lg.Not(stuff['male_lion'](x))))
    prop.append(lg.All(stuff['male_lion'], lambda x: lg.Not(stuff['female_lion'](x))))

    prop.append(lg.All(stuff['lion'], lambda x: lg.Or(stuff['male_lion'](x), stuff['female_lion'](x))))


    prop.append(lg.MuSample(stuff['lion'], stuff['has_mane']))
    prop.append(lg.MuSample(stuff['male_lion'], stuff['has_mane']))
    prop.append(lg.Not(lg.MuSample(stuff['female_lion'], stuff['has_mane'])))

    prop.append(lg.MuSample(stuff['lion'], stuff['give_live_birth']))
    prop.append(lg.MuSample(stuff['female_lion'], stuff['give_live_birth']))
    prop.append(lg.Not(lg.MuSample(stuff['male_lion'], stuff['give_live_birth'])))
    return prop

stuff, adjs = train_model(stuff, adjs, get_props, lr=0.003, N=1000)
generate_plot(stuff, adjs, draw_distribution=['lion', 'female_lion', 'male_lion'])
