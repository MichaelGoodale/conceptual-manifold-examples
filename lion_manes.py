
import torch
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm

import conceptual_manifolds.EModels as EModels
import conceptual_manifolds.language as lg
from conceptual_manifolds.train import train_model
from conceptual_manifolds.plots import generate_plot

D = 3
W = 8

stuff = {
    "lion": EPredicate(2, depth=D, width=W),
    "male_lion": EPredicate(2, depth=D, width=W),
    "female_lion": EPredicate(2, depth=D, width=W),
    'has_mane': EPredicate(2, depth=D, width=W),
    'give_live_birth': EPredicate(2, depth=D, width=W),
}


adjs = {

}

def get_props(stuff):
    prop = []
    prop.append(lg.All(stuff['male_lion'], stuff['lion']))
    prop.append(lg.All(stuff['female_lion'], stuff['lion']))
    prop.append(lg.All(stuff['female_lion'], 1-stuff['male_lion']))
    prop.append(lg.All(stuff['male_lion'], 1-stuff['female_lion']))
    prop.append(lg.All(stuff['lion'], lambda x: Or(stuff['male_lion'](x), stuff['female_lion'](x))))

    prop.append(lg.MuSample(stuff['lion'], stuff['has_mane']))
    prop.append(lg.MuSample(stuff['male_lion'], stuff['has_mane']))
    prop.append(lg.Not(lg.MuSample(stuff['female_lion'], stuff['has_mane'])))

    prop.append(lg.MuSample(stuff['lion'], stuff['give_live_birth']))
    prop.append(lg.MuSample(stuff['female_lion'], stuff['give_live_birth']))
    prop.append(lg.Not(lg.MuSample(stuff['male_lion'], stuff['give_live_birth'])))

    return prop

stuff, adjs = train_model(stuff, adjs, get_props)
generate_plot(stuff, adjs, draw_distribution=['lion', 'male_lion', 'female_lion'])
