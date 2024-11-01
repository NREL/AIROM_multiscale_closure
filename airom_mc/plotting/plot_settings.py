label_models = {
        'differentiable':'A Posteriori (II)',
        'static':'A Priori (I)',
        'comsol':'3D - COMSOL',
        'baseline':r"$\bar F'_0$ - Baseline",
    }
axis_labels = {
    "tar_yield":"Tar Yield [mol/mol]",
    "char_yield":"Char Yield [mol/mol]",
    "FL":"FL [mm]",
    'aspect':"Aspect"
}
palette_models = {
    label_models['comsol']:'k',
    label_models['baseline']:'lightgrey',
    label_models['static']:'#7fcdbb',
    label_models['differentiable']:'#2c7fb8'
}

axis_lims = {
    'tar_yield':(0.45,0.78),
    'char_yield':(0.14,0.21)
}