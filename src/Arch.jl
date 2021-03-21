using Flux

model1 = create_model_arch(
    LayerDef((9,3), 1, 8, (0,1), 1, (1,2)),
    LayerDef((1,3), 8, 16, (0,1), 1, (1,2)),
    LayerDef((1,3), 16, 32, (0,1), 1, (1,2))
)
model2 = create_model_arch(
    LayerDef((9,5), 1, 8, (0,2), 2, (1,2)),
    LayerDef((1,5), 8, 16, (0,2), 2, (1,2)),
    LayerDef((1,5), 16, 32, (0,2), 2, (1,1))
)
model3 = create_model_arch(
    LayerDef((9,9), 1, 8, (0,4), 4, (1,2)),
    LayerDef((1,9), 8, 16, (0,4), 4, (1,2)),
    LayerDef((1,9), 16, 32, (0,4), 4, (1,1))
)

function getArch()
    [model1, model2, model3]
end