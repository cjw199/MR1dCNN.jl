model1 = create_model_arch(
    LayerDef((9,3), 1, 6, (0,1), 1, (1,2)),
    LayerDef((1,3), 6, 6, (0,1), 1, (1,2)),
    LayerDef((1,3), 6, 6, (0,1), 1, (1,2)),
    LayerDef((1,3), 6, 6, (0,1), 1, (1,4)),
    LayerDef((1,3), 6, 6, (0,1), 1, (1,4))

)
model2 = create_model_arch(
    LayerDef((9,5), 1, 6, (0,2), 2, (1,2)),
    LayerDef((1,5), 6, 6, (0,2), 2, (1,2)),
    LayerDef((1,5), 6, 6, (0,2), 2, (1,2)),
    LayerDef((1,5), 6, 6, (0,2), 2, (1,1))
)
model3 = create_model_arch(
    LayerDef((9,9), 1, 6, (0,4), 4, (1,2)),
    LayerDef((1,9), 6, 6, (0,4), 4, (1,2)),
    LayerDef((1,9), 6, 6, (0,4), 4, (1,1))
)

function getArch()
    [model1, model2, model3]
end

