import numpy as np
import pytest

from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")


def test_list_models_all():
    from dipy.nn.hub import list_models

    models = list_models()
    assert isinstance(models, dict)
    assert len(models) > 0
    assert "SynthSeg" in models


def test_list_models_filter_task():
    from dipy.nn.hub import list_models

    models = list_models(task="brain_segmentation")
    assert "SynthSeg" in models
    for v in models.values():
        assert v["task"] == "brain_segmentation"


def test_list_models_filter_unknown_task():
    from dipy.nn.hub import list_models

    models = list_models(task="nonexistent_task")
    assert models == {}


def test_load_unknown_model():
    from dipy.nn.hub import load

    with pytest.raises(ValueError, match="not found in hub registry"):
        load("NonExistentModel")


def test_registry_schema():
    from dipy.nn.hub import REGISTRY

    for name, entry in REGISTRY.items():
        assert "version" in entry, f"{name} missing 'version'"
        assert "task" in entry, f"{name} missing 'task'"
        assert "backend" in entry, f"{name} missing 'backend'"
        assert "description" in entry, f"{name} missing 'description'"


@pytest.mark.skipif(not have_torch, reason="Requires PyTorch")
def test_load_synthseg():
    from dipy.nn.hub import load

    model = load("SynthSeg")
    assert model is not None
    assert hasattr(model, "predict")


@pytest.mark.skipif(not have_torch, reason="Requires PyTorch")
def test_synthseg_predict():
    from dipy.nn.hub import load

    model = load("SynthSeg")
    T1 = np.ones((64, 64, 64), dtype=np.float32)
    affine = np.eye(4)
    labels, label_dict, mask = model.predict(T1, affine)
    assert labels.shape == T1.shape
    assert isinstance(label_dict, dict)
    assert mask.shape == T1.shape
