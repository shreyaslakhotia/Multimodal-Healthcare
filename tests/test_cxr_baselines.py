import numpy as np
from icu_mm.cxr_baselines import fold_pca


def test_cxr_pca_is_fitted_on_training_rows_only(tmp_path):
    rng=np.random.default_rng(55)
    embeddings=rng.normal(size=(100,70)).astype(np.float32)
    training=np.arange(80)
    transformed,pca=fold_pca(tmp_path,embeddings,training,0,'main')
    np.testing.assert_allclose(pca.mean_,embeddings[training].mean(axis=0),rtol=1e-6,atol=1e-6)
    assert transformed.shape==(100,64)
    changed=embeddings.copy()
    changed[80:]*=1e6
    other=tmp_path/'other'
    other.mkdir()
    _,refit=fold_pca(other,changed,training,0,'main')
    np.testing.assert_array_equal(pca.mean_,refit.mean_)
    np.testing.assert_allclose(pca.components_,refit.components_,rtol=0,atol=0)
