import numpy as np
import pandas as pd

from dipy.stats.gam import gam


class TestGAM:
    """Test cases for the gam function."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)

        # Create test data
        self.n_samples = 100
        self.n_features = 5

        # Generate random design matrix
        self.Xmat = np.random.randn(self.n_samples, self.n_features)

        # Generate response variable with some relationship to predictors
        self.y = (
            0.5 * self.Xmat[:, 0]
            + 0.3 * self.Xmat[:, 1]
            + 0.1 * self.Xmat[:, 2]
            + np.random.normal(0, 0.1, self.n_samples)
        )

        # Create smoothing matrices (penalty matrices)
        self.S = [
            np.eye(self.n_features),  # Identity matrix
            np.random.randn(self.n_features, self.n_features),  # Random matrix
            np.diag(np.random.uniform(0.1, 1.0, self.n_features)),  # Diagonal matrix
        ]

        # Create smoothing parameters
        self.labda = [0.1, 0.5, 1.0]

    def test_basic_functionality(self):
        """Test that gam function works with basic inputs."""
        result = gam(self.y, self.Xmat, S=self.S)

        # Check that result is a dictionary
        assert isinstance(result, dict)

        # Check that all expected keys are present
        expected_keys = ["gam", "coefficients", "Vp", "GinvXT", "cov_params", "Z"]
        assert all(key in result for key in expected_keys)

        # Check that coefficients have correct shape
        assert result["coefficients"].shape == (self.n_features,)

        # Check that Vp has correct shape
        assert result["Vp"].shape == (self.n_features, self.n_features)

        # Check that GinvXT has correct shape
        assert result["GinvXT"].shape == (self.n_features, self.n_samples)

        # Check that Z is identity matrix
        assert np.allclose(result["Z"], np.eye(self.n_features))

    def test_with_labda_parameter(self):
        """Test that gam function works when labda parameter is provided."""
        result = gam(self.y, self.Xmat, S=self.S, labda=self.labda)

        # Should work without errors
        assert isinstance(result, dict)
        assert "coefficients" in result

        # Check that coefficients are reasonable (not all zeros or NaNs)
        assert not np.allclose(result["coefficients"], 0)
        assert not np.any(np.isnan(result["coefficients"]))

    def test_without_labda_parameter(self):
        """Test that gam function works when labda parameter is not provided."""
        result = gam(self.y, self.Xmat, S=self.S, labda=None)

        # Should work without errors
        assert isinstance(result, dict)
        assert "coefficients" in result

    def test_gam_method_parameter(self):
        """Test that gam function handles gam_method parameter"""
        # Test with different gam_method values
        result1 = gam(self.y, self.Xmat, S=self.S, gam_method="REML")
        result2 = gam(self.y, self.Xmat, S=self.S, gam_method="GCV")
        result3 = gam(self.y, self.Xmat, S=self.S, gam_method=None)

        # All should work and return similar results
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert isinstance(result3, dict)

        # Coefficients should be similar (same data, same model)
        assert np.allclose(result1["coefficients"], result3["coefficients"], rtol=1e-10)
        assert np.allclose(result2["coefficients"], result3["coefficients"], rtol=1e-10)

    def test_different_matrix_shapes(self):
        """Test that gam function works with different matrix shapes."""
        # Test with different numbers of samples
        y_small = self.y[:50]
        Xmat_small = self.Xmat[:50, :]
        S_small = [np.eye(self.n_features)]

        result = gam(y_small, Xmat_small, S=S_small)
        assert result["coefficients"].shape == (self.n_features,)

        # Test with different numbers of features
        Xmat_wide = np.random.randn(self.n_samples, 10)
        S_wide = [np.eye(10)]

        result = gam(self.y, Xmat_wide, S=S_wide)
        assert result["coefficients"].shape == (10,)

    def test_identity_smoothing_matrix(self):
        """Test that gam function works correctly with identity smoothing matrices."""
        S_identity = [np.eye(self.n_features)]

        result = gam(self.y, self.Xmat, S=S_identity)

        # Z should be identity matrix
        assert np.allclose(result["Z"], np.eye(self.n_features))

        # Coefficients should be the same as without smoothing
        result_no_smooth = gam(self.y, self.Xmat, S=S_identity, labda=[0.0])
        assert np.allclose(result["coefficients"], result_no_smooth["coefficients"])

    def test_zero_smoothing_parameters(self):
        """Test that gam function works with zero smoothing parameters."""
        result = gam(self.y, self.Xmat, S=self.S, labda=[0.0, 0.0, 0.0])

        # Should work without errors
        assert isinstance(result, dict)
        assert "coefficients" in result

    def test_large_smoothing_parameters(self):
        """Test that gam function works with large smoothing parameters."""
        large_labda = [10.0, 100.0, 1000.0]

        result = gam(self.y, self.Xmat, S=self.S, labda=large_labda)

        # Should work without errors
        assert isinstance(result, dict)
        assert "coefficients" in result

    def test_negative_smoothing_parameters(self):
        """Test that gam function handles negative smoothing parameters."""
        negative_labda = [-0.1, -0.5, -1.0]

        # Should work without errors (though may not be mathematically sound)
        result = gam(self.y, self.Xmat, S=self.S, labda=negative_labda)

        assert isinstance(result, dict)
        assert "coefficients" in result

    def test_single_smoothing_matrix(self):
        """Test that gam function works with a single smoothing matrix."""
        S_single = [np.eye(self.n_features)]

        result = gam(self.y, self.Xmat, S=S_single)

        assert isinstance(result, dict)
        assert result["coefficients"].shape == (self.n_features,)

    def test_multiple_smoothing_matrices(self):
        """Test that gam function works with multiple smoothing matrices."""
        result = gam(self.y, self.Xmat, S=self.S)

        assert isinstance(result, dict)
        assert result["coefficients"].shape == (self.n_features,)

    def test_covariance_matrix_properties(self):
        """Test that the returned covariance matrices have expected properties."""
        result = gam(self.y, self.Xmat, S=self.S)

        # Vp should be symmetric
        assert np.allclose(result["Vp"], result["Vp"].T)

        # cov_params should be symmetric
        assert np.allclose(result["cov_params"], result["cov_params"].T)

        # Both should be positive semi-definite (eigenvalues >= 0)
        Vp_eigenvals = np.linalg.eigvals(result["Vp"])
        cov_eigenvals = np.linalg.eigvals(result["cov_params"])

        assert np.all(Vp_eigenvals >= -1e-10)  # Allow small numerical errors
        assert np.all(cov_eigenvals >= -1e-10)

    def test_coefficient_consistency(self):
        """Test that coefficients are consistent across multiple calls."""
        result1 = gam(self.y, self.Xmat, S=self.S)
        result2 = gam(self.y, self.Xmat, S=self.S)

        # Coefficients should be identical
        assert np.allclose(result1["coefficients"], result2["coefficients"])

        # Vp should be identical
        assert np.allclose(result1["Vp"], result2["Vp"])

    def test_input_validation(self):
        """Test that gam function handles various input types correctly."""
        # Test with numpy arrays
        result_np = gam(np.array(self.y), np.array(self.Xmat), S=self.S)
        assert isinstance(result_np, dict)

        # Test with pandas Series
        y_pd = pd.Series(self.y)
        result_pd = gam(y_pd, self.Xmat, S=self.S)
        assert isinstance(result_pd, dict)

        # Results should be similar
        assert np.allclose(result_np["coefficients"], result_pd["coefficients"])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very small sample size
        y_tiny = self.y[:5]
        Xmat_tiny = self.Xmat[:5, :]
        S_tiny = [np.eye(self.n_features)]

        result = gam(y_tiny, Xmat_tiny, S=S_tiny)
        assert isinstance(result, dict)
        assert result["coefficients"].shape == (self.n_features,)

        # Test with single feature
        Xmat_single = self.Xmat[:, :1]
        S_single = [np.eye(1)]

        result = gam(self.y, Xmat_single, S=S_single)
        assert isinstance(result, dict)
        assert result["coefficients"].shape == (1,)

    def test_numerical_stability(self):
        """Test numerical stability with ill-conditioned matrices."""
        # Create ill-conditioned design matrix
        Xmat_ill = self.Xmat.copy()
        Xmat_ill[:, 0] = Xmat_ill[:, 0] * 1e-10  # Make first column very small

        result = gam(self.y, Xmat_ill, S=self.S)

        # Should not crash
        assert isinstance(result, dict)
        assert "coefficients" in result

        # Check for reasonable values (not inf or NaN)
        assert not np.any(np.isinf(result["coefficients"]))
        assert not np.any(np.isnan(result["coefficients"]))

    def test_return_value_types(self):
        """Test that all return values have correct types."""
        result = gam(self.y, self.Xmat, S=self.S)

        # Check types of return values
        assert hasattr(result["gam"], "params")  # Should be a fitted model
        assert isinstance(result["coefficients"], np.ndarray)
        assert isinstance(result["Vp"], np.ndarray)
        assert isinstance(result["GinvXT"], np.ndarray)
        assert isinstance(result["cov_params"], np.ndarray)
        assert isinstance(result["Z"], np.ndarray)

    def test_coefficient_significance(self):
        """Test that coefficients make sense given the data."""
        result = gam(self.y, self.Xmat, S=self.S)

        # Coefficients should not be all zero
        assert not np.allclose(result["coefficients"], 0)

        # Coefficients should be finite
        assert np.all(np.isfinite(result["coefficients"]))

        # Coefficients should not be extremely large
        assert np.all(np.abs(result["coefficients"]) < 1e6)

    def test_matrix_dimensions_consistency(self):
        """Test that all returned matrices have consistent dimensions."""
        result = gam(self.y, self.Xmat, S=self.S)

        n_features = self.n_features
        n_samples = self.n_samples

        # Check matrix dimensions
        assert result["coefficients"].shape == (n_features,)
        assert result["Vp"].shape == (n_features, n_features)
        assert result["GinvXT"].shape == (n_features, n_samples)
        assert result["cov_params"].shape == (n_features, n_features)
        assert result["Z"].shape == (n_features, n_features)

    def test_glm_fit_parameters(self):
        """Test that the GLM fit uses expected parameters."""
        # Test that the function works and returns expected structure
        result = gam(self.y, self.Xmat, S=self.S)

        # Check that we get a valid result
        assert isinstance(result, dict)
        assert "gam" in result
        assert "coefficients" in result

        # Check that the fitted model has the expected attributes
        fitted_model = result["gam"]
        assert hasattr(fitted_model, "params")
        assert hasattr(fitted_model, "cov_params")

        # Check that coefficients are reasonable
        assert result["coefficients"].shape == (self.n_features,)
        assert not np.any(np.isnan(result["coefficients"]))
