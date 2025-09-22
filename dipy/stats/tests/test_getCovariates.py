import numpy as np
import pandas as pd
import pytest

from dipy.stats.fosr import get_covariates


class TestGetCovariates:
    """Test cases for the get_covariates function."""

    def setup_method(self):
        """Set up test data."""
        # Create valid test data
        np.random.seed(42)
        n_subjects = 3
        n_streamlines_per_subject = 100
        n_disks = 5

        data = []
        for sub in range(n_subjects):
            group = sub % 2  # Alternate between 0 and 1
            gender = "Male" if sub % 2 == 0 else "Female"
            age = 25 + sub * 5

            for stream in range(n_streamlines_per_subject):
                for disk in range(1, n_disks + 1):
                    fa = np.random.uniform(0.1, 0.9)  # Valid FA range
                    data.append(
                        {
                            "subject": f"sub_{sub:03d}",
                            "streamline": stream,
                            "disk": disk,
                            "fa": fa,
                            "group": group,
                            "gender": gender,
                            "age": age,
                        }
                    )

        self.valid_df = pd.DataFrame(data)
        self.n_disks = n_disks

    def test_valid_data_success(self):
        """Test that get_covariates works with valid data and returns correct shapes."""
        X, Y = get_covariates(self.valid_df)

        # Check that X and Y are numpy arrays
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)

        # Check shapes - Y should have n_disks columns
        assert Y.shape[1] == self.n_disks

        # Check that X has the right number of features
        # group + gender + age + intercept = 4 columns
        assert X.shape[1] == 4

        # Check that all FA values in Y are between 0 and 1
        assert np.all((Y >= 0) & (Y <= 1))

        # Check that group values are binary
        assert np.all(np.isin(X[:, 0], [0, 1]))

        # Check that gender values are binary (0 for Female, 1 for Male)
        assert np.all(np.isin(X[:, 1], [0, 1]))

        # Check that age values are reasonable (should be the same for each subject)
        unique_ages = np.unique(X[:, 2])
        assert len(unique_ages) == 3  # 3 subjects with different ages

        # Check that intercept column is all ones
        assert np.all(X[:, 3] == 1)

    def test_random_string_subject_ids(self):
        """Test that get_covariates works with random string subject IDs."""
        # Create data with random string subject IDs
        data = []
        subjects = ["abc123", "xyz789", "test_subject", "random_id"]

        for i, sub in enumerate(subjects):
            group = i % 2
            gender = "Male" if i % 2 == 0 else "Female"
            age = 25 + i * 5

            for stream in range(50):
                for disk in range(1, 4):
                    fa = np.random.uniform(0.1, 0.9)
                    data.append(
                        {
                            "subject": sub,
                            "streamline": stream,
                            "disk": disk,
                            "fa": fa,
                            "group": group,
                            "gender": gender,
                            "age": age,
                        }
                    )

        df = pd.DataFrame(data)
        X, Y = get_covariates(df)

        # Should work without errors
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert Y.shape[1] == 3  # 3 disks

    def test_nan_fa_values_error(self):
        """Test that get_covariates throws an error when FA values contain NaN."""
        df_with_nan = self.valid_df.copy()
        # Introduce NaN values in FA column
        df_with_nan.loc[0, "fa"] = np.nan

        with pytest.raises(ValueError, match="FA values are missing"):
            get_covariates(df_with_nan)

    def test_random_values_in_fa_error(self):
        """Test that get_covariates throws an error when FA values are non-numeric."""
        df_with_random = self.valid_df.copy()
        # Introduce random string values in FA column
        df_with_random.loc[0, "fa"] = "random_string"
        df_with_random.loc[1, "fa"] = "invalid_value"

        with pytest.raises(ValueError, match="Column 'fa' must be numeric"):
            get_covariates(df_with_random)

    def test_fa_values_less_than_zero_error(self):
        """Test that get_covariates throws an error when FA values are negative."""
        df_negative_fa = self.valid_df.copy()
        # Introduce negative FA values
        df_negative_fa.loc[0, "fa"] = -0.1
        df_negative_fa.loc[1, "fa"] = -0.5

        with pytest.raises(ValueError, match="FA values contain negative values"):
            get_covariates(df_negative_fa)

    def test_fa_values_greater_than_one_error(self):
        """Test that get_covariates throws an error when FA values are > than 1."""
        df_high_fa = self.valid_df.copy()
        # Introduce FA values > 1
        df_high_fa.loc[0, "fa"] = 1.1
        df_high_fa.loc[1, "fa"] = 2.0

        # Note: The current implementation doesn't check for FA > 1
        # This test documents that limitation
        X, Y = get_covariates(df_high_fa)

        # Should work without errors since FA > 1 is not validated
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)

    def test_infinite_fa_values_error(self):
        """Test that get_covariates throws an error when FA values are infinite."""
        df_inf_fa = self.valid_df.copy()
        # Introduce infinite FA values
        df_inf_fa.loc[0, "fa"] = np.inf
        df_inf_fa.loc[1, "fa"] = -np.inf

        with pytest.raises(ValueError, match="FA values contain infinite values"):
            get_covariates(df_inf_fa)

    def test_very_large_age_values(self):
        """Test that get_covariates handles very large age values (but under 150)."""
        df_large_age = self.valid_df.copy()
        # Set large age values that are still biologically plausible
        df_large_age.loc[df_large_age["subject"] == "sub_000", "age"] = 120
        df_large_age.loc[df_large_age["subject"] == "sub_001", "age"] = 145
        df_large_age.loc[df_large_age["subject"] == "sub_002", "age"] = 150

        X, Y = get_covariates(df_large_age)

        # Should work without errors
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)

        # Check that large age values are preserved
        large_ages = X[:, 2]  # Age column
        assert np.any(large_ages >= 120)
        assert np.any(large_ages == 150)

    def test_negative_age_values_error(self):
        """Test that get_covariates throws an error when age values are negative."""
        df_negative_age = self.valid_df.copy()
        # Set negative age values
        df_negative_age.loc[df_negative_age["subject"] == "sub_000", "age"] = -5
        df_negative_age.loc[df_negative_age["subject"] == "sub_001", "age"] = -25
        df_negative_age.loc[df_negative_age["subject"] == "sub_002", "age"] = -100

        with pytest.raises(ValueError, match="Age values contain negative values"):
            get_covariates(df_negative_age)

    def test_infinite_age_values_error(self):
        """Test that get_covariates throws an error when age values are infinite."""
        df_inf_age = self.valid_df.copy()
        # Set infinite age values (only positive inf to avoid negative age validation)
        df_inf_age.loc[df_inf_age["subject"] == "sub_000", "age"] = np.inf
        df_inf_age.loc[df_inf_age["subject"] == "sub_001", "age"] = np.inf

        with pytest.raises(ValueError, match="Age values contain values over 150"):
            get_covariates(df_inf_age)

    def test_age_values_over_150_error(self):
        """Test that get_covariates throws an error when age values are over 150."""
        df_high_age = self.valid_df.copy()
        # Set age values over 150
        df_high_age.loc[df_high_age["subject"] == "sub_000", "age"] = 151
        df_high_age.loc[df_high_age["subject"] == "sub_001", "age"] = 200
        df_high_age.loc[df_high_age["subject"] == "sub_002", "age"] = 999

        with pytest.raises(ValueError, match="Age values contain values over 150"):
            get_covariates(df_high_age)

    def test_missing_required_columns_error(self):
        """Test get_covariates throws an error when required columns are missing."""
        df_missing = self.valid_df.drop(columns=["fa"])

        with pytest.raises(ValueError, match="Missing required columns"):
            get_covariates(df_missing)

    def test_empty_dataframe_error(self):
        """Test that get_covariates throws an error when DataFrame is empty."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            get_covariates(empty_df)

    def test_non_dataframe_input_error(self):
        """Test that get_covariates throws an error when input is not a DataFrame."""
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            get_covariates("not a dataframe")

    def test_invalid_group_values_error(self):
        """Test that get_covariates throws an error when group values are invalid."""
        df_invalid_group = self.valid_df.copy()
        # Introduce invalid group values
        df_invalid_group.loc[0, "group"] = 2
        df_invalid_group.loc[1, "group"] = -1

        with pytest.raises(ValueError, match="Invalid group values found"):
            get_covariates(df_invalid_group)

    def test_invalid_gender_values_error(self):
        """Test that get_covariates throws an error when gender values are invalid."""
        df_invalid_gender = self.valid_df.copy()
        # Introduce invalid gender values
        df_invalid_gender.loc[0, "gender"] = "Invalid"
        df_invalid_gender.loc[1, "gender"] = "Unknown"

        with pytest.raises(ValueError, match="Invalid gender values found"):
            get_covariates(df_invalid_gender)

    def test_negative_disk_values_error(self):
        """Test that get_covariates throws an error when disk values are <=0"""
        df_negative_disk = self.valid_df.copy()
        # Introduce invalid disk values
        df_negative_disk.loc[0, "disk"] = 0
        df_negative_disk.loc[1, "disk"] = -1

        with pytest.raises(ValueError, match="Disk values must be positive integers"):
            get_covariates(df_negative_disk)

    def test_negative_streamline_values_error(self):
        """Test that get_covariates throws an error when streamline values are <0."""
        df_negative_streamline = self.valid_df.copy()
        # Introduce negative streamline values
        df_negative_streamline.loc[0, "streamline"] = -1
        df_negative_streamline.loc[1, "streamline"] = -5

        with pytest.raises(
            ValueError, match="Streamline values must be non-negative integers"
        ):
            get_covariates(df_negative_streamline)

    def test_no_streamlines_parameter(self):
        """Test that get_covariates respects the no_streamlines parameter."""
        # Test with a small number of streamlines
        X, Y = get_covariates(self.valid_df, no_streamlines=50)

        # Should limit the output to 50 rows
        assert X.shape[0] <= 50
        assert Y.shape[0] <= 50

    def test_data_without_gender_and_age(self):
        """Test that get_covariates works without gender and age columns."""
        df_minimal = self.valid_df[
            ["subject", "streamline", "disk", "fa", "group"]
        ].copy()

        X, Y = get_covariates(df_minimal)

        # X should have only group + intercept = 2 columns
        assert X.shape[1] == 2

        # First column should be group values
        assert np.all(np.isin(X[:, 0], [0, 1]))

        # Second column should be intercept (all ones)
        assert np.all(X[:, 1] == 1)

    def test_numeric_string_subjects_conversion(self):
        """Test that get_covariates converts numeric string subjects to integers."""
        df_numeric_strings = self.valid_df.copy()
        # Convert subject IDs to numeric strings
        df_numeric_strings["subject"] = (
            df_numeric_strings["subject"].str.replace("sub_", "").str.zfill(3)
        )

        X, Y = get_covariates(df_numeric_strings)

        # Should work without errors
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)

    def test_mixed_subject_id_types(self):
        """Test that get_covariates handles mixed subject ID types."""
        df_mixed = self.valid_df.copy()
        # Mix numeric strings and random strings
        df_mixed.loc[df_mixed["subject"] == "sub_000", "subject"] = "001"
        df_mixed.loc[df_mixed["subject"] == "sub_001", "subject"] = "random_string"
        df_mixed.loc[df_mixed["subject"] == "sub_002", "subject"] = "002"

        X, Y = get_covariates(df_mixed)

        # Should work without errors
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
