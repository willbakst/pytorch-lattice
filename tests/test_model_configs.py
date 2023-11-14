"""Tests for the model configuration classes."""

from pytorch_lattice.model_configs import LatticeConfig, LinearConfig, _BaseModelConfig


def test_base_model_config_initialization():
    """Tests that the base model config initializes with proper defaults."""
    config = _BaseModelConfig()
    assert config.output_min is None
    assert config.output_max is None
    assert config.output_calibration_num_keypoints is None


def test_linear_config_initialization():
    """Tests that the linear config initializes with proper defaults."""
    base_config = _BaseModelConfig()
    config = LinearConfig()
    for key, value in base_config.__dict__.items():
        assert config.__dict__[key] == value
    assert config.use_bias


def test_lattice_config_initialization():
    """Tests that the lattice config initializes with proper defaults."""
    base_config = _BaseModelConfig()
    config = LatticeConfig()
    for key, value in base_config.__dict__.items():
        assert config.__dict__[key] == value
    assert config.kernel_init == "linear"
    assert config.interpolation == "simplex"
