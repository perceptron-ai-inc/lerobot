"""The registered integration must have discoverable, non-benchmark-claiming documentation."""

from pathlib import Path

from jinja2 import Template

from lerobot.policies.factory import get_policy_class
from lerobot.policies.perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    make_perceptron_isaac_pre_post_processors,
)


def test_perceptron_isaac_registration_and_processor_conventions():
    assert PerceptronIsaacConfig(device="cpu").type == "perceptron_isaac"
    assert get_policy_class("perceptron_isaac") is PerceptronIsaacPolicy
    assert PerceptronIsaacPolicy.config_class is PerceptronIsaacConfig
    assert callable(make_perceptron_isaac_pre_post_processors)


def test_perceptron_isaac_documentation_scaffolding_and_modelcard():
    root = Path(__file__).resolve().parents[3]
    tutorial = root / "docs/source/perceptron_isaac.mdx"
    assert tutorial.is_file()
    readme = root / "src/lerobot/policies/perceptron_isaac/README.md"
    assert readme.is_symlink()
    assert readme.resolve() == root / "docs/source/policy_perceptron_isaac_README.md"
    assert "local: perceptron_isaac" in (root / "docs/source/_toctree.yml").read_text()
    assert "./docs/source/perceptron_isaac.mdx" in (root / "README.md").read_text()
    content = tutorial.read_text()
    assert "MK1" in content and "unsupported" in content
    assert "mharmony[qwen35]==0.1.0" in content
    rendered = Template((root / "src/lerobot/templates/lerobot_modelcard_template.md").read_text()).render(
        model_name="perceptron_isaac"
    )
    assert "Qwen3.5" in rendered
    assert "main/en/perceptron_isaac" in rendered
