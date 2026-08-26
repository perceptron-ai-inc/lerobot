"""The processor factory's dataset_meta kwarg is policy-gated (PR #4 review)."""

from types import SimpleNamespace

from lerobot.scripts.lerobot_train import _should_pass_dataset_meta


def _cfg(is_reward_model_training=False, resume=False):
    return SimpleNamespace(is_reward_model_training=is_reward_model_training, resume=resume)


def test_groot_finetunes_keep_the_merge_base_behaviour():
    """GR00T's pipeline flips into training mode (state dropout + random crops) whenever
    dataset_meta is not None, so ordinary finetunes must not receive it."""
    assert _should_pass_dataset_meta(_cfg(), SimpleNamespace(type="groot")) is False


def test_groot_reward_model_training_still_receives_metadata():
    assert _should_pass_dataset_meta(_cfg(is_reward_model_training=True), SimpleNamespace(type="groot"))


def test_isaac_and_molmoact2_always_receive_metadata():
    # perceptron_isaac: dataset-clock validation + the resume exemption of the serving
    # split-brain check; molmoact2: feature names from the dataset.
    assert _should_pass_dataset_meta(_cfg(), SimpleNamespace(type="perceptron_isaac"))
    assert _should_pass_dataset_meta(_cfg(), SimpleNamespace(type="molmoact2"))
    assert _should_pass_dataset_meta(_cfg(), SimpleNamespace(type="act"))


def test_isaac_keeps_metadata_on_resume():
    """Resume remains exempt from the serving split-brain check because dataset metadata
    still reaches perceptron_isaac while dataset statistics are withheld."""
    isaac = SimpleNamespace(type="perceptron_isaac")
    assert _should_pass_dataset_meta(_cfg(resume=False), isaac)
    assert _should_pass_dataset_meta(_cfg(resume=True), isaac)
