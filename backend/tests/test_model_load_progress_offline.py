"""
Regression test for voicebox #434 (infinite HF retry loop on a cached model).

``model_load_progress`` is the single context manager every backend's
``_load_model_sync`` uses around its ``from_pretrained()`` call. It already
receives ``is_cached`` but never forwarded it to ``force_offline_if_cached``,
so a fully-cached model still resolved config files against huggingface.co
and ate the default 5-retry backoff per file offline. This asserts the guard
is actually active for the duration of the ``with`` block when ``is_cached``
is ``True``, and restored on exit.

NOTE: mutates process-global state in ``huggingface_hub.constants`` and
``transformers.utils.hub`` (via ``force_offline_if_cached``); run serially,
same caveat as ``test_offline_guard.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.backends.base import model_load_progress


def _hf_const():
    import huggingface_hub.constants as hf_const

    return hf_const


def _tf_hub():
    import transformers.utils.hub as tf_hub

    return tf_hub


def test_cached_model_load_forces_offline_mode():
    original_hf = _hf_const().HF_HUB_OFFLINE
    original_tf = _tf_hub()._is_offline_mode

    with model_load_progress("test-model", is_cached=True):
        assert _hf_const().HF_HUB_OFFLINE is True
        assert _tf_hub()._is_offline_mode is True

    assert original_hf == _hf_const().HF_HUB_OFFLINE
    assert original_tf == _tf_hub()._is_offline_mode


def test_uncached_model_load_does_not_force_offline_mode():
    original_hf = _hf_const().HF_HUB_OFFLINE

    with model_load_progress("test-model", is_cached=False):
        assert original_hf == _hf_const().HF_HUB_OFFLINE

    assert original_hf == _hf_const().HF_HUB_OFFLINE
