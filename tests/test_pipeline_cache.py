"""The cache of the pipeline stages must distinguish every option that changes their output."""
import dataclasses

from twws.pipeline import ORACLE_DEPENDENCIES, STAGE_B_FIELDS, PrepOptions


def test_stage_b_key_covers_every_oracle_option():
    missing = sorted({f for deps in ORACLE_DEPENDENCIES.values() for f in deps} - set(STAGE_B_FIELDS))
    assert not missing, missing


def test_first_version_oracle_options_change_the_stage_b_key():
    o = PrepOptions()
    for change in ({"signature_any": True}, {"domain_fanout_quantile": 0.95}, {"domain_fanout_min": 30.0}):
        assert dataclasses.replace(o, **change).key(STAGE_B_FIELDS) != o.key(STAGE_B_FIELDS), change
