from mitigation.hppo_agent import HierarchicalPPO, HPPOConfig


def test_hppo_step_update():
    hppo = HierarchicalPPO(HPPOConfig(steps=16, epochs=1), obs_dim=4)
    from mitigation.hppo_agent import demo_env
    buf, _ = hppo.step_env(demo_env, steps=16)
    out = hppo.update(buf)
    assert "loss" in out
