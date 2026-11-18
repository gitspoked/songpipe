"""Runs INSIDE ACE-Step .venv-ace. Reads job json, generates one song, prints wav path."""
import json, os, sys
def main():
    job = json.load(open(sys.argv[1]))
    for v in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"):
        os.environ.pop(v, None)
    sys.path.insert(0, job["ace_repo"])
    import mlx.core as mx
    # Cap MLX Metal memory BEFORE any model load. Unbounded buffer cache is
    # what spiked a 30s render past 41GB (2026-07-24): cache_limit 0 recycles
    # freed buffers immediately; memory_limit makes over-limit allocations
    # wait for in-flight GPU work instead of stacking (soft ramp, not a fail).
    lim = job.get("mlx", {})
    mx.set_cache_limit(int(lim.get("cache_limit_gb", 0) * 2**30))
    if lim.get("memory_limit_gb"):
        mx.set_memory_limit(int(lim["memory_limit_gb"] * 2**30))
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(project_root=job["ace_repo"], config_path=job["dit"],
                                     device="auto", offload_to_cpu=False)
    assert ok, f"DiT init failed: {msg}"
    llm = LLMHandler()
    msg, ok = llm.initialize(checkpoint_dir=job["checkpoint_dir"], lm_model_path=job["lm"],
                             backend="mlx", device="auto", offload_to_cpu=False, dtype=None)
    assert ok, f"LLM init failed: {msg}"
    p = job["params"]
    params = GenerationParams(task_type="text2music", thinking=True, caption=p["caption"],
        lyrics=p["lyrics"], bpm=p["bpm"], keyscale=p["keyscale"], timesignature=p["timesignature"],
        vocal_language=p["vocal_language"], duration=p["duration"],
        inference_steps=p["inference_steps"], guidance_scale=p["guidance_scale"], seed=p["seed"])
    res = generate_music(dit, llm, params=params,
                         config=GenerationConfig(batch_size=1, audio_format="wav"),
                         save_dir=job["save_dir"])
    assert res.success, f"generation failed: {res.status_message}"
    print(f"[mlx] peak_gb={mx.get_peak_memory()/2**30:.2f}")
    print(res.audios[0]["path"])
if __name__ == "__main__":
    main()
