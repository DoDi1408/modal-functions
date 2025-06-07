import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("cupy-cuda12x")
    .pip_install("torch")
    .pip_install(
        "vllm==0.8.5",
        "huggingface_hub[hf_transfer]>=0.30.0",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        "transformers>=4.51.0", # Add this line
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

#REPO_ID = "Romoamigo/Qwen3-14B-LoRA-adapters"
MODEL_NAME = "Qwen/Qwen3-32B-FP8"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App("openai-compatible-ROMOAMIGO-QWEN3-14B")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count

MINUTES = 60  # seconds
VLLM_PORT = 8000

@app.function(
    image=vllm_image,
    gpu=f"L40S:{N_GPU}",
    scaledown_window=6 * MINUTES,  # how long should we stay up with no requests?
    timeout=12 * MINUTES,  # how long should we wait for container start?
    volumes={"/root/.cache/huggingface": hf_cache_vol,"/root/.cache/vllm": vllm_cache_vol,},
    secrets=[modal.Secret.from_name("huggingface-secret"),
             modal.Secret.from_name("custom-secret")],
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    import os
    import subprocess
    import json
    from huggingface_hub import login, snapshot_download

    login(token=os.environ["HF_TOKEN"])

    """
    c_plus_plus_lora = snapshot_download(repo_id=REPO_ID)

    lora_config = {
        "name": "cplusplus-lora",
        "path": c_plus_plus_lora,
        "base_model_name": MODEL_NAME
    }
    """

    rope_scaling = {
        "rope_type": "yarn",
        "factor": 1.5,
        "original_max_position_embeddings": 32768
    }

    #lora_config_json_str = json.dumps(lora_config, ensure_ascii=False, separators=(',', ':'))
    rope_scaling_json_str = json.dumps(rope_scaling, ensure_ascii=False, separators=(',', ':'))

    #quoted_lora_config_json_str = f"'{lora_config_json_str}'"
    quoted_rope_scaling_json_str = f"'{rope_scaling_json_str}'"

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=debug",
        MODEL_NAME,
#        "--revision",
#        MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        os.environ["API_KEY"],
        "--max-model-len",
        "32768",
        "--gpu_memory_utilization",
        "0.95",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "deepseek_r1",
        "--served-model-name",
        "Qwen3-14b",
        #"--enable-lora",
        #"--lora-modules",
        #quoted_lora_config_json_str,
        #"--rope-scaling",
        #quoted_rope_scaling_json_str,
    ]

    subprocess.Popen(" ".join(cmd), shell=True)

