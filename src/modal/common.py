from modal import Stub, Image, Volume, Secret, gpu

gpu_image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:23.10-py3", add_python="3.11")
    .pip_install(
        "wheel==0.42.0",
        "transformers==4.40.2",
        "datasets==2.18.0",
        "trl==0.8.6",
        "huggingface_hub==0.20.3",
        "hf-transfer==0.1.4",
        "accelerate==0.30.1",
        "peft==0.11.1",
        "wandb==0.17.0",
        "bitsandbytes==0.43.1",
        "scipy==1.13.0",
        "openai==1.30.1",
        "sentencepiece==0.2.0",
        gpu=gpu.A100(count=1)
    )
    .apt_install("git", "build-essential", "wget")
    .run_commands(
        [
            "pip install packaging tensorboardX sentencepiece",
            "pip install  --upgrade transformers datasets accelerate evaluate bitsandbytes trl peft torch",
            "pip uninstall -y ninja && pip install ninja",
            "pip install -U flash-attn --no-build-isolation",
            "pip install --upgrade openai pyreft",
            "pip install --no-deps \"xformers<0.0.26\" peft",
            "pip install git+https://github.com/stanfordnlp/pyreft.git",
            "pip install dataclasses"
        ],
        gpu=gpu.L4(count=1)
    )
    .env(dict(
        HF_HOME="/pretrained/huggingface",
        HF_DATASETS_CACHE="/pretrained/huggingface/datasets",
        HF_HUB_ENABLE_HF_TRANSFER="True",
        WANDB__SERVICE_WAIT="300",
        WANDB_PROJECT="general-feedback-learning",
        WANDB_WATCH="false",
        TOKENIZERS_PARALLELISM="True"))
    .pip_install(
        [
            "transformers_stream_generator",
            "tiktoken",
        ]
    )
)

non_gpu_image = (
    Image.micromamba(python_version="3.11")
    .pip_install(
        "wheel==0.41.2",
        "transformers==4.36.2",
        "datasets==2.16.1",
        "trl==0.7.10",
        "huggingface_hub==0.20.2",
        "hf-transfer==0.1.4",
        "accelerate==0.25.0",
        "peft==0.7.1",
        "wandb==0.16.1",
        "bitsandbytes==0.41.3",
        "scipy==1.11.4",
        "openai==1.6.1",
        "together==0.2.10",
        "langdetect==1.0.9",
        "tenacity==8.2.3",
        "sentencepiece==0.1.99"
    )
    .apt_install("git", "build-essential", "wget")
    .env(dict(
        HF_HOME="/pretrained/huggingface",
        HF_DATASETS_CACHE="/pretrained/huggingface/datasets",
        HF_HUB_ENABLE_HF_TRANSFER="True",
        WANDB__SERVICE_WAIT="300",
        WANDB_PROJECT="general-feedback-learning",
        WANDB_WATCH="false",
        TOKENIZERS_PARALLELISM="True"))
)

api_image = (
    Image.micromamba(python_version="3.11")
    .pip_install(
        "datasets==2.16.1",
        "langdetect==1.0.9",
        "modal==0.57.43"
    )
    .env(dict(
        HF_HOME="/pretrained/huggingface",
        HF_DATASETS_CACHE="/pretrained/huggingface/datasets",
        HF_HUB_ENABLE_HF_TRANSFER="True",
        WANDB__SERVICE_WAIT="300",
        WANDB_PROJECT="general-feedback-learning",
        WANDB_WATCH="false",
        TOKENIZERS_PARALLELISM="True"))
)

# This Big is hurting me dearly
stub = Stub(
    "Ksgk", secrets=[Secret.from_name("ksgk-secret")]
)
stub.gpu_image = gpu_image
stub.non_gpu_image = non_gpu_image
stub.api_image = api_image

# Download pre-trained models into this volume.
stub.pretrained_volume = Volume.from_name("pretrained-vol-ksgk", create_if_missing=True)

# Save trained models into this volume.

stub.results_volume = Volume.from_name("results-vol-ksgk", create_if_missing=True)

VOLUME_CONFIG = {
    "/pretrained": stub.pretrained_volume,
    "/results": stub.results_volume,
}
