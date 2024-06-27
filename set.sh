# pip install -r requirements.txt
pip install transformers==4.41.0
pip install trl==0.8.6
pip install huggingface_hub==0.23.0
pip install datasets==2.18.0
pip install accelerate==0.30.0
pip install bitsanbytes==0.42.0
pip install peft==0.10.0
pip install --upgrade torch
pip install ninja packaging tensorboardX sentencepiece
pip install --upgrade openai pyreft
MAX_JOBS=4 pip install flash-attn --no-build-isolation
# pip install  --upgrade transformers datasets accelerate evaluate bitsandbytes trl peft torch
# pip install ninja packaging tensorboardX tensorboard
# pip install flash-attn
pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes
pip install git+https://github.com/stanfordnlp/pyreft.git
pip install dataclasses
pip install prometheus-eval
pip install vllm==0.4.3
# Version control is very important here ...
