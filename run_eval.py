# Inference 
from huggingface_hub import login
login(
  token="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", # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)

from src.inference import ReftInferencer, run_ft_inference
from src.inference import PeftInferencer 
from src.dataset.feedback_utils_v2 import Feedback
from src.dataset.format_v2 import to_distill_sft
from tqdm import tqdm as tqdm
from src.eval import run_eval_prometheus, process_eval_


# Load Fast Adaptor
# adaptor_id = "Ksgk-fy/reft_v1_elvf"
# f = ReftInferencer(adaptor_id)

# Load Slow Adaptor
# adaptor_id = "Ksgk-fy/feedback-adaptor-dft"
# adaptor_id = "Ksgk-fy/feedback-adaptor-sft"
adaptor_id = "Ksgk-fy/elvf_sft_v2_aug1"
f = PeftInferencer(adaptor_id)

# Load Dataset
feedback = Feedback(content = "Do not talk about elephant")
dataset = to_distill_sft(feedback)

# Run Inference
df_pred = run_ft_inference(f, dataset, feedback=feedback, train=False, run_id="1")
del f

# Basically anything above or equal 4 in score it a good response, otherwise it's bad 
feedbacks, scores = run_eval_prometheus(df_pred, feedback)

# Process Evaluation
df_eval = process_eval_(feedbacks, scores, df_pred, feedback, adaptor_id)