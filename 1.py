# «input_test»  (to ".input_test")
# «api_test»  (to ".api_test")

"""
 (eepitch-vterm)                                                                                                      
mkdir -p /tmp/runpod/                                                                                             # A ideia é a seguinte: Vai haver um script, que vai ser feito seu upload para o github, a plataforma runpod vai puxar e
cd       /tmp/runpod/                                                                                             
#  (find-dailydiag "29-10-2026.org" "112"  ".runpod_cache_vs_hugginface_cache" "linking-cache")
source ~/.venv-pytorch/bin/activate
pip install runpod transformers
ipython --no-autoindent
 (forward-line)
"""

import os; import runpod; from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
MODEL_ID = os.environ.get("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
os.environ["HF_HUB_OFFLINE"] = "1"                   # Force offline mode to use only cached models
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# (find-dailydiag "28-09-2026.org" "110"  ".runpod-doc-tutorial-model-caching" "def resolve_snapshot_path")
# def resolve_snapshot_path(model_id: str) -> str:                                                                  # Qual a necessidade dessa função? 
#     if "/" not in model_id:                                                                                       # O huggingface já não resolveria essa resolução sozinho? 
#         raise ValueError(f"MODEL_ID '{model_id}' is not in 'org/name' format")
#     org, name = model_id.split("/", 1)
#     model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
#     refs_main = os.path.join(model_root, "refs", "main")
#     snapshots_dir = os.path.join(model_root, "snapshots")
#     print(f"[ModelStore] MODEL_ID: {model_id}")
#     print(f"[ModelStore] Model root: {model_root}")
#     if os.path.isfile(refs_main):
#         with open(refs_main, "r") as f:
#             snapshot_hash = f.read().strip()
#         candidate = os.path.join(snapshots_dir, snapshot_hash)
#         if os.path.isdir(candidate):
#             print(f"[ModelStore] Using snapshot from refs/main: {candidate}")
#             return candidate
#     if not os.path.isdir(snapshots_dir):
#         raise RuntimeError(f"[ModelStore] snapshots directory not found: {snapshots_dir}")
#     versions = [
#         d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))
#     ]
#     if not versions:
#         raise RuntimeError(f"[ModelStore] No snapshot subdirectories found under {snapshots_dir}")
#     versions.sort()
#     chosen = os.path.join(snapshots_dir, versions[0])
#     print(f"[ModelStore] Using first available snapshot: {chosen}")
#     return chosen
# LOCAL_MODEL_PATH = resolve_snapshot_path(MODEL_ID); print(f"[ModelStore] Resolved local model path: {LOCAL_MODEL_PATH}")           # (find-daily "15-10-2026.org" ".runpod-local-test-download-model")
# tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=False, local_files_only=True, )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False, local_files_only=True, )
model     = AutoModelForCausalLM.from_pretrained( MODEL_ID, trust_remote_code=False, torch_dtype="auto",
                                                  device_map="auto", local_files_only=True, attn_implementation="eager", )
text_gen  = pipeline( "text-generation", model=model, tokenizer=tokenizer, )
print("[ModelStore] Model loaded from local snapshot")
# text_gen("In this course, we will teach you how to", max_length=15, num_return_sequences=1,)

def handler(job):                                                                           # Handler function that processes each inference request.
    job_input   = job.get("input", {}) or {}                                      	    # Args:
    prompt      = job_input.get("prompt", "Hello!")						    #     job: Runpod job object containing input data
    max_tokens  = int(job_input.get("max_tokens", 256))					    # Returns:
    temperature = float(job_input.get("temperature", 0.7))				    #     Dictionary with generated text or error information
    print(f"[Handler] Prompt: {prompt[:80]!r}")
    print(f"[Handler] max_tokens={max_tokens}, temperature={temperature}")
    try:
        outputs = text_gen(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
        )
        generated = outputs[0]["generated_text"]
        print(f"[Handler] Generated length: {len(generated)} chars")
        return {
            "status": "success",
            "output": generated,
        }
    except Exception as e:
        print(f"[Handler] Error during generation: {e}")
        return {
            "status": "error",
            "error": str(e),
        }

runpod.serverless.start({"handler": handler})




"""
#  (runpodocw "serverless/development/local-testing")
#  (runpodocw "serverless/development/local-testing" "## Debugging")
#  (runpodocw "serverless/development/local-testing" "## Debugging" "### Log levels" "* `DEBUG`")
# «.input_test»	(to "input_test")
 (eepitch-vterm)                                                                                                      
cd       ~/rptest/
source ~/.venv-pytorch/bin/activate
python 1.py --test_input '{"input": {"prompt": "Hello, world!"}}'

# Isso aqui não esta funcionando corretamente
# «.api_test»	(to "api_test")
 (eepitch-vterm)                                                                                                      
cd       ~/rptest/
source ~/.venv-pytorch/bin/activate
# python 1.py --rp_serve_api
python 1.py --rp_serve_api --rp_log_level DEBUG

 (eepitch-shell)
 (find-3a nil '(find-ebuffer "*vterm*") '(find-ebuffer "*shell*"))
curl -X POST http://localhost:8000/runsync \
     -H "Content-Type: application/json" \
     -d '{"input": {"prompt": "Hello, world!", "max_tokens": 15}}'


 (eepitch-vterm)                                                                                                      
cd       ~/rptest/
source ~/.venv-pytorch/bin/activate
pip install pipreqs
pipreqs ./
# (find-fline "requirements.txt")

# (find-fline "Dockerfile")
(find-man "docker")
(find-man "docker build")
(find-man "docker run")
 (eepitch-vterm)
cd       ~/rptest/
docker build . -t rpdtest1
docker run -it    rpdtest1
docker image ls


# https://github.com/new
 (eepitch-vterm)
git init
git add *
git commit -m "all-of-it"
git remote add origin git@github.com:Renderedner/runpod-test1.git
git push --set-upstream origin master
"""
