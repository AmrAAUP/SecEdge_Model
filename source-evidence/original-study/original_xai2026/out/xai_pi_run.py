
import json, os, sys, time
import numpy as np
from llama_cpp import Llama, LlamaGrammar
B = json.load(open(os.path.expanduser("~/Desktop/pi5_h3_measure/xai_pi_bundle.json")))
OUT = os.path.expanduser("~/Desktop/pi5_h3_measure/out_pi/xai_pi_results.jsonl")
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["row"])
        except Exception: pass
llm = Llama(model_path=os.path.expanduser(B["model"]), n_ctx=2048, n_threads=4, n_batch=512,
            logits_all=True, verbose=False)   # scores[] is only filled when logits_all=True
gram = LlamaGrammar.from_string(B["grammar"], verbose=False)
pre = B["prefix_ids"]; nodes = B["branch_nodes"]; kids = B["children"]
fh = open(OUT, "a")
def softmax(z):
    z = np.asarray(z, dtype=np.float64); z = z - z.max(); e = np.exp(z); return e / e.sum()
for n, r in enumerate(B["flows"], 1):
    if r["row"] in done: continue
    llm.reset()                      # fresh evaluation: order-independent numerics
    t0 = time.time()
    o = llm.create_completion(prompt=r["ids"], max_tokens=48, temperature=0.0, grammar=gram,
                              stop=["<|end|>"])
    t_dec = time.time() - t0
    text = o["choices"][0]["text"]
    t1 = time.time()
    node_probs = {}
    for node in nodes:
        llm.n_tokens = len(r["ids"])  # rewind to the prompt; KV beyond is discarded by eval
        llm.eval(pre + node)
        p = softmax(llm.scores[llm.n_tokens - 1])
        node_probs[json.dumps(node)] = {str(k): float(p[k]) for k in kids[json.dumps(node)]}
    t_trie = time.time() - t1
    fh.write(json.dumps({"row": r["row"], "text": text, "node_probs": node_probs,
                         "t_decide": round(t_dec, 3), "t_trie": round(t_trie, 3)}) + "\n")
    fh.flush()
    print(f"{n}/{len(B['flows'])} dec {t_dec:.1f}s trie {t_trie:.1f}s  {text[:50]}", flush=True)
print("DONE", flush=True)
