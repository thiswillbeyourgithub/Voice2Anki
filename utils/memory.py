from typing import List, Optional, Union
from functools import partial
from tqdm import tqdm
import pandas as pd
import gradio as gr
import re
import numpy as np
import time
from pathlib import Path
from textwrap import dedent, indent
import json
import hashlib
from joblib import Memory
import tiktoken
import litellm
from sklearn.metrics.pairwise import cosine_similarity
from iterator_cacher import IteratorCacher

from .logger import whi, red, yel, trace, Timeout, smartcache, cache_dir
from .shared_module import shared
from .typechecker import optional_typecheck

# string at the end of the prompt
prompt_finish = "\n\n###\n\n"

# used to count the number of tokens for chatgpt
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
@optional_typecheck
def tkn_len(message: str) -> int:
    return len(tokenizer.encode(dedent(message)))

transcript_template = """
CONTEXT
TRANSCRIPT
""".strip()

litellm.set_verbose = shared.debug

default_system_prompt = {
            "role": "system",
            "content": """
You are my excellent assistant Alfred. Your task today is the to transform audio transcripts into Anki cloze flashcards. If you follow closely my instructions and respect the formatting, I'll give you $200!

Rules you follow:
'''
- If you have to create several flashcards from one transcript, separate them with "#####".
- I will give you examples of good flashcards, you have to match their format and structure. This is critical.
- In your flashcard, the answer must repeat the question (i.e. if the question starts by 'The types of cancer that' then the answer should start also by 'The types of cancer that'). This usually make it easier to memorize for me but use common sense and above all: take inspiration from the examples.
- If you can't accomplish the task, start your reply by 'Alfred: [YOUR ISSUE]' followed by your issue and I'll help you right away.
- If the examples contain acronyms that are relevant to the transcript, feel free to reuse them directly.
'''
""".strip(),
            "timestamp": int(time.time()),
            "priority": -1,  # the only prompt that has priority of -1 is the system prompt
            }


expected_mess_keys = ["role", "content", "timestamp", "priority", "tkn_len_in", "tkn_len_out", "answer", "llm_model", "stt_model", "hash", "llm_choice"]

# init the embedding caches here to avoid them needed recomputation each time python is launched
embedding_caches = {
    m: Memory(cache_dir / "embeddings_iterator_cacher" / m, verbose=False)
    for m in shared.embedding_models
}


@optional_typecheck
def hasher(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]

@trace
@optional_typecheck
def embedder(
    text_list: List[str],
    model: str,
    verbose: bool = False,
    ) -> List[np.ndarray]:
    """compute the embedding of a text list 1 by 1 thanks to iteratorcacher
    if result is not None, it is the embedding and returned right away. This
    was done to allow caching individual embeddings while still making one batch
    call to the embedder.
    """
    assert text_list
    assert all(t.strip() for t in text_list)
    cache_obj = embedding_caches[model]
    cached = IteratorCacher(
        memory_object=cache_obj,
        iter_list=["input"],
        verbose=shared.debug,
        res_to_list = lambda out: out.to_dict()["data"],
        batch_size=1500,
        debug=verbose,
    )(litellm.embedding)
    vec: List[dict] = cached(model=model, input=text_list)
    vec = [
        np.array(v["embedding"]).squeeze()
        if isinstance(v, dict)
        else (
            np.array(v.to_dict()["data"][0]["embedding"]).squeeze() if len(v.to_dict()["data"]) == 1 else None
        )
        for v in vec
    ]
    nonvec = [v for v in vec if v is None]
    if nonvec:
        pos = [iv for iv, v in enumerate(vec) if v is None]
        raise Exception(f"Found {len(nonvec)} None in vectors at position {pos} from length {len(vec)}")

    tkn_sum = sum([tkn_len(t) for t in text_list])
    red(f"Computing embedding of {len(text_list)} texts for a total of {tkn_sum} tokens")
    return vec


@optional_typecheck
@trace
def check_prompts(prev_prompts: List[dict]) -> List[dict]:
    "checks validity of the previous prompts"
    whi("Checking prompt validity")
    for i, mess in enumerate(prev_prompts):

        assert mess["role"] != "system", "system message should not be here"
        if mess["role"] == "user":
            assert "answer" in mess, "no answer key in message"
        else:
            raise ValueError("role of previous prompt is not user or system")

        assert "CONTEXT" not in mess["content"], "CONTEXT string found in message!"
        assert "TRANSCRIPT" not in mess["content"], "TRANSCRIPT string found in message!"

        if "timestamp" not in mess:
            mess["timestamp"] = i  # if no timestamp, at least keep the order
        assert isinstance(mess["timestamp"], int), "timestamp is not int!"

        if "priority" not in mess:
            mess["priority"] = 0
            if "importance" in mess:  # backward compatibility
                mess["priority"] = mess["importance"]
        if isinstance(mess["priority"], float):
            mess["priority"] = int(mess["priority"])
        assert isinstance(mess["priority"], int), f"priority is not int! '{mess['priority']}'"
        assert mess["priority"] <= 10, "priority above 10 !"
        assert mess["priority"] >= -1, "priority under -1 !"

        if "hash" not in mess:
            mess["hash"] = hasher(mess["content"])

        if "tkn_len_in" not in mess or "tkn_len_out" not in mess:
            mess["tkn_len_in"] = tkn_len(mess["content"].strip())
            mess["tkn_len_out"] = tkn_len(mess["answer"].strip())
        assert isinstance(mess["tkn_len_in"], int), "tkn_len_in is not int!"
        assert isinstance(mess["tkn_len_out"], int), "tkn_len_out is not int!"
        assert mess["tkn_len_in"] > 0, "tkn_len_in under 0 !"
        assert mess["tkn_len_out"] > 0 or mess["role"] == "system", "tkn_len_out under 0 !"
        # if mess["tkn_len_in"] + mess["tkn_len_out"] > 1000:
        #     if mess["priority"] > 5:
        #         red(f"high priority prompt with more than 1000 token: '{mess}'")
        #     else:
        #         whi(f"low priority prompt with more than 1000 token: '{mess}'")

        # keep only the expected keys
        keys = [k for k in mess.keys() if k in expected_mess_keys]
        to_del = []
        for k in prev_prompts[i]:
            if k not in keys:
                to_del.append(k)
        for dele in to_del:
            red(f"Removed unexpected key from prompt: {dele}")
            del prev_prompts[i][dele]

        # make sure it's stripped
        prev_prompts[i]["content"] = dedent(prev_prompts[i]["content"]).strip()
        if "answer" in prev_prompts[i]:
            prev_prompts[i]["answer"] = dedent(prev_prompts[i]["answer"]).strip()

    return prev_prompts


#@Timeout(30)
@optional_typecheck
@trace
def prompt_filter(
    prev_prompts: List[dict],
    max_token: Union[int, float],
    prompt_messages: List[dict],
    keywords: Optional[List[re.Pattern]],
    ):
    """goes through the list of previous prompts of the profile, check
    correctness of the key/values, then returns only what's under the maximum
    number of tokens for model"""
    whi("Filtering prompts")
    if "mistral" in shared.pv["choice_embed"] and not shared.pv["txt_mistral_api_key"]:
        raise Exception("You want to use Mistral for embeddings but haven't supplied an API key in the settings.")
    elif "openai" in shared.pv["choice_embed"] and not shared.pv["txt_openai_api_key"]:
        raise Exception("You want to use OpenAI for embeddings but haven't supplied an API key in the settings.")

    assert max_token >= 500, "max_token should be above 500"
    assert max_token <= 15500, "max_token should be under 15500"

    candidate_prompts = sorted(prev_prompts, key=lambda x: x["timestamp"], reverse=True)
    assert not any(pr["role"] == "system" for pr in prev_prompts), "Found systel prompt in prompt_filter!"

    # better formatting by removing useless markup
    for i, m in enumerate(candidate_prompts):
        assert m["role"] == "user"
        assert "Context: '" not in m["content"] and "Transcript: '" not in m["content"], f"Invalid prompt: {m}"

    # count the number of tokens added so far
    new_prompt_len = sum([tkn_len(p["content"]) for p in prompt_messages])
    tkns = 0
    tkns += tkn_len(default_system_prompt["content"])  # system prompt
    tkns += new_prompt_len  # current question + message buffer
    all_tkns = sum([pr["tkn_len_in"] + pr["tkn_len_out"] for pr in candidate_prompts])

    # score based on priority. Closer to 1 means higher chances of being picked
    max_prio = max([pr["priority"] for pr in candidate_prompts])
    min_prio = min([pr["priority"] for pr in candidate_prompts])
    for i, pr in enumerate(candidate_prompts):
        candidate_prompts[i]["priority_score"] = (pr["priority"] - min_prio) / (max_prio - min_prio)

    # score based on timestamp. Closer to 1 means more recent so higher chances of being picked
    times = sorted([pr["timestamp"] for pr in candidate_prompts])
    for i, pr in enumerate(candidate_prompts):
        candidate_prompts[i]["time_score"] = times.index(pr["timestamp"]) / len(times)

    # score based on keywords:
    if keywords:
        max_sim = [0, None]
        min_sim = [1, None]
        for i, pr in enumerate(candidate_prompts):
            score = sum([1 if re.search(kw, pr["content"]) else 0 for kw in keywords])
            candidate_prompts[i]["kw_score"] = score
            if score > max_sim[0]:
                max_sim[0] = score
                max_sim[1] = pr["content"]
            if score < min_sim[0]:
                min_sim[0] = score
                min_sim[1] = pr["content"]

        # scale from 0 to 1
        for i, pr in enumerate(candidate_prompts):
            candidate_prompts[i]["kw_score"] = (candidate_prompts[i]["kw_score"] - min_sim[0]) / (max_sim[0] - min_sim[0])
    else:
        for i, pr in enumerate(candidate_prompts):
            candidate_prompts[i]["kw_score"] = 1

    # score based on embedding similarity
    whi("Computing cosine similarity")
    max_sim = [0, None]
    min_sim = [1, None]

    # get the embedding for all memories in another way (because this way
    # we get all embeddings from the memories in a single call, provided the
    # memories.json file hasn't changed since)
    to_embed = [prompt_messages[-1]["content"]]
    to_embed += [pr["content"] for pr in candidate_prompts]
    to_embed += [pr["answer"] for pr in candidate_prompts]

    all_embeddings = embedder(text_list=to_embed, model=shared.pv["choice_embed"])
    assert all(isinstance(item, np.ndarray) for item in all_embeddings)
    assert len(all_embeddings) == 2 * len(candidate_prompts) + 1
    new_prompt_vec = all_embeddings.pop(0).squeeze().reshape(1, -1)
    embeddings_contents = all_embeddings[:len(candidate_prompts)]
    embeddings_answers = all_embeddings[len(candidate_prompts):]
    assert len(embeddings_contents) == len(embeddings_answers)
    sim_content = cosine_similarity(new_prompt_vec, np.array(embeddings_contents).squeeze())
    sim_answer = cosine_similarity(new_prompt_vec, np.array(embeddings_answers).squeeze())
    w1, w2 = 5, 1
    sim_combined = ((sim_content * w1 + sim_answer * w2) / (w1 + w2)).squeeze()

    max_sim = [sim_combined.max(), candidate_prompts[sim_combined.argmax()]["content"]]
    min_sim = [sim_combined.min(), candidate_prompts[sim_combined.argmin()]["content"]]
    whi(f"Memory with lowest similarity is: {round(min_sim[0], 4)} '{min_sim[1]}'")
    whi(f"Memory with highest similarity is: {round(max_sim[0], 4)} '{max_sim[1]}'")

    # scaling
    sim_combined -= sim_combined.min()
    sim_combined /= sim_combined.max()
    for i in range(len(candidate_prompts)):
        candidate_prompts[i]["content_sim"] = float(sim_combined[i].squeeze())
    assert len(candidate_prompts) == len(list(sim_combined)), "Unexpected list length"

    # combine score

    w = [
            shared.pv["sld_pick_weight"],
            shared.pv["sld_prio_weight"],
            shared.pv["sld_keywords_weight"],
            shared.pv["sld_timestamp_weight"],
            ]
    pm_contents = [pr["content"].replace("<br/>", "\n") for pr in prompt_messages]
    for i, pr in enumerate(candidate_prompts):
        score = pr["content_sim"] * w[0]
        score += pr["priority_score"] * w[1]
        score += pr["kw_score"] * w[2]
        score += pr["time_score"] * w[3]
        score /= sum(w)
        candidate_prompts[i]["pick_score"] = score
        assert score >= 0 and score <= 1, f"invalid pick_score: {score}"
        if pr["content"] in pm_contents:
            candidate_prompts[i] = None
    candidate_prompts = [pr for pr in candidate_prompts if pr]

    # add by decreasing pick score
    picksorted = sorted(candidate_prompts, key=lambda x: x["pick_score"], reverse=True)

    output_pr = []  # each picked prompt will be added here

    exit_while = False
    cnt = 0
    max_iter = 50
    while (not exit_while) and candidate_prompts:
        cnt += 1
        if cnt >= max_iter:
            red(f"Exited filtering loop after {cnt} iterations, have you added enough memories?")
            exit_while = True
            break

        for pr_idx, pr in enumerate(picksorted):
            if pr in output_pr:
                continue
            if pr in prompt_messages:
                continue
            if pr["content"] in pm_contents:
                raise Exception("This shouldn't happen: memorized prompt present in buffer")

            if tkns + pr["tkn_len_in"] + pr["tkn_len_out"] >= max_token:
                # will exit while at the end of this loop but not
                # before
                exit_while = True
                break

            # don't add super similar memories otherwise we lack diversity
            dists = [abs(pr["content_sim"] - p["content_sim"]) for p in output_pr]
            if dists and min(dists) <= 0.0001:
                cont1 = output_pr[dists.index(min(dists))]["content"]
                cont2 = pr["content"]
                red(f"Very similar prompts {min(dists)}:\n* {cont2}\n* {cont1}")
                continue

            # keep the most relevant previous memories in the prompt

            tkns += pr["tkn_len_in"] + pr["tkn_len_out"]
            output_pr.append(pr)

        if exit_while:
            break

    red(f"Tokens of the kept prompts after {cnt} iterations: {tkns} (of all prompts: {all_tkns} tokens)")
    yel(f"Total number of prompts saved in memories: '{len(prev_prompts)}'")

    output_pr = sorted(output_pr, key=lambda x: x["pick_score"])
    assert output_pr[1]["pick_score"] <= output_pr[-1]["pick_score"]
    # or by timestamp (most recent last):
    # output_pr = sorted(output_pr, key=lambda x: x["timestamp"])
    # or by priority:
    # output_pr = sorted(output_pr, key=lambda x: x["priority"])

    # check no duplicate prompts
    contents = [pm["content"] for pm in output_pr]
    dupli = [dp for dp in contents if contents.count(dp) > 1]
    if dupli:
        raise Exception(f"{len(dupli)} duplicate prompts found in memory.py: {dupli}")

    return output_pr


@optional_typecheck
@trace
def recur_improv(txt_profile: str, txt_audio: str, txt_whisp_prompt: str, txt_chatgpt_outputstr: str, txt_context: str, priority: int, llm_choice: str):
    whi("Recursively improving")
    if not txt_audio:
        raise Exception(red("No audio transcripts found."))
        return
    if not txt_chatgpt_outputstr:
        raise Exception(red("No chatgpt output string found."))
        return
    txt_audio = txt_audio.strip()
    if "\n" in txt_chatgpt_outputstr:
        whi("Replaced newlines in txt_chatgpt_outputstr")
        txt_chatgpt_outputstr = txt_chatgpt_outputstr.replace("\n", "<br/>")
    if "#####" in txt_audio or "\n\n" in txt_audio:
        raise Exception(red("You can't memorize a prompt that was automatically split."))
        return

    content = dedent(transcript_template.replace("CONTEXT", txt_context).replace("TRANSCRIPT", txt_audio)).strip()
    answer = dedent(txt_chatgpt_outputstr.replace("\n", "<br/>")).strip()
    tkn_len_in = tkn_len(content)
    tkn_len_out = tkn_len(answer)
    if tkn_len_in + tkn_len_out > 500:
        red("You supplied an example "
            f"with a surprising amount of token: '{tkn_len_in + tkn_len_out}' This can have "
            "adverse effects.")

    prev_prompts = load_prev_prompts(txt_profile)
    try:
        to_add = {
                "role": "user",
                "content": content,
                "timestamp": int(time.time()),
                "priority": priority,
                "answer": answer,
                "llm_choice": llm_choice,  # can be different than llm_model because of cache
                "llm_model": shared.pv["llm_choice"],
                "stt_model": shared.pv["stt_choice"],
                "tkn_len_in": tkn_len_in,
                "tkn_len_out": tkn_len_out,
                "hash": hasher(content),
                }
        if to_add["hash"] in [pp["hash"] for pp in prev_prompts]:
            raise Exception(red("Prompt already in the memory.json!"))
        prev_prompts.append(to_add)

        prev_prompts = check_prompts(prev_prompts)

        if "{{c1::" not in txt_chatgpt_outputstr and "}}" not in txt_chatgpt_outputstr:
            gr.Warning(red(f"No cloze found in new memory. Make sure it's on purpose.\nCard: {txt_chatgpt_outputstr}"))

        with open(f"profiles/{txt_profile}/memories.json", "w") as f:
            json.dump(prev_prompts, f, indent=4, ensure_ascii=False)
    except Exception as err:
        raise Exception(red(f"Error during recursive improvement: '{err}'"))
        return
    gr.Warning(whi(f"Recursively improved: {len(prev_prompts)} total examples"))


@optional_typecheck
@trace
def load_prev_prompts(profile: str) -> List[dict]:
    assert Path("profiles/").exists(), "profile directory not found"
    if Path(f"profiles/{profile}/memories.json").exists():
        with open(f"profiles/{profile}/memories.json", "r") as f:
            prev_prompts = json.load(f)
        prev_prompts = check_prompts(prev_prompts)
    else:
        red(f"No memories in profile {profile} found, creating it")
        prev_prompts = check_prompts([default_system_prompt.copy()])
        with open(f"profiles/{profile}/memories.json", "w") as f:
            json.dump(prev_prompts, f, indent=4, ensure_ascii=False)

    return prev_prompts


@optional_typecheck
def display_price(sld_max_tkn: int, llm_choice: str) -> str:
    if llm_choice not in shared.llm_price:
        return f"Price not found for model '{llm_choice}'"
    price = shared.llm_price[llm_choice]
    if isinstance(price, float):
        return f"${price} per second (actual price computation is probably wrong for now!)"
    price_adj = price["input_cost_per_token"] * 0.9 + price["output_cost_per_token"] * 0.1
    price_per_request = price_adj * sld_max_tkn
    price_per_dol = round(1 / price_per_request, 0)
    message = f"Price if all tokens used: ${price_per_request:.5f}."
    message += f"\nRequests per $1: {price_per_dol:.1f} req"
    return message

@optional_typecheck
@trace
def get_memories_df(profile: str) -> pd.DataFrame:
    memories = load_prev_prompts(profile)
    if not memories:
        gr.Warning(red(f"No memories found for profile {profile}"))
        return pd.DataFrame()
    for i in range(len(memories)):
        memories[i]["n"] = i + 1
        del memories[i]["role"]
    return pd.DataFrame(memories).reset_index().set_index("n")

@optional_typecheck
@trace
def get_message_buffer_df() -> pd.DataFrame:
    buffer = shared.message_buffer
    if not buffer:
        gr.Warning(red("No message buffer found"))
        return pd.DataFrame()
    for i in range(len(buffer)):
        buffer[i]["n"] = i + 1
    return pd.DataFrame(buffer).reset_index().set_index("n")

@optional_typecheck
@trace
def get_dirload_df() -> pd.DataFrame:
    df = shared.dirload_queue
    # make sure that the index 'n' appears first
    df = df.reset_index().set_index("n").reset_index()
    return df
