import gradio as gr
import re
from tqdm import tqdm
import numpy as np
import random
import time
from pathlib import Path
from textwrap import dedent
import json
import hashlib
from joblib import Memory
import tiktoken
import openai
from openai.embeddings_utils import cosine_similarity as cos_sim

from .logger import whi, red, yel, trace, Timeout
from .shared_module import shared

# string at the end of the prompt
prompt_finish = "\n\n###\n\n"

# used to count the number of tokens for chatgpt
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokenize = tokenizer.encode

transcript_template = """
Context: 'CONTEXT'
Transcript: 'TRANSCRIPT'
""".strip()

default_system_prompt_md = {
             "role": "system",
             "content": dedent("""
             You are my excellent assistant Alfred. You always exceed my expectations. Your task today is the to transform audio transcripts into markdown formatted text.

             You follow these rules:
             - separate bullet points by a dash '-'
             - the transcript can be of poor quality, it is your job to correct transcription errors using the context.
             - If relevant, use LaTeX formatting in your answer.
             - if you're absolutely certain that you can't accomplish your task: begin your answer by 'Alfred:' and I'll take a look immediately."""),
             "timestamp": int(time.time()),
             "priority": -1,  # the only prompt that has priority of -1 is the system prompt
             }


default_system_prompt_anki = {
            "role": "system",
            "content": dedent("""You are my excellent assistant Alfred. Your task today is the to transform audio transcripts into Anki cloze flashcards. If you create several flashcards for one transcript, separate them with "#####". if you can't do it start by 'Alfred' and I'll take a look immediately."""),
            "timestamp": int(time.time()),
            "priority": -1,  # the only prompt that has priority of -1 is the system prompt
            }

if shared.backend == "anki":
    default_system_prompt = default_system_prompt_anki
    backend = "anki"
elif shared.backend == "markdown":
    default_system_prompt = default_system_prompt_md
    backend = "markdown"
else:
    raise Exception(shared.backend)

expected_mess_keys = ["role", "content", "timestamp", "priority", "tkn_len_in", "tkn_len_out", "answer", "llm_model", "tts_model", "hash"]

# embeddings using sentence transformers:
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cosine_similarity as cos_sim
# embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
# embeddings_cache = Memory(f".cache/{embedding_model_name}", verbose=0)
# embed_model = SentenceTransformer(embedding_model_name)
#
# @embeddings_cache.cache
# def embedder(text):
#     red("Computing embedding of 1 memory")
#     # remove the context before the transcript as well as the last '
#     text = text.split("Transcript: '")
#     if not len(text) == 2:
#         raise Exception(text)
#     text = text[1]
#     if not text[-1] == "'":
#         raise Exception(text)
#     text = text[:-1]
#
#     return embed_model.encode([text], show_progress_bar=False).tolist()[0]

# embeddings using ada2:
embedding_model_name = "text-embedding-ada-002"
embeddings_cache = Memory(f".cache/{embedding_model_name}", verbose=0)
@embeddings_cache.cache
def embedder(text, format):
    red("Computing embedding of 1 memory")
    # remove the context before the transcript as well as the last '
    if format == "content":
        text = text.split("Transcript: '")
        if not len(text) == 2:
            raise Exception(text)
        text = text[1]
        if not text[-1] == "'":
            raise Exception(text)
        text = text[:-1]
    elif format == "answer":
        pass
    else:
        raise ValueError(format)

    # try to bias the embedder to focus on the structure
    text = f"Pay attention to the structure of  this text: '{text}'"

    try:
        vec = openai.Embedding.create(
                model=embedding_model_name,
                input=text,
                encoding_format="float")
    except:
        time.sleep(1)
        vec = openai.Embedding.create(
                model=embedding_model_name,
                input=text,
                encoding_format="float")
    return vec["data"][0]["embedding"]

def hasher(text):
    return hashlib.sha256(text.encode()).hexdigest()[:10]

@trace
def check_prompts(prev_prompts):
    "checks validity of the previous prompts"
    whi("Checking prompt validity")
    for i, mess in enumerate(prev_prompts):

        assert mess["role"] != "system", "system message should not be here"
        if mess["role"] == "user":
            assert "answer" in mess, "no answer key in message"
            assert "{{c1::" in mess["answer"], f"No cloze found in {mess}"
            assert "}}" in mess["answer"], f"No cloze found in {mess}"
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

        mess["tkn_len_in"] = len(tokenize(dedent(mess["content"]).strip()))
        mess["tkn_len_out"] = len(tokenize(dedent(mess["answer"]).strip()))
        assert isinstance(mess["tkn_len_in"], int), "tkn_len_in is not int!"
        assert isinstance(mess["tkn_len_out"], int), "tkn_len_out is not int!"
        assert mess["tkn_len_in"] > 0, "tkn_len_in under 0 !"
        assert mess["tkn_len_out"] > 0 or mess["role"] == "system", "tkn_len_out under 0 !"
        if mess["tkn_len_in"] + mess["tkn_len_out"] > 1000:
            if mess["priority"] > 5:
                red(f"high priority prompt with more than 1000 token: '{mess}'")
            else:
                whi(f"low priority prompt with more than 1000 token: '{mess}'")

        # keep only the expected keys
        keys = [k for k in mess.keys() if k in expected_mess_keys]
        for k in prev_prompts[i]:
            if k not in keys:
                del prev_prompts[i][k]

        # make sure it's stripped
        prev_prompts[i]["content"] = dedent(prev_prompts[i]["content"]).strip()
        if "answer" in prev_prompts[i]:
            prev_prompts[i]["answer"] = dedent(prev_prompts[i]["answer"]).strip()

    return prev_prompts


#@Timeout(30)
@trace
def prompt_filter(prev_prompts, max_token, temperature, prompt_messages, keywords):
    """goes through the list of previous prompts of the profile, check
    correctness of the key/values, then returns only what's under the maximum
    number of tokens for model"""
    whi("Filtering prompts")
    if not shared.pv["txt_openai_api_key"]:
        gr.Error("No API key provided for OpenAI in the settings.")
        raise Exception("No API key provided for OpenAI in the settings.")
    openai.api_key = shared.pv["txt_openai_api_key"].strip()

    if temperature != 0:
        whi(f"Temperature is at {temperature}: making the prompt filtering non deterministic.")

    assert max_token >= 500, "max_token should be above 500"
    assert max_token <= 15500, "max_token should be under 15500"

    timesorted_pr = sorted(prev_prompts, key=lambda x: x["timestamp"], reverse=True)
    assert not any(pr["role"] == "system" for pr in prev_prompts), "Found systel prompt in prompt_filter!"

    # better formatting by removing useless markup
    for i, m in enumerate(timesorted_pr):
        assert m["role"] == "user"
        assert "Context: '" in m["content"] and "Transcript: '" in m["content"], f"Invalid prompt: {m}"
        m["content"] = m["content"].replace("Context: '", "").replace("Transcript: '", "").strip().replace("'\n", "\n")
        if m["content"][-1] == "'":
            m["content"] = m["content"][:-1]

    # count the number of tokens added so far
    new_prompt_len = sum([len(tokenize(p["content"])) for p in prompt_messages])
    tkns = 0
    tkns += len(tokenize(default_system_prompt_anki["content"]))  # system prompt
    tkns += new_prompt_len  # current question + message buffer
    all_tkns = sum([pr["tkn_len_in"] + pr["tkn_len_out"] for pr in timesorted_pr])

    # score based on priority. Closer to 1 means higher chances of being picked
    max_prio = max([pr["priority"] for pr in timesorted_pr])
    min_prio = min([pr["priority"] for pr in timesorted_pr])
    for i, pr in enumerate(timesorted_pr):
        timesorted_pr[i]["priority_score"] = (pr["priority"] - min_prio) / (max_prio - min_prio)

    # score based on keywords:
    if keywords:
        max_sim = [0, None]
        min_sim = [1, None]
        for i, pr in enumerate(timesorted_pr):
            score = sum([1 if re.search(kw, pr["content"]) else 0 for kw in keywords])
            timesorted_pr[i]["kw_score"] = score
            if score > max_sim[0]:
                max_sim[0] = score
                max_sim[1] = pr["content"]
            if score < min_sim[0]:
                min_sim[0] = score
                min_sim[1] = pr["content"]

        # scale from 0 to 1
        for i, pr in enumerate(timesorted_pr):
            timesorted_pr[i]["kw_score"] = (timesorted_pr[i]["kw_score"] - min_sim[0]) / (max_sim[0] - min_sim[0])
    else:
        for i, pr in enumerate(timesorted_pr):
            timesorted_pr[i]["kw_score"] = 1

    distances = []
    if shared.memory_metric == "length":
        for i, pr in enumerate(timesorted_pr):
            tkn_len = pr["tkn_len_in"]
            # mean of A/B and B/A is (A**2*B**2)(2AB)
            dist = ((tkn_len ** 2) + (new_prompt_len ** 2)) / (2 * tkn_len * new_prompt_len)
            # at this point: 1 means same length and anything other means
            # should not be picked.
            dist = abs(dist - 1)
            # now: 0 means same length and anything above means should not
            # be picked.
            timesorted_pr[i]["length_dist"] = dist
            distances.append(dist)

        max_dist = max(distances)
        min_dist = min(distances)

        max_sim = [0, None]
        min_sim = [1, None]
        for i, pr in enumerate(timesorted_pr):
            # make it so that the highest value is 1, lowest is 0 and
            # high means high chances of being selected
            timesorted_pr[i]["length_dist"] = 1 - ((timesorted_pr[i]["length_dist"] - min_dist) / (max_dist - min_dist))

            if timesorted_pr[i]["length_dist"] > max_sim[0]:
                max_sim[0] = timesorted_pr[i]["length_dist"]
                max_sim[1] = pr["content"]
            if timesorted_pr[i]["length_dist"] < min_sim[0]:
                min_sim[0] = timesorted_pr[i]["length_dist"]
                min_sim[1] = pr["content"]
        whi(f"Memory with lowest similarity is: {round(min_sim[0], 4)} '{min_sim[1]}'")
        whi(f"Memory with highest similarity is: {round(max_sim[0], 4)} '{max_sim[1]}'")
        assert len(timesorted_pr) == len(distances), "Unexpected list length"

    elif shared.memory_metric == "embeddings":
        whi("Computing cosine similarity")
        max_sim = [0, None]
        min_sim = [1, None]
        new_prompt_vec = embedder(prompt_messages[-1]["content"], format="content")
        for i, pr in enumerate(timesorted_pr):

            embedding = embedder(pr["content"], format="answer")
            embedding2 = embedder(pr["answer"], format="answer")
            sim = float(cos_sim(new_prompt_vec, embedding))
            sim2 = float(cos_sim(new_prompt_vec, embedding2))
            w1 = 5
            w2 = 1
            sim = (sim * 1 + sim2 * w2) / (w1 + w2)
            if sim > max_sim[0]:
                max_sim[0] = sim
                max_sim[1] = pr["content"]
            if sim < min_sim[0]:
                min_sim[0] = sim
                min_sim[1] = pr["content"]
            timesorted_pr[i]["content_sim"] = sim
            distances.append(sim)

        whi(f"Memory with lowest similarity is: {round(min_sim[0], 4)} '{min_sim[1]}'")
        whi(f"Memory with highest similarity is: {round(max_sim[0], 4)} '{max_sim[1]}'")
        assert len(timesorted_pr) == len(distances), "Unexpected list length"

        # scale the distances
        for i, pr in enumerate(timesorted_pr):
            timesorted_pr[i]["content_sim"] = (pr["content_sim"] - min_sim[0]) / (max_sim[0] - min_sim[0])

    else:
        raise ValueError(shared.memory_metric)


    # combine score
    if shared.memory_metric == "embeddings":
        score_key = "content_sim"
    elif shared.memory_metric == "length":
        score_key = "length_dist"

    w = [
            shared.pv["sld_pick_weight"],
            shared.pv["sld_prio_weight"],
            shared.pv["sld_keywords_weight"],
            ]
    pm_contents = [pr["content"].replace("<br/>", "\n") for pr in prompt_messages]
    for i, pr in enumerate(timesorted_pr):
        score = (pr[score_key] * w[0] + pr["priority_score"] * w[1] + pr["kw_score"] * w[2]) / sum(w)
        timesorted_pr[i]["pick_score"] = score
        assert score >= 0 and score <= 1, f"invalid pick_score: {score}"
        if pr["content"] in pm_contents:
            timesorted_pr[i] = None
    timesorted_pr = [pr for pr in timesorted_pr if pr]

    # add by decreasing pick score
    picksorted = sorted(timesorted_pr, key=lambda x: x["pick_score"], reverse=True)

    output_pr = []  # each picked prompt will be added here

    exit_while = False
    cnt = 0
    max_iter = 50
    while (not exit_while) and timesorted_pr:
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

            if tkns + pr["tkn_len_in"] + pr["tkn_len_out"] >= max_token:
                # will exit while at the end of this loop but not
                # before
                exit_while = True
                break

            # as the temperature incrase, increase the randomness of the picking
            if temperature <= 0.3:
                threshold = 0.05
            else:
                threshold = temperature - 0.3
            if random.random() >= pr["pick_score"] - threshold:
                continue

            # keep the most relevant previous memories in the prompt

            tkns += pr["tkn_len_in"] + pr["tkn_len_out"]
            output_pr.append(pr)

        if exit_while:
            break

    red(f"Tokens of the kept prompts: {tkns} (of all prompts: {all_tkns} tokens)")
    yel(f"Total number of prompts saved in memories: '{len(prev_prompts)}'")

    output_pr = sorted(output_pr, key=lambda x: x["pick_score"])
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


@trace
def recur_improv(txt_profile, txt_audio, txt_whisp_prompt, txt_chatgpt_outputstr, txt_context, priority, check_gpt4):
    whi("Recursively improving")
    if not txt_audio:
        red("No audio transcripts found.")
        return
    if not txt_chatgpt_outputstr:
        red("No chatgpt output string found.")
        return
    if "\n" in txt_chatgpt_outputstr:
        whi("Replaced newlines in txt_chatgpt_outputstr")
        txt_chatgpt_outputstr = txt_chatgpt_outputstr.replace("\n", "<br/>")

    content = dedent(transcript_template.replace("CONTEXT", txt_context).replace("TRANSCRIPT", txt_audio)).strip()
    answer = dedent(txt_chatgpt_outputstr.replace("\n", "<br/>")).strip()
    tkn_len_in = len(tokenize(content))
    tkn_len_out = len(tokenize(answer))
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
                "llm_model": shared.latest_llm_used,
                "tts_model": shared.latest_stt_used,
                "tkn_len_in": tkn_len_in,
                "tkn_len_out": tkn_len_out,
                "hash": hasher(content),
                }
        if to_add["hash"] in [pp["hash"] for pp in prev_prompts]:
            red("This prompt is already present in the memory!")
            return
        prev_prompts.append(to_add)

        prev_prompts = check_prompts(prev_prompts)

        with open(f"profiles/{backend}/{txt_profile}/memories.json", "w") as f:
            json.dump(prev_prompts, f, indent=4, ensure_ascii=False)
    except Exception as err:
        red(f"Error during recursive improvement: '{err}'")
        return
    whi(f"Recursively improved: {len(prev_prompts)} total examples")


@trace
def load_prev_prompts(profile):
    assert Path("profiles/").exists(), "profile directory not found"
    if Path(f"profiles/{backend}/{profile}/memories.json").exists():
        with open(f"profiles/{backend}/{profile}/memories.json", "r") as f:
            prev_prompts = json.load(f)
        prev_prompts = check_prompts(prev_prompts)
    else:
        red(f"No memories in profile {profile} found, creating it")
        prev_prompts = check_prompts([default_system_prompt.copy()])
        with open(f"profiles/{backend}/{profile}/memories.json", "w") as f:
            json.dump(prev_prompts, f, indent=4, ensure_ascii=False)

    return prev_prompts


def display_price(sld_max_tkn, check_gpt4):
    if check_gpt4:
        price = [0.01, 0.03]
    else:
        price = [0.001, 0.002]
    price_adj = price[0] * 0.9 + price[1] * 0.1
    price_per_request = price_adj * sld_max_tkn / 1000
    price_per_dol = round(1 / price_per_request, 0)
    message = f"Price if all tokens used: ${price_per_request:.5f}."
    message += f"\nRequests per $1: {price_per_dol:.1f} req"
    return message

def show_memories(profile):
    memories = load_prev_prompts(profile)
    output = [""]
    for memory in memories:
        for k, v in memory.items():
            output[-1] += f"{k.upper()}: {v}\n"
        output.append("")
    return "\n\n".join(output[:-1])
