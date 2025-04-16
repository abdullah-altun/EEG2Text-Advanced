import os

from tqdm import tqdm
import pandas as pd

import instructor
from pydantic import BaseModel
from openai import OpenAI
from langchain.prompts import PromptTemplate

from concurrent.futures import ThreadPoolExecutor, as_completed

class Evaluation(BaseModel):
    fluency: int
    adequacy: int

api_key = os.getenv("OPENAI_API_KEY")
client = instructor.from_openai(OpenAI(api_key = api_key))

prompt = """ You are a helpful language evaluator who can evaluate
input sentence2 and provide an evaluation of its fluency with a
likert scale rating of 1-5, 5 being highly fluent.
You will also have to compate two sentences and judge how adequate
is input sentence 2 with respect to input sentence 1, again with a likert scale rating of 1-5, 5 being highly adequate.
Here are the sentences: input_sentence1: {input_sentence1}, input_sentence2:{input_sentence2}"""

template = PromptTemplate(
    input_variables=["input_sentence1", "input_sentence2"],
    template=prompt,
)

def eval_single(row):
    i, r = row
    input_sentence1, input_sentence2 = r["Expected Caption"], r["Generated Caption"]
    final_prompt = template.format(input_sentence1=input_sentence1, input_sentence2=input_sentence2)
    eval_info = client.chat.completions.create(
        model="gpt-4o",
        response_model=Evaluation,
        messages=[{"role": "user", "content": final_prompt}],
    )
    return eval_info.model_dump()

def get_results_for_single_file(results_dir, file):
    file_path = os.path.join(results_dir,file)
    df = pd.read_csv(file_path)
    num_threads = 8

    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(eval_single, item) for item in df.iterrows()]
        for future in tqdm(as_completed(futures)):
            results.append(future.result())

    df["eval"] = pd.Series(results)
    df.to_csv(file.replace("csv","eval.csv"))

    average_adequacy = df["eval"].apply(lambda x: x["adequacy"]).mean()
    average_fluency = df["eval"].apply(lambda x: x["fluency"]).mean()
    return {"Avg. Fluency":average_fluency, "Avg. Adequacy": average_adequacy}


results_dir = "results"
all_res = {}
for file in os.listdir(results_dir):
    results = get_results_for_single_file(results_dir, file)
    print(results)
    all_res[file.replace("csv","")] = results

results_df = pd.DataFrame(all_res).transpose()
results_df.to_csv("all_gpt4_results.csv")