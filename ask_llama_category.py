import os
import pdb
import json
import torch
import argparse
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def ask_llama (system_message,instruction_message, pipeline):
    messages = [{"role": "system", "content": system_message},
                {"role": "user",
                    "content": instruction_message},
                ]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    with torch.no_grad():
        outputs = pipeline(prompt, max_new_tokens=1000, eos_token_id=terminators, do_sample=True, temperature=0.2,
                            top_p=0.9)

    return outputs

def make_question(finding, categories):
        question=f'Return the category of the following finding: '
        system_message = f"You are an expert trained on neurology and clinical domain! For the given finding, which category describes it best? Chose only one from the below: {' / '.join(categories)} "
        instruction_message = f'{question} {finding}'
        answer_tmp = ask_llama(system_message,instruction_message, pipel)
        return answer_tmp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distance analysis")

    parser.add_argument(
        "-input_dir",
        type=str,
        action="store",
        help="path to df with all findings",
    )
    
    parser.add_argument(
        "-output_dir",
        action="store",
        default=False,
        help="Directory to store results in",
    )

    parser.add_argument(
        "-model_dir",
        action="store",
        default=False,
        help="Directory to with models",
    )

    parser.add_argument(
        "-reverse",
        action="store",
        default=False,
        help="0 or 1 Start at the End of the Dataset to better parallelize",
    )

    # set up device
    device = "cpu" 
    if torch.cuda.is_available():
        device = "cuda" 
    # if torch.backends.mps.is_available():
    #     device = "mps"

    print(f'USING DEVICE {device}')

    # load data
    args = parser.parse_args()
    reverse = args.reverse
    input_dir = args.input_dir
    OUT_DIR = args.output_dir
    model_dir = args.model_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    master = pd.read_csv(input_dir + 'all_findings_CLEANED.csv')
    
    # load llama 
    model_name = 'ContactDoctor/Bio-Medical-Llama-3-8B'
    tokenizer = AutoTokenizer.from_pretrained(f'{model_dir}/tokenizers/{model_name}')
    model = AutoModelForCausalLM.from_pretrained(f'{model_dir}/models/{model_name}',device_map=device)
    pipel = pipeline("text-generation", model=model,tokenizer=tokenizer, model_kwargs={"torch_dtype": torch.bfloat16})
    # set model to eval mode
    model.eval()

    # load categories:
    cat = pd.read_csv(input_dir + 'categories.csv')
    level1 = np.unique(cat['Level1'])
    level2_all = np.unique(cat['Level2'])

    # start at the end if reverese = 1
    if int(reverse):
        master = master.iloc[::-1]

    for i, row in master.iterrows():
        idx = i
        finding = row['For_LLAMA2']
        answers = dict()       

        out_file = f'{OUT_DIR}finding_{idx}.json'
        
        if not os.path.isfile(out_file):
            # only compute if finding is not null
            if pd.isnull(finding):
                answers['Nullfinging'] = np.nan
                answers['Level1'] = np.nan 
                answers['Level2'] = np.nan 
                answers['Level3'] = np.nan 
                answers['Level2_free'] = np.nan 

            else:
                # NULLFINDING        
                question=f'Is the following finding a Nullfinding, meaning that it tells what was not found? Answer Yes or No: '
                system_message = f"You are an expert trained on neurology and clinical domain! Only answer Yes or No."
                instruction_message = f'{question} {finding}'
                answer_tmp = ask_llama(system_message,instruction_message, pipel)    
                answer_short = answer_tmp[0]['generated_text'].split("<|eot_id|>\n\nAssistant: ")[-1]
                answers['Nullfinging'] = answer_short.strip() 

                # LEVEL 1: 
                answer_tmp = make_question(finding, level1)
                answer_short = answer_tmp[0]['generated_text'].split("<|eot_id|>\n\nAssistant: ")[-1]
                answer_l1 = answer_short.strip() 
                answers['Level1'] = answer_l1
                print(f'Finished question {finding}. Response: {answer_short}')
                
                # LEVEL 2 free: 
                # find category in longer list of subcategories 
                answer_tmp = make_question(finding, level2_all)
                answer_short = answer_tmp[0]['generated_text'].split("<|eot_id|>\n\nAssistant: ")[-1]
                answers['Level2_free'] = answer_short.strip() 
                print(f'Finished question {finding}. Response: {answer_short}')

                # find match to Level 1 (see whether subcategories work better)
                match = cat['Level1'].str.strip() == answer_l1

                if sum(match) > 0:
                    cat_sel = cat[match]
                    level2 = np.unique(cat_sel['Level2'])

                    answer_tmp = make_question(finding, level2)
                    answer_short = answer_tmp[0]['generated_text'].split("<|eot_id|>\n\nAssistant: ")[-1]
                    answers['Level2'] = answer_short.strip() 
                    
                    match = cat['Level2'].str.strip() == answer_short.strip() 
                    if sum(match) > 0:
                        cat_sel = cat[match]
                        level3 = cat_sel['Level3'].iloc[0]
            
                        # in case there is a third level:
                        if not pd.isnull(level3):
                            question=f'Return the category of the following finding: '
                            system_message = f"You are an expert trained on neurology and clinical domain! For the given finding, which category describes it best? Chose only one from the below: {level3.replace(',',' / ')} "
                            instruction_message = f'{question} {finding}'
                            answer_tmp = ask_llama(system_message,instruction_message, pipel)    
                            answer_short = answer_tmp[0]['generated_text'].split("<|eot_id|>\n\nAssistant: ")[-1]
                            answers['Level3'] = answer_short.strip()

            # save participant_dict with scan and brain info
            with open(out_file, "w") as out: 
                json.dump(answers, out)

print('Done :)')
