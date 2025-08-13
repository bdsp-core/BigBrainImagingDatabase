import os
import json
import torch
import tarfile
import argparse
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def clean_json_string(json_str):
    # Try to load the string into a Python object
    try:
        # Attempt to load and parse the string, if valid JSON, return it
        data = json.loads(json_str)
        return data
    
    except json.JSONDecodeError:
        try:
            # If an error occurs, the string is likely incomplete, Remove the part after the last complete JSON object or array
            last_complete_json_end = json_str.rfind('}')
            if last_complete_json_end != -1:
                fixed_str = json_str[:last_complete_json_end + 1]
                return json.loads(fixed_str)
        except:
            # erroreous output
            new_str = '{"Name": "INCOMPLETE"}'
            return json.loads(new_str)


def ask_llama (system_message,instruction_message, pipeline):
    messages = [{"role": "system", "content": system_message},
                {"role": "user",
                    "content": instruction_message},
                ]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    with torch.no_grad():
        outputs = pipeline(prompt, max_new_tokens=2000, eos_token_id=terminators, do_sample=True, temperature=0.5,
                            top_p=0.9)

    return outputs

def read_from_tar(tar_path, file_name):
    with tarfile.open(tar_path, 'r') as tar:
        # Check if the file exists in the tar archive
        if file_name in tar.getnames():
            file = tar.extractfile(file_name)
            file_content = file.read()
            file_content = file_content.decode('utf-8')
            file_content = eval(file_content) # to dict
        else:
            print(f"File '{file_name}' not found in the tar archive.")
            file_content = None
    return file_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distance analysis")

    parser.add_argument(
        "-tar_data_dir",
        type=str,
        action="store",
        help="path to tar directory with data",
    )

    parser.add_argument(
        "-master_dir",
        action="store",
        default=False,
        help="Directory to csv with all patients",
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

    parser.add_argument(
        "-split",
        action="store",
        default=False,
        help="between 0 to 80000, start index to calculate ten thousand samples.",
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
    master = args.master_dir
    reverse = args.reverse
    REPORT_DIR = args.tar_data_dir
    OUT_DIR = args.output_dir
    model_dir = args.model_dir
    split = int(args.split)
    os.makedirs(OUT_DIR, exist_ok=True)
    master = pd.read_csv(master)
    
    # load llama 
    model_name = 'ContactDoctor/Bio-Medical-Llama-3-8B'
    tokenizer = AutoTokenizer.from_pretrained(f'{model_dir}/tokenizers/{model_name}')
    model = AutoModelForCausalLM.from_pretrained(f'{model_dir}/models/{model_name}',device_map=device)
    pipel = pipeline("text-generation", model=model,tokenizer=tokenizer, model_kwargs={"torch_dtype": torch.bfloat16})
    # set model to eval mode
    model.eval()

    # if MGB 
    if 'MGB' in REPORT_DIR:
        master['aws_file'] = master['aws_filename'].str.replace('/','_')

    if split:
        selection = master['aws_file'][split:split+10000]
    else:
        selection = master['aws_file']

    # start at the end if reverese = 1
    if int(reverse):
        selection = selection.iloc[::-1]

    for id_stan in selection:
        id_tmp = id_stan.split('.')[0]
        # fname = f"{REPORT_DIR}/Report_ID_{id_stan[:-4]}.json"
        out_file1 = f'{OUT_DIR}/LLAMA_out_{id_tmp}_scaninfo.json'
        out_file2 = f'{OUT_DIR}/LLAMA_out_{id_tmp}_findings.json'

        folder_name = REPORT_DIR.split('/')[-1][:-7]
        
        report = f'{folder_name}/Report_ID_{id_tmp}.json'

        # if one of the output does not exist yet:
        if not os.path.isfile(out_file1) or not os.path.isfile(out_file2):
            data_id = read_from_tar(REPORT_DIR, report)
            
            # if data_id not found in data
            if not data_id:
                data_id = 0  # do not process this one.
                # ID NOT FOUND
                with open(out_file1, "w") as out: 
                    json.dump("REPORT NOT FOUND", out)
                with open(out_file2, "w") as out: 
                    json.dump("REPORT NOT FOUND", out)
            
            # if data_id found in tar
            elif data_id:
                answers = dict()            
                # set input for model
                question = dict()
                question['Brain']="Is this report about the brain? Answer 'Yes', 'No' or 'Unknown'."
                question['Abnormality']="Is this a pathological report? Answer 'Yes', 'No' or 'Unknown'."

                question['ALL-free']= "Make a list of all mentioned pathologies in the report, start with the most clinically important one. Do not repeat pathologies. Only include pathologies which are explicitly mentioned in the report. Return a list of pathologies in JSON format. Each finding should have a 'Name', 'General Name', 'Location', 'Brain-related', 'Magnitude','Acute/Chronic' and a 'Details' field."
                
                system_message = f"You are an expert trained on neurology and clinical domain. Do only report information which is explicitely metnioed in the report."
                report = data_id['report']

                answers['report'] = report

                for qu in ['Brain','Abnormality']:
                    instruction_message = f'{report} {question[qu]}'
                    answer_tmp = ask_llama(system_message,instruction_message, pipel)
                    answer_short = answer_tmp[0]['generated_text'].split("<|eot_id|>\n\nAssistant: ")[-1]
                    answers[qu] = answer_short
                    print(f'Finished question {qu}. Response: {answer_short}')
                
                # save participant_dict with scan and brain info
                with open(out_file1, "w") as out: 
                    json.dump(answers, out)


                # now ask for the actual findings in the report

                for qu in ['ALL-free']:
                    instruction_message = f'{report} {question[qu]}'
                    answer_tmp = ask_llama(system_message,instruction_message, pipel)
                    answer_short = answer_tmp[0]['generated_text'].split("<|eot_id|>\n\nAssistant: ")[-1]
                    findings = answer_short
                    print(f'Finished question {qu}. Response: {findings}')
                                    
                    findings_list = clean_json_string(findings)

                    new_findings = []
                    if isinstance(findings_list, list):
                        if len(findings_list)>=1:
                            # Second round of LLAMA to self-correct
                            for f_tmp in findings_list: 
                                try:
                                    # ask it to self-correct 
                                    instruction_message = f"{report} Is {f_tmp['General Name']} explicitly mentioned in the report? Answer Yes or No"
                                    answer_tmp = ask_llama(system_message,instruction_message, pipel)
                                    answer_short = answer_tmp[0]['generated_text'].split("<|eot_id|>\n\nAssistant: ")[-1]
                                    f_tmp['Test'] = answer_short
                                except:
                                    f_tmp['Test'] = 'UNKNOWN'
                            
                                try:
                                    # remove names and dates from the term
                                    instruction_message = f"Does the following text contain names or dates? Answer Yes or No: {f_tmp['Details']} "
                                    answer_tmp = ask_llama(system_message,instruction_message, pipel)
                                    answer_short = answer_tmp[0]['generated_text'].split("<|eot_id|>\n\nAssistant: ")[-1]
                                    f_tmp['Details_did'] = answer_short                        
                                except:
                                    f_tmp['Details_did'] = 'UNKNOWN'

                                new_findings.append(f_tmp)

                # save participants finding with scan and brain info
                with open(out_file2, "w") as out: 
                    json.dump(new_findings, out)

    print('Done :) ')