"""
Created on Mon Nov 18 09:50:08 2024

@author: AnatolFiete.Naeher and Akhyar.Ahmed
"""

if __name__ == "__main__":
    import os
    from openai import OpenAI
    from config.settings import (
        path, batch_filename, prompt_template, cat_dict, filename
    )
    from ctd_processing.lists import d_CTD, age, gender, race, enrollment
    from ctd_processing.data_prep import data_prep2, merge
    from ctd_processing.batch_jobs_gpt import b_jobs as b_jobs_gpt
    from ctd_processing.batch_jobs_llama import llama_jobs as hf_llama_jobs

    # Data prep
    d_CTD['CTD_2'] = data_prep2(d_CTD, age, gender, race, enrollment)
    
    # GPT-based processing
    print("Starting GPT-based batch processing...")
    d_CTD['b_output'] = b_jobs_gpt(
        path=path,
        batch_filename=batch_filename + "_gpt",
        prompt_template=prompt_template,
        cat_dict=cat_dict,
        data=d_CTD['CTD_2'],
        client=OpenAI(api_key="sk-proj-XX")
    )
    # Merge GPT
    CTD_gpt = merge(d_CTD)
    # Save GPT
    gpt_csv = f"{filename}_gpt.csv"
    CTD_gpt.to_csv(os.path.join(path,gpt_csv), index=False)
    print(f"GPT-based clinical trial data processing completed. {gpt_csv} saved in {path}\n")

    # Llama-based processing
    print("Starting LLama-based batch processing...")
    hf_token = 'hf_XX'
    hf_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    d_CTD['b_output'] = hf_llama_jobs(
        path=path,
        batch_filename=batch_filename + "_llama",
        prompt_template=prompt_template,
        cat_dict=cat_dict,
        data=d_CTD['CTD_2'],
        hf_model_name=hf_model_name,
        hf_token=hf_token
    )

    # Merge Llama
    CTD_llama = merge(d_CTD)
    llama_csv = f"{filename}_llama.csv"
    CTD_llama.to_csv(os.path.join(path, llama_csv), index=False)
    print(f"LLama-based clinical trial data processing completed. {llama_csv} saved in {path}\n")
