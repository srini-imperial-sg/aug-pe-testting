import torch
import numpy as np
from tqdm import tqdm
import logging
from .api import API
from apis.biomed import get_prompt, ALL_MIMIC_TONES
import transformers
import random
from .utils import set_seed, get_subcategories, ALL_styles, ALL_OPENREVIEW_styles, ALL_PUBMED_styles, ALL_FINANCIAL_styles, ALL_FINANCIAL_luckycat_styles, ALL_ASYLEX_styles, ALL_AAAI_styles
import re
import collections
from huggingface_hub import InferenceClient

class HFAPI(API):
    def __init__(self,
                 model_type, variation_type, use_subcategory,
                 output_dir, seed, mlm_probability,
                 length, temperature, top_k, top_p, repetition_penalty, do_sample, fp16, no_cuda,
                 random_sampling_batch_size, num_beams, dry_run,
                 variation_batch_size, apply_template,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_type = model_type
        self.variation_type = variation_type
        self.output_dir = output_dir
        self.length = length
        self.temperature = temperature
        self.k = top_k
        self.p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.fp16 = fp16
        self.no_cuda = no_cuda
        self.seed = seed
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()
        self.inference_client = InferenceClient(
            base_url="http://localhost:8000/generate"
        )
        set_seed(seed=seed, n_gpu=self.n_gpu)
        self.dry_run = dry_run
        self.apply_template = apply_template
        
        self.use_subcategory = use_subcategory
        if use_subcategory:
            self.subcategory_dict = {}
            self.subcategory_dict['yelp'] = get_subcategories("yelp")
            self.subcategory_dict['pubmed'] = get_subcategories("pubmed")
            self.subcategory_dict['openreview'] = get_subcategories(
                "openreview")

        model_name_or_path = self.model_type

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, device_map="auto", padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if "gpt2" not in self.model_type:
            # use torch.float16 for large LLMs
            try:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)
            except ValueError:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, device_map="cuda", torch_dtype=torch.bfloat16)
        else:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map="auto", pad_token_id=pad_token_id)
            if self.fp16:
                self.model.half()

        self.random_sampling_batch_size = random_sampling_batch_size
        self.variation_batch_size = variation_batch_size

    @staticmethod
    def command_line_parser():
        parser = super(HFAPI, HFAPI).command_line_parser()
        parser.add_argument(
            '--model_type',
            type=str,
            default='gpt2',
            help='Which image feature extractor to use')
        parser.add_argument("--use_subcategory",
                            action="store_true", help="use subcategory")
        parser.add_argument(
            '--variation_type',
            type=str,
            default='rephrase',
            choices=["yelp_rephrase_tone", "openreview_rephrase_tone", "pubmed_rephrase_tone", "cas_paraphrase", 'psytar_rephrase_tone',
                     'hallmarks_of_cancer_rephrase_tone', "mimic_rephrase_tone", "n2c2_2008_rephrase_tone", "danielml_rephrase_tone",
                     "luckycat37_rephrase_tone", "asylex_rephrase_tone", "aaai_rephrase_tone"
                     ],
            help='Which image feature extractor to use')
        parser.add_argument("--mlm_probability", type=float, default=0.5)

        parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
        )
        parser.add_argument("--length", type=int, default=128)
        parser.add_argument("--temperature", type=float, default=1.0,)
        parser.add_argument("--repetition_penalty", type=float, default=1.0,
                            help="primarily useful for CTRL model; in that case, use 1.2")
        parser.add_argument("--top_k", type=int, default=50)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--num_beams", type=int, default=5)
        parser.add_argument("--do_sample", action="store_true",
                            help="sampling when generation")
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument("--dry_run", action="store_true", help="dry run")
        parser.add_argument(
            '--random_sampling_batch_size',
            type=int,
            default=64,
            help='The batch size for random sampling API')
        parser.add_argument(
            '--variation_batch_size',
            type=int,
            default=256,
            help='The batch size for variation API')

        parser.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        )
        parser.add_argument("--no_cuda", action="store_true",
                            help="Avoid using CUDA when available")
        
        parser.add_argument("--apply_template", action="store_true",
                            help="Whether to apply a chat template.")
        return parser

    def text_random_sampling(self, num_samples, prompt_counter=None, lens_dict=None):
        ratio_generation_training = num_samples / sum(prompt_counter.values())
        print("ratio_generation_training: ", ratio_generation_training)
        all_sequences = []
        ppls_cur = []
        additional_info = []
        sync_labels_counter = collections.Counter()

        self.model.eval()

        simulate_num = 0
        for prompt in tqdm(prompt_counter):
            # generation is proportional to the label distributions
            simulate_num_seq_to_generate =  round(
                prompt_counter[prompt] * ratio_generation_training)
            simulate_num += simulate_num_seq_to_generate

        logging.info(
            f"should -- simulated generated sequences: %d", simulate_num)
        all_prefix_prompts = []
        print("length of prompt_counter", len(prompt_counter))
        print("self.use_subcategory: ", self.use_subcategory)
        for prompt in tqdm(prompt_counter):
            # generation is proportional to the label distributions
            num_seq_to_generate = round(
                prompt_counter[prompt] * ratio_generation_training)
            if self.use_subcategory:
                if "yelp" in self.variation_type:
                    category_label = prompt.split(
                        "\t")[0].replace('Business Category: ', '')
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['yelp'][category_label]))
                    keyword = self.subcategory_dict['yelp'][category_label][rand_keyword_idx]
                    full_prompt_text = f'{prompt} with keyword {keyword}'

                elif "openreview" in self.variation_type:
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['openreview']))
                    keyword = self.subcategory_dict['openreview'][rand_keyword_idx]
                    full_prompt_text = f"Suppose that you are a {keyword}. Write a paper review based on " + prompt

                elif "pubmed" in self.variation_type:
                    full_prompt_text = "Using a variety of sentence structures, write an abstract for a medical research paper: "
                elif "asylex" in self.variation_type:
                    full_prompt_text = f"Suppose you are a legal expert. Write a sample legal case for refugee status determination where the verdict is {prompt}"
                elif "aaai" in self.variation_type:
                    full_prompt_text = f"Suppose you are medical expert. Write notes as a doctor would write for a patient where the diagnosis is given as {prompt}"
            else:
                if "cas" in self.variation_type:
                    full_prompt_text = "Écrivez une phrase en français tirée d'un essai clinique. "
                    if prompt == 'NoLabel':
                        full_prompt_text += "Elle ne doit contenir aucune négation ni aucune spéculation"
                    elif prompt == 'negation':
                        full_prompt_text += " Elle doit contenir une négation mais aucune spéculation"
                    elif prompt == 'speculation':
                        full_prompt_text += "Écrivez une phrase en français tirée d'un essai clinique. Elle ne doit contenir aucune négation mais des spéculations"
                    else:
                       full_prompt_text += "Écrivez une phrase en français tirée d'un essai clinique. Elle doit contenir des négations et des spéculations"
                    full_prompt_text += ":"
                elif 'psytar' in self.variation_type:
                    full_prompt_text = get_prompt(prompt, 'psytar')
                elif 'hallmarks_of_cancer' in self.variation_type:
                    full_prompt_text = get_prompt(prompt, 'hallmarks_of_cancer')
                elif 'mimic' in self.variation_type:
                    full_prompt_text = get_prompt(prompt, 'mimic')
                elif 'n2c2_2008' in self.variation_type:
                    full_prompt_text = get_prompt(prompt, 'n2c2_2008')
                elif 'danielml' in self.variation_type:
                    full_prompt_text = get_prompt(prompt, 'danielml')
                elif 'luckycat37' in self.variation_type:
                    full_prompt_text = get_prompt(prompt, 'luckycat37')
                else:
                    raise NotImplementedError(f"Unknown variation type: {self.variation_type}")
            
            if self.apply_template:
                if self.tokenizer.get_chat_template():
                    full_prompt_text = self.tokenizer.apply_chat_template(
                        conversation=[
                            {
                                "role": "system",
                                "content": "You are a helpful text generator. Follow the request precisely."
                                "Please don't add anything to your reply other than the requested text to generate.",
                            },
                            {"role": "user", "content": full_prompt_text},
                            {"role": "assistant", "content": "Here is the requested text:"}
                        ],
                        add_generation_prompt=False,
                        tokenize=False
                    )
                    full_prompt_text = full_prompt_text.rsplit("<|eot_id|>",1)[0]
                else:
                    print("Don't have template for this model!")
            # print("full_prompt_text", full_prompt_text)
            inputs = self.tokenizer(full_prompt_text, return_tensors='pt')
            # print(inputs)
            prompt_input_ids = inputs['input_ids']
            # print(self.tokenizer.batch_decode(prompt_input_ids))
            prompt_attn_mask = inputs['attention_mask']
            before_gen_length = len(full_prompt_text)
            if num_seq_to_generate > 0:
                # condition on the prompt
                sequences = self._generate_text(full_prompt_text, prompt_input_ids, num_seq_to_generate,
                                                max_length=self.length, batch_size=self.random_sampling_batch_size,
                                                before_gen_length=before_gen_length, prompt_attn_mask=prompt_attn_mask)
                all_sequences += sequences
            all_prefix_prompts += [full_prompt_text] * num_seq_to_generate
            additional_info += [prompt] * num_seq_to_generate
            sync_labels_counter[prompt] = num_seq_to_generate

        logging.info(f"Total generated sequences: %d", len(all_sequences))
        torch.cuda.empty_cache()
        return all_sequences,  additional_info, sync_labels_counter, all_prefix_prompts

    def _generate_text(self,full_prompt_text, prompt, seq_num, max_length, batch_size, before_gen_length, prompt_attn_mask=None):

        generated_sequences = []
        if seq_num < batch_size:
            batch_size = seq_num + 1  # TODO: improve
        all_data = []
        num_return_sequences = 2 if batch_size > 1 else 1
        print("seq_num", seq_num)
        print("batch_size", batch_size)
        print("num_return_sequences", num_return_sequences)
        print("max_length", max_length)
        for i in tqdm(range(seq_num // batch_size + 1)):
            if self.dry_run:
                generated_sequences = ["s" * max_length] * batch_size
            else:
                for _ in range(batch_size):
                    output = self.inference_client.text_generation(
                        prompt=full_prompt_text,
                        max_new_tokens=max_length,
                        temperature=self.temperature,
                        top_k=self.k,
                        top_p=self.p,
                        repetition_penalty=self.repetition_penalty,
                        do_sample=self.do_sample,
                        best_of=num_return_sequences,
                        # frequency_penalty=2,
                        details=True,
                    )

                    generated_sequences.append(output.generated_text)

                    
                    for i in output['details']['best_of_sequences']:
                        generated_sequences.append(i['generated_text'])

                
            for g in generated_sequences:
                seq = g
                seq = " ".join(seq.split())
                if seq:
                    all_data.append(seq)

        if len(all_data) > seq_num:
            all_data = random.sample(all_data, seq_num)
        return all_data

    def text_variation(self, sequences, additional_info,
                       num_variations_per_sequence, variation_degree):
        self.model.eval()
        # self.model.to(self.device)
        variations = []
        for idx in tqdm(range(num_variations_per_sequence)):
            sub_variations, var_labels = self._text_variation(
                sequences=sequences,
                labels=list(additional_info),
                variation_degree=variation_degree,
                variation_type=self.variation_type,
                batch_size=self.variation_batch_size)
            variations.append(sub_variations)
        torch.cuda.empty_cache()
        return np.stack(variations, axis=1), var_labels, [], [], []

    def _rephrase(self, label, sequence, variation_type):

        if variation_type == "yelp_rephrase_tone":
            selected_style = ALL_styles[random.randrange(len(ALL_styles))]
            prompt = "Based on {}, please rephrase the following sentences {}:\n{} \n".format(
                label, selected_style, sequence)
        elif variation_type == "asylex_rephrase_tone":
            selected_style = ALL_ASYLEX_styles[random.randrange(len(ALL_ASYLEX_styles))]
            prompt = "The case verdict is {}. Please rephrase the following sentences {}:\n{} \n".format(label, selected_style, sequence)
        elif variation_type == "aaai_rephrase_tone":
            selected_style = ALL_AAAI_styles[random.randrange(len(ALL_AAAI_styles))]
            prompt = "The diagnosis is {}. Please rephrase the following sentences {}:\n{} \n".format(label, selected_style, sequence)
        elif variation_type == 'psytar_rephrase_tone':
            label_map = {
                "ADR": "Adverse Drug Reaction",
                "DI": "Drug Indications",
                "EF": "Drug Effectiveness",
                "INF": "Drug Ineffeciveness",
                "Others": "Others",
                "SSI": "Sign/Symptoms/Illness",
                "WD": "Withdrowal Symptoms"
           }
            selected_style = ALL_styles[random.randrange(len(ALL_styles))]
            prompt = "Mentioning {}, please rephrase the following sentences {}:\n{} \n".format(
                ', '.join(label_map.get(l, 'no speficic condition') for l in label.split("|")), selected_style, sequence)
        
        elif variation_type == "openreview_rephrase_tone":
            selected_style = ALL_OPENREVIEW_styles[random.randrange(
                len(ALL_OPENREVIEW_styles))]
            prompt = "Based on {}, please rephrase the following sentences {} as a paper review:\n{} \n".format(
                label, selected_style, sequence)
        elif variation_type == "pubmed_rephrase_tone" or variation_type == 'hallmarks_of_cancer_rephrase_tone':
            selected_style = ALL_PUBMED_styles[random.randrange(
                len(ALL_PUBMED_styles))]
            prompt = "Please rephrase the following sentences {} as an abstract for medical research paper:\n{} \n".format(
                selected_style, sequence)
        elif variation_type == 'cas_paraphrase':
            prompt = "Veuillez reformuler la phrase suivante en français :\n{} \n".format(sequence)
        # elif variation_type == 'psytar_paraphrase':
        #     prompt = "Veuillez reformuler la phrase suivante en français :\n{} \n".format(sequence)
        elif 'mimic' in variation_type or 'n2c2_2008' in variation_type:
            selected_style = random.choice(ALL_MIMIC_TONES)
            prompt = f"Keeping the information about {label.replace('|', ', ')}, rephrase the note in the following style: {selected_style}.\nNote:\n{sequence}"
        elif 'danielml' in variation_type:
            selected_style = random.choice(ALL_FINANCIAL_styles)
            label_map_prompts = {
                "positive": f"Rephrase the following financial statement to emphasize growth, opportunity, or favorable outcomes, in the following style: {selected_style}.\nNote:\n{sequence}",
                "negative": f"Rephrase the following financial statement to highlight risks, setbacks, or challenges, in the following style: {selected_style}.\nNote:\n{sequence}",
                "neutral": f"Rephrase the following financial statement with an objective, professional tone, in the following style: {selected_style}.\nNote:\n{sequence}"}       
            sentiment = label
            prompt = label_map_prompts.get(sentiment, f"Rephrase the following financial statement in a professional tone, using the style: {selected_style}.\nNote:\n{sequence}")
        elif 'luckycat37' in variation_type:
            sentiment = label
            selected_style = random.choice(ALL_FINANCIAL_luckycat_styles)
            label_map_prompts = {
        "positive": f"Rephrase the following financial statement to emphasize growth, opportunity, or favorable outcomes, in the following style: {selected_style}.\nNote:\n{sequence}",
        "negative": f"Rephrase the following financial statement to highlight risks, setbacks, or challenges, in the following style: {selected_style}.\nNote:\n{sequence}",
        "neutral":  f"Rephrase the following financial statement with an objective, professional tone, in the following style: {selected_style}.\nNote:\n{sequence}"
    }
            prompt = label_map_prompts.get(sentiment, f"Rephrase the following financial statement in a professional tone, using the style: {selected_style}.\nNote:\n{sequence}")

        return prompt

    def _text_variation(self, sequences, labels, variation_degree, variation_type, batch_size):
        if self.dry_run:
            all_data = [seq+"s"*self.length for seq in sequences]
            all_labels = [lab for lab in labels]
            return all_data, all_labels

        num_seq = len(sequences)
        all_data = []
        all_labels = []

        self.model.eval()
        self.mlm_probability = variation_degree

        for i in tqdm(range(num_seq // batch_size + 1)):
            start_idx = i*batch_size
            if start_idx >= num_seq:
                break
            end_idx = num_seq if (
                i+1)*batch_size > num_seq else (i+1)*batch_size

            batch_prompt = []
            batch_labels = []
            for idx in range(start_idx, end_idx):
                prompt = self._rephrase(
                    labels[idx], sequences[idx], variation_type)
                
                if self.apply_template:
                    prompt = self.tokenizer.apply_chat_template(
                        conversation=[
                            {
                                "role": "system",
                                "content": "You are a helpful text rephraser. Follow the request precisely."
                                "Please don't include anything in your reply other than the requested text to rephrase.",
                            },
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": "Here is the rewritten text:"}
                        ],
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    prompt = prompt.rsplit("<|eot_id|>",1)[0]
                batch_prompt.append(prompt)
                batch_labels.append(labels[idx])
            generated_sequences = []
            for prompt, label in zip(batch_prompt, batch_labels):
                try_count = 0
                success_flag = False
                while try_count < 10:
                    output = self.inference_client.text_generation(
                        prompt=prompt,
                        max_new_tokens=self.length,
                        temperature=self.temperature,
                        top_k=self.k,
                        top_p=self.p,
                        repetition_penalty=self.repetition_penalty,
                        do_sample=self.do_sample,
                        # frequency_penalty=2,
                    )
                    seq = " ".join(output.split())
                    if seq:
                        all_data.append(seq)
                        lab = label.strip().split("\t")
                        all_labels.append(lab)
                        success_flag = True
                        break
                    else:
                        print("No valid sequence generated, retrying...")
                        try_count += 1
                
                if not success_flag:
                    processed_prompt = batch_prompt[idx].split("<|start_header_id|>user<|end_header_id|>\n\n")[1].split("<|eot_id|>")[0]
                    all_data.append(processed_prompt)
                    print("No valid sequence generated, using the prompt as is." , processed_prompt)
                    all_labels.append(batch_labels[idx])
            # for idx in range(len(generated_sequences)):
            #     seq = generated_sequences[idx]
            #     seq = " ".join(seq.split())
            #     lab = batch_labels[idx].strip().split("\t")
            #     if seq:
            #         all_data.append(seq)  # no lables!
            #     else:
            #         all_data.append(batch_prompt[idx])
            #     all_labels.append(lab)

        logging.info(f" _text_variation output lens  {len(all_data)}")

        return all_data, all_labels
