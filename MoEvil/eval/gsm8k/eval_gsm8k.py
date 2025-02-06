import argparse
import json
import re
from fraction import Fraction
import sys
from grader import math_equal
from tqdm import tqdm
import os

from util import last_boxed_only_string


ROOT = os.path.dirname(os.path.abspath(__file__))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def extract_answer_number(completion, direct_answer_trigger):
    # text = completion.split('The answer is: ')
    # text = completion.split('#### ')
    text = completion.split(direct_answer_trigger)
    if len(text) > 1:
        extract_ans = text[-1].strip().rstrip('.')
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if not match:
            extract_ans = text[-2].strip().rstrip('.')
            match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        # return None
        value = remove_boxed(last_boxed_only_string(completion))
        if value:
            value = value.replace(',', '')
            if is_number(value):
                return round(float(value))
            else:
                return None
        else:
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    direct_answer_trigger = 'The final answer is '

    with open(os.path.join(args.output_dir, 'generated_outputs.json'), 'r') as f:
        generated_outputs = json.load(f)
    
    invalid_outputs = []
    outputs = []
    result = []
    for idx, generated_output in tqdm(enumerate(generated_outputs)):
        prompt = generated_output['prompt']
        completion = generated_output['completion']
        prompt_answer = generated_output['answer']
        
        doc = {'question': prompt}
        y_pred = extract_answer_number(completion, direct_answer_trigger)
        outputs.append({'question': prompt, 'output': completion, 'answer': prompt_answer, 'y_pred': y_pred})
        if y_pred != None:
            result.append(float(y_pred) == float(prompt_answer) or math_equal(y_pred, prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f'{args.output_dir}/outputs.json', 'w') as f:
        json.dump(outputs, f)
    with open(f'{args.output_dir}/eval_results.json', 'w') as f:
        json.dump({'acc': acc}, f)

if __name__ == "__main__":    
    main()
