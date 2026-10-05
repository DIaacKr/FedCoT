NORMAL_ONE_SHOT_COT_PROMPT = '''
To answer the following question, provide a highly detailed and comprehensive step-by-step reasoning.
Your explanation should break down the problem into clear, logical steps, with each step on a new line.
The goal is to construct a robust and self-contained argument that fully justifies the final conclusion.
After the reasoning, present the final answer. Please model the overall structure on the provided example.

Example:
Q: 
Primary Colonization of plaque is dominated by 
A. Facultative Aerobic Gram+ve rods
B. Facultative Anaerobic Gram-ve rods
C. Facultative Aerobic Gram-ve cocci
D. Facultative Anaerobic Gram+ve cocci

A:
The oral cavity is a complex ecosystem where various microorganisms coexist, and plaque is a biofilm that forms on tooth surfaces.
Primary colonization of plaque is dominated by Streptococcus species, which are Gram-positive, facultative anaerobic cocci.
These Streptococcus species are able to adhere to the tooth surface and multiply, creating a matrix for other microorganisms to colonize.
#### D. Facultative Anaerobic Gram+ve cocci

Here is the question:
'''

def load_prompts(dataset_name=None): 

    return NORMAL_ONE_SHOT_COT_PROMPT
    

