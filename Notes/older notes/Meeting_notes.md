

# Latest Meeting Notes

TO-DOs
(High priority) Finish training runs of RPE on the reverse string on Qwen2.5-7B using composable CoT repo
Right now the training loss does not drop much, need to debug
Increase LoRA Rank (8 -> 16/32)
Try different learning rates
If none works, try Qwen2.5-0.5B
See if we can do full fine-tuning with -0.5B on the current machine
If the results look stable, we can move on
If not, we can train a harder version of reverse string where arbitrary letters, instead of binary, are used
Compare RPE with baseline encoding on composable CoT task
Start with letter concatenation (atomic) and next letter (atomic)
Data: https://github.com/fc2869/composable_cot/tree/main/data/composition/composable_cot/letter_concat_next_last_letter_composable_cot 
Baseline: just run composable cot with regular positional encoding
RPE variants
(Medium Priority) Vanilla RPE: Run the same experiment as we did on reverse string
Apply RPE to everything (instruction, output)
V2: RPE on output only
Intuition: we want the CoT to be length generalized
V3: RPE on prefix only
Given a training example of:
instruction <prefix> random </prefix> <suffix> real CoT </suffix>
(low priority) RPE on prefix only and the prefix is empty
Given a training example of:
instruction <prefix>              </prefix> <suffix> real CoT </suffix>

Does the CCoT <prefix>/<suffix> format fundamentally change RPE dynamics compared to Phase 1's raw format? (CoT traces give the model more "reasoning space" which might interact with positional encoding differently)

Does LoRA fine-tuning (updating only ~0.1% of params) interact differently with RPE vs full fine-tuning or from-scratch training? The pretrained RoPE embeddings are frozen during LoRA.

Should we also try full fine-tuning (no LoRA) as an additional condition?


# some older notes

1/19
'https://arxiv.org/pdf/2405.14722
1/13 - Recap and Next Steps for Spring 26

Manas: To discuss 
Possible Improvements for Composable CoT that might translate to GCCoT
Question: Have we tried different variants of prefixes that are more natural language-y?
Yes, but we will revisit them for TravelPlanner
Or we can have different types of prefixes in the same atomic CoT datasets
Question: Can we change the training loss during training?
Possible changes
Different SFT losses for tags (format) and atomic CoT (task completion)
Different rewards during RL based on the nature of the prefix  
Question: What is the number of positions to shift when changing the position encodings?
Depending on the distribution of the atomic CoT of the dataset of interest
For CCoT, we use random letters of length 50 - 100 (roughly 0.5n to 1.5n, where n is the length of the atomic CoT ground truth trace)
Question: Have we tried shifting the positional encoding of the instructions as well?
We can try that as well; we were thinking about only shifting the encodings of the atomic CoT, and keeping the instructions unchanged.
CCoT code running, codebase understood (for the most part)


12/22 - Project Kickoff Meeting
TO-DOs and Next Steps
Fangcong
Find some papers related to positional encodings for generalization if possible
Manas
Try to run the codebase of composable cot
Do a literature review on positional encodings for generalization (length, compositional, any kind)
(Optional) Try to implement randomized positional encoding from Ruoss et al., 2023 for decoder-only models
Take a look at the official implementation of this paper: https://github.com/google-deepmind/randomized_positional_encodings/blob/main/models/transformer.py
Ask claude/cursor/copilot to see if they can directly migrate the codebase for decoder-only models

12/02 - Project Kickoff Meeting
Status Quo: Generalized Composable CoT
https://docs.google.com/document/d/1-fcQrZ7zd8mhkf-FklikeWQU2jNuBjYRaCOco81LDIA/edit?tab=t.0#heading=h.2x8na57nbw8x 
Ongoing Direction 1: Synthesize Training Data for GCCoT
Problem Statement
TravelPlanner has very limited training data
TravelPlanner only has 45 training examples; not enough for SFT
https://huggingface.co/datasets/osunlp/TravelPlanner/viewer/train 
All existing training examples have hand-annotated gold plans by human annotators; not easy to scale up 
Each train example only has one gold plan; there is a lack of diversity in the training data.
Goals
Scale: Scaling up the training data for TravelPlanner to 100 - 500 examples.
The new training examples do not need to be hand-annotated, but need to be verified to be correct based on: (1) the constraints in the query; (2) the background information provided by the tools.
Diversity: Increase the diversity of gold plans in the training data.
Ideally, some training examples should have more than one gold plans.
Actional Items
Read the travelplanner paper (https://arxiv.org/pdf/2402.01622) and get familiar with their data construction process
Also get familiar with their codebase: https://github.com/OSU-NLP-Group/TravelPlanner 
Explore synthetic data generation methods by prompting GPT/Claude models to generate more training data. Example methods include:
Few-shot prompting: Put the existing training data (input query, additional information, and gold plans) in the context as examples and prompt LLMs to generate similar gold plans for the same query/generate similar query and plans.
Distillation: Use the same query and additional information to prompt LLMs and get multiple samples, and keep the correct LLM-generated plans as the gold plans of the new training examples.
Ongoing Direction 2: Explore Positional Encodings for Composable CoT
Problem Statement
In the ComposableCoT paper, we found that random prefixes work surprisingly well to teach LLMs to compose CoTs at inference time.
If so, can we achieve the same effect by changing the positional encodings of LLMs instead of training data augmentation?
Idea
During training, we use the atomic CoT data without any augmentation.
However, we modify the positional encodings of the atomic CoT tokens by randomly shifting them to simulate their positions in a longer, compositional CoT.
Actional Items
Read papers on changing positional encodings to improve length generalization for LLMs. For example:
https://aclanthology.org/2023.acl-short.161.pdf
https://arxiv.org/abs/2305.19466 
Implement some of the ideas of changing positional encodings on the atomic CoT data of TravelPlanner.
TO-DOs and Next Steps
Fangcong
Clean up the codebase for Composable CoT and share with Manas
Manas
Read the TravelPlanner paper: https://arxiv.org/pdf/2402.01622 
Determine which direction to work on
