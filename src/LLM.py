import os
import yaml
import re
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI

class LLMClient:
    def __init__(self):
        self.api_key = os.environ["MISTRAL_API_KEY"]
        self.model = "mistral-large-latest"
        self.llm = ChatMistralAI(model=self.model, temperature=0, api_key=self.api_key)
        with open('src/prompts.yaml', 'r') as file:
            self.prompts = yaml.safe_load(file)
    
    def call_model_api(self, input_messages):
        output = self.llm.invoke(input_messages)
        return output

    def generate_next_prompts(self, original_prompt):
        prompt = [
            (
                "system",
                self.prompts["generate_next_prompts"]
            ),
            (
                "human",
                original_prompt
            )
        ]
        print(prompt)
        return self.parse_prompts(self.call_model_api(prompt).content)

    def parse_prompts(self, prompts_text):
        
        prompt_pattern = re.compile(r"- Prompt #\d+: (.+)")
        
        parsed_prompts = [match.group(1) for match in prompt_pattern.finditer(prompts_text)]
        
        return parsed_prompts
    
    def parse_subprompts(self, prompts_text):
        print(f"Raw Prompts Text: {prompts_text}")
        prompt_pattern = re.compile(r"- Subprompt #\d+: (.+)")
        
        parsed_prompts = [match.group(1) for match in prompt_pattern.finditer(prompts_text)]
        
        return parsed_prompts

    def generate_subprompts(self, prompt_list, total_subprompts):
        base_prompts_str = "\n  ".join([f"- {prompt}" for prompt in prompt_list])
    
        prompt_template = self.prompts["generate_subprompts"]

        prompt = prompt_template.format(
            total_subprompts=total_subprompts
        )

        prompt = [
            (
                "system",
                prompt
            ),
            (
                "human",
                base_prompts_str
            )
        ]
        print(self.call_model_api(prompt).content)
        return self.parse_subprompts(self.call_model_api(prompt).content)
    def write_subprompts_to_file(self, subprompt_list, filename="subprompts.txt"):
        # Writing the subprompts into a file
        with open(filename, 'w') as file:
            for idx, subprompt in enumerate(subprompt_list, 1):
                file.write(f"Subprompt #{idx}: {subprompt}\n")
        print(f"Subprompts saved in {filename}")
    def load_subprompts_from_file(self, filename="subprompts.txt"):
        # Check if the file exists
        if not os.path.exists(filename):
            print(f"File {filename} not found!")
            return None

        # Reading the subprompts from the file
        with open(filename, 'r') as file:
            file_content = file.read()

        # Parse the subprompts from the file content
        subprompt_pattern = re.compile(r"Subprompt #\d+: (.+)")
        loaded_subprompts = [match.group(1) for match in subprompt_pattern.finditer(file_content)]
        
        return loaded_subprompts
    def generate_or_load_subprompts(self, prompt, total_subprompts, filename_prompt="prompts/prompts.txt",filename_subprompt="prompts/subprompts.txt"):
        # Try to load the subprompts from the file first
        prompts=self.load_subprompts_from_file(filename_prompt)
        subprompts = self.load_subprompts_from_file(filename_subprompt)
        
        if subprompts is not None:
            print(f"Loaded subprompts from {filename_subprompt}.")
            return prompts,subprompts
        
        # If the file doesn't exist or couldn't be loaded, generate and save the subprompts
        print(f"Generating and saving subprompts to {filename_subprompt}.")
        prompt_list = self.generate_next_prompts(prompt)
        self.write_subprompts_to_file(prompt_list, filename_prompt)
        subprompt_list = self.generate_subprompts(prompt_list, total_subprompts)
        self.write_subprompts_to_file(subprompt_list, filename_subprompt)
        return prompt_list,subprompt_list

if __name__=="__main__":
    client = LLMClient()
    prompt_list = client.generate_next_prompts("A man walking in the street")
    print("GENERATED PROMPT LIST:")
    print(prompt_list)
    subprompt_list = client.generate_subprompts(prompt_list, 4*len(prompt_list))
    print("GENERATED SUBPROMPT LIST:")
    print(subprompt_list)
    
