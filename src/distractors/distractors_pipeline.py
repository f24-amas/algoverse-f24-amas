import os
from openai import OpenAI
import json
import re
os.environ["OPENAI_API_KEY"] = "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"


# Define the OpenAI API function
def prompt_distractors(
        context,
        question,
        question_type,
        answer,
        client,
        action = "Update the question by adding new information or distractors such that the answer changes",
        model_params = {'model': 'gpt-4', 'temperature': 0.7, 'max_tokens': 200},
    ) -> str:
    '''
    This function takes in a context, question, and answer, and returns a new question with distractors added to it.
    The function uses the OpenAI API to generate a new question based on the provided context, question, and answer.
    The new question is then returned as a string.

    Parameters
    ----------
    context : str
        The context of the question.
    question : str
        The original question.
    answer : str
        The original answer.

    Returns
    -------
    str
        The new question with distractors added.
    '''
    prompt = f"""
    Context: {context}
    Original Question: {question}
    Original Answer: {answer}
    Answer Type: {question_type}

    Task: {action}. Use chain-of-thought reasoning to explain how you are modifying the question. 
    At the end, clearly provide:
    - Final Updated Question: [Your updated question here]
    - Final Updated Answer: [Your updated answer here]

    Begin reasoning below, lets think step by step but also one by one:
    """
    try:
        response = OpenAI().chat.completions.create(
            model= model_params["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature= model_params["temperature"],
            max_tokens= model_params["max_tokens"] if "max_tokens" in model_params else None,
        )
        # Return the full response
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def parse_chatgpt_response(response) -> tuple[str, str]:
    '''
    This function takes in a response from the OpenAI API and parses it to extract the updated question and answer.
    It uses regular expressions to identify the updated question and answer.
    :param response:  The response from the OpenAI API.
    :return:  The updated question and answer, or None if not found.
    '''
    # Regular expressions to capture the final updated question and answer
    print(response)
    question_pattern = r"Final Updated Question:\s*(.*)"
    answer_pattern = r"Final Updated Answer:\s*(.*)"

    # Extract using regex
    updated_question = re.search(question_pattern, response).group(1).strip() if re.search(question_pattern, response) else None
    updated_answer = re.search(answer_pattern, response).group(1).strip() if re.search(answer_pattern, response) else None

    # Return parsed values or None if not found
    return updated_question, updated_answer
