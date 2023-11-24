
import pdfplumber
from transformers import BertTokenizer, T5Tokenizer, T5ForConditionalGeneration
from collections import Counter
import random

def extract_text_from_pdf(pdf_file_path):
    
    with pdfplumber.open(pdf_file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def tokenize_text(text):
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    return tokens

def generate_distractors(text, target_entity, num_distractors=2):
    unique_words = [word.lower() for word in text.split()]
    unique_words = list(set(unique_words))
    unique_words.remove(target_entity.lower())  
    distractor_list = random.sample(unique_words, num_distractors)
    return distractor_list

def generate_mcq_questions(text, keywords, num_questions=5):
    
    t5_model_name = "t5-small"
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

    
    mcqs = []

    for keyword in keywords[:num_questions]:
        question_prompt = f"What is the role of {keyword} in the context of the text?"
        input_text = f"Generate questions: {question_prompt} {text}"
        input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512)

        question = t5_model.generate(input_ids, max_length=100, no_repeat_ngram_size=2, num_return_sequences=1)
        question_text = t5_tokenizer.decode(question[0], skip_special_tokens=True)

        correct_answer_1 = f"{keyword} "
        correct_answer_2 = f"{keyword} "

        num_incorrect_choices = 2
        distractors = generate_distractors(text, keyword, num_incorrect_choices)

        while keyword in distractors:
            random.shuffle(distractors)

        incorrect_choices = random.sample(distractors, 2)

        answer_choices = [correct_answer_1, correct_answer_2, *incorrect_choices]
        random.shuffle(answer_choices)

        mcq = {
            "question": question_text,
            "answer_choices": answer_choices
        }

        mcqs.append(mcq)

    return mcqs


pdf_file_path = "/content/chapter-2.pdf"
text = extract_text_from_pdf(pdf_file_path)
tokens = tokenize_text(text)


keywords = ["Aurangzeb", "Mughal", "rulers", "authority", "kingdoms"]
mcq_questions = generate_mcq_questions(text, keywords, num_questions=5)

for i, mcq in enumerate(mcq_questions, start=1):
    print(f"MCQ {i}:")
    print(f"Question: {mcq['question']}")
    print("Answer Choices:")
    for j, choice in enumerate(mcq['answer_choices'], start=1):
        print(f"{j}. {choice}")
    print("\n")
