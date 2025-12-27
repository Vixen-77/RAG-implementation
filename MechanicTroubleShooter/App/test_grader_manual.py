from langchain_core.documents import Document
from Services.grader import grade_hallucination, grade_answer_question

def test_hallucination():
    print("\n--- Testing Hallucination Grader ---")
    
    docs = [
        Document(page_content="The Dacia Duster requires 5W-30 oil for the 1.5 dCi engine."),
        Document(page_content="The tire pressure should be 2.2 bar for front wheels.")
    ]
    
    # 1. Grounded Generation
    gen_good = "You should use 5W-30 oil for the 1.5 dCi engine."
    result = grade_hallucination(docs, gen_good)
    print(f"Good Gen Result: {result['grounded']} (Expected: True)")
    print(f"Reasoning: {result['reasoning']}")
    
    # 2. Hallucinated Generation
    gen_bad = "The Dacia Duster has a V8 engine."
    result = grade_hallucination(docs, gen_bad)
    print(f"Bad Gen Result: {result['grounded']} (Expected: False)")
    print(f"Reasoning: {result['reasoning']}")

def test_answer_question():
    print("\n--- Testing Answer Relevance Grader ---")
    
    question = "What oil should I use?"
    
    # 1. Useful Answer
    ans_good = "You should use 5W-30 oil."
    result = grade_answer_question(question, ans_good)
    print(f"Good Answer Result: {result['useful']} (Expected: True)")
    print(f"Reasoning: {result['reasoning']}")
    
    # 2. Useless Answer
    ans_bad = "The sky is blue."
    result = grade_answer_question(question, ans_bad)
    print(f"Bad Answer Result: {result['useful']} (Expected: False)")
    print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    test_hallucination()
    test_answer_question()
