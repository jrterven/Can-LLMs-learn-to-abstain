SYSTEM = "You answer factual questions. Return only one short answer, or exactly IDK when permitted. Do not explain or list guesses."


def messages(question, tau=None, forced=False):
    if forced:
        instruction = "Give your best short answer, even if uncertain. Do not abstain."
        examples = [("What is the capital of France?", "Paris"),
                    ("What is the chemical symbol for gold?", "Au")]
    else:
        instruction = (f"Answer only if your probability of being correct is greater than {tau:.2f}. "
                       "Otherwise return exactly IDK. Use this confidence threshold for the final question.")
        examples = [("What is the capital of France?", "Paris"),
                    ("What exact integer did an unspecified person privately choose yesterday?", "IDK")]
    result = [{"role": "system", "content": SYSTEM + " " + instruction}]
    for q, a in examples:
        result.extend([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
    result.append({"role": "user", "content": question})
    return result


def confidence_messages(question, answer):
    return [{"role": "system", "content": "Judge whether the proposed answer is factually correct for the question. Reply only True or False."},
            {"role": "user", "content": f"Question: {question}\nProposed answer: {answer}"}]
