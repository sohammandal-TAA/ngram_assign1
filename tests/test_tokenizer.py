from tokenizer import Tokenizer


def run_test(text, tokenizer):
    config = tokenizer.explain()
    print("Tokenizer config:", config)
    print("Original text   :", text)
    print("Tokenized output:", tokenizer.tokenize(text))
    print("-" * 50)


if __name__ == "__main__":

    print("=== Test 1 ===")
    t = Tokenizer()
    run_test("Hello, world! It's 2024.", t)

    print("=== Test 2 ===")
    t = Tokenizer(remove_numbers=True)
    run_test("I have 2 dogs and 3 cats", t)

    print("=== Test 3 ===")
    t = Tokenizer(lowercase=False)
    run_test("Hello World", t)

    print("=== Test 4 ===")
    t = Tokenizer(remove_apostrophes=False)
    run_test("Don't stop believing", t)

    print("=== Test 5 ===")
    t = Tokenizer(remove_punctuation=False)
    run_test("Hello, world!", t)

    print("=== Test 6 ===")
    t = Tokenizer()
    run_test("Email me @ test@example.com!", t)

    print("=== Test 7 ===")
    t = Tokenizer()
    run_test("Hello     world\n\nNLP", t)

    print("=== Test 8 ===")
    t = Tokenizer()
    run_test("", t)

    print("=== Test 9: tokenize_sentences ===")
    t = Tokenizer()
    sentences = [
        "Hello World!",
        "NLP is fun.",
        "Let's tokenize text."
    ]

    print("Tokenizer config:", t.explain())
    for s, tokens in zip(sentences, t.tokenize_sentences(sentences)):
        print("Original text   :", s)
        print("Tokenized output:", tokens)
        print("-" * 50)

    print("=== Test 10 ===")
    t = Tokenizer(lowercase=False, remove_numbers=True, remove_apostrophes=False)
    run_test("Testing 123 tokenizer's CONFIG!", t)
