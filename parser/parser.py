import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# Holmes sat.
# Holmes lit a pipe.
# We arrived the day before Thursday.
# Holmes sat in the red armchair and he chuckled.
# My companion smiled an enigmatical smile.
# Holmes chuckled to himself.
# She never said a word until we were at the door here.
# Holmes sat down and lit his pipe.
# I had a country walk on Thursday and came home in a dreadful mess.
# I had a little moist red paint in the palm of my hand.

# Ironically, these grammar rules were generated via LLM
NONTERMINALS = """
S -> NP VP | NP VP Conj VP | NP VP Adv | NP VP Conj NP VP | NP VP Adv Adv
S -> NP VP Adv NP | NP VP P NP
S -> NP VP Det NP | Det NP VP | Det NP VP Det NP | Det NP VP P NP | Det NP VP Det NP P NP

NP -> N | Det NP | Adj NP | Det Adj NP | NP Conj NP | NP P NP | NP Adv | NP Adv Adv
NP -> NP P NP | NP Adv P NP

VP -> V | V NP | V NP NP | V NP Adv | V NP P NP | V Adv | V Adv Adv
VP -> VP P NP | VP Adv | VP Adv Adv

Adj -> Adj Adj | Adj Conj Adj | Adj Adv | Adj Adv Adv
Adv -> Adv Adv | Adv Conj Adv
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    s = sentence.lower()
    punctuation = ",;:.!?"
    for p in punctuation:
        s = s.replace(p, '')

    tokens = nltk.word_tokenize(s, language='english', preserve_line=True)
    words = [token for token in tokens if token.isalpha()]
    return words


def has_inner_np(tree):
    for child in tree:
        if isinstance(child, nltk.Tree):
            if child.label() == "NP":
                return True
            if has_inner_np(child):
                return True
    return False


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np_chunks = []

    # find subtrees of trees without any more np labels
    for s in tree.subtrees():
        # print(s)
        if (s.label() == "NP" and not has_inner_np(s)):
            np_chunks.append(s)

    return np_chunks


if __name__ == "__main__":
    main()


# python3 parser.py sentences/1.txt
