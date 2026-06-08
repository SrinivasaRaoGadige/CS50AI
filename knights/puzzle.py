from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# General structure rules helper to keep code clean and readable
# A person is a Knight if and only if they are NOT a Knave.
base_rules = And(
    Biconditional(AKnight, Not(AKnave)),
    Biconditional(BKnight, Not(BKnave)),
    Biconditional(CKnight, Not(CKnave))
)

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    base_rules,
    # If A is a Knight, their statement is True; if Knave, it's False.
    Biconditional(AKnight, And(AKnight, AKnave))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    base_rules,
    # A's statement translates to: both A and B are Knaves
    Biconditional(AKnight, And(AKnave, BKnave))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    base_rules,
    # A's statement: either both are Knights or both are Knaves
    Biconditional(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    # B's statement: either (A=Knight and B=Knave) or (A=Knave and B=Knight)
    Biconditional(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight)))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave.'"
# B then says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    base_rules,

    # "A says either... but you don't know which"
    # If A is a Knight, what he said is true. If he is a Knave, what he said is a lie.
    # So A's statement context boils down to:
    # Biconditional(AKnight, AKnight) OR Biconditional(AKnight, AKnave)
    Or(
        Biconditional(AKnight, AKnight),
        Biconditional(AKnight, AKnave)
    ),

    # B says: "A said 'I am a knave.'"
    # This means B is claiming that the relationship "Biconditional(AKnight, AKnave)" is true.
    Biconditional(BKnight, Biconditional(AKnight, AKnave)),

    # B says "C is a knave."
    Biconditional(BKnight, CKnave),

    # C says "A is a knight."
    Biconditional(CKnight, AKnight)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
