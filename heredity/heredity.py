import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint_prob = 1.0

    for person in people:
        # Determine how many copies of the gene the person has in this scenario
        if person in two_genes:
            target_genes = 2
        elif person in one_gene:
            target_genes = 1
        else:
            target_genes = 0

        # Determine if the person has the trait in this scenario
        target_trait = person in have_trait

        # Case A: Person has no parents listed (use unconditional distribution)
        if people[person]["mother"] is None:
            gene_prob = PROBS["gene"][target_genes]

        # Case B: Person has parents (calculate conditional distribution)
        else:
            mother = people[person]["mother"]
            father = people[person]["father"]

            # Helper to calculate probability that a specific parent passes on the gene
            parent_pass_probs = {}
            for p_name in [mother, father]:
                if p_name in two_genes:
                    # Has 2 copies, passes it unless it mutates
                    parent_pass_probs[p_name] = 1 - PROBS["mutation"]
                elif p_name in one_gene:
                    # Has 1 copy, exactly 50% chance
                    parent_pass_probs[p_name] = 0.5
                else:
                    # Has 0 copies, can only pass it if it mutates
                    parent_pass_probs[p_name] = PROBS["mutation"]

            # Combine parent probabilities based on how many genes the child needs
            if target_genes == 2:
                # Must get it from both mother AND father
                gene_prob = parent_pass_probs[mother] * parent_pass_probs[father]
            elif target_genes == 1:
                # From mother but NOT father, OR from father but NOT mother
                gene_prob = (parent_pass_probs[mother] * (1 - parent_pass_probs[father])) + \
                            ((1 - parent_pass_probs[mother]) * parent_pass_probs[father])
            else:
                # Must NOT get it from mother AND NOT get it from father
                gene_prob = (1 - parent_pass_probs[mother]) * (1 - parent_pass_probs[father])

        # Multiply by the probability that they show/don't show the trait given their genes
        trait_prob = PROBS["trait"][target_genes][target_trait]

        # Accumulate into the joint probability product
        joint_prob *= gene_prob * trait_prob

    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        # Find which category the person falls into for this specific joint scenario
        if person in two_genes:
            gene_count = 2
        elif person in one_gene:
            gene_count = 1
        else:
            gene_count = 0

        trait_status = person in have_trait

        # Add the joint probability 'p' to their running totals
        probabilities[person]["gene"][gene_count] += p
        probabilities[person]["trait"][trait_status] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        # Normalize Gene probabilities
        gene_total = sum(probabilities[person]["gene"].values())
        if gene_total > 0:
            for gene in probabilities[person]["gene"]:
                probabilities[person]["gene"][gene] /= gene_total

        # Normalize Trait probabilities
        trait_total = sum(probabilities[person]["trait"].values())
        if trait_total > 0:
            for trait in probabilities[person]["trait"]:
                probabilities[person]["trait"][trait] /= trait_total


if __name__ == "__main__":
    main()
