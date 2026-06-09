import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_dist = {}
    num_pages = len(corpus)
    links = corpus[page]

    # If the page has no outgoing links, it acts as if it links to all pages
    if not links:
        for p in corpus:
            prob_dist[p] = 1 / num_pages
        return prob_dist

    # Baseline probability for any page (random choice)
    random_prob = (1 - damping_factor) / num_pages

    # Additional probability split among specific links
    link_prob = damping_factor / len(links)

    for p in corpus:
        prob_dist[p] = random_prob
        if p in links:
            prob_dist[p] += link_prob

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Track the count of visits per page
    page_counts = {page: 0 for page in corpus}

    # The first sample is purely random
    current_page = random.choice(list(corpus.keys()))
    page_counts[current_page] += 1

    # Generate the remaining n-1 samples
    for _ in range(n - 1):
        probs = transition_model(corpus, current_page, damping_factor)

        # Unzip keys and probabilities to pass to random.choices
        pages = list(probs.keys())
        weights = list(probs.values())

        # Choose the next page based on the transition model weights
        current_page = random.choices(pages, weights=weights, k=1)[0]
        page_counts[current_page] += 1

    # Convert counts to proportions (estimated PageRank)
    pagerank = {page: count / n for page, count in page_counts.items()}
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    # Step 1: Assign each page an initial rank of 1 / N
    pagerank = {page: 1 / num_pages for page in corpus}

    while True:
        new_pagerank = {}

        for page in corpus:
            # Baseline probability
            total = (1 - damping_factor) / num_pages

            # Sum up link contributions from other pages
            for potential_linker in corpus:
                links = corpus[potential_linker]

                # Condition A: The page has links and points directly to our target page
                if page in links:
                    total += damping_factor * (pagerank[potential_linker] / len(links))
                # Condition B: The page has NO links (acts as linking to all pages)
                elif not links:
                    total += damping_factor * (pagerank[potential_linker] / num_pages)

            new_rank = total
            new_pagerank[page] = new_rank

        # Step 2: Check convergence across all pages
        converged = True
        for page in corpus:
            if abs(new_pagerank[page] - pagerank[page]) > 0.001:
                converged = False
                break

        # Update the dictionary for the next iteration step
        pagerank = new_pagerank

        if converged:
            break

    # Normalize values to perfectly sum to 1 before returning
    total_sum = sum(pagerank.values())
    return {page: rank / total_sum for page, rank in pagerank.items()}


if __name__ == "__main__":
    main()
